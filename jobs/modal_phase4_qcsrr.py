"""Bounded Modal execution for the approved 12-video Phase 4 QCSRR run."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from fractions import Fraction
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import resource
import tarfile
import tempfile
import time
from typing import Any, Mapping
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN_RELATIVE = "runs/phase4_qcsrr"
PLAN_RELATIVE = f"{RUN_RELATIVE}/execution_plan.json"
SOURCE_MANIFEST_RELATIVE = f"{RUN_RELATIVE}/source_manifest.json"
MODEL_DIR = "/root/siglip2"
MAX_CALLS = 36
MAX_ATTEMPTS = 41
MAX_RETRIES = 5
HISTORICAL_ATTEMPTS = 41
HISTORICAL_CONTINUATION_ERROR_RECORDS = 31
SOFT_STOP_SECONDS = 430.0
FUNCTION_TIMEOUT_SECONDS = 460
RESUME_SOFT_STOP_SECONDS = 840.0
RESUME_FUNCTION_TIMEOUT_SECONDS = 900
RESULTS_RELATIVE = "runs/phase4_qcsrr/results_sparse_masks"
CONTINUED_RESULTS_RELATIVE = "runs/phase4_qcsrr/results_continued"
CONTINUATION_CACHE_RELATIVE = f"{CONTINUED_RESULTS_RELATIVE}/cache"
POSTPROCESS_RESULTS_RELATIVE = f"{CONTINUED_RESULTS_RELATIVE}/postprocess"
RESUME_RESULTS_RELATIVE = "runs/phase4_qcsrr/results_resumed"
RESUME_JOURNAL_RELATIVE = f"{RUN_RELATIVE}/resume_journal.json"
RESUME_VOLUME_PATH = "/root/reva-phase4-qcsrr-journal"
RESUME_CREATE_MIN_INTERVAL_SECONDS = 11.0
RESUME_MAX_CREATE_ATTEMPTS_PER_CALL = 3
RESUME_MAX_RETRIES = 5
RESUME_MAX_PRE_DEDUP_ROWS = 10000
RESUME_HARD_WALL_MINUTES = 15
REPLICATE_SDK_VERSION = "1.0.7"
RECOVERY_RELATIVE = "runs/phase4_qcsrr/diagnostics/replicate_prediction_recovery"
RECOVERY_CALL0 = (
    "prediction-6b4y2255rhrmt0d19sybv125qm.output.bin",
    "prediction-ghznf33qahrmy0d19syrv6213r.output.bin",
)
RECOVERY_PREDICTION_IDS = (
    "6b4y2255rhrmt0d19sybv125qm",
    "ghznf33qahrmy0d19syrv6213r",
)
CACHED_CALL01_PREDICTION_ID = "z2a9knzn4nrmr0d19t39tpvjd8"


class BudgetStop(RuntimeError):
    """A bounded stop before Modal's function timeout."""


class ArchiveIncompatibility(RuntimeError):
    """A mask archive that cannot be safely mapped to the packed input."""


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Durably replace a JSON journal without leaving a partial document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _journal_save(journal: dict[str, Any], path: Path, commit: Any = None) -> None:
    journal["pending_call_indices"] = [int(item["call_index"]) for item in journal.get("calls", []) if item.get("status") not in {"succeeded", "orphaned", "terminal_failed"}]
    _atomic_write_json(path, journal)
    if commit is not None:
        commit()


def _resume_commit_callback(path: Path) -> Any:
    if not str(path).startswith(RESUME_VOLUME_PATH):
        return None

    def commit() -> None:
        import modal

        modal.Volume.from_name("reva-phase4-qcsrr-journal", create_if_missing=True).commit()

    return commit


def _resume_call_key(call: Mapping[str, Any], source_manifest: Mapping[str, Any], plan: Mapping[str, Any]) -> str:
    """Return the stable identity used to deduplicate a logical prediction."""

    qa_id = str(call["qa_id"])
    source = source_manifest[qa_id]
    frame_policy = plan["frame_policy"]
    identity = {
        "video": {
            "qa_id": qa_id,
            "video_path": str(source["video_path"]),
            "video_sha256": str(source["video_sha256"]),
        },
        "prompt": str(call["prompt"]),
        "sam3_version": str(plan["model"]["version"]),
        "frame_policy": {
            "sampling_mode": str(frame_policy["sampling_mode"]),
            "offset": float(frame_policy["offset"]),
            "frame_count": int(frame_policy["frame_count"]),
            "source_frame_indices": [int(index) for index in source["source_frame_indices"]],
        },
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resume_entry(call: Mapping[str, Any], key: str) -> dict[str, Any]:
    return {
        **dict(call),
        "call_index": int(call["call_index"]),
        "resume_key": key,
        "status": "pending",
        "prediction_ids": [],
        "attempts": [],
        "output_name": f"call-{int(call['call_index']):02d}.masks.zip",
    }


def _manifest_records(paths: list[Path]) -> dict[int, list[Mapping[str, Any]]]:
    records: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for path in paths:
        if not path.is_file():
            continue
        try:
            payload = _json(path)
        except (OSError, ValueError):
            continue
        for record in payload.get("records", []):
            if isinstance(record, Mapping) and "call_index" in record:
                records[int(record["call_index"])].append(record)
    return records


def _seed_resume_journal(root: Path, plan: Mapping[str, Any], calls: list[Mapping[str, Any]], source_manifest: Mapping[str, Any], *, path: Path | None = None, persist: bool = True) -> dict[str, Any]:
    """Seed or refresh the durable queue from plans, manifests, and mask caches."""

    journal_path = path or root / RESUME_JOURNAL_RELATIVE
    existing: dict[str, Any] = {}
    if journal_path.is_file():
        try:
            existing = _json(journal_path)
        except (OSError, ValueError):
            existing = {}
    by_key = {
        str(item.get("resume_key")): item
        for item in existing.get("calls", [])
        if isinstance(item, Mapping) and item.get("resume_key")
    }
    prior_records = _manifest_records([
        root / f"{RUN_RELATIVE}/replicate_manifest.json",
        root / f"{CONTINUED_RESULTS_RELATIVE}/replicate_manifest.json",
        root / f"{POSTPROCESS_RESULTS_RELATIVE}/replicate_manifest.json",
    ])
    cache_root = root / CONTINUATION_CACHE_RELATIVE
    journal_calls: list[dict[str, Any]] = []
    seen: dict[str, int] = {}
    for call in calls:
        key = _resume_call_key(call, source_manifest, plan)
        if key in seen:
            # Duplicate logical calls share one queue item and never create twice.
            duplicate_indices = journal_calls[seen[key]].setdefault("duplicate_call_indices", [])
            if int(call["call_index"]) not in duplicate_indices:
                duplicate_indices.append(int(call["call_index"]))
            continue
        entry = dict(by_key.get(key, _resume_entry(call, key)))
        entry.update({name: call[name] for name in ("call_index", "qa_id", "arm", "role", "entity_index", "prompt") if name in call})
        entry["call_index"] = int(call["call_index"])
        entry["resume_key"] = key
        entry.setdefault("prediction_ids", [])
        entry.setdefault("attempts", [])
        entry.setdefault("output_name", f"call-{int(call['call_index']):02d}.masks.zip")
        for record in prior_records.get(int(call["call_index"]), []):
            ids = record.get("prediction_ids") or ([record.get("prediction_id")] if record.get("prediction_id") else [])
            for prediction_id in ids:
                if prediction_id and str(prediction_id) not in entry["prediction_ids"]:
                    entry["prediction_ids"].append(str(prediction_id))
        cache_path = cache_root / str(entry["output_name"])
        resume_output_path = journal_path.parent / "outputs" / str(entry["output_name"])
        if cache_path.is_file() and cache_path.stat().st_size:
            entry["status"] = "succeeded"
            entry["output_path"] = str(cache_path.relative_to(root))
        elif resume_output_path.is_file() and resume_output_path.stat().st_size:
            entry["status"] = "succeeded"
            entry["output_path"] = f"outputs/{entry['output_name']}"
        elif entry["prediction_ids"] and entry.get("status") not in {"succeeded", "orphaned"}:
            entry["status"] = "prediction_known"
        elif entry.get("status") == "succeeded" and not entry.get("output_path"):
            entry["status"] = "prediction_known" if entry["prediction_ids"] else "pending"
        if entry.get("status") == "posting" and not entry.get("prediction_ids"):
            # A process may have died during POST. Never issue a second POST automatically.
            entry["status"] = "orphaned"
            entry["orphan_reason"] = "interrupted during create POST without a persisted prediction ID"
        journal_calls.append(entry)
        seen[key] = len(journal_calls) - 1
    journal = {
        "schema_version": 1,
        "status": "ready",
        "model_version": str(plan["model"]["version"]),
        "frame_policy": dict(plan["frame_policy"]),
        "source_manifest": str(SOURCE_MANIFEST_RELATIVE),
        "sdk": {"name": "replicate", "version": REPLICATE_SDK_VERSION, "post_create_retry": "disabled_by_scheduler"},
        "create_policy": {
            "sequential": True,
            "minimum_interval_seconds": RESUME_CREATE_MIN_INTERVAL_SECONDS,
            "max_create_attempts_per_call": RESUME_MAX_CREATE_ATTEMPTS_PER_CALL,
            "max_retries_total": RESUME_MAX_RETRIES,
        },
        "calls": journal_calls,
        "pending_call_indices": [int(item["call_index"]) for item in journal_calls if item.get("status") not in {"succeeded", "orphaned", "terminal_failed"}],
        "create_attempt_count": int(existing.get("create_attempt_count", 0)),
        "retry_count": int(existing.get("retry_count", 0)),
    }
    prior_details = existing.get("attempt_count_details") if isinstance(existing.get("attempt_count_details"), Mapping) else {}
    journal["attempt_count_details"] = {
        "journaled_resume_creates": int(journal["create_attempt_count"]),
        "untracked_legacy_fallthrough_attempts": max(
            int(prior_details.get("untracked_legacy_fallthrough_attempts", 0) or 0),
            _prior_resume_legacy_attempts(root, int(journal["create_attempt_count"])),
        ),
        "historical_attempts": HISTORICAL_ATTEMPTS,
        "historical_continuation_error_records": HISTORICAL_CONTINUATION_ERROR_RECORDS,
    }
    if persist:
        _atomic_write_json(journal_path, journal)
    return journal


def _alias_duplicate_outputs(journal: Mapping[str, Any], raw_outputs: dict[str, bytes]) -> None:
    for entry in journal.get("calls", []):
        output_name = str(entry.get("output_name", ""))
        if output_name not in raw_outputs:
            continue
        for call_index in entry.get("duplicate_call_indices", []):
            raw_outputs[f"call-{int(call_index):02d}.masks.zip"] = raw_outputs[output_name]


def _pending_resume_calls(calls: list[Mapping[str, Any]], journal: Mapping[str, Any], source_manifest: Mapping[str, Any], plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_key = {str(item["resume_key"]): item for item in journal.get("calls", []) if isinstance(item, Mapping) and item.get("resume_key")}
    pending: list[dict[str, Any]] = []
    seen: set[str] = set()
    for call in calls:
        key = _resume_call_key(call, source_manifest, plan)
        if key in seen:
            continue
        seen.add(key)
        entry = by_key.get(key)
        if entry is not None and entry.get("status") not in {"succeeded", "orphaned", "terminal_failed"}:
            pending.append(dict(call))
    return pending


def _resume_create_needed(entry: Mapping[str, Any]) -> bool:
    return not entry.get("prediction_ids") and entry.get("status") in {"pending", "retry_exhausted"}


def _response_headers(error: BaseException) -> dict[str, str]:
    response = getattr(error, "response", None)
    values: Any = getattr(response, "headers", None) or getattr(error, "headers", None) or {}
    return _allowlisted_headers(values)


def _allowlisted_headers(values: Any) -> dict[str, str]:
    try:
        items = values.items()
    except AttributeError:
        return {}
    allowed = {"retry-after", "retry-after-ms", "x-ratelimit-reset", "x-rate-limit-reset", "ratelimit-reset"}
    return {str(key).casefold(): str(value)[:80] for key, value in items if str(key).casefold() in allowed}


def _safe_detail(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value)
    text = re.sub(r"(?i)(authorization|api[_-]?key|token|secret)\s*[:=]\s*[^,\s]+", r"\1=[REDACTED]", text)
    return text[:400]


def _response_body(error: BaseException) -> str:
    response = getattr(error, "response", None)
    for value in (getattr(response, "text", None), getattr(response, "content", None), getattr(error, "detail", None), str(error)):
        if value is None:
            continue
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        return _safe_detail(value)
    return ""


def _rate_limit_info(error: BaseException, *, now: float | None = None, captured: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract safe rate metadata; never persist a credential-bearing exception."""

    now = time.time() if now is None else float(now)
    captured = captured or {}
    headers = {**_response_headers(error), **dict(captured.get("headers", {}))}
    status = captured.get("status") or getattr(getattr(error, "response", None), "status_code", None) or getattr(error, "status_code", None) or getattr(error, "status", None)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None
    delay: float | None = None
    header_order = ("retry-after", "retry-after-ms", "x-ratelimit-reset", "x-rate-limit-reset", "ratelimit-reset")
    for name in header_order:
        if name not in headers:
            continue
        value = headers[name]
        numeric: float | None = None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            if name == "retry-after":
                try:
                    parsed = parsedate_to_datetime(str(value))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    numeric = parsed.timestamp() - now
                except (TypeError, ValueError, OverflowError):
                    numeric = None
        if numeric is None:
            continue
        if name == "retry-after-ms":
            numeric /= 1000.0
        elif "reset" in name and numeric > now + 1:
            numeric -= now
        delay = max(0.0, numeric)
        break
    body = _safe_detail(captured.get("detail")) or _response_body(error)
    if delay is None:
        match = re.search(r"(?:reset|retry)[^0-9]{0,24}(\d+(?:\.\d+)?)\s*(milliseconds?|ms|seconds?|secs?|s)\b", body, flags=re.IGNORECASE)
        if match:
            delay = float(match.group(1))
            if match.group(2).casefold().startswith("ms") or match.group(2).casefold().startswith("mill"):
                delay /= 1000.0
    return {"status": status, "headers": headers, "delay_seconds": delay, "detail": body}


def _client_rate_limit_info(client: Any, error: BaseException) -> dict[str, Any]:
    transport = getattr(client, "_reva_rate_transport", None)
    captured = getattr(transport, "last_rate_limit", None) if transport is not None else None
    if transport is not None:
        transport.last_rate_limit = None
    return _rate_limit_info(error, captured=captured)


def _replicate_client(replicate: Any) -> Any:
    """Use a no-retry HTTP transport that preserves safe 429 headers for the journal."""

    import httpx

    class CaptureRateLimitTransport(httpx.HTTPTransport):
        def __init__(self) -> None:
            super().__init__(retries=0)
            self.last_rate_limit: dict[str, Any] | None = None

        def handle_request(self, request: Any) -> Any:
            response = super().handle_request(request)
            if int(response.status_code) == 429:
                try:
                    response.read()
                except Exception:
                    pass
                self.last_rate_limit = {
                    "status": int(response.status_code),
                    "headers": _allowlisted_headers(response.headers),
                    "detail": _safe_detail(getattr(response, "text", "")),
                }
            return response

    transport = CaptureRateLimitTransport()
    client = replicate.Client(transport=transport)
    client._reva_rate_transport = transport
    return client


def _resume_backoff_seconds(error: BaseException, retry_number: int, *, random_value: float | None = None, rate_info: Mapping[str, Any] | None = None) -> float:
    info = rate_info or _rate_limit_info(error)
    if info["delay_seconds"] is not None:
        return float(info["delay_seconds"])
    jitter = random.random() if random_value is None else float(random_value)
    return min(60.0, 2.0 ** max(0, retry_number)) * (0.5 + max(0.0, min(1.0, jitter)))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", str(text)).casefold().split())


def _call_specs(plan: Mapping[str, Any], selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixed = list(plan["fixed_vocabulary_by_question_order"])
    questions = selection.get("questions")
    if not isinstance(questions, list) or len(questions) != 12 or len(fixed) != 12:
        raise ValueError("Phase 4 must contain 12 questions and 12 fixed controls")
    calls: list[dict[str, Any]] = []
    for order, question in enumerate(questions):
        qa_id = str(question["qa_id"])
        phrases = [str(item).strip() for item in question["entity_prompt_phrases"]]
        if not phrases:
            raise ValueError(f"empty entity prompt list: {qa_id}")
        for entity_index, prompt in enumerate(phrases):
            calls.append({
                "call_index": len(calls), "qa_id": qa_id,
                "arm": "question-conditioned",
                "role": "question_target" if entity_index == 0 else "question_reference",
                "entity_index": entity_index, "prompt": prompt,
            })
        calls.append({
            "call_index": len(calls), "qa_id": qa_id,
            "arm": "fixed-vocabulary", "role": "fixed_control",
            "entity_index": None, "prompt": fixed[order],
        })
    if len(calls) != MAX_CALLS:
        raise ValueError(f"expected {MAX_CALLS} calls, planned {len(calls)}")
    return calls


def _preflight(root: Path) -> dict[str, Any]:
    plan = _json(root / PLAN_RELATIVE)
    selection = _json(root / "tasks/qcsrr_final_benchmark_selection.json")
    manifest = _json(root / SOURCE_MANIFEST_RELATIVE)
    calls = _call_specs(plan, selection)
    if plan["model"]["version"] != "408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269":
        raise ValueError("SAM 3 version is not pinned")
    if plan["calls"]["primary_calls"] != MAX_CALLS or plan["calls"]["maximum_attempts"] != MAX_ATTEMPTS:
        raise ValueError("Phase 4 call budget is not 36 + 5")
    if plan["candidate_policy"]["maximum_pre_dedup_rows"] != 1344:
        raise ValueError("Phase 4 candidate ceiling must be 1344")
    if plan["modal_budget"] != {
        "gpu": "A100-40GB", "cpu": 4, "memory_gib": 16,
        "max_wall_minutes": 10, "max_gpu_minutes": 10,
        "secret_name": "reva-phase3-replicate", "secret_value_logged": False,
    }:
        raise ValueError("Modal resources or secret name changed")
    missing = []
    hash_mismatches = []
    for question in selection["questions"]:
        source = root / RUN_RELATIVE / "source_videos" / str(question["video_path"])
        if not source.is_file() or source.stat().st_size == 0:
            missing.append(str(source))
        record = manifest.get(str(question["qa_id"]))
        if not isinstance(record, Mapping) or len(record.get("source_frame_indices", [])) != 16:
            raise ValueError(f"source manifest lacks 16 frames: {question['qa_id']}")
        if source.is_file():
            expected_sha256 = str(record.get("video_sha256", ""))
            actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
            if not expected_sha256 or actual_sha256 != expected_sha256:
                hash_mismatches.append({"qa_id": str(question["qa_id"]), "expected": expected_sha256, "actual": actual_sha256})
    if missing:
        raise FileNotFoundError("missing staged source videos: " + ", ".join(missing))
    if hash_mismatches:
        raise ValueError("staged source SHA-256 mismatch: " + json.dumps(hash_mismatches, sort_keys=True))
    return {
        "status": "ready", "question_count": 12,
        "question_entity_calls": sum(len(q["entity_prompt_phrases"]) for q in selection["questions"]),
        "fixed_control_calls": 12, "primary_calls": len(calls),
        "maximum_attempts": MAX_ATTEMPTS, "maximum_pre_dedup_rows": 1344,
        "gpu": "A100-40GB", "cpu": 4, "memory_gib": 16, "hard_wall_minutes": 10,
        "secret_name": "reva-phase3-replicate", "source_hashes_verified": len(selection["questions"]),
        "provider_calls": "not launched by preflight",
    }


class _Budget:
    def __init__(self, soft_stop_seconds: float = SOFT_STOP_SECONDS) -> None:
        self.started = time.perf_counter()
        self.soft_stop_seconds = float(soft_stop_seconds)

    def check(self) -> None:
        if time.perf_counter() - self.started >= self.soft_stop_seconds:
            raise BudgetStop(f"soft {self.soft_stop_seconds:.0f}-second guard reached before Modal timeout")

    def summary(self) -> dict[str, Any]:
        elapsed = time.perf_counter() - self.started
        return {
            "wall_seconds": round(elapsed, 3),
            "local_gpu_seconds_upper_bound": round(elapsed, 3),
            "time_guard": "passed" if elapsed < self.soft_stop_seconds else "stopped",
        }


def _pack_video(source: Path, destination: Path, indices: list[int]) -> tuple[int, int]:
    import av

    wanted = set(indices)
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        width, height = int(input_stream.width), int(input_stream.height)
        with av.open(str(destination), mode="w") as output_container:
            output_stream = output_container.add_stream("mpeg4", rate=Fraction("4.0"))
            output_stream.width, output_stream.height = width, height
            output_stream.pix_fmt = "yuv420p"
            for source_index, frame in enumerate(input_container.decode(input_stream)):
                if source_index not in wanted:
                    continue
                output_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                for packet in output_stream.encode(output_frame):
                    output_container.mux(packet)
                written += 1
            for packet in output_stream.encode():
                output_container.mux(packet)
    if written != len(indices):
        raise RuntimeError(f"packed {written} source frames; expected {len(indices)}")
    return width, height


def _read_source_frames(source: Path, indices: list[int]) -> dict[int, Any]:
    import av

    wanted = set(indices)
    result: dict[int, Any] = {}
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        for source_index, frame in enumerate(container.decode(stream)):
            if source_index in wanted:
                result[source_index] = frame.to_ndarray(format="rgb24")
    if set(result) != wanted:
        raise RuntimeError("source frame decode did not return exact sampled indices")
    return result


def _download_output(output: Any) -> bytes:
    import urllib.request

    if isinstance(output, (bytes, bytearray)):
        return bytes(output)
    url = output if isinstance(output, str) else getattr(output, "url", None)
    if url:
        with urllib.request.urlopen(str(url), timeout=180) as response:
            return response.read()
    if hasattr(output, "read"):
        return output.read()
    try:
        return b"".join(bytes(chunk) for chunk in output)
    except Exception as exc:
        raise TypeError("Replicate returned unsupported output type") from exc


def _validate_archive(data: bytes, dimensions: tuple[int, int], source_indices: list[int], *, require_full_frames: bool = False) -> dict[str, Any]:
    import numpy as np
    from PIL import Image
    import sys

    sys.path.insert(0, str(Path(WORK) / "code"))
    import replicate_sam3 as R

    try:
        entries, overlays = R._read_mask_archive(data)
    except Exception as exc:
        raise ArchiveIncompatibility(f"SAM 3 mask archive parser rejected output ({type(exc).__name__})") from exc
    width, height = dimensions
    frame_indices = sorted(int(entry["frame_index"]) for entry in entries)
    if any(index < 0 or index >= len(source_indices) for index in frame_indices):
        raise ArchiveIncompatibility("SAM 3 output frame index is outside packed 16-frame input")
    if len(frame_indices) < len(source_indices):
        for entry in entries:
            member = str(entry.get("member", ""))
            if re.search(r"(?:frame|mask)[-_]?\d+", Path(member).stem, flags=re.IGNORECASE) is None:
                raise ArchiveIncompatibility("sparse SAM 3 mask archive requires explicit frame indices")
    masks: list[dict[str, Any]] = []
    for entry in entries:
        with Image.open(io.BytesIO(entry["payload"])) as image:
            if image.size != (width, height):
                raise ArchiveIncompatibility(f"SAM 3 mask dimensions {image.size} != packed {(width, height)}")
            values = np.asarray(image.convert("L"))
            unique = np.unique(values)
            if unique.size > 2:
                raise ArchiveIncompatibility("SAM 3 mask is not binary")
            mask = values > 0
        masks.append({
            "packed_frame_index": int(entry["frame_index"]),
            "source_frame_index": int(source_indices[int(entry["frame_index"])]),
            "mask": mask,
            "mask_values": [int(value) for value in unique],
        })
    ordered_masks = sorted(masks, key=lambda item: item["packed_frame_index"])
    mapped = [item["source_frame_index"] for item in ordered_masks]
    return {
        "mask_count": len(entries), "packed_frame_indices": frame_indices,
        "source_frame_indices": mapped,
        "dimensions": {"width": width, "height": height},
        "binary_mask_values": sorted({value for item in masks for value in item["mask_values"]}),
        "sampled_frame_count": len(source_indices),
        "mask_bearing_frame_count": len(frame_indices),
        "proposal_miss_frame_count": max(0, len(source_indices) - len(frame_indices)),
        "overlay_only": not entries and bool(overlays),
        "overlay_members": [dict(item) for item in overlays],
        "sparse": len(frame_indices) < len(source_indices),
        "masks": masks,
    }


def _mask_digest(mask: Any) -> str:
    import numpy as np

    values = np.asarray(mask, dtype=np.uint8)
    return hashlib.sha256(str(tuple(values.shape)).encode("ascii") + np.packbits(values, axis=None).tobytes()).hexdigest()


def _geometry(mask: Any) -> tuple[int, dict[str, int], dict[str, float]]:
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return 0, {"left": 0, "top": 0, "right": 0, "bottom": 0}, {"x": 0.0, "y": 0.0}
    height, width = mask.shape
    return int(len(xs)), {"left": int(xs.min()), "top": int(ys.min()), "right": int(xs.max()) + 1, "bottom": int(ys.max()) + 1}, {"x": float(xs.mean() / width), "y": float(ys.mean() / height)}


def _component_masks(mask: Any) -> list[Any]:
    import numpy as np
    from scipy import ndimage

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    components = [labels == label for label in range(1, int(count) + 1)]
    return sorted(components, key=lambda item: (-int(item.sum()), _mask_digest(item)))


def _crop(frame: Any, box: Mapping[str, int]) -> tuple[Any, dict[str, int]]:
    left, top, right, bottom = (int(box[key]) for key in ("left", "top", "right", "bottom"))
    height, width = frame.shape[:2]
    pad_x, pad_y = max(1, round((right - left) * 0.10)), max(1, round((bottom - top) * 0.10))
    crop_box = {"left": max(0, left - pad_x), "top": max(0, top - pad_y), "right": min(width, right + pad_x), "bottom": min(height, bottom + pad_y)}
    return frame[crop_box["top"]:crop_box["bottom"], crop_box["left"]:crop_box["right"]].copy(), crop_box


def _build_candidates(plan: Mapping[str, Any], calls: list[Mapping[str, Any]], parsed: Mapping[str, Mapping[str, Any]], source_manifest: Mapping[str, Any], source_frames: Mapping[str, Mapping[int, Any]], *, component_cap: int | None | object = -1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # ``None`` is the recovery policy: retain every feasible connected component.
    cap = int(plan["candidate_policy"]["target_component_cap_per_prompt_frame"]) if component_cap == -1 else component_cap
    dedup: dict[tuple[str, int, str], dict[str, Any]] = {}
    pre_union, pre_components = 0, 0
    omitted: list[dict[str, Any]] = []
    for call in calls:
        qa_id, call_key = str(call["qa_id"]), str(call["call_index"])
        # Continuation runs may intentionally skip a prior failed call after
        # the shared retry budget is exhausted.  Keep all available proposals
        # and let the corresponding arm have an empty pool downstream.
        if call_key not in parsed:
            continue
        for item in parsed[call_key]["masks"]:
            mask = item["mask"]
            packed, source_index = int(item["packed_frame_index"]), int(item["source_frame_index"])
            frame = source_frames[qa_id][source_index]
            pre_union += 1
            candidates_for_call = [(mask, "union", None)]
            if str(call["role"]) in {"question_target", "fixed_control"}:
                components = _component_masks(mask)
                retained_components = components if cap is None else components[:int(cap)]
                pre_components += len(retained_components)
                candidates_for_call.extend((component, "component", index) for index, component in enumerate(retained_components))
                omitted_components = [] if cap is None else components[int(cap):]
                if cap is not None:
                    for index, component in enumerate(omitted_components, start=int(cap)):
                        area, box, centroid = _geometry(component)
                        omitted.append({"qa_id": qa_id, "call_index": int(call["call_index"]), "prompt": str(call["prompt"]), "packed_frame_index": packed, "component_index": index, "area_pixels": area, "box": box, "centroid": centroid, "mask_sha256": _mask_digest(component)})
            for candidate_mask, kind, component_index in candidates_for_call:
                area, box, centroid = _geometry(candidate_mask)
                if area <= 0:
                    continue
                digest = _mask_digest(candidate_mask)
                arm = str(call["arm"])
                key = (qa_id, arm, source_index, digest)
                provenance = {"call_index": int(call["call_index"]), "arm": str(call["arm"]), "role": str(call["role"]), "prompt": str(call["prompt"]), "proposal_kind": kind, "component_index": component_index}
                existing = dedup.get(key)
                if existing is not None:
                    existing["prompt_provenance"].append(provenance)
                    if kind not in existing["proposal_kinds"]:
                        existing["proposal_kinds"].append(kind)
                    continue
                crop, crop_box = _crop(frame, box)
                dedup[key] = {"candidate_id": f"{qa_id}-{arm[:2]}-f{packed:02d}-{digest[:16]}", "qa_id": qa_id, "arm": arm, "packed_frame_index": packed, "source_frame_index": source_index, "timestamp": round(source_index / float(source_manifest[qa_id]["fps"]), 6), "frame_dimensions": {"width": int(source_manifest[qa_id]["width"]), "height": int(source_manifest[qa_id]["height"])}, "mask_sha256": digest, "mask_area_pixels": area, "mask_area_fraction": round(area / float(mask.shape[0] * mask.shape[1]), 9), "source_box_pixels": box, "centroid_normalized": centroid, "crop_box_pixels": crop_box, "proposal_kinds": [kind], "prompt_provenance": [provenance], "mask": candidate_mask, "crop": crop}
    pre_dedup = pre_union + pre_components
    maximum_pre_dedup = int(plan["candidate_policy"]["maximum_pre_dedup_rows"]) if cap is not None else RESUME_MAX_PRE_DEDUP_ROWS
    if pre_dedup > maximum_pre_dedup:
        raise ValueError(f"candidate ceiling exceeded: {pre_dedup} > {maximum_pre_dedup}")
    candidates = sorted(dedup.values(), key=lambda row: (row["qa_id"], row["packed_frame_index"], row["candidate_id"]))
    return candidates, {"union_rows_pre_dedup": pre_union, "target_component_rows_pre_dedup": pre_components, "rows_pre_dedup": pre_dedup, "rows_post_dedup": len(candidates), "exact_duplicates_removed": pre_dedup - len(candidates), "component_omitted_count": len(omitted), "component_omitted": omitted, "component_policy": "all_feasible_components" if cap is None else f"deterministic_first_{cap}_by_area_then_sha256", "annotation_independent": True, "maximum_pre_dedup_rows": maximum_pre_dedup}


def _mask_iou(left: Any, right: Any) -> float:
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    return float(intersection / union) if union else 0.0


def _box_iou(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    x1, y1 = max(left["left"], right["left"]), max(left["top"], right["top"])
    x2, y2 = min(left["right"], right["right"]), min(left["bottom"], right["bottom"])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_left = (left["right"] - left["left"]) * (left["bottom"] - left["top"])
    area_right = (right["right"] - right["left"]) * (right["bottom"] - right["top"])
    return float(overlap / max(area_left + area_right - overlap, 1))


def _tracks(candidates: list[dict[str, Any]]) -> tuple[dict[str, str], list[dict[str, Any]]]:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    by_video: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for candidate in candidates:
        by_video[candidate["qa_id"]][int(candidate["packed_frame_index"])].append(candidate)
    track_ids = {candidate["candidate_id"]: candidate["candidate_id"] for candidate in candidates}
    links: list[dict[str, Any]] = []
    for qa_id, frames in by_video.items():
        for frame_index in range(15):
            left = sorted(frames.get(frame_index, []), key=lambda item: item["candidate_id"])
            right = sorted(frames.get(frame_index + 1, []), key=lambda item: item["candidate_id"])
            if not left or not right:
                continue
            diagonal = math.hypot(left[0]["mask"].shape[1], left[0]["mask"].shape[0])
            cost = np.full((len(left), len(right)), 10.0, dtype=np.float64)
            eligible: dict[tuple[int, int], tuple[float, float]] = {}
            for i, source in enumerate(left):
                for j, target in enumerate(right):
                    iou = _mask_iou(source["mask"], target["mask"])
                    dx = target["centroid_normalized"]["x"] - source["centroid_normalized"]["x"]
                    dy = target["centroid_normalized"]["y"] - source["centroid_normalized"]["y"]
                    displacement = math.hypot(dx, dy) * diagonal
                    area_ratio = target["mask_area_pixels"] / max(source["mask_area_pixels"], 1)
                    if iou >= 0.25 or (displacement <= 0.03 and 0.5 <= area_ratio <= 2.0):
                        cost[i, j] = -iou + (i * len(right) + j) * 1e-10
                        eligible[(i, j)] = (iou, displacement)
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                if (i, j) not in eligible:
                    continue
                source, target = left[i], right[j]
                old_track, new_track = track_ids[source["candidate_id"]], track_ids[target["candidate_id"]]
                chosen = min(old_track, new_track)
                for candidate_id, track_id in list(track_ids.items()):
                    if track_id in {old_track, new_track}:
                        track_ids[candidate_id] = chosen
                iou, displacement = eligible[(i, j)]
                links.append({"qa_id": qa_id, "from": source["candidate_id"], "to": target["candidate_id"], "iou": round(iou, 6), "normalized_centroid_displacement": round(displacement / diagonal, 6)})
    return track_ids, links


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:8], 16)


def _public_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key not in {"mask", "crop"}}


def _edge_distance(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    dx = max(left["left"] - right["right"], right["left"] - left["right"], 0)
    dy = max(left["top"] - right["bottom"], right["top"] - left["bottom"], 0)
    return float(math.hypot(dx, dy))


def _center_distance(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    left_center = ((left["left"] + left["right"]) / 2.0, (left["top"] + left["bottom"]) / 2.0)
    right_center = ((right["left"] + right["right"]) / 2.0, (right["top"] + right["bottom"]) / 2.0)
    return float(math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1]))


def _relation_geometry(qa_id: str, target: Mapping[str, Any], refs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute only the frozen, answer-neutral relations named in the plan."""

    result: dict[str, Any] = {"status": "neutral", "satisfied": True, "reference_candidate_ids": [ref["candidate_id"] for ref in refs]}
    if qa_id == "train_005918":
        return {**result, "status": "unassessable_orientation", "satisfied": True}
    if qa_id == "train_014154" and refs:
        delta = target["centroid_normalized"]["x"] - refs[0]["centroid_normalized"]["x"]
        return {**result, "status": "computed_left_right", "relation": "left" if delta < 0 else "right" if delta > 0 else "overlap", "delta_x": round(delta, 6)}
    if qa_id == "train_011679" and len(refs) >= 2:
        edge_distances = [_edge_distance(target["source_box_pixels"], ref["source_box_pixels"]) for ref in refs[:2]]
        center_distances = [_center_distance(target["source_box_pixels"], ref["source_box_pixels"]) for ref in refs[:2]]
        return {**result, "status": "computed_distance_comparison", "edge_distances_pixels": [round(value, 3) for value in edge_distances], "center_distances_pixels": [round(value, 3) for value in center_distances], "distance_order": "first_closer" if edge_distances[0] < edge_distances[1] else "second_closer" if edge_distances[1] < edge_distances[0] else "equal"}
    if qa_id == "train_012957" and len(refs) >= 3:
        inside_intersection = _box_iou(target["source_box_pixels"], refs[0]["source_box_pixels"]) > 0
        edge_distances = [_edge_distance(target["source_box_pixels"], ref["source_box_pixels"]) for ref in refs[1:3]]
        center_distances = [_center_distance(target["source_box_pixels"], ref["source_box_pixels"]) for ref in refs[1:3]]
        return {**result, "status": "computed_group_distance_comparison", "satisfied": inside_intersection, "within_intersection": inside_intersection, "anchor_edge_distances_pixels": [round(value, 3) for value in edge_distances], "anchor_center_distances_pixels": [round(value, 3) for value in center_distances], "distance_order": "traffic_lights_closer" if edge_distances[0] < edge_distances[1] else "green_roof_building_closer" if edge_distances[1] < edge_distances[0] else "equal"}
    if qa_id == "train_002092" and refs:
        inside_median = _box_iou(target["source_box_pixels"], refs[0]["source_box_pixels"]) > 0
        return {**result, "status": "computed_median_containment", "satisfied": inside_median, "inside_median": inside_median, "box_iou": round(_box_iou(target["source_box_pixels"], refs[0]["source_box_pixels"]), 6)}
    if qa_id == "train_003224" and refs:
        left_of_field = target["centroid_normalized"]["x"] < refs[0]["centroid_normalized"]["x"]
        return {**result, "status": "computed_left_side", "satisfied": left_of_field, "left_of_baseball_field": left_of_field}
    return result


def _tokenization_audit(processor: Any, torch: Any, phrases: list[str], encoder_revision: str) -> dict[str, Any]:
    """Record the exact query tokenization without running a second embedding pass."""

    actual = processor(text=list(phrases), padding="max_length", truncation=True, return_tensors="pt")
    explicit = processor(text=list(phrases), padding="max_length", truncation=True, max_length=64, return_tensors="pt")
    actual_ids = actual.get("input_ids")
    explicit_ids = explicit.get("input_ids")
    if actual_ids is None or explicit_ids is None:
        raise RuntimeError("the pinned query processor did not return input_ids")
    if not hasattr(actual_ids, "shape") or not hasattr(explicit_ids, "shape"):
        raise RuntimeError("the pinned query processor returned non-tensor input_ids")
    tokenizer = getattr(processor, "tokenizer", processor)

    def _tokenizer_value(name: str) -> Any:
        value = getattr(tokenizer, name, None)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)

    def _tokenizer_token(name: str) -> Any:
        value = getattr(tokenizer, name, None)
        return None if value is None else str(value)

    attention_mask = actual.get("attention_mask")
    if attention_mask is not None:
        non_pad_counts = [int(value) for value in attention_mask.sum(dim=1).detach().cpu().tolist()]
    else:
        padding_id = _tokenizer_value("pad_token_id")
        if padding_id is None:
            non_pad_counts = []
        else:
            non_pad_counts = [int(value) for value in (actual_ids != padding_id).sum(dim=1).detach().cpu().tolist()]
    actual_shape = [int(value) for value in actual_ids.shape]
    explicit_shape = [int(value) for value in explicit_ids.shape]
    return {
        "encoder_revision": str(encoder_revision),
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "normalized_phrases": list(phrases),
        "tokenization_inputs": {"padding": "max_length", "truncation": True, "return_tensors": "pt"},
        "explicit_comparison_inputs": {"padding": "max_length", "truncation": True, "max_length": 64, "return_tensors": "pt"},
        "actual_tokenized_shape": actual_shape,
        "actual_tokenized_length": actual_shape[-1] if actual_shape else None,
        "padding_id": _tokenizer_value("pad_token_id"),
        "bos_eos_pad": {
            "bos_token_id": _tokenizer_value("bos_token_id"),
            "bos_token": _tokenizer_token("bos_token"),
            "eos_token_id": _tokenizer_value("eos_token_id"),
            "eos_token": _tokenizer_token("eos_token"),
            "pad_token_id": _tokenizer_value("pad_token_id"),
            "pad_token": _tokenizer_token("pad_token"),
        },
        "attention_mask_present": attention_mask is not None,
        "attention_mask_shape": [int(value) for value in attention_mask.shape] if attention_mask is not None else None,
        "non_pad_token_counts": non_pad_counts,
        "max_non_pad_token_count": max(non_pad_counts, default=0),
        "explicit_max_length": 64,
        "explicit_max_length_shape": explicit_shape,
        "input_ids_equal_explicit_max_length_64": bool(torch.equal(actual_ids, explicit_ids)),
    }


def _rank(plan: Mapping[str, Any], selection: Mapping[str, Any], calls: list[Mapping[str, Any]], candidates: list[dict[str, Any]], budget: _Budget) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import sys

    sys.path.insert(0, str(Path(WORK) / "code"))
    import build_feature_cache as B

    if not candidates:
        return [], {"status": "no_candidates"}
    model_root = B.validate_local_model_dir(Path(MODEL_DIR))
    torch, processor, model = B._load_local_encoder(model_root, "cuda")
    budget.check()
    image_features = B.encode_images(
        processor.image_processor,
        model,
        [candidate["crop"] for candidate in candidates],
        torch=torch,
        device="cuda",
        batch_size=32,
        input_data_format="channels_last",
    )
    phrases = sorted({_norm(str(call["prompt"])) for call in calls})
    budget.check()
    tokenization_audit = _tokenization_audit(processor, torch, phrases, B.SIGLIP2_MODEL_REVISION)
    text_features = dict(zip(phrases, B.encode_queries(processor, model, phrases, torch=torch, device="cuda", batch_size=32)))
    by_video_arm: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_video_arm[(candidate["qa_id"], candidate["arm"])].append(index)
    rankings: list[dict[str, Any]] = []
    all_links: list[dict[str, Any]] = []
    track_count = 0
    fixed_prompts = list(plan["fixed_vocabulary_by_question_order"])
    for question_order, question in enumerate(selection["questions"]):
        budget.check()
        qa_id = str(question["qa_id"])
        phrases_for_question = [str(item) for item in question["entity_prompt_phrases"]]
        queries_to_rank = [(phrase, "question-conditioned", "target" if i == 0 else "reference", i) for i, phrase in enumerate(phrases_for_question)]
        queries_to_rank.append((fixed_prompts[question_order], "fixed-vocabulary", "fixed_control", 0))
        qc_pool = by_video_arm[(qa_id, "question-conditioned")]
        fixed_pool = by_video_arm[(qa_id, "fixed-vocabulary")]
        target_prompt_by_arm = {"question-conditioned": _norm(phrases_for_question[0]), "fixed-vocabulary": _norm(fixed_prompts[question_order])}
        target_role_by_arm = {"question-conditioned": "question_target", "fixed-vocabulary": "fixed_control"}
        for phrase, arm, phrase_role, phrase_index in queries_to_rank:
            normalized = _norm(phrase)
            pool = qc_pool if arm == "question-conditioned" else fixed_pool
            reference_pool = qc_pool
            query = np.asarray(text_features[normalized], dtype=np.float32)
            scores = np.asarray([float(np.dot(image_features[index], query)) for index in pool], dtype=np.float32)
            order = sorted(range(len(pool)), key=lambda item: (-float(scores[item]), candidates[pool[item]]["candidate_id"]))
            ranked_indices = [pool[item] for item in order]
            seed = _seed(qa_id, arm, normalized)
            rng = random.Random(seed)
            random_top5 = [[candidates[pool[item]]["candidate_id"] for item in rng.sample(range(len(pool)), min(5, len(pool)))] for _ in range(1000)]
            target_indices = {index for index in pool if any(item.get("role") == target_role_by_arm[arm] and _norm(str(item.get("prompt"))) == target_prompt_by_arm[arm] for item in candidates[index]["prompt_provenance"])}
            broad_scene_target = qa_id in {"train_005918", "train_008163", "train_010200", "train_002092"}
            component_target_indices = {index for index in target_indices if "component" in candidates[index]["proposal_kinds"]}
            if component_target_indices and not broad_scene_target:
                target_indices = component_target_indices
            target_candidates = [candidates[index] for index in sorted(target_indices)]
            tracks, links = _tracks(target_candidates) if target_candidates else ({}, [])
            all_links.extend(links)
            track_count += len(set(tracks.values()))
            r1_order = ranked_indices
            geometry_rows: dict[str, dict[str, Any]] = {}
            geometry_enforced = False
            if phrase_role in {"target", "fixed_control"} and len(phrases_for_question) > 1:
                scored: list[tuple[bool, float, str, int]] = []
                ref_phrases = phrases_for_question[1:]
                for target_index in pool:
                    if target_index not in target_indices:
                        continue
                    target = candidates[target_index]
                    same_frame = [index for index in reference_pool if candidates[index]["packed_frame_index"] == target["packed_frame_index"] and candidates[index]["candidate_id"] != target["candidate_id"]]
                    refs: list[dict[str, Any]] = []
                    for ref_phrase in ref_phrases:
                        ref_normalized = _norm(ref_phrase)
                        matching_refs = [index for index in reference_pool if any(item.get("role") == "question_reference" and _norm(str(item.get("prompt"))) == ref_normalized for item in candidates[index]["prompt_provenance"])]
                        same_frame_matching = [index for index in same_frame if index in matching_refs]
                        ref_pool = same_frame_matching or matching_refs
                        ref_query = np.asarray(text_features[_norm(ref_phrase)], dtype=np.float32)
                        ref_index = max(ref_pool, key=lambda index: (float(np.dot(image_features[index], ref_query)), candidates[index]["candidate_id"])) if ref_pool else None
                        if ref_index is not None:
                            refs.append(candidates[ref_index])
                    geometry = _relation_geometry(qa_id, target, refs)
                    geometry["pool_relation"] = "asymmetric_fixed_target_qc_references" if arm == "fixed-vocabulary" else "question_target_qc_references"
                    geometry_rows[target["candidate_id"]] = geometry
                    geometry_enforced = geometry_enforced or qa_id in {"train_003224", "train_002092", "train_012957"}
                    target_score = float(np.dot(image_features[target_index], query))
                    scored.append((not bool(geometry.get("satisfied", True)), -target_score, target["candidate_id"], target_index))
                if scored:
                    r1_order = [item[3] for item in sorted(scored)] + [index for index in ranked_indices if index not in {item[3] for item in scored}]
            r2_order = [index for index in r1_order if index in target_indices] if phrase_role in {"target", "fixed_control"} else []
            temporal_support: dict[str, Any] = {"status": "not_temporal_query"}
            if phrase_role in {"target", "fixed_control"} and question["question_only_decomposition"].get("temporal_operation"):
                by_track: dict[str, list[int]] = defaultdict(list)
                for index in target_indices:
                    track_id = tracks.get(candidates[index]["candidate_id"])
                    if track_id is not None:
                        by_track[track_id].append(index)
                track_rows = []
                for track_id, indices in by_track.items():
                    indices.sort(key=lambda index: candidates[index]["packed_frame_index"])
                    track_rows.append({"track_id": track_id, "sampled_packed_indices": [candidates[index]["packed_frame_index"] for index in indices], "source_frame_indices": [candidates[index]["source_frame_index"] for index in indices], "first_visible_packed_index": candidates[indices[0]]["packed_frame_index"], "last_visible_packed_index": candidates[indices[-1]]["packed_frame_index"], "max_similarity": round(max(float(np.dot(image_features[index], query)) for index in indices), 7)})
                track_rows.sort(key=lambda row: (-row["max_similarity"], row["track_id"]))
                temporal_support = {"status": "sampled_visibility_bracket", "temporal_operation": question["question_only_decomposition"]["temporal_operation"], "tracks": track_rows[:8], "sample_count": 16, "interpolation": "none"}
            rankings.append({
                "qa_id": qa_id, "arm": arm, "phrase": normalized, "phrase_role": phrase_role, "candidate_count": len(pool), "comparison_pool_arm": "question-conditioned" if arm == "fixed-vocabulary" else arm, "similarity": "cosine_on_l2_normalized_siglip2_features",
                "R0": {"top5": [{**_public_candidate(candidates[index]), "similarity": round(float(np.dot(image_features[index], query)), 7)} for index in ranked_indices[:5]], "random_seed": seed, "random_draws": 1000, "random_top5_candidate_ids": random_top5},
                "R1": {"top5": [{**_public_candidate(candidates[index]), "similarity": round(float(np.dot(image_features[index], query)), 7), "geometry": geometry_rows.get(candidates[index]["candidate_id"], {"status": "neutral_or_reference"})} for index in r1_order[:5]], "geometry_source": "question_only_decomposition", "geometry_constraints_enforced": geometry_enforced, "orientation_status": "unassessable" if qa_id == "train_005918" else "computed_or_neutral"},
                "R2": {"status": "target_only_association" if r2_order else "not_applicable_to_reference_rank", "scope": "component_target_tracks" if component_target_indices and not broad_scene_target else "union_and_component_tracks_for_broad_scene", "rank_effect": "track metadata is attached to the unchanged R1 deterministic order", "top5": [{"candidate_id": candidates[index]["candidate_id"], "track_id": tracks.get(candidates[index]["candidate_id"]), "track_length": sum(track == tracks.get(candidates[index]["candidate_id"]) for track in tracks.values()), "source_frame_index": candidates[index]["source_frame_index"], "similarity": round(float(np.dot(image_features[index], query)), 7)} for index in r2_order[:5]], "temporal_support": temporal_support, "association_thresholds": {"iou": 0.25, "normalized_centroid_displacement": 0.03, "area_ratio": [0.5, 2.0]}},
                "native_sam3_score": {"status": "unavailable", "reason": "return_zip had no attributable per-mask scores"},
            })
    return rankings, {"status": "complete", "text_phrase_count": len(phrases), "image_feature_count": len(image_features), "candidate_counts_by_arm": {"question-conditioned": sum(len(value) for (video, arm), value in by_video_arm.items() if arm == "question-conditioned"), "fixed-vocabulary": sum(len(value) for (video, arm), value in by_video_arm.items() if arm == "fixed-vocabulary")}, "r2_link_count": len(all_links), "r2_track_count": track_count, "r2_links": all_links, "tokenization_audit": tokenization_audit}


def _gzip_json(value: Any) -> bytes:
    return gzip.compress((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(), compresslevel=6)


def _archive(outputs: Mapping[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as tar:
        for name, data in outputs.items():
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            tar.addfile(info, io.BytesIO(data))
    return stream.getvalue()


def _emit(state: Mapping[str, Any], budget: _Budget, gpu_name: str | None = None, peak: tuple[int, int] | None = None) -> dict[str, bytes]:
    raw_outputs = dict(state.get("raw_outputs", {}))
    records = list(state.get("records", []))
    resume_state = state.get("resume")
    is_resume = isinstance(resume_state, Mapping) and resume_state.get("mode") == "resume"
    hard_wall_minutes = RESUME_HARD_WALL_MINUTES if is_resume else 10
    resource_based_max_estimate = {
        "status": "unit_bound_only_provider_rates_unavailable",
        "max_estimate_basis": "the full function wall ceiling with requested Modal resources",
        "gpu": "A100-40GB",
        "gpu_minutes": hard_wall_minutes,
        "cpu_minutes": hard_wall_minutes * 4,
        "memory_gib_minutes": hard_wall_minutes * 16,
        "max_estimate_usd": None,
    }
    prediction_ids: list[str] = []
    for record in records:
        ids = record.get("prediction_ids") or ([record.get("prediction_id")] if record.get("prediction_id") else [])
        for prediction_id in ids:
            if prediction_id and prediction_id not in prediction_ids:
                prediction_ids.append(str(prediction_id))
    completed_primary_calls = sum(
        1 + len(record.get("duplicate_call_indices") or [])
        for record in records
        if record.get("status") == "succeeded"
    )
    attempt_count_details = dict(state.get("attempt_count_details") or {})
    if is_resume:
        attempt_count_details.setdefault("journaled_resume_creates", int(state.get("attempt_count", 0)))
        attempt_count_details.setdefault("untracked_legacy_fallthrough_attempts", 0)
        attempt_count_details.setdefault("historical_attempts", HISTORICAL_ATTEMPTS)
        attempt_count_details.setdefault("historical_continuation_error_records", HISTORICAL_CONTINUATION_ERROR_RECORDS)
    manifest = {
        "schema_version": 1,
        "status": state.get("status", "failed"),
        "error_type": state.get("error_type"),
        "error_phase": state.get("error_phase"),
        "error_message": state.get("error_message"),
        "model": state.get("model"),
        "planned_primary_calls": MAX_CALLS,
        "completed_primary_calls": completed_primary_calls,
        "attempt_count": int(state.get("attempt_count", 0)),
        "attempt_count_details": attempt_count_details,
        "retry_count": int(state.get("retry_count", 0)),
        "records": records,
        "resume": state.get("resume"),
        "first_archive_validation": state.get("first_archive_validation"),
        "native_score_status": "unavailable: return_zip masks have no attributable per-mask scores",
        "qwen_scoring": "not-run",
    }
    candidates = state.get("candidate_public", [])
    rankings = state.get("rankings", [])
    report = {
        "schema_version": 1,
        "status": manifest["status"],
        "scope": "Phase 4 QCSRR 12-video experiment",
        "error_message": manifest["error_message"],
        "replicate": {"primary_calls": MAX_CALLS, "completed": manifest["completed_primary_calls"], "attempts": manifest["attempt_count"], "retries": manifest["retry_count"], "prediction_ids": prediction_ids, "attempt_count_details": attempt_count_details},
        "resume": state.get("resume"),
        "candidate_count": len(candidates),
        "candidate_summary": state.get("candidate_summary"),
        "ranking_summary": state.get("ranking_summary"),
        "native_score_status": manifest["native_score_status"],
        "runtime": {**budget.summary(), "gpu_device": gpu_name, "gpu_count": 1 if gpu_name else None, "peak_cuda_memory_allocated_bytes": peak[0] if peak else None, "peak_cuda_memory_reserved_bytes": peak[1] if peak else None, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024},
        "hardware": {"requested_gpu": "A100-40GB", "requested_cpus": 4, "requested_memory_gib": 16, "hard_wall_minutes": hard_wall_minutes, "function_timeout_seconds": int(state.get("function_timeout_seconds", FUNCTION_TIMEOUT_SECONDS)), "resource_based_max_estimate": resource_based_max_estimate},
        "cost": {"actual_cost_usd": None, "actual_cost_status": "provider billing is not exposed to the Modal function", "replicate_list_price_estimate_usd": round(float(state.get("attempt_count", 0)) * 0.0017, 4), "replicate_list_price_basis": "approximately $0.0017 per attempt; variable provider input price", "modal_estimate_status": "resource-based billing requires provider reconciliation", "modal_resource_based_max_estimate": resource_based_max_estimate},
        "qwen_scoring": "not-run",
        "forbidden_work": ["Qwen", "training", "DA3", "VGGT", "Codabench", "other providers"],
    }
    outputs = {"replicate_outputs.tar.gz": _archive(raw_outputs), "replicate_manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(), "candidates.json.gz": _gzip_json(candidates), "rankings.json.gz": _gzip_json(rankings), "modal_report.json": (json.dumps(report, indent=2) + "\n").encode()}
    if state.get("resume_journal"):
        outputs["resume_journal.json"] = (json.dumps(state["resume_journal"], indent=2, sort_keys=True) + "\n").encode()
    return outputs


def _resume_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key not in {"output_path"}}


def _resume_sleep(seconds: float, budget: _Budget) -> None:
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        budget.check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(1.0, remaining))


def _persist_prediction_id(journal: dict[str, Any], entry: dict[str, Any], prediction_id: str, path: Path, commit: Any = None) -> None:
    """Write the ID before the first poll so a restart never creates twice."""

    entry["prediction_ids"] = list(dict.fromkeys([*entry.get("prediction_ids", []), str(prediction_id)]))
    entry["prediction_id"] = str(prediction_id)
    entry["status"] = "prediction_created"
    journal["status"] = "running"
    journal["pending_call_indices"] = [int(item["call_index"]) for item in journal["calls"] if item.get("status") not in {"succeeded", "orphaned", "terminal_failed"}]
    _journal_save(journal, path, commit)


def _resume_run_queue(*, client: Any, R: Any, packed: Mapping[str, Path], calls: list[Mapping[str, Any]], source_manifest: Mapping[str, Any], journal: dict[str, Any], journal_path: Path, state: dict[str, Any], budget: _Budget, commit: Any = None) -> None:
    """Run only pending queue items, persisting every create transition."""

    by_key = {str(item["resume_key"]): item for item in journal["calls"]}
    last_create_at: float | None = None
    for call in calls:
        key = _resume_call_key(call, source_manifest, {"model": {"version": journal["model_version"]}, "frame_policy": journal["frame_policy"]})
        entry = by_key[key]
        if entry.get("status") in {"succeeded", "orphaned"}:
            continue
        qa_id = str(call["qa_id"])
        output_name = str(entry["output_name"])
        # Existing IDs are always retrieved first.  A create that returned an ID is never relaunched.
        prediction: Any = None
        if entry.get("prediction_ids"):
            for prediction_id in entry["prediction_ids"]:
                try:
                    prediction = client.predictions.get(str(prediction_id))
                    while str(getattr(prediction, "status", "")) in {"starting", "processing", "queued"}:
                        budget.check()
                        time.sleep(1.0)
                        prediction = client.predictions.get(str(prediction_id))
                    if str(getattr(prediction, "status", "")) == "succeeded":
                        break
                except Exception as exc:
                    entry["retrieve_error"] = _client_rate_limit_info(client, exc)
                    _journal_save(journal, journal_path, commit)
            if prediction is not None and str(getattr(prediction, "status", "")) != "succeeded":
                entry["status"] = "terminal_failed"
                entry["terminal_status"] = str(getattr(prediction, "status", "unknown"))
                _journal_save(journal, journal_path, commit)
                continue
        while prediction is None and _resume_create_needed(entry):
            budget.check()
            if entry.get("attempts") and int(journal.get("retry_count", 0)) >= RESUME_MAX_RETRIES:
                entry["status"] = "retry_exhausted"
                _journal_save(journal, journal_path, commit)
                break
            if len(entry.get("attempts", [])) >= RESUME_MAX_CREATE_ATTEMPTS_PER_CALL:
                entry["status"] = "retry_exhausted"
                _journal_save(journal, journal_path, commit)
                break
            if last_create_at is not None:
                _resume_sleep(RESUME_CREATE_MIN_INTERVAL_SECONDS - (time.monotonic() - last_create_at), budget)
            attempt_number = len(entry.get("attempts", [])) + 1
            attempt = {"attempt": attempt_number, "status": "posting", "started_at": datetime.now(timezone.utc).isoformat()}
            entry.setdefault("attempts", []).append(attempt)
            journal["create_attempt_count"] = int(journal.get("create_attempt_count", 0)) + 1
            state["attempt_count"] = int(journal["create_attempt_count"])
            _journal_save(journal, journal_path, commit)
            last_create_at = time.monotonic()
            transport = getattr(client, "_reva_rate_transport", None)
            if transport is not None:
                transport.last_rate_limit = None
            try:
                payload = R.prediction_input(packed[qa_id], str(call["prompt"]))
                with payload["video"].open("rb") as stream:
                    prediction = client.predictions.create(version=R.REPLICATE_VERSION, input={**payload, "video": stream})
                prediction_id = str(getattr(prediction, "id", "")) or None
                if prediction_id is None:
                    attempt.update({"status": "created_without_id", "finished_at": datetime.now(timezone.utc).isoformat()})
                    entry["status"] = "orphaned"
                    entry["orphan_reason"] = "provider response did not include prediction ID"
                    _journal_save(journal, journal_path, commit)
                    break
                attempt.update({"status": "created", "prediction_id": prediction_id, "finished_at": datetime.now(timezone.utc).isoformat()})
                _persist_prediction_id(journal, entry, prediction_id, journal_path, commit)
            except Exception as exc:
                info = _client_rate_limit_info(client, exc)
                attempt.update({"status": "failed", "finished_at": datetime.now(timezone.utc).isoformat(), "error": info})
                # A response without a known rate-limit rejection may mean the
                # provider accepted the POST but the client lost its response.
                # Do not issue a second create in that uncertain state.
                if info.get("status") != 429:
                    attempt["status"] = "unknown_post_outcome" if info.get("status") is None else "failed_without_retry"
                    entry["status"] = "orphaned" if info.get("status") is None else "terminal_failed"
                    entry["orphan_reason"] = "create POST outcome was not a confirmed HTTP 429 rejection"
                    _journal_save(journal, journal_path, commit)
                    break
                journal["retry_count"] = int(journal.get("retry_count", 0)) + 1
                state["retry_count"] = int(journal["retry_count"])
                if len(entry.get("attempts", [])) >= RESUME_MAX_CREATE_ATTEMPTS_PER_CALL or int(journal["retry_count"]) >= RESUME_MAX_RETRIES:
                    entry["status"] = "retry_exhausted"
                else:
                    entry["status"] = "pending"
                _journal_save(journal, journal_path, commit)
                if entry["status"] == "retry_exhausted":
                    break
                _resume_sleep(_resume_backoff_seconds(exc, len(entry["attempts"]), rate_info=info), budget)
                continue
            # The ID was persisted before entering this polling loop.
            while str(getattr(prediction, "status", "")) in {"starting", "processing", "queued"}:
                budget.check()
                time.sleep(1.0)
                prediction = client.predictions.get(str(entry["prediction_id"]))
            break
        if prediction is None or str(getattr(prediction, "status", "")) != "succeeded":
            if prediction is not None:
                entry["status"] = "terminal_failed"
                entry["terminal_status"] = str(getattr(prediction, "status", "unknown"))
                _journal_save(journal, journal_path, commit)
            continue
        try:
            output_bytes = _download_output(getattr(prediction, "output", None))
        except Exception as exc:
            entry["status"] = "prediction_known"
            entry["output_error"] = _rate_limit_info(exc)
            _journal_save(journal, journal_path, commit)
            continue
        state["raw_outputs"][output_name] = output_bytes
        _alias_duplicate_outputs(journal, state["raw_outputs"])
        _atomic_write_bytes(journal_path.parent / "outputs" / output_name, output_bytes)
        entry.update({"status": "succeeded", "output_path": f"outputs/{output_name}", "output_sha256": hashlib.sha256(output_bytes).hexdigest(), "output_size_bytes": len(output_bytes), "completed_at": datetime.now(timezone.utc).isoformat()})
        _journal_save(journal, journal_path, commit)
    journal["status"] = "ready"
    journal["pending_call_indices"] = [int(item["call_index"]) for item in journal["calls"] if item.get("status") not in {"succeeded", "orphaned", "terminal_failed"}]
    _journal_save(journal, journal_path, commit)
    state["records"] = [_resume_record(item) for item in journal["calls"]]


def _execute(work: str = WORK, *, resume_sparse: bool = False, continue_no_retry: bool = False, postprocess_cached: bool = False, resume: bool = False, journal_path: str | None = None) -> dict[str, bytes]:
    import sys

    work_root = Path(work)
    plan = _json(work_root / PLAN_RELATIVE)
    selection = _json(work_root / "tasks/qcsrr_final_benchmark_selection.json")
    source_manifest = _json(work_root / SOURCE_MANIFEST_RELATIVE)
    calls = _call_specs(plan, selection)
    budget = _Budget(RESUME_SOFT_STOP_SECONDS if resume else SOFT_STOP_SECONDS)
    model_metadata = dict(plan["model"])
    model_metadata["first_archive_gate"] = "validate returned frame indices within packed 0..15, explicit indices for sparse archives, source mapping, source dimensions, and binary mask values; sparse outputs count absent sampled frames as proposal misses"
    state: dict[str, Any] = {"status": "running", "records": [], "raw_outputs": {}, "attempt_count": 0, "retry_count": 0, "model": model_metadata, "function_timeout_seconds": RESUME_FUNCTION_TIMEOUT_SECONDS if resume else FUNCTION_TIMEOUT_SECONDS, "resume": {"mode": "resume" if resume else "postprocess_cached" if postprocess_cached else "continued_no_retry" if continue_no_retry else "sparse_recovery" if resume_sparse else "fresh", "recovered_prediction_ids": list(RECOVERY_PREDICTION_IDS) if (resume_sparse or continue_no_retry or postprocess_cached) else [], "attempt_count_scope": "cumulative resume journal" if resume else "run"}}
    try:
        import torch
        import replicate

        sys.path.insert(0, str(work_root / "code"))
        import replicate_sam3 as R

        with tempfile.TemporaryDirectory(prefix="reva_phase4_", dir="/tmp") as temp:
            temp_root = Path(temp)
            packed: dict[str, Path] = {}
            source_frames: dict[str, dict[int, Any]] = {}
            for question in selection["questions"]:
                budget.check()
                qa_id = str(question["qa_id"])
                record = source_manifest[qa_id]
                source = work_root / RUN_RELATIVE / "source_videos" / str(record["video_path"])
                indices = [int(index) for index in record["source_frame_indices"]]
                packed[qa_id] = temp_root / f"{qa_id}.legacy-offset-0.mp4"
                _pack_video(source, packed[qa_id], indices)
                source_frames[qa_id] = _read_source_frames(source, indices)
            client = None if postprocess_cached else _replicate_client(replicate) if resume else replicate.Client()
            calls_to_run = calls
            if resume:
                resume_path = Path(journal_path) if journal_path else work_root / RESUME_JOURNAL_RELATIVE
                commit = _resume_commit_callback(resume_path)
                journal = _seed_resume_journal(work_root, plan, calls, source_manifest, path=resume_path, persist=True)
                if commit is not None:
                    commit()
                state["attempt_count"] = int(journal.get("create_attempt_count", 0))
                state["retry_count"] = int(journal.get("retry_count", 0))
                state["resume"]["attempt_count_scope"] = "journaled resume creates only; see attempt_count_details"
                state["attempt_count_details"] = dict(journal.get("attempt_count_details", {}))
                state["resume_journal"] = journal
                for raw_path in sorted((work_root / CONTINUATION_CACHE_RELATIVE).glob("call-*.masks.zip")):
                    state["raw_outputs"][raw_path.name] = raw_path.read_bytes()
                for entry in journal.get("calls", []):
                    output_path = resume_path.parent / "outputs" / str(entry.get("output_name", ""))
                    if output_path.is_file() and output_path.stat().st_size:
                        state["raw_outputs"][str(entry["output_name"])] = output_path.read_bytes()
                _alias_duplicate_outputs(journal, state["raw_outputs"])
                calls_to_run = _pending_resume_calls(calls, journal, source_manifest, plan)
                _resume_run_queue(client=client, R=R, packed=packed, calls=calls_to_run, source_manifest=source_manifest, journal=journal, journal_path=resume_path, state=state, budget=budget, commit=commit)
                state["attempt_count_details"]["journaled_resume_creates"] = int(journal.get("create_attempt_count", 0))
                state["attempt_count"] = int(journal.get("create_attempt_count", 0))
                state["resume_journal"] = journal
            elif postprocess_cached:
                cache_root = work_root / CONTINUATION_CACHE_RELATIVE
                prior_manifest = _json(work_root / f"{CONTINUED_RESULTS_RELATIVE}/replicate_manifest.json")
                state["records"] = list(prior_manifest.get("records", []))
                state["attempt_count"] = int(prior_manifest.get("attempt_count", 0))
                state["retry_count"] = int(prior_manifest.get("retry_count", 0))
                for raw_path in sorted(cache_root.glob("call-*.masks.zip")):
                    state["raw_outputs"][raw_path.name] = raw_path.read_bytes()
                calls_to_run = []
                for call in calls:
                    raw_name = f"call-{int(call['call_index']):02d}.masks.zip"
                    if raw_name not in state["raw_outputs"]:
                        continue
                    qa_id = str(call["qa_id"])
                    dimensions = (int(source_manifest[qa_id]["width"]), int(source_manifest[qa_id]["height"]))
                    validation = _validate_archive(state["raw_outputs"][raw_name], dimensions, [int(index) for index in source_manifest[qa_id]["source_frame_indices"]])
                    if state.get("first_archive_validation") is None:
                        state["first_archive_validation"] = {key: value for key, value in validation.items() if key != "masks"}
            elif resume_sparse:
                recovered_path = work_root / RECOVERY_RELATIVE / RECOVERY_CALL0[0]
                recovered_bytes = recovered_path.read_bytes()
                recovered_call = calls[0]
                dimensions = (int(source_manifest[recovered_call["qa_id"]]["width"]), int(source_manifest[recovered_call["qa_id"]]["height"]))
                validation = _validate_archive(recovered_bytes, dimensions, [int(index) for index in source_manifest[recovered_call["qa_id"]]["source_frame_indices"]])
                state["first_archive_validation"] = {key: value for key, value in validation.items() if key != "masks"}
                state["raw_outputs"]["call-00.masks.zip"] = recovered_bytes
                state["attempt_count"], state["retry_count"] = 2, 1
                state["records"].append({**recovered_call, "status": "succeeded", "prediction_id": RECOVERY_PREDICTION_IDS[0], "prediction_ids": list(RECOVERY_PREDICTION_IDS), "attempts": [{"prediction_id": RECOVERY_PREDICTION_IDS[0], "status": "succeeded", "runtime_seconds": 4.334, "metrics": {"predict_time": 2.010364307, "total_time": 2.073212952}, "recovered": True}, {"prediction_id": RECOVERY_PREDICTION_IDS[1], "status": "succeeded", "runtime_seconds": 2.0, "metrics": {"predict_time": 1.936667421, "total_time": 1.947250615}, "recovered": True}], "mask_count": validation["mask_count"], "packed_frame_indices": validation["packed_frame_indices"], "source_frame_indices": validation["source_frame_indices"], "recovered": True})
                calls_to_run = calls[1:]
            elif continue_no_retry:
                cache_root = work_root / CONTINUATION_CACHE_RELATIVE
                call0_bytes = (cache_root / "call-00.masks.zip").read_bytes()
                call1_bytes = (cache_root / "call-01.masks.zip").read_bytes()
                call0, call1, skipped = calls[0], calls[1], calls[2]
                call0_dims = (int(source_manifest[call0["qa_id"]]["width"]), int(source_manifest[call0["qa_id"]]["height"]))
                call0_validation = _validate_archive(call0_bytes, call0_dims, [int(index) for index in source_manifest[call0["qa_id"]]["source_frame_indices"]])
                call1_dims = (int(source_manifest[call1["qa_id"]]["width"]), int(source_manifest[call1["qa_id"]]["height"]))
                call1_validation = _validate_archive(call1_bytes, call1_dims, [int(index) for index in source_manifest[call1["qa_id"]]["source_frame_indices"]])
                state["first_archive_validation"] = {key: value for key, value in call0_validation.items() if key != "masks"}
                state["raw_outputs"].update({"call-00.masks.zip": call0_bytes, "call-01.masks.zip": call1_bytes})
                state["attempt_count"], state["retry_count"] = 8, 5
                state["records"].append({**call0, "status": "succeeded", "prediction_id": RECOVERY_PREDICTION_IDS[0], "prediction_ids": list(RECOVERY_PREDICTION_IDS), "attempts": [{"prediction_id": RECOVERY_PREDICTION_IDS[0], "status": "succeeded", "recovered": True}, {"prediction_id": RECOVERY_PREDICTION_IDS[1], "status": "succeeded", "recovered": True}], "mask_count": call0_validation["mask_count"], "packed_frame_indices": call0_validation["packed_frame_indices"], "source_frame_indices": call0_validation["source_frame_indices"], "recovered": True})
                state["records"].append({**call1, "status": "succeeded", "prediction_id": CACHED_CALL01_PREDICTION_ID, "attempts": [{"prediction_id": CACHED_CALL01_PREDICTION_ID, "status": "succeeded", "recovered": True}], "mask_count": call1_validation["mask_count"], "packed_frame_indices": call1_validation["packed_frame_indices"], "source_frame_indices": call1_validation["source_frame_indices"], "recovered": True})
                state["records"].append({**skipped, "status": "skipped_prior_failure", "error_type": "ReplicateError", "error_message": "prior call exhausted the five shared retries before continuation"})
                calls_to_run = calls[3:]
            # Resume has its own durable, rate-aware queue.  Never fall
            # through into the legacy back-to-back create loop after it.
            if _legacy_create_loop_allowed(resume):
                for call in calls_to_run:
                    budget.check()
                    if int(state["attempt_count"]) >= MAX_ATTEMPTS:
                        raise BudgetStop("Replicate attempt cap reached")
                    qa_id, call_key = str(call["qa_id"]), str(call["call_index"])
                    attempts_for_call: list[dict[str, Any]] = []
                    while True:
                        budget.check()
                        state["attempt_count"] += 1
                        started = time.perf_counter()
                        prediction_id = None
                        try:
                            payload = R.prediction_input(packed[qa_id], str(call["prompt"]))
                            with payload["video"].open("rb") as stream:
                                prediction = client.predictions.create(version=R.REPLICATE_VERSION, input={**payload, "video": stream})
                            prediction_id = str(getattr(prediction, "id", "")) or None
                            while str(getattr(prediction, "status", "")) in {"starting", "processing", "queued"}:
                                budget.check()
                                time.sleep(1.0)
                                prediction = client.predictions.get(prediction_id)
                            status = str(getattr(prediction, "status", ""))
                            raw_metrics = getattr(prediction, "metrics", {}) or {}
                            metrics = {key: float(raw_metrics[key]) for key in ("predict_time", "total_time") if key in raw_metrics and isinstance(raw_metrics[key], (int, float))}
                            attempt = {"prediction_id": prediction_id, "status": status, "runtime_seconds": round(time.perf_counter() - started, 3), "metrics": metrics}
                            attempts_for_call.append(attempt)
                            if status != "succeeded":
                                raise RuntimeError("Replicate prediction did not succeed")
                            output_bytes = _download_output(getattr(prediction, "output", None))
                            state["raw_outputs"][f"call-{int(call['call_index']):02d}.masks.zip"] = output_bytes
                            dimensions = (int(source_manifest[qa_id]["width"]), int(source_manifest[qa_id]["height"]))
                            validation = _validate_archive(output_bytes, dimensions, [int(index) for index in source_manifest[qa_id]["source_frame_indices"]])
                            if state.get("first_archive_validation") is None:
                                state["first_archive_validation"] = {key: value for key, value in validation.items() if key != "masks"}
                            state["records"].append({**call, "status": "succeeded", "prediction_id": prediction_id, "attempts": attempts_for_call, "mask_count": validation["mask_count"], "packed_frame_indices": validation["packed_frame_indices"], "source_frame_indices": validation["source_frame_indices"], "runtime_seconds": round(time.perf_counter() - started, 3), "metrics": metrics})
                            break
                        except BudgetStop:
                            attempts_for_call.append({"prediction_id": prediction_id, "status": "stopped_by_budget", "runtime_seconds": round(time.perf_counter() - started, 3)})
                            state["records"].append({**call, "status": "stopped_by_budget", "attempts": attempts_for_call})
                            raise
                        except ArchiveIncompatibility as exc:
                            attempts_for_call.append({"prediction_id": prediction_id, "status": "incompatible_archive", "error_type": type(exc).__name__, "runtime_seconds": round(time.perf_counter() - started, 3)})
                            state["records"].append({**call, "status": "incompatible_archive", "attempts": attempts_for_call, "error_type": type(exc).__name__, "error_message": str(exc)})
                            if continue_no_retry:
                                break
                            raise
                        except Exception as exc:
                            attempts_for_call.append({"prediction_id": prediction_id, "status": "error", "error_type": type(exc).__name__, "error_message": str(exc), "runtime_seconds": round(time.perf_counter() - started, 3)})
                            if continue_no_retry:
                                state["records"].append({**call, "status": "failed", "attempts": attempts_for_call, "error_type": type(exc).__name__, "error_message": str(exc)})
                                break
                            if int(state["retry_count"]) >= MAX_RETRIES or int(state["attempt_count"]) >= MAX_ATTEMPTS:
                                state["records"].append({**call, "status": "failed", "attempts": attempts_for_call, "error_type": type(exc).__name__, "error_message": str(exc)})
                                raise
                            state["retry_count"] += 1
            parsed: dict[str, dict[str, Any]] = {}
            for call in calls:
                call_key = str(call["call_index"])
                raw_name = f"call-{int(call['call_index']):02d}.masks.zip"
                if raw_name not in state["raw_outputs"]:
                    continue
                raw = state["raw_outputs"][raw_name]
                parsed[call_key] = _validate_archive(raw, (int(source_manifest[str(call["qa_id"])]["width"]), int(source_manifest[str(call["qa_id"])]["height"])), [int(index) for index in source_manifest[str(call["qa_id"])]["source_frame_indices"]])
            candidates, summary = _build_candidates(plan, calls, parsed, source_manifest, source_frames, component_cap=None if resume else -1)
            state["candidate_summary"] = summary
            state["candidate_public"] = [_public_candidate(candidate) for candidate in candidates]
            rankings, ranking_summary = _rank(plan, selection, calls, candidates, budget)
            state["rankings"], state["ranking_summary"] = rankings, ranking_summary
            if not torch.cuda.is_available():
                raise RuntimeError("Modal function did not receive CUDA")
            state["status"] = "complete_available_outputs" if (resume or continue_no_retry or postprocess_cached) else "complete"
            peak = (int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved()))
            return _emit(state, budget, str(torch.cuda.get_device_name(0)), peak)
    except BudgetStop as exc:
        state["status"], state["error_type"], state["error_phase"] = "partial_budget_stop", type(exc).__name__, "execution"
        return _emit(state, budget)
    except Exception as exc:
        state["status"], state["error_type"], state["error_phase"], state["error_message"] = "failed", type(exc).__name__, "execution", str(exc)
        return _emit(state, budget)


def _remote_entrypoint(*, resume_sparse: bool = False, continue_no_retry: bool = False, postprocess_cached: bool = False, resume: bool = False) -> dict[str, bytes]:
    """Import the copied job module inside Modal to keep the closure tiny."""

    import importlib.util

    path = "/root/modal_phase4_qcsrr.py"
    spec = importlib.util.spec_from_file_location("reva_phase4_remote", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("remote Phase 4 job module is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._execute("/root/ReVA", resume_sparse=resume_sparse, continue_no_retry=continue_no_retry, postprocess_cached=postprocess_cached, resume=resume, journal_path=f"{RESUME_VOLUME_PATH}/resume_journal.json" if resume else None)


def _build_modal_app(*, resume_sparse: bool = False, continue_no_retry: bool = False, postprocess_cached: bool = False, resume: bool = False) -> tuple[Any, Any]:
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.14.1", "torchvision==0.29.1", "transformers==5.19.0", "accelerate==1.15.0", "huggingface-hub==2.2.0", "av==18.1.0", "numpy", "pillow", "scipy", f"replicate=={REPLICATE_SDK_VERSION}")
        .run_commands(
            "python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/siglip2-base-patch16-224', revision='022b6f71160ffb0169ca4709e2d7e25be659598a', local_dir='/root/siglip2', allow_patterns=['config.json','model.safetensors','preprocessor_config.json','special_tokens_map.json','tokenizer.json','tokenizer.model','tokenizer_config.json'])\"",
            "python -c \"from pathlib import Path; Path('/root/siglip2/.reva_encoder_revision').write_text('022b6f71160ffb0169ca4709e2d7e25be659598a\\n')\"",
        )
        .env({"WORK": WORK, "HF_HOME": f"{WORK}/hf_cache"})
        .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
        .add_local_file(str(ROOT / PLAN_RELATIVE), f"{WORK}/{PLAN_RELATIVE}", copy=True)
        .add_local_file(str(ROOT / "tasks/qcsrr_final_benchmark_selection.json"), f"{WORK}/tasks/qcsrr_final_benchmark_selection.json", copy=True)
        .add_local_file(str(ROOT / SOURCE_MANIFEST_RELATIVE), f"{WORK}/{SOURCE_MANIFEST_RELATIVE}", copy=True)
        .add_local_file(str(ROOT / "jobs/modal_phase4_qcsrr.py"), "/root/modal_phase4_qcsrr.py", copy=True)
        .add_local_dir(str(ROOT / RUN_RELATIVE / "source_videos"), f"{WORK}/{RUN_RELATIVE}/source_videos", copy=True)
        .add_local_dir(str(ROOT / RECOVERY_RELATIVE), f"{WORK}/{RECOVERY_RELATIVE}", copy=True)
        .add_local_dir(str(ROOT / CONTINUATION_CACHE_RELATIVE), f"{WORK}/{CONTINUATION_CACHE_RELATIVE}", copy=True)
        .add_local_file(str(ROOT / CONTINUED_RESULTS_RELATIVE / "replicate_manifest.json"), f"{WORK}/{CONTINUED_RESULTS_RELATIVE}/replicate_manifest.json", copy=True)
    )
    if resume:
        image = image.add_local_file(str(ROOT / RESUME_JOURNAL_RELATIVE), f"{WORK}/{RESUME_JOURNAL_RELATIVE}", copy=True)
    journal_volume = modal.Volume.from_name("reva-phase4-qcsrr-journal", create_if_missing=True) if resume else None
    app = modal.App("reva-phase4-qcsrr", tags={"experiment": "phase4-qcsrr"})

    function_kwargs = {"image": image, "gpu": "A100-40GB", "cpu": 4, "memory": 16384, "timeout": RESUME_FUNCTION_TIMEOUT_SECONDS if resume else FUNCTION_TIMEOUT_SECONDS, "serialized": True, "secrets": [modal.Secret.from_name("reva-phase3-replicate")]}
    if journal_volume is not None:
        function_kwargs["volumes"] = {RESUME_VOLUME_PATH: journal_volume}

    @app.function(**function_kwargs)
    def phase4_qcsrr() -> dict[str, bytes]:
        return _remote_entrypoint(resume_sparse=resume_sparse, continue_no_retry=continue_no_retry, postprocess_cached=postprocess_cached, resume=resume)

    return app, phase4_qcsrr


def _run_recovery(root: Path, prediction_id: str) -> int:
    """Fetch an existing prediction output without creating a new prediction."""

    import modal

    image = modal.Image.debian_slim(python_version="3.11").pip_install("replicate==1.0.7")
    app = modal.App("reva-phase4-qcsrr-recovery", tags={"experiment": "phase4-qcsrr", "purpose": "archive-recovery"})

    @app.function(image=image, cpu=1, memory=1024, timeout=120, serialized=True, secrets=[modal.Secret.from_name("reva-phase3-replicate")])
    def recover_prediction(existing_prediction_id: str) -> dict[str, Any]:
        import replicate

        client = replicate.Client()
        prediction = client.predictions.get(existing_prediction_id)
        if prediction.status not in {"succeeded", "failed", "canceled"}:
            prediction.wait()
        output = prediction.output

        def collect(value: Any) -> list[bytes]:
            if value is None:
                return []
            if hasattr(value, "read"):
                return [bytes(value.read())]
            if isinstance(value, (bytes, bytearray)):
                return [bytes(value)]
            if isinstance(value, (list, tuple)):
                result: list[bytes] = []
                for item in value:
                    result.extend(collect(item))
                return result
            if isinstance(value, dict):
                result = []
                for item in value.values():
                    result.extend(collect(item))
                return result
            return []

        return {
            "prediction_id": prediction.id,
            "status": prediction.status,
            "metrics": prediction.metrics or {},
            "error": prediction.error,
            "output_files": collect(output),
        }

    destination = root / RUN_RELATIVE / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    with app.run():
        result = recover_prediction.remote(prediction_id)
    files = list(result.pop("output_files", []))
    stamp = str(time.time_ns())
    metadata_path = destination / f"prediction-{prediction_id}-recovery-{stamp}.json"
    metadata = {**result, "output_file_count": len(files), "output_files": []}
    for index, data in enumerate(files):
        output_path = destination / f"prediction-{prediction_id}-recovery-{stamp}-{index:02d}.bin"
        output_path.write_bytes(data)
        metadata["output_files"].append({"path": str(output_path.relative_to(root)), "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "recovered", "prediction_id": prediction_id, "metadata": str(metadata_path.relative_to(root)), "output_files": metadata["output_files"]}, indent=2))
    return 0


def _next_resume_destination(root: Path) -> Path:
    base = root / RESUME_RESULTS_RELATIVE
    artifact_names = ("replicate_outputs.tar.gz", "replicate_manifest.json", "candidates.json.gz", "rankings.json.gz", "modal_report.json")
    for suffix in range(1000):
        candidate = base if suffix == 0 else base.with_name(f"{base.name}-{suffix:03d}")
        if not any((candidate / name).exists() for name in artifact_names):
            return candidate
    raise RuntimeError("could not find an unused versioned resume output directory")


def _prior_resume_legacy_attempts(root: Path, journal_attempt_count: int) -> int:
    """Recover the known pre-guard fall-through count without adding it to 30."""

    maximum = 0
    for manifest_path in sorted(root.glob(f"{RESUME_RESULTS_RELATIVE}*/replicate_manifest.json")):
        try:
            manifest = _json(manifest_path)
        except (OSError, ValueError):
            continue
        resume = manifest.get("resume")
        if not isinstance(resume, Mapping) or resume.get("mode") != "resume":
            continue
        try:
            maximum = max(maximum, int(manifest.get("attempt_count", 0)) - int(journal_attempt_count))
        except (TypeError, ValueError):
            continue
    return max(0, maximum)


def _legacy_create_loop_allowed(resume: bool) -> bool:
    """Keep the legacy create path unavailable to every durable resume."""

    return not resume


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate the plan and staged source only")
    parser.add_argument("--recover-prediction-id", help="fetch an existing Replicate output without creating a prediction")
    parser.add_argument("--resume-sparse", action="store_true", help="resume from the recovered sparse call-00 archive")
    parser.add_argument("--continue-no-retry", action="store_true", help="continue call indices 3..35 once each from cached call 0/1 outputs")
    parser.add_argument("--postprocess-cached", action="store_true", help="rank only cached successful archives; never call Replicate")
    parser.add_argument("--resume", action="store_true", help="rate-aware durable resume of only missing logical calls")
    args = parser.parse_args()
    if args.recover_prediction_id:
        return _run_recovery(ROOT, args.recover_prediction_id)
    if sum(bool(value) for value in (args.resume_sparse, args.continue_no_retry, args.postprocess_cached, args.resume)) > 1:
        raise ValueError("choose one continuation mode")
    readiness = _preflight(ROOT)
    if args.resume:
        plan = _json(ROOT / PLAN_RELATIVE)
        selection = _json(ROOT / "tasks/qcsrr_final_benchmark_selection.json")
        source_manifest = _json(ROOT / SOURCE_MANIFEST_RELATIVE)
        calls = _call_specs(plan, selection)
        journal_path = ROOT / RESUME_JOURNAL_RELATIVE
        journal = _seed_resume_journal(ROOT, plan, calls, source_manifest, path=journal_path, persist=not args.dry_run)
        pending = _pending_resume_calls(calls, journal, source_manifest, plan)
        if args.dry_run:
            print(json.dumps({"status": "ready", "mode": "resume", "pending_count": len(pending), "pending_calls": pending, "minimum_interval_seconds": RESUME_CREATE_MIN_INTERVAL_SECONDS, "max_create_attempts_per_call": RESUME_MAX_CREATE_ATTEMPTS_PER_CALL, "max_retries_total": RESUME_MAX_RETRIES, "soft_stop_seconds": RESUME_SOFT_STOP_SECONDS, "function_timeout_seconds": RESUME_FUNCTION_TIMEOUT_SECONDS}, indent=2))
            return 0
    elif args.dry_run:
        print(json.dumps(readiness, indent=2))
        return 0
    destination = _next_resume_destination(ROOT) if args.resume else ROOT / POSTPROCESS_RESULTS_RELATIVE if args.postprocess_cached else ROOT / CONTINUED_RESULTS_RELATIVE if args.continue_no_retry else ROOT / RESULTS_RELATIVE
    artifact_names = ["replicate_outputs.tar.gz", "replicate_manifest.json", "candidates.json.gz", "rankings.json.gz", "modal_report.json"]
    existing = [name for name in artifact_names if (destination / name).exists()]
    if existing:
        raise RuntimeError("refusing to overwrite existing Phase 4 artifacts: " + ", ".join(existing))
    import modal

    if args.resume_sparse:
        recovery_dir = ROOT / RECOVERY_RELATIVE
        for name in RECOVERY_CALL0:
            path = recovery_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing recovered sparse archive: {path}")
    if args.continue_no_retry:
        cache_dir = ROOT / CONTINUATION_CACHE_RELATIVE
        for name in ("call-00.masks.zip", "call-01.masks.zip"):
            path = cache_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing continuation cache archive: {path}")
    if args.postprocess_cached:
        cache_dir = ROOT / CONTINUATION_CACHE_RELATIVE
        if not (ROOT / CONTINUED_RESULTS_RELATIVE / "replicate_manifest.json").is_file():
            raise FileNotFoundError("missing continuation manifest for cached postprocess")
        cached = sorted(cache_dir.glob("call-*.masks.zip"))
        if not cached:
            raise FileNotFoundError(f"missing cached successful archives: {cache_dir}")
    app, function = _build_modal_app(resume_sparse=args.resume_sparse, continue_no_retry=args.continue_no_retry, postprocess_cached=args.postprocess_cached, resume=args.resume)
    with modal.enable_output():
        with app.run():
            result = function.remote()
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in result.items():
        path = ROOT / RESUME_JOURNAL_RELATIVE if args.resume and name == "resume_journal.json" else destination / name
        if path.exists():
            if not (args.resume and name == "resume_journal.json"):
                raise RuntimeError(f"refusing to overwrite returned artifact: {path}")
        path.write_bytes(data)
    print(json.dumps({"status": "returned", "artifacts": sorted(result), "preflight": readiness}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
