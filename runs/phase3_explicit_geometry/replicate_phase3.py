"""Create question-entity-only SAM3 masks for the approved Phase 3 scope.

The token is read only from the process environment. It is never printed,
written to a manifest, or passed as a command-line argument.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
import tempfile
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(RUN))
import replicate_sam3 as R
import sam3_overlay as M
import common as C
from geometry_relations import question_entities


# Replicate currently throttles low-balance accounts to roughly six
# prediction-create requests per minute. Keep a wider local interval between
# request starts so a resumed run does not immediately hit the same limit.
SAM3_REQUEST_INTERVAL_SECONDS = 10.5
SAM3_429_MAX_RETRIES = 2
SAM3_429_BACKOFF_SECONDS = (5.5, 10.5)


def _wait_for_request_slot(
    last_start: float | None,
    *,
    clock=time.monotonic,
    sleeper=time.sleep,
    interval: float = SAM3_REQUEST_INTERVAL_SECONDS,
) -> float:
    """Wait until the next request may start and return its start timestamp."""

    now = float(clock())
    if last_start is not None:
        delay = float(interval) - (now - float(last_start))
        if delay > 0:
            sleeper(delay)
            now = float(clock())
    return now


def _is_transient_429(exc: BaseException) -> bool:
    """Recognize Replicate rate-limit failures without persisting exception text."""

    text = f"{type(exc).__name__}: {exc}".casefold()
    return "429" in text or "rate limit" in text or "too many requests" in text


def _self_check_request_schedule() -> None:
    """Check that the injected scheduler enforces the required interval."""

    current = [0.0]
    sleeps: list[float] = []

    def clock() -> float:
        return current[0]

    def sleeper(delay: float) -> None:
        sleeps.append(float(delay))
        current[0] += float(delay)

    first = _wait_for_request_slot(None, clock=clock, sleeper=sleeper)
    current[0] += 1.0
    second = _wait_for_request_slot(first, clock=clock, sleeper=sleeper)
    if second - first < SAM3_REQUEST_INTERVAL_SECONDS:
        raise AssertionError("SAM3 request scheduler did not enforce interval")
    if not sleeps or sleeps[0] <= 0:
        raise AssertionError("SAM3 request scheduler did not sleep for a busy slot")
    if not _is_transient_429(RuntimeError("HTTP 429 Too Many Requests")):
        raise AssertionError("SAM3 429 classifier failed")
    if len(SAM3_429_BACKOFF_SECONDS) < SAM3_429_MAX_RETRIES or any(
        float(delay) <= 0 for delay in SAM3_429_BACKOFF_SECONDS[:SAM3_429_MAX_RETRIES]
    ):
        raise AssertionError("SAM3 429 backoff policy is not bounded and positive")


_self_check_request_schedule()


def _download_output(output: Any, destination: Path) -> None:
    url = output if isinstance(output, str) else getattr(output, "url", None)
    if url:
        with urllib.request.urlopen(str(url), timeout=300) as response:
            destination.write_bytes(response.read())
        return
    if hasattr(output, "read"):
        destination.write_bytes(output.read())
        return
    if isinstance(output, (bytes, bytearray)):
        destination.write_bytes(bytes(output))
        return
    try:
        destination.write_bytes(b"".join(bytes(chunk) for chunk in output))
    except Exception as exc:
        raise TypeError("Replicate returned an unsupported output type") from exc


def _selected(selection: dict[str, Any], split: str, scope: str) -> list[dict[str, Any]]:
    rows = list(selection["selected_questions"][split])
    if scope == "tranche":
        wanted = set(selection["tranche"]["ids"][split])
        rows = [row for row in rows if row["qa_id"] in wanted]
    return rows


def _archive_name(video_path: str, prompt: str) -> str:
    video_key = hashlib.sha256(video_path.encode()).hexdigest()[:10]
    safe = "-".join(part for part in prompt.casefold().split() if part.isalnum()) or "entity"
    return f"{video_key}.{safe}.masks.zip"


def _legacy_union_indices(video_info: dict[str, Any]) -> list[int]:
    indices: set[int] = set()
    for offset in (0.0, 0.5):
        indices.update(
            C.sample_index_list(
                int(video_info["n_frames"]), float(video_info["fps"]), offset, mode="legacy"
            )
        )
    return sorted(indices)


def _write_legacy_video(source: Path, destination: Path, source_indices: list[int]) -> None:
    """Encode exactly the Stage A legacy 0/0.5 union for a SAM3 call."""

    import av

    wanted = set(source_indices)
    if len(wanted) != len(source_indices) or not source_indices:
        raise ValueError("legacy source indices must be non-empty and unique")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        with av.open(str(destination), mode="w") as output_container:
            output_stream = output_container.add_stream("mpeg4", rate=Fraction(str(C.FPS_TARGET)))
            output_stream.width = int(input_stream.width)
            output_stream.height = int(input_stream.height)
            output_stream.pix_fmt = "yuv420p"
            for source_index, frame in enumerate(input_container.decode(input_stream)):
                if source_index not in wanted:
                    continue
                output_frame = av.VideoFrame.from_ndarray(frame.to_ndarray(format="rgb24"), format="rgb24")
                for packet in output_stream.encode(output_frame):
                    output_container.mux(packet)
                written += 1
                if written == len(source_indices):
                    break
            for packet in output_stream.encode():
                output_container.mux(packet)
    if written != len(source_indices):
        raise RuntimeError(f"legacy frame extraction wrote {written}; expected {len(source_indices)}")


def run(*, scope: str, data_root: Path, manifest_path: Path, selection_path: Path, output_root: Path) -> dict[str, Any]:
    if scope not in {"tranche", "full"}:
        raise ValueError("scope must be tranche or full")
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN is required for SAM3; no API call was made")
    try:
        import replicate
    except ModuleNotFoundError as exc:
        raise RuntimeError("the official Replicate Python package is required") from exc
    selection = json.loads(selection_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    rows = [row for split in ("valQ", "valT") for row in _selected(selection, split, scope)]
    by_video_entities: dict[str, set[str]] = {}
    for row in rows:
        entities = question_entities(row["question"])
        if set(entities) != set(row["sam3_prompt_entities"]):
            raise RuntimeError(f"selection prompt entities drifted for {row['qa_id']}")
        by_video_entities.setdefault(row["video_path"], set()).update(entities)
    raw_root = output_root / "sam3_raw"
    overlay_root = output_root / "overlay_cache"
    raw_root.mkdir(parents=True, exist_ok=True)
    calls: list[dict[str, Any]] = []
    planned_pairs = [
        (video_path, prompt)
        for video_path in sorted(by_video_entities)
        for prompt in sorted(by_video_entities[video_path])
    ]
    planned = len(planned_pairs)
    max_attempts = int(selection["cost_gate"]["max_paid_attempts_under_cap"])
    if planned > max_attempts:
        raise RuntimeError(f"planned SAM3 calls exceed hard attempt cap: {planned} > {max_attempts}")
    prior_manifests = [output_root / "sam3_api_manifest.json", output_root / "sam3_api_manifest.partial.json"]
    prior_paid = int(selection["cost_gate"].get("prior_paid_attempt_count", 0))
    attempt_log: list[dict[str, Any]] = []
    for prior in prior_manifests:
        if prior.is_file():
            try:
                prior_info = json.loads(prior.read_text())
                prior_paid = max(prior_paid, int(prior_info.get("paid_attempt_count_total", 0)))
                prior_attempts = prior_info.get("attempt_log", [])
                if isinstance(prior_attempts, list):
                    attempt_log.extend(item for item in prior_attempts if isinstance(item, dict))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
    existing_archives = {
        raw_root / _archive_name(video_path, prompt)
        for video_path, prompt in planned_pairs
        if (raw_root / _archive_name(video_path, prompt)).is_file()
        and R.cached_mask_info(raw_root / _archive_name(video_path, prompt)) is not None
    }
    # A valid archive is treated as one already-paid attempt if no progress
    # manifest exists. This conservative count prevents a resumed run from
    # spending past the cap after an interrupted process.
    prior_paid = max(prior_paid, len(existing_archives))
    fresh_planned = planned - len(existing_archives)
    if prior_paid + fresh_planned > max_attempts:
        raise RuntimeError(
            f"resumed SAM3 attempts would exceed hard cap: {prior_paid} prior + "
            f"{fresh_planned} new > {max_attempts}"
        )
    paid_attempts = prior_paid
    last_request_start: float | None = None

    def write_partial() -> None:
        (output_root / "sam3_api_manifest.partial.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scope": scope,
                    "model": R.REPLICATE_MODEL,
                    "version": R.REPLICATE_VERSION,
                    "planned_call_count": planned,
                    "paid_attempt_count_total": paid_attempts,
                    "request_interval_seconds": SAM3_REQUEST_INTERVAL_SECONDS,
                    "max_429_retries": SAM3_429_MAX_RETRIES,
                    "prompt_policy": "question entity strings only; no options, gold, or expected locations",
                    "attempt_log": attempt_log,
                    "calls": calls,
                    "complete": False,
                },
                indent=2,
            )
            + "\n"
        )

    manifest_info = json.loads(manifest_path.read_text())
    with tempfile.TemporaryDirectory(prefix="phase3_sam3_", dir="/tmp") as temp_dir:
      temp_root = Path(temp_dir)
      for video_path in sorted(by_video_entities):
        source = data_root / video_path
        if not source.is_file():
            raise FileNotFoundError(source)
        if video_path not in manifest_info:
            raise RuntimeError(f"video manifest is missing selected clip: {video_path}")
        source_indices = _legacy_union_indices(manifest_info[video_path])
        source_key = hashlib.sha256(video_path.encode()).hexdigest()[:10]
        legacy_video = temp_root / f"{source_key}.legacy-union.mp4"
        _write_legacy_video(source, legacy_video, source_indices)
        for prompt in sorted(by_video_entities[video_path]):
            archive = raw_root / _archive_name(video_path, prompt)
            mapping_path = archive.with_suffix(".mapping.json")
            cached = R.cached_mask_info(archive)
            mapping = None
            if mapping_path.is_file():
                try:
                    mapping = json.loads(mapping_path.read_text())
                except (OSError, TypeError, ValueError, json.JSONDecodeError):
                    mapping = None
            reusable = (
                cached is not None
                and isinstance(mapping, dict)
                and mapping.get("source_frame_indices") == source_indices
                and all(0 <= int(index) < len(source_indices) for index in cached["frame_indices"])
            )
            if reusable:
                calls.append(
                    {
                        "video_path": video_path,
                        "prompt": prompt,
                        "path": archive.name,
                        "reused": True,
                        "input_policy": "legacy-union-fallback",
                        "source_frame_indices": source_indices,
                    }
                )
                continue
            payload = R.prediction_input(legacy_video, prompt)
            output = None
            attempts_for_prompt = 0
            for retry_index in range(SAM3_429_MAX_RETRIES + 1):
                if paid_attempts >= max_attempts:
                    raise RuntimeError(f"SAM3 paid attempt cap reached before {video_path}/{prompt}")
                if retry_index:
                    backoff = SAM3_429_BACKOFF_SECONDS[min(retry_index - 1, len(SAM3_429_BACKOFF_SECONDS) - 1)]
                    time.sleep(float(backoff))
                last_request_start = _wait_for_request_slot(last_request_start)
                paid_attempts += 1
                attempts_for_prompt += 1
                attempt = {
                    "video_path": video_path,
                    "prompt": prompt,
                    "retry_index": retry_index,
                    "status": "started",
                }
                attempt_log.append(attempt)
                # Persist the increment before entering the API call.  A
                # process killed during the request therefore resumes within
                # the hard cap and does not retry a counted attempt blindly.
                write_partial()
                try:
                    with legacy_video.open("rb") as video_stream:
                        output = replicate.run(R.prediction_ref(), input={**payload, "video": video_stream})
                except Exception as exc:
                    rate_limited = _is_transient_429(exc)
                    attempt["status"] = "rate_limited" if rate_limited else "failed"
                    write_partial()
                    if rate_limited and retry_index < SAM3_429_MAX_RETRIES:
                        continue
                    if rate_limited:
                        raise RuntimeError(
                            f"Replicate SAM3 rate limit persisted for {video_path}/{prompt} "
                            f"after {attempts_for_prompt} attempts"
                        ) from exc
                    raise RuntimeError(
                        f"Replicate SAM3 prediction failed for {video_path}/{prompt} "
                        f"({type(exc).__name__})"
                    ) from exc
                attempt["status"] = "succeeded"
                write_partial()
                break
            if output is None:
                raise RuntimeError(f"Replicate returned no output for {video_path}/{prompt}")
            _download_output(output, archive)
            info = R.cached_mask_info(archive)
            if info is None:
                raise RuntimeError(f"Replicate returned invalid mask archive for {video_path}/{prompt}")
            if any(not 0 <= int(index) < len(source_indices) for index in info["frame_indices"]):
                raise RuntimeError(f"Replicate returned frame outside legacy input for {video_path}/{prompt}")
            mapping_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "mode": "legacy-union-fallback",
                        "source_frame_indices": source_indices,
                        "input_frame_indices": list(range(len(source_indices))),
                    },
                    indent=2,
                )
                + "\n"
            )
            calls.append(
                {
                    "video_path": video_path,
                    "prompt": prompt,
                    "path": archive.name,
                    "reused": False,
                    "mask_count": info["mask_count"],
                    "prediction_attempts": attempts_for_prompt,
                    "input_policy": "legacy-union-fallback",
                    "source_frame_indices": source_indices,
                }
            )
            write_partial()
      for row in rows:
          prompts = []
          for prompt in row["sam3_prompt_entities"]:
              archive = raw_root / _archive_name(row["video_path"], prompt)
              mapping = json.loads(archive.with_suffix(".mapping.json").read_text())
              prompts.append((prompt, archive, dict(zip(mapping["input_frame_indices"], mapping["source_frame_indices"]))))
          info = manifest[row["video_path"]]
          M.write_overlay_cache(
              overlay_root,
              qa_id=row["qa_id"],
              video_path=row["video_path"],
              prompt_zips=prompts,
              fps=float(info["fps"]),
              frame_count=int(info["n_frames"]),
          )
    if len(calls) != planned:
        raise RuntimeError(f"SAM3 call manifest count mismatch: {len(calls)} != {planned}")
    result = {
        "schema_version": 1,
        "scope": scope,
        "model": R.REPLICATE_MODEL,
        "version": R.REPLICATE_VERSION,
        "planned_call_count": planned,
        "call_count": sum(not call["reused"] for call in calls),
        "reused_call_count": sum(bool(call["reused"]) for call in calls),
        "paid_attempt_count_total": paid_attempts,
        "prediction_attempt_count_total": paid_attempts,
        "max_paid_attempts_under_cap": max_attempts,
        "request_interval_seconds": SAM3_REQUEST_INTERVAL_SECONDS,
        "max_429_retries": SAM3_429_MAX_RETRIES,
        "observed_conservative_cost_usd_per_call": float(selection["cost_gate"]["conservative_rate_usd_per_call"]),
        "conservative_cost_usd": sum(not call["reused"] for call in calls) * float(selection["cost_gate"]["conservative_rate_usd_per_call"]),
        "cap_usd": float(selection["cost_gate"]["cap_usd"]),
        "prompt_policy": "question entity strings only; no options, gold, or expected locations",
        "attempt_log": attempt_log,
        "calls": calls,
        "selected_question_count": len(rows),
        "complete": True,
    }
    (output_root / "sam3_api_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    (output_root / "sam3_api_manifest.partial.json").unlink(missing_ok=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("tranche", "full"), required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--selection", default="runs/phase3_explicit_geometry/selection_manifest.json")
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    result = run(
        scope=args.scope,
        data_root=Path(args.data_root),
        manifest_path=Path(args.manifest),
        selection_path=ROOT / args.selection,
        output_root=Path(args.output_root),
    )
    print(json.dumps({key: result[key] for key in ("scope", "planned_call_count", "call_count", "reused_call_count", "conservative_cost_usd")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
