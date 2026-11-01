"""Pure planning and preflight checks for the isolated Phase 3A smoke.

This module has no provider imports and never reads a credential value.  The
Replicate and Modal entrypoints use it before staging data or making a call.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any, Mapping

import common as C


PLAN_RELATIVE_PATH = Path("tasks/phase3a_retrieval_smoke_plan.json")
PILOT_RELATIVE_PATH = Path("tasks/phase3_grounding_pilot_ids.json")
MAX_QUESTION_PHRASES = 4
FIXED_PROMPTS = ("vehicle", "building", "road", "river", "water", "field", "court", "boat", "tree", "pool")
COLORS = {"black", "blue", "red", "yellow", "white", "green", "gray", "grey"}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON manifest must contain an object: {path}")
    return value


def load_plan(root: str | Path) -> dict[str, Any]:
    plan = _read_json(Path(root) / PLAN_RELATIVE_PATH)
    if plan.get("schema_version") != 1:
        raise ValueError("unsupported Phase 3A smoke plan schema")
    if plan.get("model", {}).get("replicate_version") != "408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269":
        raise ValueError("Phase 3A smoke must use the pinned SAM 3 version")
    if plan.get("frame_policy") != {
        "sampling_mode": "legacy",
        "offset": 0.0,
        "frame_count": 16,
        "timestamp_source": "original_source_frame_index_divided_by_source_fps",
    }:
        raise ValueError("Phase 3A smoke frame policy must be the exact legacy offset-0 16-frame policy")
    videos = plan.get("videos")
    if not isinstance(videos, list) or len(videos) != 2:
        raise ValueError("Phase 3A smoke plan must select exactly two videos")
    ids = [item.get("qa_id") for item in videos if isinstance(item, Mapping)]
    if len(ids) != 2 or len(set(ids)) != 2:
        raise ValueError("Phase 3A smoke video IDs must be unique")
    if plan.get("tile_probe", {}).get("tile_count") != 4:
        raise ValueError("Phase 3A smoke tile probe must contain four calls")
    if plan.get("modal_budget", {}).get("gpu") != "A100":
        raise ValueError("Phase 3A smoke must use the approved Modal A100")
    if plan.get("modal_budget", {}).get("monetary_cap_waived_by_user") is not True:
        raise ValueError("Phase 3A smoke monetary cap waiver is missing")
    if plan.get("replicate_budget", {}).get("monetary_cap_waived_by_user") is not True:
        raise ValueError("Phase 3A Replicate monetary cap waiver is missing")
    return plan


def _overlap_spans(spans: list[dict[str, Any]], start: int, end: int) -> bool:
    return any(start < int(item["end"]) and int(item["start"]) < end for item in spans)


def extract_question_entities(question: str, vocabulary: list[str]) -> list[dict[str, Any]]:
    """Extract at most four longest question-only entity phrases in text order."""

    text = str(question)
    lowered = text.casefold()
    candidates: list[dict[str, Any]] = []
    for raw_phrase in sorted({str(item).strip() for item in vocabulary if str(item).strip()}, key=lambda item: (-len(item), item)):
        pattern = re.compile(rf"(?<![a-z0-9]){re.escape(raw_phrase.casefold())}(?![a-z0-9])")
        for match in pattern.finditer(lowered):
            phrase = text[match.start() : match.end()]
            start, end = match.start(), match.end()
            attributes: list[str] = []
            if raw_phrase.casefold() == "car":
                prefix = re.search(r"([a-z]+)\s+$", lowered[:start])
                if prefix and prefix.group(1) in COLORS:
                    start = prefix.start(1)
                    phrase = text[start:end]
                    attributes.append(prefix.group(1))
            if _overlap_spans(candidates, start, end):
                continue
            candidates.append({"phrase": phrase.casefold(), "start": start, "end": end, "attributes": attributes})
    candidates.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    return candidates[:MAX_QUESTION_PHRASES]


def _pilot_rows(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    pilot = _read_json(root / PILOT_RELATIVE_PATH)
    expected_checksums = pilot.get("source_split_checksums", {})
    rows_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split in ("valQ", "valT"):
        rows = _read_json(root / "data" / "splits" / f"{split}.json").get("QA")
        if not isinstance(rows, list):
            raise ValueError(f"{split} split has no QA list")
        if C.split_checksum(rows) != expected_checksums.get(split):
            raise ValueError(f"{split} checksum does not match the fixed pilot manifest")
        rows_by_split[split] = {str(row.get("qa_id")): row for row in rows}
    return pilot, rows_by_split


def validate_plan_rows(root: str | Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate selected rows and question-only phrases against fixed splits."""

    pilot, rows_by_split = _pilot_rows(Path(root))
    vocabulary = list(pilot.get("prompt_entities", [])) + list(FIXED_PROMPTS)
    resolved: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    for spec in plan["videos"]:
        split = str(spec.get("split"))
        qa_id = str(spec.get("qa_id"))
        row = rows_by_split.get(split, {}).get(qa_id)
        if row is None or qa_id not in set(pilot.get("splits", {}).get(split, {}).get("ids", [])):
            raise ValueError(f"smoke selection is not in the fixed pilot: {split}/{qa_id}")
        video_path = str(row.get("video_path", ""))
        if not video_path or video_path in seen_videos:
            raise ValueError("Phase 3A smoke videos must be distinct")
        seen_videos.add(video_path)
        extracted = extract_question_entities(str(row.get("question", "")), vocabulary)
        extracted_phrases = [item["phrase"] for item in extracted]
        planned_phrases = [str(item).casefold() for item in spec.get("question_phrases", [])]
        if extracted_phrases != planned_phrases:
            raise ValueError(
                f"question-only decomposition changed for {qa_id}: {extracted_phrases!r} != {planned_phrases!r}"
            )
        fixed = [str(item) for item in spec.get("fixed_prompts", [])]
        if not fixed or len(fixed) > len(FIXED_PROMPTS) or any(item not in FIXED_PROMPTS for item in fixed):
            raise ValueError(f"invalid fixed prompt selection for {qa_id}")
        resolved.append({"spec": dict(spec), "row": row, "entities": extracted})
    return resolved


def build_call_plan(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for item in plan["videos"]:
        qa_id = str(item["qa_id"])
        for prompt in item["fixed_prompts"]:
            calls.append({"qa_id": qa_id, "arm": "fixed-vocabulary", "prompt": str(prompt), "tile": None})
        for phrase in item["question_phrases"]:
            calls.append({"qa_id": qa_id, "arm": "question-conditioned", "prompt": str(phrase), "tile": None})
    tile = plan["tile_probe"]
    for index in range(int(tile["tile_count"])):
        calls.append(
            {
                "qa_id": str(tile["qa_id"]),
                "arm": "tile-probe",
                "prompt": str(tile["question_phrase"]),
                "tile": index,
            }
        )
    if len(calls) > int(plan["replicate_budget"]["max_attempts"]):
        raise ValueError("planned Phase 3A calls exceed the Replicate attempt cap")
    return calls


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any("token" in str(key).casefold() or "secret" in str(key).casefold() or "password" in str(key).casefold() or _contains_secret_key(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False


def preflight(
    root: str | Path,
    plan: Mapping[str, Any],
    attestation_path: str | Path | None,
    *,
    backend: str = "modal",
    environ: Mapping[str, str] | None = None,
    modal_cli: str | None = None,
) -> dict[str, Any]:
    """Return a safe, non-secret readiness result; never return credential values."""

    if backend not in {"modal", "hpc"}:
        raise ValueError(f"unsupported Phase 3A preflight backend: {backend}")
    env = os.environ if environ is None else environ
    reasons: list[str] = []
    attestation: dict[str, Any] = {}
    if attestation_path is None:
        reasons.append("no provider preflight attestation was supplied")
    else:
        try:
            attestation = _read_json(Path(attestation_path))
        except (OSError, RuntimeError, ValueError) as exc:
            reasons.append(str(exc))
        if _contains_secret_key(attestation):
            reasons.append("preflight attestation contains a credential-like key; credentials must not be persisted")

    credential_rotated = attestation.get("replicate_credential_rotated") is True
    credential_reuse_authorized = attestation.get("replicate_credential_reuse_authorized") is True
    if not credential_rotated and not credential_reuse_authorized:
        reasons.append(
            "preflight attestation requires replicate_credential_rotated=true "
            "or replicate_credential_reuse_authorized=true"
        )

    required_true = [
        "replicate_endpoint_schema_verified",
        "replicate_version_verified",
    ]
    if backend == "modal":
        required_true.extend(("modal_credentials_available", "modal_runtime_available"))
    for key in required_true:
        if attestation.get(key) is not True:
            reasons.append(f"preflight attestation is missing {key}=true")
    if not env.get("REPLICATE_API_TOKEN"):
        reasons.append("REPLICATE_API_TOKEN is absent from the process environment")
    if backend == "modal" and modal_cli is None:
        modal_cli = shutil.which("modal")
    if backend == "modal" and not modal_cli:
        reasons.append("modal CLI is unavailable")

    return {
        "ready": not reasons,
        "backend": "Modal" if backend == "modal" else "HPC",
        "replicate_credential_status": (
            "rotated" if credential_rotated else "reuse_authorized" if credential_reuse_authorized else "unverified"
        ),
        "credential_present": bool(env.get("REPLICATE_API_TOKEN")),
        "modal_cli_present": bool(modal_cli) if backend == "modal" else False,
        "reasons": reasons,
        "planned_replicate_calls": len(build_call_plan(plan)),
        "monetary_cap_status": "waived_by_user_for_two_video_smoke; provider charges are not locally bounded",
        "operational_limits": {
            "replicate_attempts": int(plan["replicate_budget"]["max_attempts"]),
            "replicate_retries": int(plan["replicate_budget"]["max_retries"]),
            "modal_gpu_minutes": int(plan["modal_budget"]["max_gpu_minutes"]),
            "modal_wall_minutes": int(plan["modal_budget"]["max_wall_minutes"]),
        },
    }


def tile_bounds(width: int, height: int, overlap: float = 0.25) -> list[dict[str, int]]:
    """Return four source-pixel tiles with overlap measured within each tile."""

    if width <= 0 or height <= 0 or not 0 < overlap < 1:
        raise ValueError("tile dimensions and overlap must be positive")
    tile_width = min(width, math.ceil((width / 2) / (1 - overlap)))
    tile_height = min(height, math.ceil((height / 2) / (1 - overlap)))
    xs = [0, max(0, width - tile_width)]
    ys = [0, max(0, height - tile_height)]
    return [{"left": x, "top": y, "right": x + tile_width, "bottom": y + tile_height} for y in ys for x in xs]


def source_frame_policy(video_info: Mapping[str, Any]) -> dict[str, Any]:
    indices = C.sample_index_list(int(video_info["n_frames"]), float(video_info["fps"]), 0.0, mode="legacy")
    if len(indices) != 16:
        raise ValueError("Phase 3A smoke requires exactly 16 legacy offset-0 source frames")
    return {
        "sampling_mode": "legacy",
        "offset": 0.0,
        "source_frame_indices": indices,
        "source_timestamps": [round(index / max(float(video_info["fps"]), 1e-6), 9) for index in indices],
    }

