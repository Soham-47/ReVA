"""Pure helpers for the isolated Replicate SAM 3 video pilot."""

from __future__ import annotations

import io
import re
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


REPLICATE_MODEL = "lucataco/sam3-video"
# Pinned from the public Replicate versions page on 2026-10-16.
REPLICATE_VERSION = "408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269"
REPLICATE_API_BASE = "https://api.replicate.com/v1"
REPLICATE_APPROX_RUN_USD = 0.018
OUTPUT_REQUIRED_INPUTS = {"mask_only": True, "return_zip": True}


def prediction_ref() -> str:
    return f"{REPLICATE_MODEL}:{REPLICATE_VERSION}"


def prediction_input(video: str | Path, prompt: str) -> dict[str, Any]:
    path = Path(video)
    if not path.is_file():
        raise FileNotFoundError(path)
    prompt = str(prompt).strip()
    if not prompt:
        raise ValueError("SAM 3 prompt must be non-empty")
    return {
        "video": path,
        "prompt": prompt,
        "mask_only": True,
        "return_zip": True,
    }


def _safe_member(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def frame_index_from_name(name: str, fallback: int) -> int:
    stem = Path(name).stem
    match = re.search(r"(?:frame|mask)[-_]?(\d+)", stem, flags=re.IGNORECASE)
    return int(match.group(1)) if match else int(fallback)


def map_frame_indices(frame_indices: Iterable[int], source_frame_indices: Iterable[int]) -> list[int]:
    """Map output-video frame ordinals back to original source-frame indices."""

    source = [int(index) for index in source_frame_indices]
    mapped: list[int] = []
    for index in frame_indices:
        ordinal = int(index)
        if not 0 <= ordinal < len(source):
            raise ValueError(f"output frame index {ordinal} is outside fallback frame range")
        mapped.append(source[ordinal])
    return mapped


def _read_mask_archive(
    data: bytes | bytearray | io.BufferedIOBase,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate a mask ZIP and return PNG entries plus optional empty-output video."""

    if hasattr(data, "read"):
        data = data.read()
    raw = bytes(data)
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if any(not _safe_member(item.filename) for item in members):
            raise ValueError("SAM 3 output ZIP contains an unsafe member path")
        names = [item.filename for item in members if item.filename.lower().endswith(".png")]
        overlay_names = [
            item.filename
            for item in members
            if Path(item.filename).name.casefold() == "overlay.mp4"
        ]
        if not names and not overlay_names:
            raise ValueError("SAM 3 output ZIP contains neither PNG masks nor overlay.mp4")
        overlays = []
        for name in overlay_names:
            payload = archive.read(name)
            if not payload:
                raise ValueError(f"SAM 3 empty-output overlay is empty: {name}")
            overlays.append({"member": name, "bytes": len(payload)})
        entries = []
        for ordinal, name in enumerate(names):
            payload = archive.read(name)
            if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"SAM 3 output member is not a PNG: {name}")
            entries.append(
                {
                    "member": name,
                    "frame_index": frame_index_from_name(name, ordinal),
                    "payload": payload,
                    "bytes": len(payload),
                }
            )
    frame_indices = [int(entry["frame_index"]) for entry in entries]
    if len(frame_indices) != len(set(frame_indices)):
        raise ValueError("SAM 3 output ZIP maps multiple masks to one frame")
    return entries, overlays


def read_mask_zip(data: bytes | bytearray | io.BufferedIOBase) -> list[dict[str, Any]]:
    """Read and validate the model's ``return_zip=true`` frame-mask output."""

    entries, _ = _read_mask_archive(data)
    return entries


def inspect_mask_zip(data: bytes | bytearray | io.BufferedIOBase) -> dict[str, Any]:
    """Return compact metadata after validating a ``return_zip`` output."""

    entries, overlays = _read_mask_archive(data)
    return {
        "mask_count": len(entries),
        "frame_indices": [int(entry["frame_index"]) for entry in entries],
        "members": [
            {key: value for key, value in entry.items() if key != "payload"}
            for entry in entries
        ],
        "overlay_members": overlays,
    }


def cached_mask_info(path: str | Path) -> dict[str, Any] | None:
    """Return validated metadata for a reusable output ZIP, if one exists."""

    candidate = Path(path)
    if not candidate.is_file() or candidate.stat().st_size == 0:
        return None
    try:
        return inspect_mask_zip(candidate.read_bytes())
    except (OSError, ValueError, zipfile.BadZipFile):
        return None


def subset_counts(subset: Mapping[str, Any]) -> tuple[int, int]:
    """Return selected question and prompt-call counts from a run manifest."""

    split_specs = subset.get("splits")
    prompts = subset.get("prompts")
    if not isinstance(split_specs, Mapping) or not isinstance(prompts, Mapping):
        raise ValueError("SAM 3 subset must contain splits and prompts")
    ids: list[str] = []
    for split, spec in split_specs.items():
        if not isinstance(spec, Mapping) or not isinstance(spec.get("ids"), list):
            raise ValueError(f"SAM 3 subset split {split!r} must contain an ids list")
        split_ids = [str(qa_id) for qa_id in spec["ids"]]
        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"SAM 3 subset split {split!r} contains duplicate IDs")
        ids.extend(split_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("SAM 3 subset contains duplicate IDs across splits")
    if set(prompts) != set(ids):
        raise ValueError("SAM 3 subset prompts must cover exactly the selected IDs")
    call_count = 0
    for qa_id in ids:
        values = prompts[qa_id]
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)):
            raise ValueError(f"SAM 3 prompts for {qa_id!r} must be a string list")
        clean = [str(value).strip() for value in values if str(value).strip()]
        if not clean:
            raise ValueError(f"SAM 3 prompts for {qa_id!r} are empty")
        call_count += len(clean)
    return len(ids), call_count


def validate_sam3_manifest(manifest: Mapping[str, Any], subset: Mapping[str, Any]) -> tuple[int, int]:
    """Validate a completed API manifest against its own staged subset."""

    expected_ids, expected_calls = subset_counts(subset)
    completed_calls = int(manifest.get("call_count", 0)) + int(manifest.get("reused_call_count", 0))
    if (
        not manifest.get("complete")
        or int(manifest.get("planned_call_count", 0)) != expected_calls
        or completed_calls != expected_calls
        or int(manifest.get("selected_video_count", 0)) != expected_ids
        or int(manifest.get("full_subset_video_count", 0)) != expected_ids
    ):
        raise RuntimeError("SAM 3 API manifest is incomplete or does not match its subset")
    return expected_ids, expected_calls


def selected_rows(
    split_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    prompts: Mapping[str, str | Iterable[str]],
) -> list[dict[str, Any]]:
    """Resolve and validate a fixed question subset from split JSON."""

    resolved: list[dict[str, Any]] = []
    seen_videos: set[str] = set()
    for split, rows in split_rows.items():
        by_id = {str(row.get("qa_id")): row for row in rows}
        for qa_id, prompt in prompts.items():
            row = by_id.get(qa_id)
            if row is None:
                continue
            video = str(row.get("video_path", ""))
            if video in seen_videos:
                raise ValueError(f"subset contains duplicate video: {video}")
            seen_videos.add(video)
            prompt_list = [str(prompt)] if isinstance(prompt, str) else [str(item) for item in prompt]
            prompt_list = [item.strip() for item in prompt_list if item.strip()]
            if not prompt_list:
                raise ValueError(f"subset prompt list is empty: {qa_id}")
            resolved.append(
                {"split": split, "prompt": prompt_list[0], "prompts": prompt_list, **dict(row)}
            )
    if len(resolved) != len(prompts):
        missing = sorted(set(prompts) - {row["qa_id"] for row in resolved})
        raise ValueError(f"subset IDs missing from split files: {missing}")
    return resolved
