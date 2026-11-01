"""Direct SAM 3 mask caches for visual-only VLM scoring.

This cache intentionally does not use Grounding DINO entities or SAM 2 cache
identity.  Every mask returned for a direct text prompt is available to the
overlay scorer, which keeps the direct-prompt arm independent of detector
selection.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

import replicate_sam3 as R


SAM3_OVERLAY_VERSION = "sam3-direct-overlay-v1"
SAM3_CACHE_SCHEMA_VERSION = 1


def output_path(root: str | Path, qa_id: str) -> Path:
    return Path(root) / f"{qa_id}.json"


def score_output_path(score_path: str | Path) -> Path:
    """Return a versioned sibling score path dedicated to direct SAM 3 masks."""

    path = Path(score_path)
    suffix = path.suffix or ".jsonl"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    tag = f"segmentation.{SAM3_OVERLAY_VERSION}"
    if path.name.endswith(f".{tag}{suffix}"):
        return path
    return path.with_name(f"{stem}.{tag}{suffix}")


def _content_digest(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("content_digest", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mask_geometry(image: Image.Image) -> tuple[float, dict[str, float], dict[str, float]]:
    values = np.asarray(image.convert("L")) > 0
    height, width = values.shape[:2]
    area_pixels = int(values.sum())
    if not area_pixels:
        return 0.0, {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}, {"x": 0.0, "y": 0.0}
    ys, xs = np.nonzero(values)
    area = area_pixels / float(width * height)
    box = {
        "left": float(xs.min() / width),
        "top": float(ys.min() / height),
        "right": float((xs.max() + 1) / width),
        "bottom": float((ys.max() + 1) / height),
    }
    centroid = {"x": float(xs.mean() / width), "y": float(ys.mean() / height)}
    return area, box, centroid


def write_overlay_cache(
    root: str | Path,
    *,
    qa_id: str,
    video_path: str,
    prompt_zips: Iterable[tuple[str, str | Path] | tuple[str, str | Path, Mapping[int, int]]],
    fps: float,
    frame_count: int,
) -> Path:
    """Expand prompt ZIPs into a cache consumable by the VLM overlay scorer."""

    cache_root = Path(root)
    mask_root = cache_root / "masks" / qa_id
    mask_root.mkdir(parents=True, exist_ok=True)
    prompts: list[str] = []
    frames: dict[int, dict[str, Any]] = {}
    total_masks = 0
    for call_index, call in enumerate(prompt_zips):
        if len(call) == 2:
            prompt, zip_path = call
            frame_map: Mapping[int, int] = {}
        elif len(call) == 3:
            prompt, zip_path, frame_map = call
        else:
            raise ValueError("SAM 3 prompt ZIP entries must contain prompt, path, and optional frame map")
        prompt = str(prompt).strip()
        if not prompt:
            raise ValueError("SAM 3 prompts must be non-empty")
        prompts.append(prompt)
        entries = R.read_mask_zip(Path(zip_path).read_bytes())
        for ordinal, entry in enumerate(entries):
            raw_frame_index = int(entry["frame_index"])
            frame_index = int(frame_map.get(raw_frame_index, raw_frame_index))
            if not 0 <= frame_index < int(frame_count):
                raise ValueError(f"SAM 3 frame index {frame_index} is outside video frame range")
            payload = entry["payload"]
            with Image.open(io.BytesIO(payload)) as image:
                image = image.convert("L")
                area, box, centroid = _mask_geometry(image)
                member_path = mask_root / f"call-{call_index:02d}-frame-{frame_index:06d}-{ordinal:04d}.png"
                image.save(member_path)
                height, width = image.height, image.width
            relative = member_path.relative_to(cache_root).as_posix()
            mask = {
                "mask_id": f"sam3-call-{call_index:02d}-frame-{frame_index:06d}-{ordinal:04d}",
                "frame_index": frame_index,
                "timestamp": frame_index / max(float(fps), 1e-6),
                "detection_index": total_masks,
                "label": prompt,
                "confidence": 1.0,
                "area": area,
                "centroid": centroid,
                "box": box,
                "mask_path": relative,
            }
            frames.setdefault(
                frame_index,
                {
                    "frame_index": frame_index,
                    "timestamp": mask["timestamp"],
                    "width": width,
                    "height": height,
                    "masks": [],
                },
            )["masks"].append(mask)
            total_masks += 1
    # Questions with no extracted entities make no SAM 3 requests.  Preserve
    # the normal cache schema with an empty prompt/mask set so downstream
    # relation-context generation can mark the evidence insufficient and use
    # the frozen baseline path.
    record: dict[str, Any] = {
        "schema_version": SAM3_CACHE_SCHEMA_VERSION,
        "overlay_version": SAM3_OVERLAY_VERSION,
        "qa_id": qa_id,
        "video_path": video_path,
        "prompts": prompts,
        "frame_count": int(frame_count),
        "fps": float(fps),
        "mask_count": total_masks,
        "frames": [frames[index] for index in sorted(frames)],
    }
    record["content_digest"] = _content_digest(record)
    path = output_path(cache_root, qa_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return path


def load_overlay_cache(root: str | Path, qa_id: str) -> dict[str, Any]:
    path = output_path(root, qa_id)
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"SAM 3 overlay cache is missing for {qa_id}: {path}") from exc
    if record.get("schema_version") != SAM3_CACHE_SCHEMA_VERSION or record.get("overlay_version") != SAM3_OVERLAY_VERSION:
        raise ValueError(f"unsupported SAM 3 overlay cache: {path}")
    if record.get("qa_id") != qa_id or record.get("content_digest") != _content_digest(record):
        raise ValueError(f"SAM 3 overlay cache digest or question mismatch: {path}")
    masks: list[dict[str, Any]] = []
    for frame in record.get("frames", []):
        for mask in frame.get("masks", []):
            mask = dict(mask)
            mask_path = Path(str(mask.get("mask_path", "")))
            if mask_path.is_absolute() or ".." in mask_path.parts or not (Path(root) / mask_path).is_file():
                raise ValueError(f"SAM 3 overlay mask file is missing or unsafe: {mask_path}")
            masks.append(mask)
    return {
        "cache_key": str(record["content_digest"]),
        "cache_content_digest": str(record["content_digest"]),
        "prompts": list(record["prompts"]),
        "masks": masks,
        "mask_count": len(masks),
        "path": str(path),
    }


def masks_for_frame_indices(
    masks: Iterable[Mapping[str, Any]], frame_indices: Iterable[int]
) -> list[dict[str, Any]]:
    """Select only masks that overlap the VLM's sampled source frames."""

    selected = {int(index) for index in frame_indices}
    return [
        dict(mask)
        for mask in masks
        if int(mask.get("frame_index", -1)) in selected
    ]
