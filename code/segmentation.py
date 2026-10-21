"""Deterministic SAM 2 mask caches and question-conditioned visual overlays."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

import grounding as G


SEGMENTATION_SCHEMA_VERSION = 1
# Keep visual-overlay scores distinct from the earlier text-context design.
SEGMENTATION_VERSION = "sam2-box-overlay-v1"
SEGMENTATION_MODEL_ID = "facebook/sam2-hiera-tiny"
# Immutable HF revision; the local marker is checked before loading weights.
SEGMENTATION_MODEL_REVISION = "7c218beaf0bb87874785f32b582f640134fc1c09"
SEGMENTATION_MODEL_REVISION_ID = (
    f"{SEGMENTATION_MODEL_ID}@{SEGMENTATION_MODEL_REVISION}"
)
SEGMENTATION_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "processor_config.json",
)
SEGMENTATION_CACHE_MARKER = ".reva_sam2_revision"
SEGMENTATION_CACHE_KEY_FIELDS = (
    "video_checksum",
    "segmentation_revision",
    "grounding_cache_key",
    "frame_policy",
    "mask_policy",
)
DEFAULT_MAX_MASKS_PER_FRAME = 6
DEFAULT_MASK_THRESHOLD = 0.0
SEGMENTATION_MASK_ALGORITHM = "sam2-box-prompt"
SEGMENTATION_MODEL_OUTPUT = "single-mask"


def segmentation_mask_policy(max_masks_per_frame: int = DEFAULT_MAX_MASKS_PER_FRAME) -> dict[str, Any]:
    """Return the cache identity shared by mask building and score loading."""

    if int(max_masks_per_frame) <= 0:
        raise ValueError("max_masks_per_frame must be positive")
    return {
        "algorithm": SEGMENTATION_MASK_ALGORITHM,
        "max_masks_per_frame": int(max_masks_per_frame),
        "model_output": SEGMENTATION_MODEL_OUTPUT,
    }


def _payload(
    video_checksum: str,
    segmentation_revision: str,
    grounding_cache_key: str,
    frame_policy: Mapping[str, Any],
    mask_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if not all(isinstance(value, Mapping) for value in (frame_policy, mask_policy)):
        raise ValueError("segmentation cache policies must be mappings")
    payload = {
        "video_checksum": str(video_checksum),
        "segmentation_revision": str(segmentation_revision),
        "grounding_cache_key": str(grounding_cache_key),
        "frame_policy": dict(frame_policy),
        "mask_policy": dict(mask_policy),
    }
    if not all(payload[key] for key in ("video_checksum", "segmentation_revision", "grounding_cache_key")):
        raise ValueError("segmentation cache identity fields are required")
    try:
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("segmentation cache identity must be JSON-serializable") from exc
    return payload


def segmentation_cache_key(
    video_checksum: str,
    segmentation_revision: str = SEGMENTATION_MODEL_REVISION_ID,
    grounding_cache_key: str = "",
    frame_policy: Mapping[str, Any] | None = None,
    mask_policy: Mapping[str, Any] | None = None,
) -> str:
    payload = _payload(
        video_checksum,
        segmentation_revision,
        grounding_cache_key,
        frame_policy or {},
        mask_policy or {},
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def segmentation_cache_path(
    cache_dir: str | Path,
    video_checksum: str,
    segmentation_revision: str,
    grounding_cache_key: str,
    frame_policy: Mapping[str, Any],
    mask_policy: Mapping[str, Any],
) -> Path:
    return Path(cache_dir) / (
        f"{segmentation_cache_key(video_checksum, segmentation_revision, grounding_cache_key, frame_policy, mask_policy)}.json"
    )


def segmentation_output_path(score_path: str | Path) -> Path:
    """Return a sibling score path that cannot overwrite grounding/baseline scores."""

    path = Path(score_path)
    suffix = path.suffix or ".jsonl"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    tag = f"segmentation.{SEGMENTATION_VERSION}"
    if path.name.endswith(f".{tag}{suffix}"):
        return path
    return path.with_name(f"{stem}.{tag}{suffix}")


def cache_content_digest(record: Mapping[str, Any]) -> str:
    payload = dict(record)
    payload.pop("cache_content_digest", None)
    try:
        encoded = json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("segmentation cache content must be JSON-serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _mask_row(mask: Mapping[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    if not isinstance(mask, Mapping):
        raise ValueError("segmentation masks must be mappings")
    label = str(mask.get("label", "")).strip()
    path = str(mask.get("mask_path", "")).strip()
    if not label or not path or Path(path).is_absolute():
        raise ValueError("segmentation masks require relative mask_path and label")
    confidence = _finite(mask.get("confidence", 0.0), name="mask confidence")
    area = _finite(mask.get("area", 0.0), name="mask area")
    if not 0 <= confidence <= 1 or not 0 <= area <= 1:
        raise ValueError("mask confidence and area must be in [0, 1]")
    box = G.normalise_box(mask.get("box", {}))
    centroid = mask.get("centroid", {})
    if not isinstance(centroid, Mapping):
        raise ValueError("mask centroid must be a mapping")
    cx = _finite(centroid.get("x"), name="mask centroid x")
    cy = _finite(centroid.get("y"), name="mask centroid y")
    if not 0 <= cx <= 1 or not 0 <= cy <= 1:
        raise ValueError("mask centroid must be in [0, 1]")
    try:
        frame_index = int(mask["frame_index"])
        timestamp = _finite(mask["timestamp"], name="mask timestamp")
        detection_index = int(mask.get("detection_index", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("mask timing and detection index are required") from exc
    if frame_index < 0 or detection_index < 0 or timestamp < 0:
        raise ValueError("mask timing and detection index must be non-negative")
    if root is not None and not (root / path).is_file():
        raise ValueError(f"segmentation mask file is missing: {root / path}")
    return {
        "mask_id": str(mask.get("mask_id", f"frame-{frame_index:06d}-det-{detection_index:04d}")),
        "frame_index": frame_index,
        "timestamp": timestamp,
        "detection_index": detection_index,
        "label": label,
        "confidence": confidence,
        "area": area,
        "centroid": {"x": cx, "y": cy},
        "box": box,
        "mask_path": path,
    }


def write_segmentation_cache(
    cache_dir: str | Path,
    *,
    video_checksum: str,
    segmentation_revision: str,
    grounding_cache_key: str,
    frame_policy: Mapping[str, Any],
    mask_policy: Mapping[str, Any],
    frames: Iterable[Mapping[str, Any]],
    contact_sheet: str | None = None,
) -> Path:
    payload = _payload(
        video_checksum,
        segmentation_revision,
        grounding_cache_key,
        frame_policy,
        mask_policy,
    )
    root = Path(cache_dir)
    canonical_frames: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError("segmentation frames must be mappings")
        frame_index = int(frame["frame_index"])
        if frame_index in seen_indices:
            raise ValueError("segmentation frame indices must be unique")
        seen_indices.add(frame_index)
        width = int(frame["width"])
        height = int(frame["height"])
        if frame_index < 0 or width <= 0 or height <= 0:
            raise ValueError("segmentation frame geometry is invalid")
        masks = [_mask_row(mask, root=root) for mask in frame.get("masks", [])]
        canonical_frames.append(
            {
                "frame_index": frame_index,
                "timestamp": _finite(frame["timestamp"], name="frame timestamp"),
                "width": width,
                "height": height,
                "masks": masks,
            }
        )
    if not canonical_frames:
        raise ValueError("segmentation cache requires at least one frame")
    canonical_frames.sort(key=lambda row: row["frame_index"])
    expected_indices = payload["frame_policy"].get("frame_indices")
    if expected_indices is not None and [row["frame_index"] for row in canonical_frames] != [int(index) for index in expected_indices]:
        raise ValueError("segmentation cache frame policy does not match frames")
    record = {
        "schema_version": SEGMENTATION_SCHEMA_VERSION,
        "segmentation_version": SEGMENTATION_VERSION,
        "cache_key": segmentation_cache_key(**payload),
        **payload,
        "contact_sheet": contact_sheet,
        "frames": canonical_frames,
    }
    record["cache_content_digest"] = cache_content_digest(record)
    json.dumps(record, sort_keys=True, allow_nan=False)
    path = segmentation_cache_path(cache_dir, **payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(path)
    return path


def load_segmentation_cache(path: str | Path, *, expected_cache_key: str | None = None) -> dict[str, Any]:
    cache_path = Path(path)
    try:
        record = json.loads(cache_path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"segmentation scoring requires an offline cache at {cache_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid segmentation cache JSON: {cache_path}") from exc
    if not isinstance(record, Mapping) or record.get("schema_version") != SEGMENTATION_SCHEMA_VERSION:
        raise ValueError(f"unsupported segmentation cache schema: {cache_path}")
    if record.get("cache_content_digest") != cache_content_digest(record):
        raise ValueError(f"segmentation cache content digest mismatch: {cache_path}")
    identity = _payload(
        record.get("video_checksum", ""),
        record.get("segmentation_revision", ""),
        record.get("grounding_cache_key", ""),
        record.get("frame_policy", {}),
        record.get("mask_policy", {}),
    )
    actual_key = segmentation_cache_key(**identity)
    if record.get("cache_key") != actual_key or (expected_cache_key and expected_cache_key != actual_key):
        raise ValueError(f"segmentation cache key mismatch: {cache_path}")
    frames = record.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"segmentation cache has no frames: {cache_path}")
    expected_indices = identity["frame_policy"].get("frame_indices")
    actual_indices: list[int] = []
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise ValueError(f"invalid segmentation frame: {cache_path}")
        actual_indices.append(int(frame["frame_index"]))
        for mask in frame.get("masks", []):
            _mask_row(mask, root=cache_path.parent)
    if expected_indices is not None and actual_indices != [int(index) for index in expected_indices]:
        raise ValueError(f"segmentation cache frame policy mismatch: {cache_path}")
    return dict(record)


def flatten_masks(cache: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in cache.get("frames", []):
        if not isinstance(frame, Mapping):
            continue
        for mask in frame.get("masks", []):
            if isinstance(mask, Mapping):
                rows.append(dict(mask))
    return rows


def filter_question_masks(
    masks: Iterable[Mapping[str, Any]],
    question_entities: Iterable[str],
    *,
    min_confidence: float = DEFAULT_MASK_THRESHOLD,
    max_masks: int = 8,
) -> list[dict[str, Any]]:
    if not 0 <= min_confidence <= 1 or max_masks < 0:
        raise ValueError("mask confidence must be in [0, 1] and max_masks cannot be negative")
    entities = [str(entity) for entity in question_entities if str(entity).strip()]
    if not entities or max_masks == 0:
        return []
    selected: list[tuple[float, int, dict[str, Any]]] = []
    for order, mask in enumerate(masks):
        if not isinstance(mask, Mapping):
            continue
        label = str(mask.get("label", ""))
        if not any(G._contains_phrase(label, entity) for entity in entities):
            continue
        try:
            confidence = float(mask.get("confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or confidence < min_confidence:
            continue
        selected.append((confidence, order, dict(mask)))
    selected.sort(key=lambda item: (-item[0], int(item[2].get("frame_index", 0)), item[1]))
    return [row for _, _, row in selected[:max_masks]]


def format_segmentation_context(
    masks: Iterable[Mapping[str, Any]], *, max_chars: int = 1000, max_masks: int = 8
) -> str:
    if max_chars <= 0 or max_masks < 0:
        raise ValueError("segmentation context bounds are invalid")
    lines = [
        "Segmentation context: uncertain SAM 2 box-prompted masks; each displayed frame keeps the original on the left and a mask overlay on the right. Normalized image shape evidence is fallible and gives no physical depth."
    ]
    for mask in list(masks)[:max_masks]:
        if not isinstance(mask, Mapping):
            continue
        centroid = mask.get("centroid", {})
        lines.append(
            "mask={mask} label={label} confidence={confidence:.3f} area={area:.3f} "
            "centroid(x={x:.3f},y={y:.3f}) frame={frame} time={time:.3f}s".format(
                mask=mask.get("mask_id", "?"),
                label=str(mask.get("label", "unknown")),
                confidence=float(mask.get("confidence", 0.0)),
                area=float(mask.get("area", 0.0)),
                x=float(centroid.get("x", 0.0)),
                y=float(centroid.get("y", 0.0)),
                frame=mask.get("frame_index", "?"),
                time=float(mask.get("timestamp", 0.0)),
            )
        )
    return "\n".join(lines)[:max_chars]


format_mask_context = format_segmentation_context


def compose_mask_overlay_video(
    frames: Any,
    frame_indices: Sequence[int],
    masks: Iterable[Mapping[str, Any]],
    cache_dir: str | Path,
) -> np.ndarray:
    """Keep each original frame and append a same-size mask overlay beside it."""

    source = np.asarray(frames)
    if source.ndim != 4 or source.shape[-1] != 3:
        raise ValueError("frames must have shape [time, height, width, 3]")
    if len(frame_indices) != len(source):
        raise ValueError("frame_indices must match the number of frames")
    by_frame: dict[int, list[Mapping[str, Any]]] = {}
    for mask in masks:
        if not isinstance(mask, Mapping):
            continue
        try:
            by_frame.setdefault(int(mask["frame_index"]), []).append(mask)
        except (KeyError, TypeError, ValueError):
            continue
    root = Path(cache_dir)
    overlays: list[np.ndarray] = []
    for frame_index, frame in zip(frame_indices, source):
        overlay = frame.copy()
        for mask in by_frame.get(int(frame_index), []):
            mask_path = mask.get("mask_path")
            if not isinstance(mask_path, str) or Path(mask_path).is_absolute():
                continue
            try:
                from PIL import Image

                mask_image = Image.open(root / mask_path).convert("L")
                mask_array = np.asarray(mask_image) > 0
            except (OSError, ValueError):
                continue
            if mask_array.shape != overlay.shape[:2]:
                from PIL import Image

                mask_array = np.asarray(
                    Image.fromarray(mask_array.astype(np.uint8) * 255).resize(
                        (overlay.shape[1], overlay.shape[0]), Image.Resampling.NEAREST
                    )
                ) > 0
            overlay[mask_array] = (
                0.55 * overlay[mask_array].astype(np.float32)
                + 0.45 * np.asarray((255, 64, 32), dtype=np.float32)
            ).astype(np.uint8)
        overlays.append(np.concatenate((frame, overlay), axis=1))
    return np.stack(overlays)


def augment_video_with_masks(
    frames: Any,
    meta: Any,
    masks: Iterable[Mapping[str, Any]],
    cache_dir: str | Path,
) -> tuple[np.ndarray, Any]:
    """Append overlays while preserving the sampled frame count and timestamps."""

    augmented = compose_mask_overlay_video(frames, list(meta.frames_indices), masks, cache_dir)
    import common as C

    augmented_meta = C._video_metadata(
        total_num_frames=int(meta.total_num_frames),
        fps=float(meta.fps),
        frames_indices=list(meta.frames_indices),
        width=int(augmented.shape[2]),
        height=int(augmented.shape[1]),
        duration=float(getattr(meta, "duration", 0.0)),
    )
    return augmented, augmented_meta
