"""Interfaces for evidence-grounded inference artifacts.

Model loading and video decoding stay in the command-line builders. This
module owns the deterministic cache identity, validation, box normalization,
tracking, and compact prompt context shared by those builders and scoring.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

EVIDENCE_SCHEMA_VERSION = 1
SELECTOR_VERSION = "selector-v2"
GROUNDING_VERSION = "grounding-v1"
DETECTOR_VERSION = "detector-v1"
GROUNDING_DETECTOR_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
# Immutable HF commit from the official model repository. The model page is
# allowed to move; this cache identity must not.
GROUNDING_DETECTOR_MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
GROUNDING_DETECTOR_REVISION = (
    f"{GROUNDING_DETECTOR_MODEL_ID}@{GROUNDING_DETECTOR_MODEL_REVISION}"
)
GROUNDING_DETECTOR_PREPROCESSING = {
    "image_processor": "GroundingDinoImageProcessor",
    "color_space": "rgb",
    "dtype": "float32",
    "do_resize": True,
    "size": {"shortest_edge": 800, "longest_edge": 1333},
    "resample": 2,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
    "text_lowercase": True,
    "text_terminal_period": True,
}
GROUNDING_DETECTOR_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
GROUNDING_CACHE_SCHEMA_VERSION = 1
GROUNDING_CACHE_KEY_FIELDS = (
    "video_checksum",
    "detector_revision",
    "prompt",
    "preprocessing",
    "thresholds",
    "frame_policy",
    "tracker",
)
DEFAULT_DETECTOR_THRESHOLD = 0.4
DEFAULT_DETECTOR_TEXT_THRESHOLD = 0.3
DEFAULT_TRACK_IOU_THRESHOLD = 0.3
DEFAULT_GROUNDING_PROMPT = (
    "person. bicycle. car. motorcycle. bus. truck. boat. building. house. "
    "bridge. road. parking lot. court. basketball court. soccer field. "
    "baseball field. tennis court. sports field. field. pool. swimming pool. "
    "barn. water. river. tree. path. train. train track."
)
GROUNDING_CACHE_MARKER = ".reva_detector_revision"
DEFAULT_CANDIDATE_RATE = 4.0
DEFAULT_CANDIDATE_CAP = 64
CANDIDATE_GRID_ALGORITHM = "uniform-endpoints"
CANDIDATE_GRID_VERSION = "candidate-grid-v1"
MAX_ADAPTIVE_FRAMES = 16
FEATURE_CACHE_SCHEMA_VERSION = 1
FEATURE_ENCODER_MODEL_ID = "google/siglip2-base-patch16-224"
FEATURE_ENCODER_REVISION = (
    f"{FEATURE_ENCODER_MODEL_ID}@022b6f71160ffb0169ca4709e2d7e25be659598a"
)
FEATURE_PREPROCESSING = {
    "model_id": FEATURE_ENCODER_MODEL_ID,
    "image_processor": "SiglipImageProcessor",
    "color_space": "rgb",
    "dtype": "float32",
    "do_convert_rgb": None,
    "do_resize": True,
    "size": {"height": 224, "width": 224},
    "resample": 2,
    "do_rescale": True,
    "rescale_factor": 1 / 255,
    "do_normalize": True,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "feature_normalization": "l2",
    "text_padding": "max_length",
    "text_truncation": True,
}
FEATURE_MODEL_REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)
FEATURE_CACHE_KEY_FIELDS = (
    "video_checksum",
    "encoder_revision",
    "candidate_policy",
    "preprocessing",
)


def _safe_version(value: str) -> str:
    """Make a version safe to use in a filename without accepting path parts."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip(".-")
    return cleaned or "unknown"


def evidence_output_path(
    score_path: str | Path,
    *,
    selector_version: str = SELECTOR_VERSION,
    grounding_version: str = GROUNDING_VERSION,
) -> Path:
    """Return a versioned sibling path that cannot overwrite a legacy score file."""

    path = Path(score_path)
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    if not suffix:
        suffix = ".jsonl"
    tag = f"evidence.{_safe_version(selector_version)}.{_safe_version(grounding_version)}"
    if path.name.endswith(f".{tag}{suffix}"):
        return path
    return path.with_name(f"{stem}.{tag}{suffix}")


def evidence_manifest_path(output_path: str | Path) -> Path:
    """Return the manifest sidecar path for an evidence score file."""

    path = Path(output_path)
    return path.with_name(path.name + ".manifest.json")


def grounding_output_path(score_path: str | Path) -> Path:
    """Return a detector-only sibling path that cannot overwrite any baseline."""

    path = Path(score_path)
    suffix = path.suffix or ".jsonl"
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    tag = f"grounding.{_safe_version(DETECTOR_VERSION)}"
    if path.name.endswith(f".{tag}{suffix}"):
        return path
    return path.with_name(f"{stem}.{tag}{suffix}")


def build_evidence_manifest(
    score_manifest: Mapping[str, Any] | None = None,
    *,
    selector_version: str = SELECTOR_VERSION,
    grounding_version: str = GROUNDING_VERSION,
    selector_enabled: bool = True,
    grounding_enabled: bool = True,
    candidate_rate: float = DEFAULT_CANDIDATE_RATE,
    candidate_cap: int = DEFAULT_CANDIDATE_CAP,
    selection_params: Mapping[str, Any] | None = None,
    grounding_params: Mapping[str, Any] | None = None,
    feature_encoder_revision: str = FEATURE_ENCODER_REVISION,
    feature_preprocessing: Mapping[str, Any] | None = None,
    output_kind: str = "evidence_score",
) -> dict[str, Any]:
    """Copy a score manifest and add a versioned, JSON-serializable evidence spec."""

    if candidate_rate <= 0 or candidate_cap <= 0:
        raise ValueError("candidate rate and cap must be positive")
    manifest = dict(score_manifest or {})
    manifest.update(
        {
            "evidence_schema_version": EVIDENCE_SCHEMA_VERSION,
            "output_kind": str(output_kind),
            "selector_version": str(selector_version),
            "grounding_version": str(grounding_version),
            "feature_encoder_revision": str(feature_encoder_revision),
            "feature_cache_key_fields": list(FEATURE_CACHE_KEY_FIELDS),
            "selector": {
                "version": str(selector_version),
                "enabled": bool(selector_enabled),
                "candidate_rate": float(candidate_rate),
                "candidate_cap": int(candidate_cap),
                "feature_encoder_revision": str(feature_encoder_revision),
                "feature_preprocessing": dict(feature_preprocessing or {}),
                "parameters": dict(selection_params or {}),
            },
            "grounding": {
                "version": str(grounding_version),
                "enabled": bool(grounding_enabled),
                "parameters": dict(grounding_params or {}),
            },
        }
    )
    # Fail at interface construction rather than when a job writes its manifest.
    json.dumps(manifest, sort_keys=True)
    return manifest


def _cache_payload(
    video_checksum: str,
    encoder_revision: str,
    candidate_policy: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(candidate_policy, Mapping) or not isinstance(preprocessing, Mapping):
        raise ValueError("candidate policy and preprocessing must be mappings")
    payload = {
        "video_checksum": str(video_checksum),
        "encoder_revision": str(encoder_revision),
        "candidate_policy": dict(candidate_policy),
        "preprocessing": dict(preprocessing),
    }
    if not payload["video_checksum"] or not payload["encoder_revision"]:
        raise ValueError("video checksum and encoder revision are required")
    try:
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("feature cache identity must be JSON-serializable") from exc
    return payload


def feature_cache_key(
    video_checksum: str,
    encoder_revision: str,
    candidate_policy: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
) -> str:
    """Return the stable identity for one video's frozen candidate features."""

    payload = _cache_payload(video_checksum, encoder_revision, candidate_policy, preprocessing)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def feature_cache_path(
    cache_dir: str | Path,
    video_checksum: str,
    encoder_revision: str,
    candidate_policy: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
) -> Path:
    """Return the cache file whose name is derived from all cache identity fields."""

    return Path(cache_dir) / f"{feature_cache_key(video_checksum, encoder_revision, candidate_policy, preprocessing)}.json"


def _detector_payload(
    video_checksum: str,
    detector_revision: str,
    prompt: str,
    preprocessing: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    frame_policy: Mapping[str, Any],
    tracker: Mapping[str, Any],
) -> dict[str, Any]:
    if not all(isinstance(value, Mapping) for value in (preprocessing, thresholds, frame_policy, tracker)):
        raise ValueError("detector cache settings must be mappings")
    normalized_prompt = normalize_detector_prompt(prompt)
    payload = {
        "video_checksum": str(video_checksum),
        "detector_revision": str(detector_revision),
        "prompt": normalized_prompt,
        "preprocessing": dict(preprocessing),
        "thresholds": dict(thresholds),
        "frame_policy": dict(frame_policy),
        "tracker": dict(tracker),
    }
    if not payload["video_checksum"] or not payload["detector_revision"]:
        raise ValueError("video checksum and detector revision are required")
    try:
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("detector cache identity must be JSON-serializable") from exc
    return payload


def normalize_detector_prompt(prompt: str) -> str:
    """Return lowercase, period-separated labels required by Grounding DINO."""

    if not isinstance(prompt, str):
        raise ValueError("detector prompt must be text")
    labels = [part.strip().casefold() for part in prompt.split(".") if part.strip()]
    if not labels:
        raise ValueError("detector prompt must contain at least one label")
    unique: list[str] = []
    for label in labels:
        if label not in unique:
            unique.append(label)
    return ". ".join(unique) + "."


def detector_cache_key(
    video_checksum: str,
    detector_revision: str,
    prompt: str,
    preprocessing: Mapping[str, Any] = GROUNDING_DETECTOR_PREPROCESSING,
    thresholds: Mapping[str, Any] | None = None,
    frame_policy: Mapping[str, Any] | None = None,
    tracker: Mapping[str, Any] | None = None,
) -> str:
    """Return the stable identity for one detector cache."""

    payload = _detector_payload(
        video_checksum,
        detector_revision,
        prompt,
        preprocessing,
        thresholds or {
            "box_threshold": DEFAULT_DETECTOR_THRESHOLD,
            "text_threshold": DEFAULT_DETECTOR_TEXT_THRESHOLD,
        },
        frame_policy or {},
        tracker or {"algorithm": "adjacent-greedy-iou", "iou_threshold": DEFAULT_TRACK_IOU_THRESHOLD},
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def detector_cache_path(
    cache_dir: str | Path,
    video_checksum: str,
    detector_revision: str,
    prompt: str,
    preprocessing: Mapping[str, Any] = GROUNDING_DETECTOR_PREPROCESSING,
    thresholds: Mapping[str, Any] | None = None,
    frame_policy: Mapping[str, Any] | None = None,
    tracker: Mapping[str, Any] | None = None,
) -> Path:
    """Return the separate detector cache path derived from all settings."""

    return Path(cache_dir) / (
        f"{detector_cache_key(video_checksum, detector_revision, prompt, preprocessing, thresholds, frame_policy, tracker)}.json"
    )


# Noun-first aliases keep the cache boundary discoverable to callers.
grounding_cache_key = detector_cache_key
grounding_cache_path = detector_cache_path


def candidate_index_list(
    n_frames: int,
    fps: float,
    rate: float = DEFAULT_CANDIDATE_RATE,
    cap: int = DEFAULT_CANDIDATE_CAP,
) -> list[int]:
    """Return a bounded, endpoint-inclusive candidate grid for one video."""

    n = int(n_frames)
    frames_per_second = float(fps)
    if n <= 0 or not math.isfinite(frames_per_second) or frames_per_second <= 0:
        raise ValueError("n_frames and fps must be positive")
    if not math.isfinite(float(rate)) or rate <= 0:
        raise ValueError("candidate rate must be positive")
    try:
        cap_value = int(cap)
    except (TypeError, ValueError) as exc:
        raise ValueError("candidate cap must be a positive integer") from exc
    if isinstance(cap, bool) or cap_value != cap or cap_value <= 0:
        raise ValueError("candidate cap must be a positive integer")
    count = min(n, cap_value, max(1, math.ceil(n / frames_per_second * float(rate))))
    if count == 1:
        return [0]
    return [round(index * (n - 1) / (count - 1)) for index in range(count)]


def candidate_policy(rate: float = DEFAULT_CANDIDATE_RATE, cap: int = DEFAULT_CANDIDATE_CAP) -> dict[str, Any]:
    """Return the versioned candidate-grid policy used in cache identities."""

    if rate <= 0 or cap <= 0:
        raise ValueError("candidate rate and cap must be positive")
    return {
        "grid_algorithm": CANDIDATE_GRID_ALGORITHM,
        "grid_version": CANDIDATE_GRID_VERSION,
        "rate": float(rate),
        "cap": int(cap),
    }


def detector_frame_policy(
    frame_indices: Iterable[int], offsets: Iterable[float], sampling_mode: str
) -> dict[str, Any]:
    """Describe the exact baseline source frames represented by a detector cache."""

    indices = [int(index) for index in frame_indices]
    if not indices or any(index < 0 for index in indices):
        raise ValueError("detector frame policy requires non-negative frame indices")
    normalized_offsets = [float(offset) for offset in offsets]
    if not normalized_offsets or any(not 0 <= offset < 1 for offset in normalized_offsets):
        raise ValueError("detector offsets must be in [0, 1)")
    if not isinstance(sampling_mode, str) or not sampling_mode:
        raise ValueError("detector sampling mode is required")
    return {
        "algorithm": "common.sample_index_list",
        "sampling_mode": sampling_mode,
        "offsets": normalized_offsets,
        "frame_indices": sorted(set(indices)),
    }


def candidate_grid_matches(candidates: Iterable[Mapping[str, Any]], expected: Sequence[int]) -> bool:
    """Return whether cached candidate indices exactly match the current grid."""

    try:
        actual = [int(candidate["frame_index"]) for candidate in candidates]
    except (KeyError, TypeError, ValueError):
        return False
    return actual == [int(index) for index in expected]


def query_input_fingerprint(query: Mapping[str, Any]) -> str:
    """Fingerprint question text and options for validating cached query features."""

    if not isinstance(query, Mapping):
        raise ValueError("query must be a mapping")
    payload = {
        "question": query.get("question", ""),
        "options": query.get("options", {}),
    }
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("query text and options must be JSON-serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def feature_cache_content_digest(record: Mapping[str, Any]) -> str:
    """Hash the complete cache record, excluding its self-reported digest."""

    payload = dict(record)
    payload.pop("cache_content_digest", None)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("feature cache content must be JSON-serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def _feature_vector(value: Any, *, name: str) -> list[float]:
    if isinstance(value, (str, bytes)) or value is None:
        raise ValueError(f"{name} must be a numeric sequence")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    if not vector or not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain finite values")
    return vector


def write_feature_cache(
    cache_dir: str | Path,
    *,
    video_checksum: str,
    encoder_revision: str,
    candidate_policy: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    query_features: Mapping[str, Sequence[float]] | None = None,
    query_fingerprints: Mapping[str, str] | None = None,
) -> Path:
    """Write a portable offline feature cache and return its keyed path.

    The caller supplies frozen features. This module intentionally does not load
    an encoder or fetch model weights.
    """

    payload = _cache_payload(video_checksum, encoder_revision, candidate_policy, preprocessing)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("feature cache candidates must be mappings")
        index = _frame_index(candidate)
        if index is None:
            raise ValueError("feature cache candidates require non-negative integer frame_index")
        row = dict(candidate)
        row["frame_index"] = index
        row["feature"] = _feature_vector(row.get("feature", row.get("embedding")), name="feature")
        if "timestamp" in row:
            try:
                row["timestamp"] = float(row["timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError("feature cache timestamps must be numeric") from exc
        rows.append(row)
    if not rows:
        raise ValueError("feature cache requires at least one candidate")
    try:
        candidate_cap = int(candidate_policy.get("cap", len(rows)))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("candidate policy cap must be an integer") from exc
    if candidate_cap <= 0 or len(rows) > candidate_cap:
        raise ValueError("feature cache candidates exceed the candidate policy cap")
    queries = {
        str(query_id): _feature_vector(feature, name=f"query_features[{query_id}]")
        for query_id, feature in (query_features or {}).items()
    }
    fingerprints = {str(query_id): str(value) for query_id, value in (query_fingerprints or {}).items()}
    if set(queries) != set(fingerprints):
        raise ValueError("query features and query fingerprints must have the same IDs")
    record = {
        "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "cache_key": feature_cache_key(**payload),
        **payload,
        "candidates": rows,
        "query_features": queries,
        "query_fingerprints": fingerprints,
    }
    record["cache_content_digest"] = feature_cache_content_digest(record)
    json.dumps(record, sort_keys=True, allow_nan=False)
    path = feature_cache_path(cache_dir, **payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)
    return path


def load_feature_cache(
    path: str | Path,
    *,
    expected_cache_key: str | None = None,
) -> dict[str, Any]:
    """Load and validate one offline feature cache before adaptive scoring."""

    cache_path = Path(path)
    try:
        with cache_path.open() as stream:
            record = json.load(stream)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"adaptive scoring requires an offline feature cache at {cache_path}; "
            "populate it with a pinned encoder before scoring (no model is downloaded)"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid feature cache JSON: {cache_path}") from exc
    if not isinstance(record, Mapping) or record.get("schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported feature cache schema: {cache_path}")
    if record.get("cache_content_digest") != feature_cache_content_digest(record):
        raise ValueError(f"feature cache content digest mismatch: {cache_path}")
    identity = _cache_payload(
        record.get("video_checksum", ""),
        record.get("encoder_revision", ""),
        record.get("candidate_policy", {}),
        record.get("preprocessing", {}),
    )
    actual_key = feature_cache_key(**identity)
    if record.get("cache_key") != actual_key or (expected_cache_key and expected_cache_key != actual_key):
        raise ValueError(f"feature cache key mismatch: {cache_path}")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"feature cache has no candidates: {cache_path}")
    try:
        candidate_cap = int(record["candidate_policy"].get("cap", len(candidates)))
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid candidate policy in feature cache: {cache_path}") from exc
    if candidate_cap <= 0 or len(candidates) > candidate_cap:
        raise ValueError(f"feature cache exceeds its candidate policy cap: {cache_path}")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"invalid candidate in feature cache: {cache_path}")
        if _frame_index(candidate) is None:
            raise ValueError(f"invalid frame index in feature cache: {cache_path}")
        _feature_vector(candidate.get("feature", candidate.get("embedding")), name="feature")
        if "timestamp" in candidate:
            try:
                timestamp = float(candidate["timestamp"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid timestamp in feature cache: {cache_path}") from exc
            if not math.isfinite(timestamp):
                raise ValueError(f"invalid timestamp in feature cache: {cache_path}")
    query_features = record.get("query_features", {})
    if not isinstance(query_features, Mapping):
        raise ValueError(f"invalid query_features in feature cache: {cache_path}")
    for query_id, feature in query_features.items():
        _feature_vector(feature, name=f"query_features[{query_id}]")
    query_fingerprints = record.get("query_fingerprints", {})
    if not isinstance(query_fingerprints, Mapping) or set(query_features) != set(query_fingerprints):
        raise ValueError(f"query feature fingerprints are missing or mismatched: {cache_path}")
    if any(not isinstance(value, str) or not value for value in query_fingerprints.values()):
        raise ValueError(f"invalid query feature fingerprint in cache: {cache_path}")
    return dict(record)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return finite cosine similarity for two equal-length feature vectors."""

    first = _feature_vector(left, name="left feature")
    second = _feature_vector(right, name="right feature")
    if len(first) != len(second):
        raise ValueError("feature vectors must have equal length")
    left_norm = math.sqrt(sum(value * value for value in first))
    right_norm = math.sqrt(sum(value * value for value in second))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(first, second)) / (left_norm * right_norm)


def rank_query_candidates(
    candidates: Iterable[Mapping[str, Any]],
    query_feature: Sequence[float],
) -> list[dict[str, Any]]:
    """Attach deterministic cosine relevance scores to cached frame features."""

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        row = dict(candidate)
        row["relevance"] = cosine_similarity(
            row.get("feature", row.get("embedding")), query_feature
        )
        ranked.append(row)
    return ranked


def select_adaptive_frames(
    candidates: Iterable[Mapping[str, Any]],
    query_feature: Sequence[float],
    *,
    limit: int = 16,
    anchor_count: int = 4,
    min_spacing: int | None = None,
    temporal_offset: float = 0.0,
) -> list[dict[str, Any]]:
    """Select query-relevant frames while retaining uniform temporal anchors."""

    if not 0.0 <= temporal_offset < 1.0:
        raise ValueError("temporal_offset must be in [0, 1)")
    limit = min(limit, MAX_ADAPTIVE_FRAMES)
    return select_candidate_frames(
        rank_query_candidates(candidates, query_feature),
        limit=limit,
        anchor_count=anchor_count,
        min_spacing=min_spacing,
        anchor_offset=temporal_offset,
    )


def _finite_score(row: Mapping[str, Any]) -> float:
    for key in ("relevance", "score", "similarity", "quality"):
        value = row.get(key)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            return parsed
    return 0.0


def _frame_index(row: Mapping[str, Any]) -> int | None:
    value = row.get("frame_index", row.get("frame"))
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0 or isinstance(value, float) and parsed != value:
        return None
    return parsed


def select_candidate_frames(
    candidates: Iterable[Mapping[str, Any]],
    *,
    limit: int = 16,
    anchor_count: int = 4,
    min_spacing: int | None = None,
    anchor_offset: float = 0.0,
) -> list[dict[str, Any]]:
    """Choose uniform temporal anchors, then high-scoring spaced candidates.

    Invalid frame indices are ignored. Duplicate indices keep the highest score,
    with the first row winning ties. The returned rows are copies sorted by time.
    """

    if limit <= 0 or anchor_count < 0:
        raise ValueError("limit must be positive and anchor_count cannot be negative")
    if min_spacing is not None and min_spacing < 1:
        raise ValueError("min_spacing must be positive")
    if not 0.0 <= anchor_offset < 1.0:
        raise ValueError("anchor_offset must be in [0, 1)")

    by_index: dict[int, tuple[float, int, dict[str, Any]]] = {}
    for order, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            continue
        index = _frame_index(candidate)
        if index is None:
            continue
        score = _finite_score(candidate)
        current = by_index.get(index)
        if current is None or score > current[0]:
            row = dict(candidate)
            row["frame_index"] = index
            by_index[index] = (score, order, row)
    ordered = [item[2] for item in sorted(by_index.values(), key=lambda item: item[2]["frame_index"])]
    if not ordered:
        return []
    limit = min(limit, len(ordered))
    anchor_count = min(anchor_count, limit)
    first = ordered[0]["frame_index"]
    last = ordered[-1]["frame_index"]
    selected: dict[int, dict[str, Any]] = {}
    if anchor_count == 1:
        anchor_positions = [0]
    elif anchor_count:
        if anchor_offset == 0.0 or anchor_count == 2:
            anchor_positions = [round(index * (len(ordered) - 1) / (anchor_count - 1)) for index in range(anchor_count)]
        else:
            interior = [
                min(
                    len(ordered) - 1,
                    round((index + anchor_offset) * (len(ordered) - 1) / (anchor_count - 1)),
                )
                for index in range(1, anchor_count - 1)
            ]
            anchor_positions = [0, *interior, len(ordered) - 1]
    else:
        anchor_positions = []
    for position in anchor_positions:
        row = ordered[position]
        selected[row["frame_index"]] = row

    span = last - first
    spacing = min_spacing if min_spacing is not None else max(1, span // max(limit * 2, 1))

    def far_enough(index: int) -> bool:
        return all(abs(index - selected_index) >= spacing for selected_index in selected)

    ranked = sorted(
        enumerate(ordered),
        key=lambda item: (-_finite_score(item[1]), item[1]["frame_index"], item[0]),
    )
    for _, row in ranked:
        if len(selected) >= limit:
            break
        index = row["frame_index"]
        if index not in selected and far_enough(index):
            selected[index] = row
    # A caller-specified spacing is a preference when it prevents filling the
    # frame budget; unique candidates still have deterministic priority.
    if len(selected) < limit:
        for _, row in ranked:
            if len(selected) >= limit:
                break
            index = row["frame_index"]
            if index not in selected:
                selected[index] = row
    return [selected[index] for index in sorted(selected)]


def pad_frame_selection(selected: Sequence[Mapping[str, Any]], target_count: int) -> list[dict[str, Any]]:
    """Repeat selected rows for short clips, matching common sampler behavior."""

    if target_count <= 0:
        raise ValueError("target_count must be positive")
    rows = [dict(row) for row in selected]
    if not rows:
        return []
    return [dict(rows[index % len(rows)]) for index in range(target_count)]


_WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def _words(value: Any) -> list[str]:
    return _WORD_RE.findall(str(value).casefold())


def _contains_phrase(label: str, entity: str) -> bool:
    label_words = _words(label)
    entity_words = _words(entity)
    if not entity_words or len(entity_words) > len(label_words):
        return False
    return any(label_words[start : start + len(entity_words)] == entity_words for start in range(len(label_words) - len(entity_words) + 1))


def _phrase_variants(entity: str) -> tuple[str, ...]:
    """Return a label plus one conservative English plural variant."""

    words = _words(entity)
    if not words:
        return ()
    last = words[-1]
    if last.endswith(("s", "x", "z", "ch", "sh")):
        plural = last + "es"
    elif len(last) > 1 and last.endswith("y") and last[-2] not in "aeiou":
        plural = last[:-1] + "ies"
    else:
        plural = last + "s"
    return (" ".join(words), " ".join((*words[:-1], plural)))


def _track_labels(track: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ("label", "class_name", "phrase", "entity", "name", "labels", "phrases"):
        value = track.get(key)
        if isinstance(value, (list, tuple)):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    return values


def filter_question_tracks(
    tracks: Iterable[Mapping[str, Any]],
    question_entities: Iterable[str],
    *,
    min_confidence: float = 0.0,
    max_tracks: int = 8,
) -> list[dict[str, Any]]:
    """Keep confident tracks whose labels contain a named question entity."""

    if not 0 <= min_confidence <= 1 or max_tracks < 0:
        raise ValueError("min_confidence must be in [0, 1] and max_tracks cannot be negative")
    entities = [str(entity) for entity in question_entities if _words(entity)]
    if not entities or max_tracks == 0:
        return []
    relevant: list[tuple[float, int, dict[str, Any]]] = []
    for order, track in enumerate(tracks):
        if not isinstance(track, Mapping):
            continue
        if not any(_contains_phrase(label, entity) for label in _track_labels(track) for entity in entities):
            continue
        try:
            confidence = float(track.get("confidence", track.get("score", 0.0)))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or confidence < min_confidence:
            continue
        relevant.append((confidence, order, dict(track)))
    relevant.sort(key=lambda item: (-item[0], item[1]))
    return [track for _, _, track in relevant[:max_tracks]]


# Short alias for callers that prefer the noun-first form.
filter_relevant_tracks = filter_question_tracks


def question_track_entities(
    query: Mapping[str, Any], tracks: Iterable[Mapping[str, Any]]
) -> list[str]:
    """Find detector labels explicitly named by a question or one of its options."""

    if not isinstance(query, Mapping):
        raise ValueError("query must be a mapping")
    text_parts = [str(query.get("question", ""))]
    options = query.get("options", {})
    if isinstance(options, Mapping):
        text_parts.extend(str(value) for value in options.values())
    text = " ".join(text_parts)
    entities: list[str] = []
    for track in tracks:
        for label in _track_labels(track) if isinstance(track, Mapping) else []:
            candidate = re.sub(r"^(?:a|an|the)\s+", "", label.casefold()).strip()
            if candidate and any(_contains_phrase(text, variant) for variant in _phrase_variants(candidate)) and candidate not in entities:
                entities.append(candidate)
    return entities


def normalise_box(
    box: Sequence[float] | Mapping[str, Any],
    width: float | None = None,
    height: float | None = None,
) -> dict[str, float]:
    """Convert a pixel box to clipped left/top/right/bottom image-relative values."""

    if isinstance(box, Mapping):
        keys = (("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax"), ("left", "top", "right", "bottom"))
        values = next((tuple(box[key] for key in names) for names in keys if all(key in box for key in names)), None)
        if values is None:
            raise ValueError("box must contain x1/y1/x2/y2 or left/top/right/bottom")
        already_relative = keys[-1] == tuple(next(names for names in keys if all(key in box for key in names)))
    else:
        if len(box) != 4:
            raise ValueError("box must have four coordinates")
        values = tuple(box)
        already_relative = False
    try:
        left, top, right, bottom = (float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("box coordinates must be numeric") from exc
    if not all(math.isfinite(value) for value in (left, top, right, bottom)):
        raise ValueError("box coordinates must be finite")
    if left > right or top > bottom:
        raise ValueError("box coordinates must be ordered")
    if already_relative:
        if any(value < 0 or value > 1 for value in (left, top, right, bottom)):
            raise ValueError("relative box coordinates must be in [0, 1]")
    else:
        if width is None or height is None or width <= 0 or height <= 0:
            raise ValueError("pixel boxes require positive width and height")
        left, right = left / float(width), right / float(width)
        top, bottom = top / float(height), bottom / float(height)
    return {
        "left": max(0.0, min(1.0, left)),
        "top": max(0.0, min(1.0, top)),
        "right": max(0.0, min(1.0, right)),
        "bottom": max(0.0, min(1.0, bottom)),
    }


normalize_box = normalise_box


def box_iou(left: Mapping[str, Any] | Sequence[float], right: Mapping[str, Any] | Sequence[float]) -> float:
    """Return IoU for two normalized image-relative boxes."""

    first = normalise_box(left)
    second = normalise_box(right)
    overlap_left = max(first["left"], second["left"])
    overlap_top = max(first["top"], second["top"])
    overlap_right = min(first["right"], second["right"])
    overlap_bottom = min(first["bottom"], second["bottom"])
    overlap = max(0.0, overlap_right - overlap_left) * max(0.0, overlap_bottom - overlap_top)
    first_area = (first["right"] - first["left"]) * (first["bottom"] - first["top"])
    second_area = (second["right"] - second["left"]) * (second["bottom"] - second["top"])
    union = first_area + second_area - overlap
    return overlap / union if union else 0.0


def _canonical_detection(
    detection: Mapping[str, Any], *, width: float, height: float, order: int
) -> dict[str, Any] | None:
    raw_box = detection.get("box", detection.get("bbox", detection.get("bounding_box")))
    if raw_box is None:
        return None
    label = detection.get("label", detection.get("class_name", detection.get("phrase")))
    if label is None or not str(label).strip():
        return None
    try:
        confidence = float(detection.get("confidence", detection.get("score", 0.0)))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None
    try:
        box = normalise_box(raw_box, width, height)
    except (TypeError, ValueError):
        return None
    return {
        "detection_index": int(detection.get("detection_index", order)),
        "label": str(label).strip(),
        "confidence": confidence,
        "box": box,
    }


def _canonical_frame(frame: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(frame, Mapping):
        raise ValueError("detector frames must be mappings")
    try:
        frame_index = int(frame["frame_index"])
        timestamp = float(frame["timestamp"])
        width = float(frame["width"])
        height = float(frame["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("detector frames require frame_index, timestamp, width, and height") from exc
    if frame_index < 0 or not all(math.isfinite(value) and value > 0 for value in (width, height)):
        raise ValueError("detector frame geometry is invalid")
    if not math.isfinite(timestamp) or timestamp < 0:
        raise ValueError("detector timestamps must be finite and non-negative")
    detections: list[dict[str, Any]] = []
    raw_detections = frame.get("detections", [])
    if not isinstance(raw_detections, Sequence) or isinstance(raw_detections, (str, bytes)):
        raise ValueError("detector frame detections must be a sequence")
    for order, detection in enumerate(raw_detections):
        if isinstance(detection, Mapping):
            canonical = _canonical_detection(detection, width=width, height=height, order=order)
            if canonical is not None:
                detections.append(canonical)
    return {
        "frame_index": frame_index,
        "timestamp": timestamp,
        "width": int(width),
        "height": int(height),
        "detections": detections,
    }


def associate_detections(
    frames: Iterable[Mapping[str, Any]],
    *,
    iou_threshold: float = DEFAULT_TRACK_IOU_THRESHOLD,
) -> list[dict[str, Any]]:
    """Associate same-label detections across adjacent sampled frames by greedy IoU."""

    if not 0 <= float(iou_threshold) <= 1:
        raise ValueError("track IoU threshold must be in [0, 1]")
    ordered_frames = sorted((_canonical_frame(frame) for frame in frames), key=lambda row: row["frame_index"])
    tracks: list[dict[str, Any]] = []
    active: list[dict[str, Any]] = []
    for frame in ordered_frames:
        candidates: list[tuple[float, int, int]] = []
        for detection_order, detection in enumerate(frame["detections"]):
            for active_order, track in enumerate(active):
                previous = track["observations"][-1]
                if str(previous["label"]).casefold() != str(detection["label"]).casefold():
                    continue
                score = box_iou(previous["box"], detection["box"])
                if score >= float(iou_threshold):
                    candidates.append((-score, active_order, detection_order))
        matched_active: set[int] = set()
        matched_detections: set[int] = set()
        next_active: list[dict[str, Any]] = []
        for negative_iou, active_order, detection_order in sorted(candidates):
            if active_order in matched_active or detection_order in matched_detections:
                continue
            track = active[active_order]
            detection = frame["detections"][detection_order]
            track["observations"].append(
                {
                    "frame_index": frame["frame_index"],
                    "timestamp": frame["timestamp"],
                    "box": detection["box"],
                    "confidence": detection["confidence"],
                    "label": detection["label"],
                    "association_iou": -negative_iou,
                }
            )
            track["confidence"] = max(track["confidence"], detection["confidence"])
            track["uncertain"] = False
            matched_active.add(active_order)
            matched_detections.add(detection_order)
            next_active.append(track)
        for detection_order, detection in enumerate(frame["detections"]):
            if detection_order in matched_detections:
                continue
            track = {
                "track_id": f"track-{len(tracks) + 1:04d}",
                "label": detection["label"],
                "confidence": detection["confidence"],
                "uncertain": True,
                "observations": [
                    {
                        "frame_index": frame["frame_index"],
                        "timestamp": frame["timestamp"],
                        "box": detection["box"],
                        "confidence": detection["confidence"],
                        "label": detection["label"],
                    }
                ],
            }
            tracks.append(track)
            next_active.append(track)
        active = next_active
    # Matched tracks are added when first created; sort by numeric ID for stable output.
    return sorted(tracks, key=lambda track: str(track["track_id"]))


def detector_cache_content_digest(record: Mapping[str, Any]) -> str:
    """Hash detector cache contents, excluding the self-reported digest."""

    payload = dict(record)
    payload.pop("cache_content_digest", None)
    try:
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ValueError("detector cache content must be JSON-serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


def write_detector_cache(
    cache_dir: str | Path,
    *,
    video_checksum: str,
    detector_revision: str,
    prompt: str,
    preprocessing: Mapping[str, Any] = GROUNDING_DETECTOR_PREPROCESSING,
    thresholds: Mapping[str, Any] | None = None,
    frame_policy: Mapping[str, Any] | None = None,
    tracker: Mapping[str, Any] | None = None,
    frames: Iterable[Mapping[str, Any]],
    tracks: Iterable[Mapping[str, Any]],
) -> Path:
    """Write a validated detector/track cache atomically."""

    payload = _detector_payload(
        video_checksum,
        detector_revision,
        prompt,
        preprocessing,
        thresholds or {
            "box_threshold": DEFAULT_DETECTOR_THRESHOLD,
            "text_threshold": DEFAULT_DETECTOR_TEXT_THRESHOLD,
        },
        frame_policy or {},
        tracker or {"algorithm": "adjacent-greedy-iou", "iou_threshold": DEFAULT_TRACK_IOU_THRESHOLD},
    )
    canonical_frames = sorted((_canonical_frame(frame) for frame in frames), key=lambda row: row["frame_index"])
    if not canonical_frames:
        raise ValueError("detector cache requires at least one frame")
    expected_indices = payload["frame_policy"].get("frame_indices")
    actual_indices = [row["frame_index"] for row in canonical_frames]
    if expected_indices is not None and actual_indices != [int(index) for index in expected_indices]:
        raise ValueError("detector cache frame policy does not match detector frames")
    canonical_tracks = [dict(track) for track in tracks]
    record = {
        "schema_version": GROUNDING_CACHE_SCHEMA_VERSION,
        "cache_key": detector_cache_key(**payload),
        **payload,
        "frames": canonical_frames,
        "tracks": canonical_tracks,
    }
    record["cache_content_digest"] = detector_cache_content_digest(record)
    json.dumps(record, sort_keys=True, allow_nan=False)
    path = detector_cache_path(cache_dir, **payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(record, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)
    return path


def load_detector_cache(path: str | Path, *, expected_cache_key: str | None = None) -> dict[str, Any]:
    """Load and validate a detector cache before it is used in scoring."""

    cache_path = Path(path)
    try:
        with cache_path.open() as stream:
            record = json.load(stream)
    except FileNotFoundError as exc:
        raise RuntimeError(f"grounding scoring requires an offline detector cache at {cache_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid detector cache JSON: {cache_path}") from exc
    if not isinstance(record, Mapping) or record.get("schema_version") != GROUNDING_CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported detector cache schema: {cache_path}")
    if record.get("cache_content_digest") != detector_cache_content_digest(record):
        raise ValueError(f"detector cache content digest mismatch: {cache_path}")
    identity = _detector_payload(
        record.get("video_checksum", ""),
        record.get("detector_revision", ""),
        record.get("prompt", ""),
        record.get("preprocessing", {}),
        record.get("thresholds", {}),
        record.get("frame_policy", {}),
        record.get("tracker", {}),
    )
    actual_key = detector_cache_key(**identity)
    if record.get("cache_key") != actual_key or (expected_cache_key and expected_cache_key != actual_key):
        raise ValueError(f"detector cache key mismatch: {cache_path}")
    frames = record.get("frames")
    tracks = record.get("tracks")
    if not isinstance(frames, list) or not frames or not isinstance(tracks, list):
        raise ValueError(f"detector cache frames/tracks are invalid: {cache_path}")
    canonical = [_canonical_frame(frame) for frame in frames]
    expected_indices = identity["frame_policy"].get("frame_indices")
    if expected_indices is not None and [row["frame_index"] for row in canonical] != [int(index) for index in expected_indices]:
        raise ValueError(f"detector cache frame policy mismatch: {cache_path}")
    for track in tracks:
        if not isinstance(track, Mapping) or not isinstance(track.get("observations"), list):
            raise ValueError(f"detector cache track is invalid: {cache_path}")
        for observation in track["observations"]:
            if not isinstance(observation, Mapping):
                raise ValueError(f"detector cache observation is invalid: {cache_path}")
            canonical_observation = _canonical_detection(observation, width=1, height=1, order=0)
            if canonical_observation is None or "box" not in observation:
                raise ValueError(f"detector cache observation has no box: {cache_path}")
            try:
                frame_index = int(observation["frame_index"])
                timestamp = float(observation["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"detector cache observation timing is invalid: {cache_path}") from exc
            if frame_index < 0 or not math.isfinite(timestamp) or timestamp < 0:
                raise ValueError(f"detector cache observation timing is invalid: {cache_path}")
            normalise_box(observation["box"])
    return dict(record)


write_grounding_cache = write_detector_cache
load_grounding_cache = load_detector_cache


def _observations(track: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("observations", "detections", "boxes"):
        value = track.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [item for item in value if isinstance(item, Mapping)]
    if "box" in track or "bbox" in track or "bounding_box" in track:
        return [track]
    return []


def _format_observation(track: Mapping[str, Any], observation: Mapping[str, Any]) -> str | None:
    raw_box = observation.get("box", observation.get("bbox", observation.get("bounding_box")))
    if raw_box is None:
        return None
    width = observation.get("width", observation.get("frame_width", track.get("width", track.get("frame_width"))))
    height = observation.get("height", observation.get("frame_height", track.get("height", track.get("frame_height"))))
    try:
        box = normalise_box(raw_box, width, height)
    except (TypeError, ValueError):
        return None
    frame = observation.get("frame_index", observation.get("frame", "?"))
    timestamp = observation.get("timestamp", observation.get("time", "?"))
    try:
        timestamp_text = f"{float(timestamp):.3f}s"
    except (TypeError, ValueError):
        timestamp_text = str(timestamp)
    return (
        f"frame={frame} time={timestamp_text} "
        f"box(left={box['left']:.3f},top={box['top']:.3f},"
        f"right={box['right']:.3f},bottom={box['bottom']:.3f})"
    )


def format_grounding_context(
    tracks: Iterable[Mapping[str, Any]],
    *,
    max_chars: int = 1200,
    max_tracks: int = 8,
    max_observations: int = 4,
) -> str:
    """Format a small, explicitly 2D grounding context for the original video prompt."""

    if max_chars <= 0 or max_tracks < 0 or max_observations < 0:
        raise ValueError("context bounds must be non-negative, with max_chars positive")
    header = (
        "Grounding context: uncertain normalized [0,1] 2D image coordinates; "
        "detector evidence is fallible."
    )
    lines = [header]
    for track in list(tracks)[:max_tracks]:
        if not isinstance(track, Mapping):
            continue
        track_id = track.get("track_id", "?")
        label = next(iter(_track_labels(track)), "unknown")
        try:
            confidence = float(track.get("confidence", track.get("score", 0.0)))
            confidence_text = f"{confidence:.3f}"
        except (TypeError, ValueError):
            confidence_text = "?"
        uncertainty = "true" if track.get("uncertain", True) else "false"
        lines.append(f"track={track_id} label={label} confidence={confidence_text} uncertain={uncertainty}")
        observations = _observations(track)
        for observation in observations[:max_observations]:
            formatted = _format_observation(track, observation)
            if formatted is not None:
                lines.append(f"  {formatted}")
    return "\n".join(lines)[:max_chars]


format_grounding_text = format_grounding_context
