"""Build and validate the Phase 7 canonical source-frame manifest.

The existing dataset manifest records aggregate video metadata.  Earlier
experiments sometimes add per-frame timestamps in a run-local source
manifest, so this module normalises both forms without changing either input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import common as C


SCHEMA_VERSION = 1
MANIFEST_KIND = "reva-phase7-frame-manifest"
TIMESTAMP_TOLERANCE = 1e-6
SAMPLE_OFFSETS = (0.0, 0.5)


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _content_digest(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "content_digest"}
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _source_timestamps(info: Mapping[str, Any], video_path: str) -> tuple[list[float], str]:
    n_frames = int(info.get("n_frames", info.get("frame_count", 0)))
    fps = _finite_float(info.get("fps"), f"{video_path}.fps")
    if n_frames < 1 or fps <= 0:
        raise ValueError(f"{video_path}: n_frames and fps must be positive")

    raw = info.get("frame_timestamps", info.get("source_timestamps_s"))
    if raw is None:
        return [index / fps for index in range(n_frames)], "index_divided_by_fps"
    if not isinstance(raw, list) or len(raw) != n_frames:
        raise ValueError(f"{video_path}: frame timestamp count does not match n_frames")
    timestamps = [_finite_float(value, f"{video_path}.frame_timestamps[{index}]") for index, value in enumerate(raw)]
    if any(a > b + TIMESTAMP_TOLERANCE for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{video_path}: frame timestamps must be nondecreasing")
    source = str(info.get("timestamp_source", "presentation_timestamp"))
    if not source:
        raise ValueError(f"{video_path}: timestamp_source must be non-empty")
    return timestamps, source


def _record(video_path: str, info: Mapping[str, Any]) -> dict[str, Any]:
    n_frames = int(info.get("n_frames", info.get("frame_count", 0)))
    fps = _finite_float(info.get("fps"), f"{video_path}.fps")
    width = int(info.get("width", 0))
    height = int(info.get("height", 0))
    duration = _finite_float(info.get("duration", info.get("duration_s")), f"{video_path}.duration_s")
    if n_frames < 1 or width < 1 or height < 1 or duration <= 0:
        raise ValueError(f"{video_path}: invalid frame dimensions or duration")
    timestamps, timestamp_source = _source_timestamps(info, video_path)
    if timestamps[0] < -TIMESTAMP_TOLERANCE or timestamps[-1] > duration + max(TIMESTAMP_TOLERANCE, 1 / fps):
        raise ValueError(f"{video_path}: frame timestamps exceed the declared duration")

    record: dict[str, Any] = {
        "video_path": video_path,
        "n_frames": n_frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_s": duration,
        "timestamp_source": timestamp_source,
        "frame_timestamps_s": timestamps,
        "frame_index_origin": 0,
    }
    if info.get("video_sha256"):
        record["video_sha256"] = str(info["video_sha256"])
    samples: dict[str, dict[str, Any]] = {}
    for offset in SAMPLE_OFFSETS:
        indices = C.sample_index_list(n_frames, fps, offset, mode="legacy")
        samples[str(offset)] = {
            "frame_indices": indices,
            "timestamps_s": [timestamps[index] for index in indices],
        }
    record["legacy_samples"] = samples
    return record


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _records_from_source(value: Any, source: Path) -> dict[str, Mapping[str, Any]]:
    if isinstance(value, dict) and isinstance(value.get("records"), dict):
        value = value["records"]
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected a mapping of video path to metadata")
    if any(not isinstance(info, Mapping) for info in value.values()):
        raise ValueError(f"{source}: every video record must be a metadata mapping")
    return {str(path): info for path, info in value.items()}


def _selected_paths(paths: Iterable[str] | None) -> set[str] | None:
    if paths is None:
        return None
    selected = {str(path).strip() for path in paths if str(path).strip()}
    if not selected:
        raise ValueError("video selection must contain at least one path")
    return selected


def build_manifest(
    base_manifest: Mapping[str, Mapping[str, Any]],
    *,
    source_manifests: Iterable[Mapping[str, Mapping[str, Any]]] = (),
    video_paths: Iterable[str] | None = None,
    dataset_revision: str = "unknown",
) -> dict[str, Any]:
    """Normalise metadata and exact source timestamps into a versioned record."""

    merged = {str(path): info for path, info in base_manifest.items()}
    for source in source_manifests:
        for path, info in source.items():
            if path in merged:
                base = _record(path, merged[path])
                candidate = _record(path, info)
                if (base["n_frames"], base["width"], base["height"]) != (
                    candidate["n_frames"], candidate["width"], candidate["height"]
                ):
                    raise ValueError(f"conflicting frame metadata for {path}")
            merged[path] = info

    selected = _selected_paths(video_paths)
    if selected is not None:
        missing = sorted(selected - merged.keys())
        if missing:
            raise ValueError(f"selected videos are absent from all manifests: {missing[:3]}")
        merged = {path: merged[path] for path in sorted(selected)}
    if not merged:
        raise ValueError("cannot build an empty Phase 7 frame manifest")

    records = {path: _record(path, merged[path]) for path in sorted(merged)}
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": MANIFEST_KIND,
        "dataset_revision": str(dataset_revision),
        "sampling_policy": {
            "mode": "legacy",
            "offsets": list(SAMPLE_OFFSETS),
            "frame_budget": C.MAX_FRAMES,
            "timestamp_binding": "source_frame_index_to_manifest_timestamp",
        },
        "records": records,
    }
    manifest["content_digest"] = _content_digest(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Raise on malformed or tampered canonical manifests."""

    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("manifest_kind") != MANIFEST_KIND:
        raise ValueError("unsupported Phase 7 frame manifest schema")
    if manifest.get("content_digest") != _content_digest(manifest):
        raise ValueError("Phase 7 frame manifest content digest mismatch")
    policy = manifest.get("sampling_policy")
    if not isinstance(policy, Mapping) or policy.get("mode") != "legacy":
        raise ValueError("Phase 7 frame manifest sampling policy is invalid")
    if list(policy.get("offsets", [])) != list(SAMPLE_OFFSETS) or policy.get("frame_budget") != C.MAX_FRAMES:
        raise ValueError("Phase 7 frame manifest sampling policy is not the frozen Stage A policy")
    records = manifest.get("records")
    if not isinstance(records, Mapping) or not records:
        raise ValueError("Phase 7 frame manifest must contain records")
    for video_path, record in records.items():
        if not isinstance(video_path, str) or not isinstance(record, Mapping):
            raise ValueError("Phase 7 frame manifest record is invalid")
        n_frames = int(record.get("n_frames", 0))
        fps = _finite_float(record.get("fps"), f"{video_path}.fps")
        width = int(record.get("width", 0))
        height = int(record.get("height", 0))
        duration = _finite_float(record.get("duration_s"), f"{video_path}.duration_s")
        if width < 1 or height < 1 or duration <= 0 or not str(record.get("timestamp_source", "")):
            raise ValueError(f"{video_path}: invalid canonical metadata")
        timestamps = record.get("frame_timestamps_s")
        if n_frames < 1 or fps <= 0 or not isinstance(timestamps, list) or len(timestamps) != n_frames:
            raise ValueError(f"{video_path}: invalid canonical frame timeline")
        parsed = [_finite_float(value, f"{video_path}.frame_timestamps_s") for value in timestamps]
        if parsed[0] < -TIMESTAMP_TOLERANCE or parsed[-1] > duration + max(TIMESTAMP_TOLERANCE, 1 / fps):
            raise ValueError(f"{video_path}: canonical frame timestamps exceed duration")
        if any(a > b + TIMESTAMP_TOLERANCE for a, b in zip(parsed, parsed[1:])):
            raise ValueError(f"{video_path}: canonical frame timeline is not monotone")
        samples = record.get("legacy_samples")
        if not isinstance(samples, Mapping):
            raise ValueError(f"{video_path}: legacy samples are missing")
        for offset in SAMPLE_OFFSETS:
            sample = samples.get(str(offset))
            if not isinstance(sample, Mapping):
                raise ValueError(f"{video_path}: legacy sample {offset} is missing")
            indices = sample.get("frame_indices")
            sample_times = sample.get("timestamps_s")
            if not isinstance(indices, list) or not indices or not isinstance(sample_times, list) or len(indices) != len(sample_times):
                raise ValueError(f"{video_path}: malformed legacy sample {offset}")
            if any(not isinstance(index, int) or index < 0 or index >= n_frames for index in indices):
                raise ValueError(f"{video_path}: legacy sample {offset} has an invalid frame index")
            sample_values = [_finite_float(value, f"{video_path}.legacy_samples[{offset}]") for value in sample_times]
            if any(abs(value - parsed[index]) > TIMESTAMP_TOLERANCE for value, index in zip(sample_values, indices)):
                raise ValueError(f"{video_path}: legacy sample {offset} timestamps do not bind to source frames")


def write_manifest(manifest: Mapping[str, Any], destination: str | os.PathLike[str]) -> Path:
    validate_manifest(manifest)
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(target)
    return target


def _split_video_paths(path: Path) -> set[str]:
    payload = _load_json(path)
    rows = payload.get("QA") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a QA list")
    paths = {row.get("video_path") for row in rows if isinstance(row, Mapping)}
    if any(not isinstance(value, str) or not value for value in paths):
        raise ValueError(f"{path}: every QA row must have a video_path")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", required=True, type=Path)
    parser.add_argument("--source-manifest", action="append", type=Path, default=[])
    parser.add_argument("--split", action="append", type=Path, default=[])
    parser.add_argument("--video", action="append", default=[])
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    base = _records_from_source(_load_json(args.base_manifest), args.base_manifest)
    sources = [_records_from_source(_load_json(path), path) for path in args.source_manifest]
    selected = set(args.video)
    for split in args.split:
        selected.update(_split_video_paths(split))
    revision = args.dataset_revision
    if revision is None:
        revision_path = args.base_manifest.parent / "dataset_revision.json"
        revision = str(_load_json(revision_path).get("revision", "unknown")) if revision_path.is_file() else "unknown"
    manifest = build_manifest(base, source_manifests=sources, video_paths=selected or None, dataset_revision=revision)
    path = write_manifest(manifest, args.out)
    print(f"wrote {path} ({len(manifest['records'])} videos, digest {manifest['content_digest']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
