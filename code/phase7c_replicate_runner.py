"""Prepare and validate the isolated Phase 7C Replicate smoke.

The default operation is a local dry run.  It reads the already staged,
source-reviewed archway and marked-van clips, binds every planned request to
the canonical source timeline, and writes a resumable manifest.  API calls
are available only through ``--execute`` and require a token in the process
environment; the token is never persisted or printed.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
from typing import Any
import urllib.request
import zipfile
from email.utils import parsedate_to_datetime

import phase7_manifest as canonical


SCHEMA_VERSION = 1
MANIFEST_KIND = "reva-phase7c-replicate-smoke"
RUN_RELATIVE = Path("runs/phase7C_replicate_smoke")
CANONICAL_RELATIVE = Path("runs/phase7A_feasibility/canonical_frame_manifest.json")
CHECKSUM_RELATIVE = Path("runs/phase7B_cache_validation/source_checksum_manifest.json")
REPLICATE_API_BASE = "https://api.replicate.com/v1"
REPLICATE_API_SOURCES = (
    "https://replicate.com/docs/topics/predictions/create-a-prediction",
    "https://replicate.com/docs/reference/http/",
)
REPLICATE_SDK_VERSION = "1.0.7"
REPLICATE_SDK_SOURCES = (
    "https://github.com/replicate/replicate-python/tree/1.0.7/replicate",
    "https://raw.githubusercontent.com/replicate/replicate-python/1.0.7/replicate/client.py",
    "https://raw.githubusercontent.com/replicate/replicate-python/1.0.7/replicate/helpers.py",
)
REQUIREMENTS_RELATIVE = RUN_RELATIVE / "requirements.txt"

MERGED_MASK_MODEL = "lucataco/sam3-video"
MERGED_MASK_VERSION = "408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269"
ID_PRESERVING_MODEL = "8digit/gener8-sam3"
ID_PRESERVING_VERSION = "a6f50615da4cfa8b03fe95ec56681bc252a5e60c2ed368938f72bfa4f55dfcd1"

FIXED_CONTROL_PROMPTS = ("vehicle", "building", "road")
MAX_REQUESTS = 16
MAX_RETRY_ATTEMPTS = 4
MAX_RETRIES_PER_REQUEST = 1
POLL_INTERVAL_SECONDS = 2.0
MAX_POLL_CYCLES_PER_REQUEST = 300
MODEL_PLANNING_COST_USD = {
    "merged_mask_reference": 0.0026,
    "id_preserving_candidate": 0.078,
}

_VIDEO_SPECS = (
    {
        "qa_id": "train_000047",
        "split": "valT",
        "role": "scene_structure_under_camera_motion",
        "video_path": "VisDrone/uav0000077_00720_v_01.mp4",
        "question": "At what point in the video does the large archway with Chinese characters first become visible?",
        "question_prompt": "large archway with Chinese characters",
    },
    {
        "qa_id": "train_001138",
        "split": "valT",
        "role": "visually_distinctive_object_identity",
        "video_path": "VisDrone/uav0000243_00001_v_04.mp4",
        "question": "At what point in the video does the white van with blue markings on its side first appear?",
        "question_prompt": "white van with blue markings on its side",
    },
)

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "merged_mask_reference": {
        "model": MERGED_MASK_MODEL,
        "version": MERGED_MASK_VERSION,
        "version_ref": f"{MERGED_MASK_MODEL}:{MERGED_MASK_VERSION}",
        "role": "pinned_existing_merged_mask_reference",
        "input_schema": {
            "video": "file",
            "prompt": "string",
            "mask_only": "boolean",
            "return_zip": "boolean",
        },
        "input_defaults": {"mask_only": True, "return_zip": True},
        "output_schema": {
            "kind": "zip",
            "required": "one PNG mask per full source frame, each explicitly named with a frame or mask index",
            "optional": ["overlay.mp4"],
            "timestamp_mapping": "unverified unless endpoint output metadata binds frame count and fps",
        },
        "schema_sources": [
            "https://replicate.com/lucataco/sam3-video",
            "code/replicate_sam3.py",
        ],
    },
    "id_preserving_candidate": {
        "model": ID_PRESERVING_MODEL,
        "version": ID_PRESERVING_VERSION,
        "version_ref": f"{ID_PRESERVING_MODEL}:{ID_PRESERVING_VERSION}",
        "role": "third_party_id_preserving_candidate_unverified",
        "input_schema": {
            "video": "file",
            "prompt": "string",
            "preview": "boolean",
        },
        "input_defaults": {"preview": False},
        "output_schema": {
            "kind": "zip",
            "required": ["objects.json", "labels/labels_%05d.png for every source frame"],
            "label_encoding": "16-bit PNG; zero background and nonzero object labels",
            "timestamp_mapping": "verified only when objects.json frame count/fps match the source contract",
            "optional": ["preview.mp4"],
        },
        "schema_sources": [
            "https://replicate.com/8digit/gener8-sam3",
            "https://replicate.com/8digit/gener8-sam3/versions/"
            + ID_PRESERVING_VERSION,
        ],
    },
}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _content_digest(value: Mapping[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "content_digest"}
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _safe_member(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def _png_header(payload: bytes) -> tuple[int, int, int]:
    if len(payload) < 29 or not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("output member is not a PNG")
    if payload[12:16] != b"IHDR":
        raise ValueError("PNG is missing IHDR")
    width, height, bit_depth = struct.unpack(">IIB", payload[16:25])
    if width < 1 or height < 1:
        raise ValueError("PNG dimensions must be positive")
    return width, height, bit_depth


def _frame_number(name: str) -> int:
    match = re.search(r"(?:frame|mask)[-_]?(\d+)", Path(name).stem, re.IGNORECASE)
    if not match:
        raise ValueError(f"output mask has no frame index: {name}")
    return int(match.group(1))


def _probe_overlay(payload: bytes) -> dict[str, Any]:
    """Read the optional output video metadata without trusting archive order."""

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4") as stream:
            stream.write(payload)
            stream.flush()
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,nb_frames,nb_read_frames,avg_frame_rate,r_frame_rate",
                    "-of",
                    "json",
                    stream.name,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
    except OSError as exc:
        raise ValueError("merged-mask overlay metadata could not be validated") from exc
    if result.returncode:
        raise ValueError("merged-mask overlay metadata could not be validated")
    try:
        streams = json.loads(result.stdout).get("streams", [])
        metadata = streams[0]
        width = int(metadata["width"])
        height = int(metadata["height"])
        frame_count = int(metadata.get("nb_frames") or metadata.get("nb_read_frames"))
        rate = str(metadata.get("avg_frame_rate") or metadata.get("r_frame_rate") or "")
        numerator, denominator = rate.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
        raise ValueError("merged-mask overlay metadata could not be validated") from exc
    if frame_count < 1 or width < 1 or height < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError("merged-mask overlay metadata could not be validated")
    return {"width": width, "height": height, "frame_count": frame_count, "fps": fps}


def _validate_overlay(
    archive: zipfile.ZipFile,
    names: list[str],
    source: Mapping[str, Any],
    source_frame_count: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    overlay_names = [name for name in names if Path(name).name.casefold() == "overlay.mp4"]
    if len(overlay_names) != 1:
        raise ValueError("sparse merged-mask output requires overlay.mp4 metadata")
    metadata = _probe_overlay(archive.read(overlay_names[0]))
    if metadata["frame_count"] != source_frame_count:
        raise ValueError("merged-mask overlay frame count does not match source frame count")
    source_fps = float(source["fps"])
    if abs(float(metadata["fps"]) - source_fps) > 1e-6:
        raise ValueError("merged-mask overlay fps does not match source fps")
    # The observed SAM3 overlay is padded from 640x360 to 640x368.
    if metadata["width"] != width or metadata["height"] not in {height, height + 8}:
        raise ValueError("merged-mask overlay dimensions are incompatible with source")
    metadata.update(
        {
            "member": overlay_names[0],
            "padding": {
                "width": metadata["width"] - width,
                "height": metadata["height"] - height,
            },
        }
    )
    return metadata


def _frame_mapping(source: Mapping[str, Any], output_indices: list[int]) -> list[dict[str, Any]]:
    mapping = _validate_source_frame_contract(source)
    rows: list[dict[str, Any]] = []
    for output_index in output_indices:
        if output_index < 0 or output_index >= len(mapping):
            raise ValueError(f"output frame {output_index} is outside source frame mapping")
        row = mapping[output_index]
        source_index = int(row["source_frame_index"])
        timestamp = float(row["timestamp_s"])
        if not math.isfinite(timestamp):
            raise ValueError("source frame mapping timestamp is not finite")
        rows.append(
            {
                "output_frame_index": output_index,
                "source_frame_index": source_index,
                "timestamp_s": timestamp,
            }
        )
    return rows


def _validate_source_frame_contract(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    mapping = source.get("frame_mapping")
    if not isinstance(mapping, list) or len(mapping) != int(source.get("n_frames", -1)):
        raise ValueError("source frame mapping does not match source frame count")
    try:
        fps = float(source["fps"])
        width = int(source["width"])
        height = int(source["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("source frame metadata is incomplete") from exc
    if not math.isfinite(fps) or fps <= 0 or width < 1 or height < 1:
        raise ValueError("source frame metadata is invalid")
    rows: list[dict[str, Any]] = []
    last_timestamp = -math.inf
    for index, row in enumerate(mapping):
        if not isinstance(row, Mapping) or int(row.get("source_frame_index", -1)) != index:
            raise ValueError("source frame mapping indices are not contiguous")
        timestamp = float(row.get("timestamp_s"))
        if not math.isfinite(timestamp) or timestamp < last_timestamp:
            raise ValueError("source frame mapping timestamps are not monotone")
        rows.append({"source_frame_index": index, "timestamp_s": timestamp})
        last_timestamp = timestamp
    return rows


def _zip_members(data: bytes | bytearray | io.BufferedIOBase) -> tuple[zipfile.ZipFile, list[str]]:
    raw = data.read() if hasattr(data, "read") else bytes(data)
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError("Replicate output is not a readable ZIP") from exc
    names = [item.filename for item in archive.infolist() if not item.is_dir()]
    if any(not _safe_member(name) for name in names):
        archive.close()
        raise ValueError("Replicate output ZIP contains an unsafe member path")
    return archive, names


def validate_output_zip(
    model_key: str,
    data: bytes | bytearray | io.BufferedIOBase,
    source: Mapping[str, Any],
    *,
    expected_prompt: str | None = None,
) -> dict[str, Any]:
    """Validate one model-specific ZIP and bind its frames to source timestamps."""

    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown Phase 7C model key: {model_key}")
    archive, names = _zip_members(data)
    try:
        source_mapping = _validate_source_frame_contract(source)
        width, height = int(source["width"]), int(source["height"])
        source_frame_count = len(source_mapping)
        if model_key == "merged_mask_reference":
            mask_names = [name for name in names if name.casefold().endswith(".png")]
            if not mask_names:
                raise ValueError("merged-mask output ZIP contains no PNG masks")
            rows: list[dict[str, Any]] = []
            seen: set[int] = set()
            for name in mask_names:
                # Require explicit names; sparse output is accepted only after
                # its overlay validates the complete source timeline.
                output_index = _frame_number(name)
                if output_index in seen:
                    raise ValueError("merged-mask output maps multiple masks to one frame")
                seen.add(output_index)
                member_width, member_height, _ = _png_header(archive.read(name))
                if (member_width, member_height) != (width, height):
                    raise ValueError(f"merged-mask PNG dimensions do not match source: {name}")
                rows.append({"member": name, "output_frame_index": output_index})
            rows.sort(key=lambda item: int(item["output_frame_index"]))
            output_indices = [int(item["output_frame_index"]) for item in rows]
            if any(index < 0 or index >= source_frame_count for index in output_indices):
                raise ValueError("merged-mask output frame index is outside source frame range")
            dense = output_indices == list(range(source_frame_count))
            overlay = None
            missing_source_frame_indices: list[int] = []
            if not dense:
                overlay = _validate_overlay(
                    archive, names, source, source_frame_count, width, height
                )
                missing_source_frame_indices = [
                    index for index in range(source_frame_count) if index not in seen
                ]
                for row in rows:
                    row["source_frame_index"] = int(row["output_frame_index"])
            return {
                "output_kind": "merged_mask_zip",
                "mask_count": len(rows),
                "source_frame_count": source_frame_count,
                "source_fps": float(source["fps"]),
                "source_width": width,
                "source_height": height,
                "output_frame_indices": output_indices,
                "frame_mapping": [] if dense else [
                    {
                        "output_frame_index": int(row["output_frame_index"]),
                        "source_frame_index": int(row["source_frame_index"]),
                        "timestamp_s": float(
                            source_mapping[int(row["source_frame_index"])]
                            ["timestamp_s"]
                        ),
                    }
                    for row in rows
                ],
                "source_frame_indices": output_indices,
                "missing_source_frame_indices": missing_source_frame_indices,
                "timestamp_mapping_status": "unverified_no_endpoint_metadata",
                "frame_mapping_provenance": (
                    "named zero-based mask indices joined to the source frame mapping; "
                    "overlay frame count and fps matched the source; endpoint timestamp "
                    "metadata unavailable"
                    if not dense
                    else "named zero-based mask indices; endpoint timestamp metadata unavailable"
                ),
                **({"overlay_metadata": overlay} if not dense else {}),
                "members": rows,
            }

        object_names = [name for name in names if Path(name).name == "objects.json"]
        if len(object_names) != 1:
            raise ValueError("ID-preserving output must contain exactly one objects.json")
        try:
            metadata = json.loads(archive.read(object_names[0]))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("objects.json is not valid JSON") from exc
        if not isinstance(metadata, Mapping):
            raise ValueError("objects.json must contain an object")
        if expected_prompt is not None and str(metadata.get("prompt", "")) != expected_prompt:
            raise ValueError("objects.json prompt does not match the planned request")
        if int(metadata.get("width", -1)) != width or int(metadata.get("height", -1)) != height:
            raise ValueError("objects.json dimensions do not match source")
        label_names: list[tuple[int, str]] = []
        for name in names:
            match = re.fullmatch(r"labels/labels_(\d{5})\.png", name)
            if match:
                label_names.append((int(match.group(1)), name))
        if not label_names:
            raise ValueError("ID-preserving output contains no labels/*.png frames")
        label_names.sort()
        output_indices = [index for index, _ in label_names]
        if output_indices != list(range(len(output_indices))):
            raise ValueError("ID-preserving labels must be contiguous from frame zero")
        if int(metadata.get("frames", -1)) != source_frame_count:
            raise ValueError("objects.json frame count does not match source frame count")
        try:
            output_fps = float(metadata["fps"])
            source_fps = float(source["fps"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("objects.json fps metadata is missing") from exc
        if not math.isfinite(output_fps) or abs(output_fps - source_fps) > 1e-6:
            raise ValueError("objects.json fps does not match source fps")
        if output_indices != list(range(source_frame_count)):
            raise ValueError("ID-preserving labels must cover every source frame")
        for _, name in label_names:
            member_width, member_height, bit_depth = _png_header(archive.read(name))
            if (member_width, member_height) != (width, height):
                raise ValueError(f"ID-preserving label dimensions do not match source: {name}")
            if bit_depth != 16:
                raise ValueError("ID-preserving labels must be 16-bit PNG")
        objects = metadata.get("objects", metadata.get("instances"))
        if not isinstance(objects, list):
            raise ValueError("objects.json must contain an objects list")
        required = {
            "label", "sam_id", "frames", "first_frame", "last_frame",
            "mean_score", "best_frame", "best_box", "best_area",
            "best_area_fraction", "best_score",
        }
        for item in objects:
            if not isinstance(item, Mapping) or not required.issubset(item):
                raise ValueError("objects.json object entry is incomplete")
            first, last = int(item["first_frame"]), int(item["last_frame"])
            if not 0 <= first <= last < len(output_indices):
                raise ValueError("objects.json object frame interval is out of range")
            best_box = item["best_box"]
            if not isinstance(best_box, list) or len(best_box) != 4:
                raise ValueError("objects.json best_box must contain four coordinates")
        return {
            "output_kind": "id_preserving_label_zip",
            "label_frame_count": len(label_names),
            "source_frame_count": source_frame_count,
            "source_fps": float(source["fps"]),
            "source_width": width,
            "source_height": height,
            "output_frame_count": int(metadata["frames"]),
            "output_fps": output_fps,
            "output_frame_indices": output_indices,
            "frame_mapping": _frame_mapping(source, output_indices),
            "timestamp_mapping_status": "verified_output_frame_count_and_fps",
            "objects": [dict(item) for item in objects],
            "members": [name for _, name in label_names],
        }
    finally:
        archive.close()


def _source_videos(root: Path, canonical_path: Path, checksum_path: Path) -> list[dict[str, Any]]:
    canonical_manifest = _load_json(canonical_path)
    canonical.validate_manifest(canonical_manifest)
    checksum_manifest = _load_json(checksum_path)
    entries = {
        str(item.get("video_path")): item
        for item in checksum_manifest.get("entries", [])
        if isinstance(item, Mapping)
    }
    videos: list[dict[str, Any]] = []
    for spec in _VIDEO_SPECS:
        path = spec["video_path"]
        record = canonical_manifest["records"].get(path)
        entry = entries.get(path)
        if not isinstance(record, Mapping) or not isinstance(entry, Mapping):
            raise ValueError(f"Phase 7C source is absent from validated manifests: {path}")
        if entry.get("hpc_verification_status") != "match" or entry.get("local_source_status") != "match":
            raise ValueError(f"Phase 7C source checksum is not independently verified: {path}")
        source_path = Path(str(entry.get("local_source_path", "")))
        if not source_path.is_file():
            raise ValueError(f"Phase 7C staged source is missing: {source_path}")
        actual_checksum = _sha256(source_path)
        if actual_checksum != str(entry.get("local_source_sha256")) or actual_checksum != str(entry.get("hpc_source_sha256")):
            raise ValueError(f"Phase 7C source checksum mismatch: {path}")
        timestamps = [float(value) for value in record["frame_timestamps_s"]]
        mapping = [
            {"source_frame_index": index, "timestamp_s": timestamp}
            for index, timestamp in enumerate(timestamps)
        ]
        question_prompt = spec["question_prompt"]
        if question_prompt.casefold() not in spec["question"].casefold():
            raise ValueError(f"question prompt is not a literal question span: {spec['qa_id']}")
        videos.append(
            {
                **spec,
                "source_path": str(source_path.resolve().relative_to(root.resolve())),
                "source_sha256": actual_checksum,
                "width": int(record["width"]),
                "height": int(record["height"]),
                "fps": float(record["fps"]),
                "n_frames": int(record["n_frames"]),
                "duration_s": float(record["duration_s"]),
                "timestamp_source": str(record["timestamp_source"]),
                "frame_mapping": mapping,
                "fixed_control_prompts": list(FIXED_CONTROL_PROMPTS),
            }
        )
    return videos


def build_run_manifest(root: str | Path) -> dict[str, Any]:
    """Build the deterministic dry-run manifest for exactly two reviewed clips."""

    root_path = Path(root).resolve()
    canonical_path = root_path / CANONICAL_RELATIVE
    checksum_path = root_path / CHECKSUM_RELATIVE
    videos = _source_videos(root_path, canonical_path, checksum_path)
    requests: list[dict[str, Any]] = []
    for video in videos:
        prompt_rows = [("question-conditioned", video["question_prompt"])] + [
            ("fixed-vocabulary", prompt) for prompt in FIXED_CONTROL_PROMPTS
        ]
        for model_key in MODEL_SPECS:
            for prompt_arm, prompt in prompt_rows:
                request_id = f"{video['qa_id']}--{model_key}--{prompt_arm}--{prompt.replace(' ', '_')}"
                requests.append(
                    {
                        "request_id": request_id,
                        "qa_id": video["qa_id"],
                        "video_path": video["video_path"],
                        "source_path": video["source_path"],
                        "source_sha256": video["source_sha256"],
                        "model_key": model_key,
                        "model": MODEL_SPECS[model_key]["model"],
                        "version": MODEL_SPECS[model_key]["version"],
                        "prompt_arm": prompt_arm,
                        "prompt": prompt,
                        "input": {
                            "video": video["source_path"],
                            "prompt": prompt,
                            **MODEL_SPECS[model_key]["input_defaults"],
                        },
                        "frame_policy": "full_source_timeline",
                        "frame_mapping_digest": hashlib.sha256(_json_bytes(video["frame_mapping"])).hexdigest(),
                        "output_relpath": f"outputs/{request_id}.zip",
                        "status": "planned",
                        "attempts": [],
                    }
                )
    if len(requests) != MAX_REQUESTS:
        raise ValueError(f"Phase 7C request count changed unexpectedly: {len(requests)}")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": MANIFEST_KIND,
        "status": "dry_run",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_revision": _load_json(root_path / "data/dataset_revision.json").get("revision", "unknown"),
        "canonical_manifest": str(CANONICAL_RELATIVE),
        "source_checksum_manifest": str(CHECKSUM_RELATIVE),
        "selection": {
            "video_count": len(videos),
            "qa_ids": [video["qa_id"] for video in videos],
            "basis": "source-reviewed archway camera-motion and marked-van identity diagnostics",
            "answer_labels_used": False,
            "baseline_predictions_used": False,
            "manual_target_boxes_used": False,
            "model_ranked_identity_used": False,
        },
        "prompt_policy": {
            "question_conditioned": "literal phrase spans from the question wording",
            "fixed_vocabulary": list(FIXED_CONTROL_PROMPTS),
            "answer_options_or_labels_in_prompts": False,
        },
        "models": MODEL_SPECS,
        "execution": {
            "resumable": True,
            "scheduler": "sequential_async_predictions_create_poll",
            "no_parallel_submissions": True,
            "max_primary_prediction_creates": MAX_REQUESTS,
            "max_retry_prediction_creates": MAX_RETRY_ATTEMPTS,
            "max_prediction_create_attempts_total": MAX_REQUESTS + MAX_RETRY_ATTEMPTS,
            "max_new_prediction_attempts": MAX_REQUESTS,
            "max_retries_per_request": MAX_RETRIES_PER_REQUEST,
            "poll_interval_seconds": POLL_INTERVAL_SECONDS,
            "max_poll_cycles_per_request": MAX_POLL_CYCLES_PER_REQUEST,
            "paid_calls_made": 0,
            "prediction_create_attempts": 0,
            "prediction_ids_created": 0,
            "prediction_ids_observed": 0,
            "successful_predictions": 0,
            "retry_attempts": 0,
            "billing_status": "no_calls",
            "billing_uncertainty": False,
            "billable_cost_estimate_usd": 0.0,
            "dry_run": True,
            "api_base": REPLICATE_API_BASE,
            "api_sources": list(REPLICATE_API_SOURCES),
            "runtime_dependency": {
                "package": "replicate",
                "version": REPLICATE_SDK_VERSION,
                "requirements_file": str(REQUIREMENTS_RELATIVE),
                "api_surface": [
                    "replicate.Client",
                    "client.predictions.create(version=..., input=...)",
                    "client.predictions.get(prediction_id)",
                    "local file handles encoded through client.files.create",
                    "FileOutput.read()",
                ],
                "sources": list(REPLICATE_SDK_SOURCES),
            },
        },
        "credential_status": "present" if os.environ.get("REPLICATE_API_TOKEN") else "absent",
        "videos": videos,
        "requests": requests,
    }
    manifest["content_digest"] = _content_digest(manifest)
    validate_manifest(manifest, root_path, rehash_sources=False)
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any],
    root: str | Path,
    *,
    rehash_sources: bool = False,
) -> None:
    """Validate the run manifest and immutable source/frame bindings."""

    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("manifest_kind") != MANIFEST_KIND:
        raise ValueError("unsupported Phase 7C manifest schema")
    if manifest.get("content_digest") != _content_digest(manifest):
        raise ValueError("Phase 7C manifest content digest mismatch")
    execution = manifest.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("Phase 7C execution contract is missing")
    if execution.get("scheduler") != "sequential_async_predictions_create_poll" or execution.get("no_parallel_submissions") is not True:
        raise ValueError("Phase 7C scheduler must be sequential and asynchronous")
    if tuple(execution.get("api_sources", ())) != REPLICATE_API_SOURCES:
        raise ValueError("Phase 7C Replicate API source references changed")
    runtime_dependency = execution.get("runtime_dependency")
    if not isinstance(runtime_dependency, Mapping):
        raise ValueError("Phase 7C runtime dependency manifest is missing")
    if runtime_dependency.get("package") != "replicate" or runtime_dependency.get("version") != REPLICATE_SDK_VERSION:
        raise ValueError("Phase 7C Replicate SDK pin changed")
    if runtime_dependency.get("requirements_file") != str(REQUIREMENTS_RELATIVE):
        raise ValueError("Phase 7C runtime dependency path changed")
    if tuple(runtime_dependency.get("sources", ())) != REPLICATE_SDK_SOURCES:
        raise ValueError("Phase 7C Replicate SDK source references changed")
    if int(execution.get("max_primary_prediction_creates", -1)) != MAX_REQUESTS:
        raise ValueError("Phase 7C primary create cap changed")
    if int(execution.get("max_retry_prediction_creates", -1)) != MAX_RETRY_ATTEMPTS:
        raise ValueError("Phase 7C shared retry cap changed")
    if int(execution.get("max_prediction_create_attempts_total", -1)) != MAX_REQUESTS + MAX_RETRY_ATTEMPTS:
        raise ValueError("Phase 7C total create cap changed")
    if int(execution.get("max_retries_per_request", -1)) != MAX_RETRIES_PER_REQUEST:
        raise ValueError("Phase 7C per-request retry cap changed")
    if float(execution.get("poll_interval_seconds", -1.0)) != POLL_INTERVAL_SECONDS:
        raise ValueError("Phase 7C poll interval changed")
    if int(execution.get("max_poll_cycles_per_request", -1)) != MAX_POLL_CYCLES_PER_REQUEST:
        raise ValueError("Phase 7C poll cycle cap changed")
    if int(execution.get("prediction_create_attempts", 0)) < 0 or int(execution.get("retry_attempts", 0)) < 0:
        raise ValueError("Phase 7C create attempt counters cannot be negative")
    if int(execution.get("retry_attempts", 0)) > MAX_RETRY_ATTEMPTS:
        raise ValueError("Phase 7C shared retry cap exceeded")
    if int(execution.get("prediction_create_attempts", 0)) > MAX_REQUESTS + MAX_RETRY_ATTEMPTS:
        raise ValueError("Phase 7C total create cap exceeded")
    for counter in ("prediction_ids_created", "prediction_ids_observed", "successful_predictions"):
        if int(execution.get(counter, 0)) < 0:
            raise ValueError(f"Phase 7C {counter} cannot be negative")
    if int(execution.get("successful_predictions", 0)) > int(execution.get("prediction_ids_observed", 0)):
        raise ValueError("Phase 7C successful predictions exceed observed prediction IDs")
    try:
        estimate = float(execution.get("billable_cost_estimate_usd", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Phase 7C billable cost estimate is invalid") from exc
    if not math.isfinite(estimate) or estimate < 0:
        raise ValueError("Phase 7C billable cost estimate is invalid")
    if int(manifest.get("selection", {}).get("video_count", -1)) != 2:
        raise ValueError("Phase 7C manifest must contain exactly two videos")
    models = manifest.get("models")
    if not isinstance(models, Mapping) or set(models) != set(MODEL_SPECS):
        raise ValueError("Phase 7C model set is not exact")
    for key, expected in MODEL_SPECS.items():
        if models[key].get("model") != expected["model"] or models[key].get("version") != expected["version"]:
            raise ValueError(f"Phase 7C model pin changed: {key}")
    videos = manifest.get("videos")
    if not isinstance(videos, list) or len(videos) != 2:
        raise ValueError("Phase 7C manifest video list is invalid")
    video_by_id = {str(video.get("qa_id")): video for video in videos if isinstance(video, Mapping)}
    if set(video_by_id) != {spec["qa_id"] for spec in _VIDEO_SPECS}:
        raise ValueError("Phase 7C selected video IDs changed")
    for video in videos:
        question = str(video.get("question", ""))
        phrase = str(video.get("question_prompt", ""))
        if not phrase or phrase.casefold() not in question.casefold():
            raise ValueError("question-conditioned prompt is not a question span")
        mapping = video.get("frame_mapping")
        if not isinstance(mapping, list) or len(mapping) != int(video.get("n_frames", -1)):
            raise ValueError(f"invalid source frame mapping for {video.get('qa_id')}")
        source_relative = Path(str(video.get("source_path", "")))
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise ValueError("source path must remain inside the workspace")
        last_timestamp = -math.inf
        for index, row in enumerate(mapping):
            if not isinstance(row, Mapping) or int(row.get("source_frame_index", -1)) != index:
                raise ValueError("source frame mapping indices are not contiguous")
            timestamp = float(row.get("timestamp_s"))
            if not math.isfinite(timestamp) or timestamp < last_timestamp:
                raise ValueError("source frame mapping timestamps are not monotone")
            last_timestamp = timestamp
        if rehash_sources:
            source_path = Path(str(root)) / str(video["source_path"])
            if _sha256(source_path) != str(video["source_sha256"]):
                raise ValueError(f"source checksum changed: {video.get('qa_id')}")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or len(requests) != MAX_REQUESTS:
        raise ValueError("Phase 7C request count does not match capped plan")
    seen: set[str] = set()
    for request in requests:
        if not isinstance(request, Mapping):
            raise ValueError("Phase 7C request is malformed")
        request_id = str(request.get("request_id", ""))
        if not request_id or request_id in seen:
            raise ValueError("Phase 7C request IDs must be unique")
        seen.add(request_id)
        if request.get("model_key") not in MODEL_SPECS:
            raise ValueError("Phase 7C request model key is invalid")
        video = video_by_id.get(str(request.get("qa_id")))
        if video is None:
            raise ValueError("Phase 7C request points to an unselected video")
        prompt = str(request.get("prompt", ""))
        allowed = {str(video["question_prompt"]), *FIXED_CONTROL_PROMPTS}
        if prompt not in allowed:
            raise ValueError("Phase 7C request prompt is outside question/fixed vocabulary")
        model = MODEL_SPECS[str(request["model_key"])]
        if request.get("model") != model["model"] or request.get("version") != model["version"]:
            raise ValueError("Phase 7C request model pin changed")
        if request.get("source_path") != video.get("source_path") or request.get("source_sha256") != video.get("source_sha256"):
            raise ValueError("Phase 7C request source binding changed")
        request_input = request.get("input")
        if not isinstance(request_input, Mapping) or request_input.get("video") != video.get("source_path") or request_input.get("prompt") != prompt:
            raise ValueError("Phase 7C request input is not bound to its source/prompt")
        if request.get("prompt_arm") == "question-conditioned" and prompt != video.get("question_prompt"):
            raise ValueError("question-conditioned request prompt changed")
        if request.get("prompt_arm") == "fixed-vocabulary" and prompt not in FIXED_CONTROL_PROMPTS:
            raise ValueError("fixed-vocabulary request prompt changed")
        output_relative = Path(str(request.get("output_relpath", "")))
        if output_relative.is_absolute() or ".." in output_relative.parts or not output_relative.parts or output_relative.parts[0] != "outputs":
            raise ValueError("Phase 7C output path must remain under outputs/")
        expected_mapping_digest = hashlib.sha256(_json_bytes(video["frame_mapping"])).hexdigest()
        if request.get("frame_mapping_digest") != expected_mapping_digest:
            raise ValueError("Phase 7C request frame mapping digest changed")
        if request.get("status") not in {
            "planned", "creating", "submitted", "polling", "succeeded", "reused",
            "failed", "blocked_cap", "unresolved_creation", "canceled",
        }:
            raise ValueError("Phase 7C request status is invalid")
        if not isinstance(request.get("attempts", []), list):
            raise ValueError("Phase 7C request attempts must be a list")
        try:
            create_attempt_count = int(request.get("create_attempt_count", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("Phase 7C request create attempt count is invalid") from exc
        if not 0 <= create_attempt_count <= 1 + MAX_RETRIES_PER_REQUEST:
            raise ValueError("Phase 7C request create attempt count exceeds its cap")
        prediction_id = request.get("provider_prediction_id")
        if request.get("status") in {"submitted", "polling", "succeeded", "reused", "canceled"} and not prediction_id:
            raise ValueError("Phase 7C provider prediction ID is required for resumed/terminal requests")


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["content_digest"] = _content_digest(manifest)
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=target.parent, delete=False) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(target)


def _download_output(output: Any, destination: Path) -> None:
    if isinstance(output, (list, tuple)):
        # gener8-sam3 documents a ZIP first and an optional preview MP4
        # second.  Select the ZIP explicitly so the validator receives the
        # label bundle even when preview output is enabled by a future run.
        candidates = list(output)
        for candidate in candidates:
            url = candidate if isinstance(candidate, str) else getattr(candidate, "url", None)
            if url and str(url).casefold().split("?", 1)[0].endswith(".zip"):
                _download_output(candidate, destination)
                return
        if candidates:
            _download_output(candidates[0], destination)
            return
        raise TypeError("Replicate returned an empty output list")
    if isinstance(output, (bytes, bytearray)):
        destination.write_bytes(bytes(output))
        return
    if hasattr(output, "read"):
        payload = output.read()
        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("Replicate FileOutput.read() did not return bytes")
        destination.write_bytes(bytes(payload))
        return
    # URL fallback is intentionally unauthenticated. Replicate output files
    # are served from signed delivery URLs; the API bearer must never cross
    # from the API client to a model-output host.
    url = output if isinstance(output, str) else getattr(output, "url", None)
    if url:
        request = urllib.request.Request(str(url))
        with urllib.request.urlopen(request, timeout=600) as response:
            destination.write_bytes(response.read())
        return
    try:
        destination.write_bytes(b"".join(bytes(chunk) for chunk in output))
    except Exception as exc:
        raise TypeError("Replicate returned an unsupported output type") from exc


def _provider_field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _prediction_id(prediction: Any) -> str | None:
    value = _provider_field(prediction, "id")
    return str(value) if value else None


def _prediction_status(prediction: Any) -> str:
    value = _provider_field(prediction, "status", "")
    return str(value or "").strip().casefold()


def _prediction_output(prediction: Any) -> Any:
    return _provider_field(prediction, "output")


def _prediction_snapshot(prediction: Any) -> dict[str, Any]:
    """Return provider telemetry without copying inputs, outputs, logs, or errors."""

    snapshot: dict[str, Any] = {}
    prediction_id = _prediction_id(prediction)
    if prediction_id:
        snapshot["id"] = prediction_id
    status = _prediction_status(prediction)
    if status:
        snapshot["status"] = status
    for key in ("model", "version", "created_at", "started_at", "completed_at"):
        value = _provider_field(prediction, key)
        if value is not None:
            snapshot[key] = str(value)
    metrics = _provider_field(prediction, "metrics")
    if isinstance(metrics, Mapping):
        safe_metrics: dict[str, float] = {}
        for key in ("predict_time", "total_time"):
            value = metrics.get(key)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                safe_metrics[key] = numeric
        if safe_metrics:
            snapshot["metrics"] = safe_metrics
    return snapshot


def _provider_status_code(exc: BaseException) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "status", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def _retry_after_seconds(exc: BaseException, *, now: float) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = getattr(exc, "retry_after", None)
    if value is None and isinstance(headers, Mapping):
        value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            target = parsedate_to_datetime(str(value)).timestamp()
            seconds = target - now
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _rate_limit_delay(exc: BaseException, retry_index: int, *, now: float) -> float:
    exponential = min(60.0, 2.0 ** max(0, retry_index))
    server_delay = _retry_after_seconds(exc, now=now)
    return max(exponential, server_delay or 0.0)


def _error_type(exc: BaseException) -> str:
    return type(exc).__name__


def execute_manifest(
    path: str | Path,
    root: str | Path,
    *,
    client: Any | None = None,
    sleep_fn: Any = time.sleep,
    clock_fn: Any = time.time,
) -> dict[str, Any]:
    """Run one request at a time using durable asynchronous predictions.

    A manifest is written before every create and immediately after its
    prediction ID is observed.  A ``creating`` request without an ID is
    therefore treated as unresolved after a crash and is never recreated.
    """

    manifest_path = Path(path)
    root_path = Path(root).resolve()
    manifest = _load_json(manifest_path)
    validate_manifest(manifest, root_path, rehash_sources=True)
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        manifest["status"] = "blocked_missing_credential"
        manifest["blocker"] = "REPLICATE_API_TOKEN is absent from the process environment; no API call was made"
        _write_manifest(manifest_path, manifest)
        return manifest
    manifest["credential_status"] = "present"
    if client is None:
        try:
            import replicate  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("the official Replicate Python package is required for --execute") from exc
        client = replicate.Client()

    video_by_id = {str(video["qa_id"]): video for video in manifest["videos"]}
    execution = manifest["execution"]
    max_creates = int(execution["max_prediction_create_attempts_total"])
    max_retries = int(execution["max_retry_prediction_creates"])
    poll_interval = float(execution["poll_interval_seconds"])
    max_poll_cycles = int(execution["max_poll_cycles_per_request"])
    existing_prediction_ids = sum(bool(request.get("provider_prediction_id")) for request in manifest["requests"])
    execution["prediction_ids_observed"] = max(
        int(execution.get("prediction_ids_observed", 0)), existing_prediction_ids
    )

    def persist() -> None:
        _write_manifest(manifest_path, manifest)

    execution["dry_run"] = False
    manifest["status"] = "running"
    execution["paid_calls_made"] = None
    execution["billing_status"] = "estimate_only_provider_billing_unavailable"
    execution["billing_uncertainty"] = bool(execution.get("billing_uncertainty", False))
    persist()

    def record_provider(request: dict[str, Any], prediction: Any) -> None:
        snapshot = _prediction_snapshot(prediction)
        if snapshot:
            request["provider_snapshot"] = snapshot
            if snapshot.get("id") and not request.get("provider_prediction_id"):
                request["provider_prediction_id"] = snapshot["id"]

    def mark_attempt(request: dict[str, Any], **fields: Any) -> dict[str, Any]:
        attempts = request.setdefault("attempts", [])
        attempt = {"attempt_index": len(attempts), **fields}
        attempts.append(attempt)
        return attempt

    def download_and_validate(request: dict[str, Any], source: Mapping[str, Any], prediction: Any, output_path: Path) -> bool:
        output = _prediction_output(prediction)
        if output is None:
            request["status"] = "failed"
            request["error_type"] = "MissingPredictionOutput"
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".part")
        try:
            _download_output(output, temporary)
            validated = validate_output_zip(
                str(request["model_key"]), temporary.read_bytes(), source,
                expected_prompt=str(request["prompt"]),
            )
            temporary.replace(output_path)
        except Exception as exc:
            temporary.unlink(missing_ok=True)
            request["status"] = "failed"
            request["error_type"] = _error_type(exc)
            return False
        request["status"] = "succeeded"
        if not request.get("provider_success_recorded"):
            execution["successful_predictions"] = int(execution.get("successful_predictions", 0)) + 1
            request["provider_success_recorded"] = True
        request["output_sha256"] = _sha256(output_path)
        request["output_validation"] = validated
        return True

    def poll_request(request: dict[str, Any], source: Mapping[str, Any], prediction: Any | None = None) -> str:
        poll_cycles = 0
        rate_limit_retries = 0
        while poll_cycles < max_poll_cycles:
            if prediction is None:
                try:
                    prediction = client.predictions.get(request["provider_prediction_id"])
                except Exception as exc:
                    if _provider_status_code(exc) == 429 and rate_limit_retries < max_retries - int(execution.get("retry_attempts", 0)):
                        delay = _rate_limit_delay(exc, rate_limit_retries, now=float(clock_fn()))
                        rate_limit_retries += 1
                        execution["retry_attempts"] = int(execution.get("retry_attempts", 0)) + 1
                        request["last_poll_rate_limit"] = {"error_type": _error_type(exc), "delay_s": delay}
                        persist()
                        sleep_fn(delay)
                        continue
                    request["status"] = "polling"
                    request["last_poll_error_type"] = _error_type(exc)
                    persist()
                    return "paused"
            record_provider(request, prediction)
            status = _prediction_status(prediction)
            if status in {"starting", "processing", "queued"}:
                request["status"] = "polling"
                persist()
                prediction = None
                poll_cycles += 1
                sleep_fn(poll_interval)
                continue
            if status == "succeeded":
                if download_and_validate(request, source, prediction, output_path):
                    request["attempts"][-1]["status"] = "succeeded"
                    persist()
                    return "succeeded"
                request["attempts"][-1]["status"] = "failed"
                persist()
                return "failed"
            if status == "failed":
                request["status"] = "failed"
                request["error_type"] = "ProviderFailed"
                request["attempts"][-1]["status"] = "failed"
                persist()
                return "failed"
            if status == "canceled":
                request["status"] = "canceled"
                request["attempts"][-1]["status"] = "canceled"
                persist()
                return "canceled"
            request["status"] = "polling"
            request["last_poll_status"] = status or "unknown"
            persist()
            return "paused"
        request["status"] = "polling"
        request["poll_cycle_limit_reached"] = True
        persist()
        return "paused"

    for request in manifest["requests"]:
        output_path = root_path / RUN_RELATIVE / str(request["output_relpath"])
        source = video_by_id[str(request["qa_id"])]
        status = str(request.get("status"))
        if status in {"succeeded", "reused"} and output_path.is_file():
            try:
                validated = validate_output_zip(
                    str(request["model_key"]), output_path.read_bytes(), source,
                    expected_prompt=str(request["prompt"]),
                )
            except ValueError:
                # A terminal provider result with a missing/invalid local file
                # is resumed by polling its durable provider ID, never by create.
                if request.get("provider_prediction_id"):
                    request["status"] = "polling"
                else:
                    request["status"] = "planned"
            else:
                request["status"] = "reused"
                request["output_validation"] = validated
                persist()
                continue
        elif status in {"succeeded", "reused"} and request.get("provider_prediction_id"):
            request["status"] = "polling"
        if request.get("status") == "unresolved_creation":
            manifest["status"] = "blocked_unresolved_creation"
            manifest["blocker"] = "a prior create returned without a durable prediction ID; manual provider reconciliation is required"
            persist()
            return manifest
        if request.get("status") == "creating" and not request.get("provider_prediction_id"):
            request["status"] = "unresolved_creation"
            request["blocker"] = "create was marked in progress without a persisted provider prediction ID"
            manifest["status"] = "blocked_unresolved_creation"
            persist()
            return manifest
        if request.get("status") in {"planned", "creating", "submitted", "polling", "succeeded", "reused", "canceled"} and request.get("provider_prediction_id"):
            if not request.get("attempts"):
                mark_attempt(request, status="polling", kind="resume_poll", retry=False)
                persist()
            outcome = poll_request(request, source)
            if outcome == "paused":
                manifest["status"] = "partial"
                persist()
                return manifest
            continue
        if request.get("status") in {"failed", "blocked_cap", "canceled"}:
            continue
        if int(execution.get("prediction_create_attempts", 0)) >= max_creates:
            request["status"] = "blocked_cap"
            persist()
            continue

        # Create a prediction only from a planned request.  The creating marker
        # is durable before the network call so a crash cannot trigger a blind
        # duplicate submission on the next invocation.  Explicit 429 responses
        # are the sole automatic create retry; all other no-ID outcomes stop.
        while True:
            create_attempt = int(request.get("create_attempt_count", 0))
            is_retry = create_attempt > 0
            if is_retry and int(execution.get("retry_attempts", 0)) >= max_retries:
                request["status"] = "blocked_cap"
                persist()
                break
            request["status"] = "creating"
            request["create_attempt_count"] = create_attempt + 1
            execution["prediction_create_attempts"] = int(execution.get("prediction_create_attempts", 0)) + 1
            if is_retry:
                execution["retry_attempts"] = int(execution.get("retry_attempts", 0)) + 1
            attempt = mark_attempt(
                request,
                status="started",
                retry=is_retry,
                kind="prediction_create",
            )
            persist()
            model = MODEL_SPECS[str(request["model_key"])]
            try:
                with (root_path / str(source["source_path"])).open("rb") as stream:
                    prediction = client.predictions.create(
                        version=model["version"],
                        input={**request["input"], "video": stream},
                    )
            except Exception as exc:
                if _provider_status_code(exc) == 429 and not is_retry and int(execution.get("retry_attempts", 0)) < max_retries:
                    delay = _rate_limit_delay(exc, 0, now=float(clock_fn()))
                    attempt.update({"status": "rate_limited", "error_type": _error_type(exc), "delay_s": delay})
                    request["status"] = "planned"
                    persist()
                    sleep_fn(delay)
                    continue
                attempt.update({"status": "unresolved", "error_type": _error_type(exc)})
                request["status"] = "unresolved_creation"
                request["blocker"] = "provider create failed without a prediction ID; no automatic duplicate retry"
                request["potential_billing_uncertainty"] = True
                execution["billing_uncertainty"] = True
                manifest["status"] = "blocked_unresolved_creation"
                persist()
                return manifest
            prediction_id = _prediction_id(prediction)
            had_provider_id = bool(request.get("provider_prediction_id"))
            record_provider(request, prediction)
            if not prediction_id:
                attempt.update({"status": "missing_id", "uncertainty": "provider_response_without_prediction_id"})
                request["status"] = "unresolved_creation"
                request["blocker"] = "provider create returned no prediction ID; no automatic duplicate retry"
                request["potential_billing_uncertainty"] = True
                execution["billing_uncertainty"] = True
                manifest["status"] = "blocked_unresolved_creation"
                persist()
                return manifest
            if not had_provider_id:
                execution["prediction_ids_created"] = int(execution.get("prediction_ids_created", 0)) + 1
                execution["prediction_ids_observed"] = int(execution.get("prediction_ids_observed", 0)) + 1
                execution["billable_cost_estimate_usd"] = round(
                    float(execution.get("billable_cost_estimate_usd", 0.0))
                    + MODEL_PLANNING_COST_USD[str(request["model_key"])],
                    4,
                )
            request["provider_prediction_id"] = prediction_id
            request["status"] = "submitted"
            attempt["status"] = "submitted"
            persist()
            outcome = poll_request(request, source, prediction)
            if outcome == "paused":
                manifest["status"] = "partial"
                persist()
                return manifest
            break

    manifest["status"] = "complete" if all(item["status"] in {"succeeded", "reused"} for item in manifest["requests"]) else "partial"
    persist()
    return manifest


def write_dry_run(root: str | Path, output: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    manifest = build_run_manifest(root_path)
    requirements_path = root_path / REQUIREMENTS_RELATIVE
    if not requirements_path.is_file() or f"replicate=={REPLICATE_SDK_VERSION}" not in requirements_path.read_text():
        raise ValueError(f"Phase 7C isolated runtime dependency pin is missing: {requirements_path}")
    _write_manifest(output_path, manifest)
    source_entries = []
    for video in manifest["videos"]:
        stem = str(video["video_path"]).replace("/", "__") + ".jpg"
        contact_sheet = root_path / "runs/phase7B_cache_validation/contact_sheets/source_only" / stem
        if not contact_sheet.is_file():
            raise ValueError(f"reviewed source-only contact sheet is missing: {contact_sheet}")
        source_entries.append(
            {
                "qa_id": video["qa_id"],
                "video_path": video["video_path"],
                "source_path": video["source_path"],
                "source_sha256": video["source_sha256"],
                "source_frame_count": video["n_frames"],
                "timestamp_source": video["timestamp_source"],
                "contact_sheet_path": str(contact_sheet.resolve().relative_to(root_path)),
                "contact_sheet_sha256": _sha256(contact_sheet),
                "contact_sheet_provenance": "reused Phase 7B source-only review artifact",
            }
        )
    source_manifest: dict[str, Any] = {
        "schema_version": 1,
        "manifest_kind": "reva-phase7c-source-artifacts",
        "dataset_revision": manifest["dataset_revision"],
        "entries": source_entries,
    }
    source_manifest["content_digest"] = _content_digest(source_manifest)
    source_manifest_path = output_path.with_name("source_manifest.json")
    _write_manifest(source_manifest_path, source_manifest)
    report = {
        "schema_version": 1,
        "kind": "reva-phase7c-replicate-smoke-dry-run",
        "manifest": str(output_path.relative_to(root_path)),
        "source_manifest": str(source_manifest_path.relative_to(root_path)),
        "status": "blocked_missing_credential" if manifest["credential_status"] == "absent" else "ready_for_explicit_execute",
        "credential_status": manifest["credential_status"],
        "paid_calls_made": 0,
        "billing_status": "no_calls",
        "billable_cost_estimate_usd": 0.0,
        "planned_requests": len(manifest["requests"]),
        "model_request_counts": {
            key: sum(item["model_key"] == key for item in manifest["requests"])
            for key in MODEL_SPECS
        },
        "resumable": True,
        "scheduler": manifest["execution"]["scheduler"],
        "runtime_dependency": manifest["execution"]["runtime_dependency"],
        "max_primary_prediction_creates": int(manifest["execution"]["max_primary_prediction_creates"]),
        "max_retry_prediction_creates": int(manifest["execution"]["max_retry_prediction_creates"]),
        "max_prediction_create_attempts_total": int(manifest["execution"]["max_prediction_create_attempts_total"]),
        "request_count_cap_verified": len(manifest["requests"]) == MAX_REQUESTS,
        "output_validation_contract": {
            "merged_mask_reference": "explicit full contiguous named frames; timestamps unverified without endpoint metadata",
            "id_preserving_candidate": "objects.json plus 16-bit labels for every frame with matching source count/fps",
        },
        "planning_estimate_usd": round(
            sum(MODEL_PLANNING_COST_USD[item["model_key"]] for item in manifest["requests"]), 4
        ),
        "planning_estimate_basis": "16 primary requests at official pinned model page estimates; retries excluded",
        "blocker": (
            "REPLICATE_API_TOKEN is absent from the process environment; no paid call was attempted"
            if manifest["credential_status"] == "absent"
            else None
        ),
    }
    report["content_digest"] = _content_digest(report)
    report_path = output_path.with_name("dry_run_report.json")
    _write_manifest(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.dry_run == args.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    output = args.manifest or (args.root / RUN_RELATIVE / "manifest.json")
    if args.dry_run:
        report = write_dry_run(args.root, output)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    result = execute_manifest(output, args.root)
    print(json.dumps({"status": result["status"], "credential_status": result["credential_status"], "paid_calls_made": result["execution"]["paid_calls_made"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
