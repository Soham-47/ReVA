"""Build the offline, annotation-independent Phase 3B Gate 1 inputs.

This script reads only the saved Phase 3A frames/mask archives and the Gate 0
component CSV.  It does not import Modal, Transformers, Replicate, or any
model code.  The generated directory is self-contained apart from the source
video paths recorded for provenance.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import os
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PHASE3B_GATE1_OUT", str(ROOT / "runs/phase3b_gate1"))).resolve()
LEGACY_RUN = ROOT / "runs/phase3a_retrieval_smoke"
RECOVERY_RUN = ROOT / "runs/phase3b_proposal_recovery"
WIDTH, HEIGHT = 640, 360
FRAME_COUNT = 16
PAD_FRACTION = 0.10
CONTEXT_PADDING = (0.20, 0.40)
SIGLIP_REVISION = "022b6f71160ffb0169ca4709e2d7e25be659598a"
QUERY_PHRASES = ("black car", "parking lot", "baseball field")

MASK_SPECS = {
    "black_car_qcond": {
        "qa_id": "train_010397",
        "call": 10,
        "arm": "question-conditioned",
        "prompt": "black car",
    },
    "vehicle_fixed": {
        "qa_id": "train_010397",
        "call": 0,
        "arm": "fixed-vocabulary",
        "prompt": "vehicle",
    },
    "parking_lot_qcond": {
        "qa_id": "train_010397",
        "call": 11,
        "arm": "question-conditioned",
        "prompt": "parking lot",
    },
    "baseball_field_qcond": {
        "qa_id": "train_009869",
        "call": 15,
        "arm": "question-conditioned",
        "prompt": "baseball field",
    },
    "field_fixed": {
        "qa_id": "train_009869",
        "call": 12,
        "arm": "fixed-vocabulary",
        "prompt": "field",
    },
}

QUERY_BY_QA = {
    "train_010397": ["black car", "parking lot"],
    "train_009869": ["baseball field"],
}

PROTECTED_ARTIFACTS = (
    "plan.json",
    "manifest.json",
    "candidates.jsonl",
    "candidate_masks.zip",
    "annotations/manual_target_annotations.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _box(values: Iterable[int | float]) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4:
        raise ValueError(f"expected xyxy box, got {result!r}")
    return result  # type: ignore[return-value]


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _expand_box(box: tuple[int, int, int, int], fraction: float) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    dx = max(1, int(round((right - left) * fraction)))
    dy = max(1, int(round((bottom - top) * fraction)))
    return max(0, left - dx), max(0, top - dy), min(WIDTH, right + dx), min(HEIGHT, bottom + dy)


def _mask_png(mask: np.ndarray) -> bytes:
    image = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def _load_archive_masks(path: Path) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            match = re.fullmatch(r"mask_(\d{5})\.png", Path(member).name)
            if not match:
                continue
            packed = int(match.group(1))
            with Image.open(io.BytesIO(archive.read(member))) as image:
                mask = np.asarray(image.convert("L")) > 0
            if mask.shape != (HEIGHT, WIDTH):
                raise AssertionError(f"unexpected mask shape in {path}: {mask.shape}")
            masks[packed] = mask
    return masks


def _frame_info(source_manifest: dict[str, Any], qa_id: str, packed: int) -> dict[str, Any]:
    policy = source_manifest["source_policies"][qa_id]
    indices = policy["source_frame_indices"]
    timestamps = policy["source_timestamps"]
    if len(indices) != FRAME_COUNT or len(timestamps) != FRAME_COUNT:
        raise AssertionError(f"{qa_id} does not have the fixed 16-frame policy")
    return {
        "source_frame_index": int(indices[packed]),
        "timestamp_s": float(timestamps[packed]),
        "source_video": str(policy["video_path"]),
        "source_video_checksum": str(policy["video_checksum"]),
    }


def _filter_reason(mask: np.ndarray, box: tuple[int, int, int, int]) -> str | None:
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    area = int(mask.sum())
    if area == 0:
        return "empty_mask"
    if not (0 <= left < right <= WIDTH and 0 <= top < bottom <= HEIGHT):
        return "invalid_box"
    if width < 2 or height < 2:
        return "degenerate_box"
    if area < 16:
        return "area_lt_16_pixels"
    if area > int(WIDTH * HEIGHT * 0.95):
        return "near_full_frame_mask"
    return None


def _copy_source_files(source_manifest: dict[str, Any]) -> dict[str, Any]:
    """Copy only the two selected frame sets and five consumed mask archives."""

    inputs = OUT / "inputs"
    (inputs / "masks").mkdir(parents=True, exist_ok=True)
    (OUT / "source_frames").mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for source, destination, kind in (
        (LEGACY_RUN / "manifest.json", inputs / "phase3a_manifest.json", "phase3a_manifest"),
        (RECOVERY_RUN / "components.csv", inputs / "gate0_components.csv", "gate0_components"),
        (RECOVERY_RUN / "frames.csv", inputs / "gate0_frames.csv", "gate0_frames"),
        (RECOVERY_RUN / "summary.json", inputs / "gate0_summary.json", "gate0_summary"),
    ):
        shutil.copy2(source, destination)
        files.append(
            {
                "kind": kind,
                "path": str(destination.relative_to(OUT)),
                "source_path": str(source.relative_to(ROOT)),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )

    for qa_id in sorted(QUERY_BY_QA):
        for packed in range(FRAME_COUNT):
            source = LEGACY_RUN / "review/manual_audit/source_frames" / qa_id / f"frame-{packed + 1:02d}.png"
            destination = OUT / "source_frames" / qa_id / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            files.append(
                {
                    "kind": "source_frame",
                    "qa_id": qa_id,
                    "packed_frame_index": packed,
                    "path": str(destination.relative_to(OUT)),
                    "source_path": str(source.relative_to(ROOT)),
                    "sha256": _sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )

    for mask_name, spec in MASK_SPECS.items():
        qa_id = str(spec["qa_id"])
        source = LEGACY_RUN / "replicate" / qa_id / f"call-{int(spec['call']):02d}.masks.zip"
        destination = inputs / "masks" / qa_id / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        files.append(
            {
                "kind": "source_mask_archive",
                "mask_name": mask_name,
                "qa_id": qa_id,
                "call": int(spec["call"]),
                "path": str(destination.relative_to(OUT)),
                "source_path": str(source.relative_to(ROOT)),
                "sha256": _sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    return {"files": files, "source_policies": source_manifest["source_policies"]}


def _candidate(
    *,
    source_manifest: dict[str, Any],
    candidate_id: str,
    qa_id: str,
    arm: str,
    prompt: str,
    candidate_type: str,
    mask_kind: str,
    mask_name: str,
    packed: int,
    mask: np.ndarray,
    proposal_box: tuple[int, int, int, int],
    source_member: str,
    archive_path: Path,
    context_padding_fraction: float | None = None,
    context_basis: str | None = None,
    source_component_id: int | None = None,
) -> tuple[dict[str, Any], bytes]:
    frame = _frame_info(source_manifest, qa_id, packed)
    crop_box = _expand_box(proposal_box, PAD_FRACTION)
    spec = MASK_SPECS[mask_name]
    public = {
        "candidate_id": candidate_id,
        "qa_id": qa_id,
        "arm": arm,
        "proposal_prompt": prompt,
        "candidate_type": candidate_type,
        "mask_kind": mask_kind,
        "mask_encoding": "binary_png_0_255",
        "packed_frame_index": packed,
        **frame,
        "source_frame": f"source_frames/{qa_id}/frame-{packed + 1:02d}.png",
        "source_mask_name": mask_name,
        "source_mask_archive": str(archive_path.relative_to(OUT)),
        "source_mask_archive_original": str(
            (LEGACY_RUN / "replicate" / qa_id / archive_path.name).relative_to(ROOT)
        ),
        "source_mask_member": source_member,
        "proposal_box_pixels": {
            "left": proposal_box[0],
            "top": proposal_box[1],
            "right": proposal_box[2],
            "bottom": proposal_box[3],
        },
        "crop_box_pixels": {
            "left": crop_box[0],
            "top": crop_box[1],
            "right": crop_box[2],
            "bottom": crop_box[3],
        },
        "mask_area_pixels": int(mask.sum()),
        "mask_area_fraction": round(float(mask.mean()), 8),
        "query_phrases": QUERY_BY_QA[qa_id],
        "native_score_status": "unavailable: saved archives contain binary masks without attributable per-mask scores",
        "manual_annotation_status": "separate_pending_blind_annotation",
    }
    if context_padding_fraction is not None:
        public["context_padding_fraction"] = context_padding_fraction
        public["context_basis"] = context_basis or "original_unsplit_proposal_bbox"
    if source_component_id is not None:
        public["source_component_id_frame_local"] = source_component_id
    public["mask_member"] = f"candidate_masks/{candidate_id}.png"
    payload = _mask_png(mask)
    public["mask_sha256"] = hashlib.sha256(payload).hexdigest()
    return public, payload


def _load_component_rows() -> list[dict[str, Any]]:
    with (RECOVERY_RUN / "components.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = set(MASK_SPECS)
    if {str(row["mask"]) for row in rows} != expected:
        raise AssertionError("Gate 0 component CSV mask names changed")
    return rows


def _build_candidates(source_manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, bytes], Counter[str], Counter[str]]:
    staged_archives: dict[str, Path] = {}
    archive_masks: dict[str, dict[int, np.ndarray]] = {}
    for mask_name, spec in MASK_SPECS.items():
        archive = OUT / "inputs" / "masks" / str(spec["qa_id"]) / f"call-{int(spec['call']):02d}.masks.zip"
        staged_archives[mask_name] = archive
        archive_masks[mask_name] = _load_archive_masks(archive)

    candidates: list[dict[str, Any]] = []
    mask_payloads: dict[str, bytes] = {}
    filtered: Counter[str] = Counter()
    raw_by_type: Counter[str] = Counter()
    seen_ids: set[str] = set()

    def add_candidate(
        *,
        mask_name: str,
        packed: int,
        mask: np.ndarray,
        candidate_type: str,
        mask_kind: str,
        proposal_box: tuple[int, int, int, int],
        suffix: str,
        context_padding_fraction: float | None = None,
        context_basis: str | None = None,
        source_component_id: int | None = None,
    ) -> None:
        spec = MASK_SPECS[mask_name]
        arm, prompt, qa_id = str(spec["arm"]), str(spec["prompt"]), str(spec["qa_id"])
        raw_by_type[candidate_type] += 1
        reason = _filter_reason(mask, proposal_box)
        if reason:
            filtered[reason] += 1
            return
        candidate_id = f"{candidate_type[:3]}-{arm[:3]}-{qa_id}-{mask_name}-{packed:02d}-{suffix}"
        if candidate_id in seen_ids:
            raise AssertionError(f"duplicate candidate id: {candidate_id}")
        seen_ids.add(candidate_id)
        source_member = f"mask_{packed:05d}.png"
        record, payload = _candidate(
            source_manifest=source_manifest,
            candidate_id=candidate_id,
            qa_id=qa_id,
            arm=arm,
            prompt=prompt,
            candidate_type=candidate_type,
            mask_kind=mask_kind,
            mask_name=mask_name,
            packed=packed,
            mask=mask,
            proposal_box=proposal_box,
            source_member=source_member,
            archive_path=staged_archives[mask_name],
            context_padding_fraction=context_padding_fraction,
            context_basis=context_basis,
            source_component_id=source_component_id,
        )
        candidates.append(record)
        mask_payloads[record["mask_member"]] = payload

    # Component rows are a Gate 0 artifact. Recompute labels only to materialize
    # the exact component pixels; the source labels/boxes remain authoritative.
    for row in _load_component_rows():
        mask_name = str(row["mask"])
        packed = int(row["packed_frame"])
        component_id = int(row["component_id_frame_local"])
        union = archive_masks[mask_name].get(packed)
        if union is None:
            raise AssertionError(f"component row points to missing mask frame: {row}")
        labels, count = ndimage.label(union, structure=np.ones((3, 3), dtype=np.uint8))
        if not 1 <= component_id <= int(count):
            raise AssertionError(f"component id out of range: {row}")
        component = labels == component_id
        recorded_box = _box(ast.literal_eval(str(row["bbox_xyxy"])))
        if _bbox(component) != recorded_box or int(component.sum()) != int(row["area_px"]):
            raise AssertionError(f"Gate 0 component mismatch: {row}")
        expected_timestamp = _frame_info(source_manifest, str(MASK_SPECS[mask_name]["qa_id"]), packed)["timestamp_s"]
        if abs(float(row["timestamp_s"]) - float(expected_timestamp)) > 1e-6:
            raise AssertionError(f"Gate 0 timestamp mismatch: {row}")
        add_candidate(
            mask_name=mask_name,
            packed=packed,
            mask=component,
            candidate_type="connected_component",
            mask_kind="8_connected_component",
            proposal_box=recorded_box,
            suffix=f"component-{component_id:04d}",
        )

    # Keep each union proposal as an explicit candidate alongside its split components.
    for mask_name, spec in MASK_SPECS.items():
        for packed, union in sorted(archive_masks[mask_name].items()):
            proposal_box = _bbox(union)
            if proposal_box is None:
                continue
            add_candidate(
                mask_name=mask_name,
                packed=packed,
                mask=union,
                candidate_type="original_unsplit",
                mask_kind="saved_union_mask",
                proposal_box=proposal_box,
                suffix="union",
            )

    # Context candidates use only a deterministic padding of the saved
    # parking-lot/baseball-field union boxes. Include both the question
    # conditioned field proposals and the fixed-vocabulary field proposals;
    # no target coordinates are used.
    for mask_name in ("parking_lot_qcond", "baseball_field_qcond", "field_fixed"):
        for packed, union in sorted(archive_masks[mask_name].items()):
            proposal_box = _bbox(union)
            if proposal_box is None:
                continue
            previous: tuple[int, int, int, int] | None = None
            for fraction in CONTEXT_PADDING:
                context_box = _expand_box(proposal_box, fraction)
                if context_box == previous:
                    continue
                previous = context_box
                context = np.zeros((HEIGHT, WIDTH), dtype=bool)
                left, top, right, bottom = context_box
                context[top:bottom, left:right] = True
                add_candidate(
                    mask_name=mask_name,
                    packed=packed,
                    mask=context,
                    candidate_type="context_region",
                    mask_kind="deterministic_rectangle",
                    proposal_box=context_box,
                    suffix=f"pad-{int(round(fraction * 100)):02d}",
                    context_padding_fraction=fraction,
                )

    # The fixed-vocabulary field union reaches the frame edges, so its
    # expanded rectangle is rejected by the near-full-frame image filter.
    # Preserve a deterministic fixed-field context arm by falling back to
    # each Gate 0 component box when a frame produced no union context.
    for packed, union in sorted(archive_masks["field_fixed"].items()):
        frame_contexts = [
            item
            for item in candidates
            if item["candidate_type"] == "context_region"
            and item["source_mask_name"] == "field_fixed"
            and int(item["packed_frame_index"]) == packed
        ]
        if frame_contexts:
            continue
        labels, count = ndimage.label(union, structure=np.ones((3, 3), dtype=np.uint8))
        component_rows = [
            row
            for row in _load_component_rows()
            if row["mask"] == "field_fixed" and int(row["packed_frame"]) == packed
        ]
        if len(component_rows) != int(count):
            raise AssertionError("field_fixed component count changed while building contexts")
        for row in component_rows:
            component_id = int(row["component_id_frame_local"])
            component_box = _box(ast.literal_eval(str(row["bbox_xyxy"])))
            component = labels == component_id
            if _bbox(component) != component_box:
                raise AssertionError("field_fixed component box changed while building contexts")
            previous: tuple[int, int, int, int] | None = None
            for fraction in CONTEXT_PADDING:
                context_box = _expand_box(component_box, fraction)
                if context_box == previous:
                    continue
                previous = context_box
                context = np.zeros((HEIGHT, WIDTH), dtype=bool)
                left, top, right, bottom = context_box
                context[top:bottom, left:right] = True
                add_candidate(
                    mask_name="field_fixed",
                    packed=packed,
                    mask=context,
                    candidate_type="context_region",
                    mask_kind="deterministic_rectangle",
                    proposal_box=context_box,
                    suffix=f"component-{component_id:04d}-pad-{int(round(fraction * 100)):02d}",
                    context_padding_fraction=fraction,
                    context_basis="field_fixed_component_bbox",
                    source_component_id=component_id,
                )

    if not candidates:
        raise AssertionError("no candidates survived deterministic filters")
    required_types = {"connected_component", "original_unsplit", "context_region"}
    if {str(item["candidate_type"]) for item in candidates} != required_types:
        raise AssertionError("all Gate 1 candidate types must be represented")
    if {str(item["arm"]) for item in candidates} != {"question-conditioned", "fixed-vocabulary"}:
        raise AssertionError("both candidate arms must be represented")
    return candidates, mask_payloads, raw_by_type, filtered


def _write_mask_archive(path: Path, payloads: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for member in sorted(payloads):
            info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            archive.writestr(info, payloads[member])


def _write_candidates(path: Path, candidates: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        for candidate in candidates:
            stream.write(json.dumps(candidate, sort_keys=True, separators=(",", ":")) + "\n")


def _plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "prepared",
        "gate": "phase3b_gate1",
        "scope": "offline candidate staging plus one deferred Modal SigLIP2 ranking job",
        "models_called": 0,
        "paid_inference": "not_run",
        "human_target_annotations_used": False,
        "questions": {
            "train_010397": {
                "question": "At what time does the black car appear in the lower right section of the parking lot and when does it disappear?",
                "query_phrases": ["black car", "parking lot"],
            },
            "train_009869": {
                "question": "From which timestamp to which timestamp is the baseball field continuously visible in the video?",
                "query_phrases": ["baseball field"],
            },
        },
        "candidate_arms": ["question-conditioned", "fixed-vocabulary"],
        "candidate_types": ["connected_component", "original_unsplit", "context_region"],
        "context_policy": {
            "source": "saved parking_lot/baseball_field union mask bounding boxes from qcond and fixed field proposals",
            "padding_fractions": list(CONTEXT_PADDING),
            "manual_coordinates": False,
        },
        "filters": {
            "connectivity": 8,
            "minimum_component_area_pixels": 16,
            "minimum_box_width_pixels": 2,
            "minimum_box_height_pixels": 2,
            "maximum_mask_area_fraction": 0.95,
            "inputs": "image-only mask area/box checks; no target labels or annotations",
        },
        "crop_styles": ["padded_rgb_box", "masked_rgb"],
        "ranking": {
            "model": "google/siglip2-base-patch16-224",
            "revision": SIGLIP_REVISION,
            "download": "exact public commit inside the deferred Modal function; no local snapshot is required",
            "verification": "code/build_feature_cache.py:validate_local_model_dir",
            "query_text_source": "exact question-derived phrases only",
            "query_phrases": list(QUERY_PHRASES),
            "similarity": "cosine_on_l2_normalized_features",
            "random_draws": 1000,
            "ranking_scopes": ["video_global", "per_frame"],
            "random_pool": "identical (qa_id, arm) or (qa_id, arm, packed_frame_index) candidate pool for each query",
        },
        "modal_budget": {
            "gpu": "A100-40GB",
            "gpu_memory_gib": 40,
            "cpu": 4,
            "memory_gib": 16,
            "max_gpu_minutes": 10,
            "max_wall_minutes": 10,
        },
        "outputs": {
            "prepared_manifest": "manifest.json",
            "execution_manifest": "execution_manifest.json",
            "modal_report": "modal_report.json",
            "rankings": "rankings.json",
            "random_baseline": "random_baseline.json",
            "runtime": "runtime.json",
            "vram": "vram.json",
            "storage": "storage.json",
        },
        "manual_evaluation": {
            "status": "pending_blind_annotation",
            "annotation_file": "annotations/manual_target_annotations.json",
            "category_metrics": "N/A until manual labels are frozen",
            "correct_instance_metrics": "uncertain/N/A because target identity is unresolved",
        },
    }


def _manual_status(source_manifest: dict[str, Any]) -> dict[str, Any]:
    black_car_frames = []
    for packed in range(FRAME_COUNT):
        frame = _frame_info(source_manifest, "train_010397", packed)
        black_car_frames.append(
            {
                "packed_frame_index": packed,
                "source_frame_index": frame["source_frame_index"],
                "timestamp_s": frame["timestamp_s"],
                "parked_lower_right_vehicle_visible": True,
                "visible_category": "likely_car",
                "question_target_identity": "unresolved",
                "occlusion_ambiguity_note": "Persistent parked lower-right vehicle co-occurs with other moving cars; the requested identity cannot be uniquely verified from this 16-frame sample.",
            }
        )
    return {
        "schema_version": 1,
        "scope": "evaluation-only status; this file is never an inference input",
        "status": "unresolved",
        "used_for_candidates": False,
        "reason": "Visual audit found a persistent parked lower-right car and other moving cars; the requested target cannot be uniquely verified from the 16 source frames.",
        "queries": {
            "train_010397|black car": {
                "target_identity": "unresolved",
                "correct_instance_metrics": {"status": "uncertain", "value": None},
                "frames": black_car_frames,
            },
            "train_010397|parking lot": {
                "target_identity": "region identity pending annotation",
                "correct_instance_metrics": {"status": "N/A", "value": None},
            },
            "train_009869|baseball field": {
                "target_identity": "region boundary pending annotation",
                "correct_instance_metrics": {"status": "N/A", "value": None},
            },
        },
    }


def build() -> dict[str, Any]:
    existing = [name for name in PROTECTED_ARTIFACTS if (OUT / name).exists()]
    if existing:
        raise RuntimeError(
            "refusing to overwrite existing Gate 1 build artifacts: "
            + ", ".join(existing)
            + "; use a new run directory for a fresh build"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((LEGACY_RUN / "manifest.json").read_text())
    if source_manifest.get("status") != "complete" or source_manifest.get("qwen_scoring") != "not-run":
        raise RuntimeError("Phase 3A source manifest is not an eligible inference-free input")
    staged = _copy_source_files(source_manifest)
    plan = _plan()
    _write_json(OUT / "plan.json", plan)
    _write_json(OUT / "annotations" / "manual_target_annotations.json", _manual_status(source_manifest))
    candidates, payloads, raw_by_type, filtered = _build_candidates(source_manifest)
    _write_candidates(OUT / "candidates.jsonl", candidates)
    _write_mask_archive(OUT / "candidate_masks.zip", payloads)

    counts_by_type = Counter(str(item["candidate_type"]) for item in candidates)
    counts_by_arm = Counter(str(item["arm"]) for item in candidates)
    pools = Counter((str(item["qa_id"]), str(item["arm"])) for item in candidates)
    same_frame_mask_counts = Counter(
        (
            str(item["qa_id"]),
            int(item["packed_frame_index"]),
            str(item["mask_sha256"]),
        )
        for item in candidates
    )
    duplicate_groups = [count for count in same_frame_mask_counts.values() if count > 1]
    artifact_paths = [
        OUT / "plan.json",
        OUT / "annotations" / "manual_target_annotations.json",
        OUT / "candidates.jsonl",
        OUT / "candidate_masks.zip",
    ]
    artifact_files = [
        {"path": str(path.relative_to(OUT)), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(artifact_paths)
    ]
    manifest = {
        "schema_version": 1,
        "status": "prepared",
        "execution_status": "not_run",
        "scope": "Phase 3B Gate 1 offline candidate staging",
        "models_called": 0,
        "paid_inference": "not_run",
        "source": {
            "phase3a_manifest": "inputs/phase3a_manifest.json",
            "gate0_components": "inputs/gate0_components.csv",
            "gate0_summary": "inputs/gate0_summary.json",
            "selected_videos": sorted(QUERY_BY_QA),
            "dimensions_px": [WIDTH, HEIGHT],
            "frame_count_per_video": FRAME_COUNT,
            "mask_connectivity": 8,
            "staged_files": staged["files"],
        },
        "candidates": {
            "metadata": "candidates.jsonl",
            "mask_archive": "candidate_masks.zip",
            "raw_gate0_component_rows": sum(raw_by_type.values()) - sum(
                raw_by_type[key] for key in ("original_unsplit", "context_region")
            ),
            "raw_by_type": dict(sorted(raw_by_type.items())),
            "retained_total": len(candidates),
            "retained_by_type": dict(sorted(counts_by_type.items())),
            "retained_by_arm": dict(sorted(counts_by_arm.items())),
            "retained_by_qa_arm": {
                f"{qa_id}|{arm}": count for (qa_id, arm), count in sorted(pools.items())
            },
            "deduplication": "none; same-frame exact-mask duplicates remain separate candidates and are identified by qa_id, packed_frame_index, candidate_id, and mask_sha256",
            "same_frame_mask_key_unique_count": len(same_frame_mask_counts),
            "same_frame_exact_mask_duplicate_group_count": len(duplicate_groups),
            "same_frame_exact_mask_duplicate_candidate_count": sum(duplicate_groups),
            "same_frame_exact_mask_duplicate_extra_count": sum(count - 1 for count in duplicate_groups),
            "filtered_by_reason": dict(sorted(filtered.items())),
            "filter_policy": plan["filters"],
        },
        "queries": {
            "exact_question_phrases": list(QUERY_PHRASES),
            "by_qa": QUERY_BY_QA,
            "answer_choices_or_gold_used": False,
        },
        "model": {
            "name": "google/siglip2-base-patch16-224",
            "revision": SIGLIP_REVISION,
            "download_inside_modal_job": True,
            "verification": "code/build_feature_cache.py:validate_local_model_dir",
            "load_local_files_only_after_verification": True,
        },
        "modal_budget": plan["modal_budget"],
        "manual_evaluation": plan["manual_evaluation"],
        "artifact_files": artifact_files,
        "assertions": {
            "all_candidate_masks_binary": True,
            "all_boxes_within_source_frame": True,
            "all_source_frame_timestamps_from_phase3a_manifest": True,
            "human_annotations_used_for_candidates": False,
        },
    }
    _write_json(OUT / "manifest.json", manifest)

    # Self-checks are intentionally assert-based so a changed source schema
    # cannot silently produce a ranking pool with missing provenance.
    assert len(candidates) == manifest["candidates"]["retained_total"]
    assert set(counts_by_type) == {"connected_component", "original_unsplit", "context_region"}
    assert set(counts_by_arm) == {"question-conditioned", "fixed-vocabulary"}
    assert all(item["query_phrases"] for item in candidates)
    assert all(len(item["mask_sha256"]) == 64 for item in candidates)
    assert all(0 <= item["proposal_box_pixels"]["left"] < item["proposal_box_pixels"]["right"] <= WIDTH for item in candidates)
    assert all(0 <= item["proposal_box_pixels"]["top"] < item["proposal_box_pixels"]["bottom"] <= HEIGHT for item in candidates)
    assert len(payloads) == len(candidates)
    return manifest


if __name__ == "__main__":
    result = build()
    print(json.dumps({
        "status": result["status"],
        "candidate_count": result["candidates"]["retained_total"],
        "retained_by_type": result["candidates"]["retained_by_type"],
        "retained_by_arm": result["candidates"]["retained_by_arm"],
        "filtered_by_reason": result["candidates"]["filtered_by_reason"],
        "models_called": result["models_called"],
    }, indent=2))
