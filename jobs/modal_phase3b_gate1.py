#!/usr/bin/env python3
"""Bounded Modal A100 ranking job for the prepared Phase 3B Gate 1 pools.

The default mode is a CPU-only preflight.  Use ``--launch`` only after the
prepared artifacts and the deferred remote download have been explicitly
approved.  The launched function downloads and verifies the pinned SigLIP2
snapshot; it never mounts the evaluation-only annotation file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping
import zipfile


ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN_RELATIVE = os.environ.get("PHASE3B_GATE1_RUN", "runs/phase3b_gate1")
MODEL_RELATIVE = os.environ.get(
    "SIGLIP2_DIR", "hf_cache/siglip2-base-patch16-224-022b6f7"
)
SIGLIP_REVISION = "022b6f71160ffb0169ca4709e2d7e25be659598a"
SIGLIP_REPO_ID = "google/siglip2-base-patch16-224"
QUERY_PHRASES = ("black car", "parking lot", "baseball field")
FRAME_COUNT = 16
GPU = "A100-40GB"
GPU_FAMILY = "A100"
GPU_MEMORY_GIB = 40
CPUS = 4
MEMORY_GIB = 16
MAX_SECONDS = 10 * 60
RANDOM_DRAWS = 1000


def _safe_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_candidates(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (run_root / "candidates.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("candidate metadata must contain JSON objects")
            rows.append(row)
    return rows


def _preflight(work_root: Path) -> dict[str, Any]:
    run_root = work_root / RUN_RELATIVE
    errors: list[str] = []
    warnings: list[str] = []
    required = (
        "plan.json",
        "manifest.json",
        "candidates.jsonl",
        "candidate_masks.zip",
        "source_frames",
        "inputs",
    )
    missing = [name for name in required if not (run_root / name).exists()]
    if missing:
        errors.append("missing Gate 1 artifacts: " + ", ".join(missing))
    plan: dict[str, Any] = {}
    manifest: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    if not missing:
        plan = _safe_json(run_root / "plan.json")
        manifest = _safe_json(run_root / "manifest.json")
        candidates = _load_candidates(run_root)
        if plan.get("status") != "prepared" or manifest.get("status") != "prepared":
            errors.append("Gate 1 plan/manifest must remain in prepared status")
        if manifest.get("execution_status") != "not_run":
            errors.append("prepared manifest execution_status is not not_run")
        if manifest.get("models_called") != 0 or manifest.get("paid_inference") != "not_run":
            errors.append("prepared artifacts claim model/API work was already run")
        model = plan.get("ranking", {})
        if model.get("model") != SIGLIP_REPO_ID or model.get("revision") != SIGLIP_REVISION:
            errors.append("plan does not pin the required SigLIP2 revision")
        if model.get("query_phrases") != list(QUERY_PHRASES):
            errors.append("plan query phrases are not the exact approved phrases")
        budget = plan.get("modal_budget", {})
        if (
            budget.get("gpu") != GPU
            or int(budget.get("cpu", -1)) != CPUS
            or int(budget.get("memory_gib", -1)) != MEMORY_GIB
            or int(budget.get("max_gpu_minutes", -1)) > 10
            or int(budget.get("max_wall_minutes", -1)) > 10
        ):
            errors.append("plan resources exceed the 10-minute A100/4-vCPU/16-GiB gate")

        ids = [str(row.get("candidate_id")) for row in candidates]
        if len(ids) != len(set(ids)):
            errors.append("candidate IDs are not unique")
        if not candidates:
            errors.append("candidate pool is empty")
        if {str(row.get("candidate_type")) for row in candidates} != {
            "connected_component",
            "original_unsplit",
            "context_region",
        }:
            errors.append("candidate pool is missing a required candidate type")
        if {str(row.get("arm")) for row in candidates} != {
            "question-conditioned",
            "fixed-vocabulary",
        }:
            errors.append("candidate pool is missing a required arm")
        if {str(row.get("proposal_prompt")) for row in candidates} - {
            "black car",
            "parking lot",
            "baseball field",
            "vehicle",
            "field",
        }:
            errors.append("candidate prompt vocabulary changed")
        for row in candidates:
            source_frame = run_root / str(row.get("source_frame", ""))
            if not source_frame.is_file():
                errors.append(f"missing source frame for {row.get('candidate_id')}")
                break
            box = row.get("proposal_box_pixels", {})
            if not (
                0 <= int(box.get("left", -1)) < int(box.get("right", -1)) <= 640
                and 0 <= int(box.get("top", -1)) < int(box.get("bottom", -1)) <= 360
            ):
                errors.append(f"out-of-bounds candidate box: {row.get('candidate_id')}")
                break
        try:
            with zipfile.ZipFile(run_root / "candidate_masks.zip") as archive:
                archive_names = set(archive.namelist())
            missing_masks = [row["mask_member"] for row in candidates if row["mask_member"] not in archive_names]
            if missing_masks:
                errors.append(f"candidate mask archive is missing {len(missing_masks)} members")
            if len(archive_names) != len(candidates):
                errors.append("candidate mask archive contains unexpected members")
        except (OSError, zipfile.BadZipFile) as exc:
            errors.append(f"candidate mask archive is unreadable: {type(exc).__name__}")

    model_dir = work_root / MODEL_RELATIVE
    required_model_files = (
        ".reva_encoder_revision",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
    )
    model_snapshot_present = model_dir.is_dir() and all(
        (model_dir / name).is_file() for name in required_model_files
    )
    if not model_snapshot_present:
        warnings.append(
            "the Modal function will download the exact public SigLIP2 commit inside the 10-minute job and verify it locally"
        )
    return {
        "status": "deferred" if not errors else "blocked",
        "backend": "Modal",
        "provider_launch": "skipped",
        "launch_requires_explicit_approval": True,
        "model_calls": 0,
        "paid_inference": "not_run",
        "run_root": str(run_root),
        "run_manifest_present": (run_root / "manifest.json").is_file(),
        "model_snapshot_present": model_snapshot_present,
        "remote_model_download_required": True,
        "remote_model": {
            "repo_id": SIGLIP_REPO_ID,
            "revision": SIGLIP_REVISION,
            "verification": "code/build_feature_cache.py:validate_local_model_dir",
        },
        "model_revision": SIGLIP_REVISION,
        "requested_resources": {
            "gpu": GPU,
            "gpu_memory_gib": GPU_MEMORY_GIB,
            "cpu": CPUS,
            "memory_gib": MEMORY_GIB,
            "max_gpu_minutes": 10,
            "max_wall_minutes": 10,
        },
        "candidate_count": len(candidates),
        "errors": errors,
        "warnings": warnings,
    }


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return dict(row)


def _stable_seed(value: str) -> int:
    # Avoid Python's process-randomized hash so random baselines are reproducible.
    return sum((index + 1) * ord(char) for index, char in enumerate(value)) & 0xFFFFFFFF


def _random_baseline(candidate_ids: list[str], seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    return [
        [candidate_ids[index] for index in rng.sample(range(len(candidate_ids)), min(5, len(candidate_ids)))]
        for _ in range(RANDOM_DRAWS)
    ]


def _run(work: str = WORK) -> dict[str, bytes]:
    """Encode staged crops and return immutable output files."""

    budget_started = time.monotonic()
    import sys

    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(Path(work) / "code"))
    import build_feature_cache as B

    work_path = Path(work)
    run_root = work_path / RUN_RELATIVE
    plan = _safe_json(run_root / "plan.json")
    manifest = _safe_json(run_root / "manifest.json")
    candidates = _load_candidates(run_root)
    if plan.get("ranking", {}).get("revision") != SIGLIP_REVISION:
        raise RuntimeError("refusing to run with an unpinned SigLIP2 revision")
    if manifest.get("execution_status") != "not_run":
        raise RuntimeError("refusing to overwrite an already executed Gate 1 manifest")
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 1 requires the allocated A100 CUDA device")
    gpu_name = str(torch.cuda.get_device_name(0))
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    if GPU_FAMILY.casefold() not in gpu_name.casefold() or total_memory < 35 * 1024**3:
        raise RuntimeError(f"Gate 1 requires an A100 40GB device; allocated {gpu_name} ({total_memory} bytes)")

    model_dir = work_path / MODEL_RELATIVE
    from huggingface_hub import snapshot_download

    # shortcut: no persistent volume, so a later approved rerun may download
    # the pinned snapshot again; use a verified Modal Volume if repeat cost matters.
    snapshot_download(
        repo_id=SIGLIP_REPO_ID,
        revision=SIGLIP_REVISION,
        local_dir=str(model_dir),
        allow_patterns=list(B.SNAPSHOT_FILE_INTEGRITY),
        max_workers=4,
    )
    (model_dir / B.REVISION_MARKER).write_text(SIGLIP_REVISION + "\n")
    model_dir = B.validate_local_model_dir(model_dir)

    def check_budget() -> None:
        if time.monotonic() - budget_started >= MAX_SECONDS:
            raise TimeoutError("Phase 3B Gate 1 10-minute wall/GPU guard reached")

    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    by_frame_key: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(candidates):
        by_key[(str(row["qa_id"]), str(row["arm"]))].append(index)
        by_frame_key[(str(row["qa_id"]), str(row["arm"]), int(row["packed_frame_index"]))].append(index)

    image_features: dict[str, list[list[float]]] = {}
    torch_module, processor, model = B._load_local_encoder(model_dir, "cuda")
    torch.cuda.reset_peak_memory_stats()
    with zipfile.ZipFile(run_root / "candidate_masks.zip") as masks_archive:
        frame_cache: dict[str, np.ndarray] = {}

        def style_crops(style: str) -> list[np.ndarray]:
            crops: list[np.ndarray] = []
            for row in candidates:
                check_budget()
                frame_path = run_root / str(row["source_frame"])
                frame = frame_cache.get(str(frame_path))
                if frame is None:
                    with Image.open(frame_path) as image:
                        frame = np.asarray(image.convert("RGB")).copy()
                    frame_cache[str(frame_path)] = frame
                payload = masks_archive.read(str(row["mask_member"]))
                with Image.open(io.BytesIO(payload)) as image:
                    mask = np.asarray(image.convert("L")) > 0
                box = row["crop_box_pixels"]
                left, top, right, bottom = (int(box[key]) for key in ("left", "top", "right", "bottom"))
                crop = frame[top:bottom, left:right].copy()
                if style == "masked_rgb":
                    crop[~mask[top:bottom, left:right]] = 0
                crops.append(crop)
            return crops

        for style in ("padded_rgb_box", "masked_rgb"):
            check_budget()
            crops = style_crops(style)
            image_features[style] = B.encode_images(
                processor,
                model,
                crops,
                torch=torch_module,
                device="cuda",
                batch_size=32,
            )
            del crops

    phrases = list(QUERY_PHRASES)
    text_features = dict(
        zip(
            phrases,
            B.encode_queries(processor, model, phrases, torch=torch_module, device="cuda", batch_size=8),
        )
    )
    rankings: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    questions = plan["questions"]

    def add_ranking_group(
        *,
        qa_id: str,
        phrase: str,
        arm: str,
        scope: str,
        packed_frame_index: int | None,
        pool: list[int],
    ) -> None:
        pool_ids = [str(candidates[index]["candidate_id"]) for index in pool]
        suffix = "global" if packed_frame_index is None else f"frame-{packed_frame_index:02d}"
        random_key = f"{scope}|{qa_id}|{arm}|{phrase}|{suffix}"
        seed = _stable_seed(random_key)
        random_rows.append(
            {
                "baseline_key": random_key,
                "rank_scope": scope,
                "qa_id": qa_id,
                "arm": arm,
                "phrase": phrase,
                "packed_frame_index": packed_frame_index,
                "candidate_count": len(pool),
                "candidate_pool": "identical_per_arm_pool" if packed_frame_index is None else "identical_per_arm_frame_pool",
                "pool_candidate_ids": pool_ids,
                "random_seed": seed,
                "draws": RANDOM_DRAWS,
                "top5_candidate_ids": _random_baseline(pool_ids, seed),
            }
        )
        frame_row = None
        if packed_frame_index is not None:
            frame_pool = by_frame_key.get((qa_id, "fixed-vocabulary", packed_frame_index), [])
            if not frame_pool:
                frame_pool = by_frame_key.get((qa_id, arm, packed_frame_index), [])
            if frame_pool:
                frame_row = candidates[frame_pool[0]]
        frame_metadata = {
            "source_frame_index": None if frame_row is None else frame_row["source_frame_index"],
            "timestamp_s": None if frame_row is None else frame_row["timestamp_s"],
        }
        query_feature = np.asarray(text_features[phrase], dtype=np.float32)
        for style in ("padded_rgb_box", "masked_rgb"):
            check_budget()
            if pool:
                features = np.asarray([image_features[style][index] for index in pool], dtype=np.float32)
                scores = features @ query_feature
                order = sorted(
                    range(len(pool)),
                    key=lambda item: (-float(scores[item]), str(candidates[pool[item]]["candidate_id"])),
                )
                ranked = [
                    {
                        **_public_candidate(candidates[pool[item]]),
                        "similarity": float(scores[item]),
                    }
                    for item in order[:5]
                ]
            else:
                ranked = []
            rankings.append(
                {
                    "rank_scope": scope,
                    "qa_id": qa_id,
                    "phrase": phrase,
                    "arm": arm,
                    "crop_style": style,
                    "packed_frame_index": packed_frame_index,
                    **frame_metadata,
                    "candidate_count": len(pool),
                    "candidate_pool": "identical_per_arm_pool" if packed_frame_index is None else "identical_per_arm_frame_pool",
                    "pool_candidate_ids": pool_ids,
                    "similarity": "cosine_on_l2_normalized_features",
                    "siglip_top1": ranked[0] if ranked else None,
                    "siglip_top5": ranked,
                    "random_baseline_key": random_key,
                    "manual_labels": {
                        "status": "pending_blind_annotation",
                        "semantic_category_metrics": "N/A",
                        "correct_instance_metrics": "uncertain/N/A",
                    },
                    "native_score_ranking": {
                        "status": "unavailable",
                        "reason": "saved masks contain no attributable native per-mask scores",
                    },
                }
            )

    for qa_id in sorted(questions):
        for phrase in questions[qa_id]["query_phrases"]:
            if phrase not in text_features:
                raise RuntimeError(f"query phrase was not encoded: {phrase}")
            for arm in ("fixed-vocabulary", "question-conditioned"):
                check_budget()
                add_ranking_group(
                    qa_id=qa_id,
                    phrase=phrase,
                    arm=arm,
                    scope="video_global",
                    packed_frame_index=None,
                    pool=by_key.get((qa_id, arm), []),
                )
                for packed in range(FRAME_COUNT):
                    add_ranking_group(
                        qa_id=qa_id,
                        phrase=phrase,
                        arm=arm,
                        scope="per_frame",
                        packed_frame_index=packed,
                        pool=by_frame_key.get((qa_id, arm, packed), []),
                    )

    elapsed = time.monotonic() - budget_started
    staged_bytes = sum(path.stat().st_size for path in run_root.rglob("*") if path.is_file())
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    runtime = {
        "wall_seconds": round(elapsed, 3),
        "gpu_seconds_upper_bound": round(elapsed, 3),
        "time_guard": "passed" if elapsed < MAX_SECONDS else "stopped",
        "includes_model_download_and_load": True,
        "gpu_device": gpu_name,
        "gpu_count": int(torch.cuda.device_count()),
        "allocated_cpus": CPUS,
        "allocated_memory_gib": MEMORY_GIB,
    }
    vram = {
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
        "device_total_memory_bytes": total_memory,
    }
    storage = {
        "staged_input_bytes": staged_bytes,
        "model_snapshot_bytes": sum(
            path.stat().st_size for path in model_dir.rglob("*") if path.is_file()
        ),
        "candidate_count": len(candidates),
        "candidate_mask_archive_bytes": (run_root / "candidate_masks.zip").stat().st_size,
        "source_frame_bytes": sum(
            path.stat().st_size for path in (run_root / "source_frames").rglob("*.png")
        ),
        "candidate_metadata_bytes": (run_root / "candidates.jsonl").stat().st_size,
    }
    execution_manifest = {
        "schema_version": 1,
        "status": "complete",
        "backend": "Modal",
        "prepared_manifest_sha256": _sha256(run_root / "manifest.json"),
        "model": {
            "name": SIGLIP_REPO_ID,
            "revision": SIGLIP_REVISION,
            "download_inside_job": True,
            "verification": "code/build_feature_cache.py:validate_local_model_dir",
            "load_local_files_only_after_verification": True,
        },
        "candidate_count": len(candidates),
        "ranking_rows": len(rankings),
        "random_baseline_rows": len(random_rows),
        "ranking_rows_by_scope": {
            scope: sum(row["rank_scope"] == scope for row in rankings)
            for scope in ("video_global", "per_frame")
        },
        "random_baseline_rows_by_scope": {
            scope: sum(row["rank_scope"] == scope for row in random_rows)
            for scope in ("video_global", "per_frame")
        },
        "outputs": {
            "rankings": "rankings.json",
            "random_baseline": "random_baseline.json",
            "runtime": "runtime.json",
            "vram": "vram.json",
            "storage": "storage.json",
        },
        "manual_evaluation": {
            "status": "pending_blind_annotation",
            "correct_instance_metrics": "uncertain/N/A because target identity is unresolved",
        },
        "runtime": runtime,
        "vram": vram,
        "storage": storage,
        "budget": {
            "gpu": GPU,
            "gpu_memory_gib": GPU_MEMORY_GIB,
            "cpu": CPUS,
            "memory_gib": MEMORY_GIB,
            "max_gpu_minutes": 10,
            "max_wall_minutes": 10,
        },
    }
    execution_bytes = (json.dumps(execution_manifest, indent=2) + "\n").encode()
    return {
        "execution_manifest.json": execution_bytes,
        "modal_report.json": execution_bytes,
        "rankings.json": (json.dumps(rankings, indent=2) + "\n").encode(),
        "random_baseline.json": (json.dumps(random_rows, indent=2) + "\n").encode(),
        "runtime.json": (json.dumps(runtime, indent=2) + "\n").encode(),
        "vram.json": (json.dumps(vram, indent=2) + "\n").encode(),
        "storage.json": (json.dumps(storage, indent=2) + "\n").encode(),
    }


def _build_modal_app() -> Any:
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.14.1",
            "torchvision==0.29.1",
            "transformers==5.19.0",
            "accelerate==1.15.0",
            "huggingface-hub==2.2.0",
            "numpy",
            "pillow",
        )
        .env({"WORK": WORK, "HF_HOME": f"{WORK}/hf_cache"})
        .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
        # Mount only inference inputs. The evaluation-only annotations file is
        # deliberately left outside the image.
        .add_local_file(
            str(ROOT / RUN_RELATIVE / "plan.json"),
            f"{WORK}/{RUN_RELATIVE}/plan.json",
            copy=True,
        )
        .add_local_file(
            str(ROOT / RUN_RELATIVE / "manifest.json"),
            f"{WORK}/{RUN_RELATIVE}/manifest.json",
            copy=True,
        )
        .add_local_file(
            str(ROOT / RUN_RELATIVE / "candidates.jsonl"),
            f"{WORK}/{RUN_RELATIVE}/candidates.jsonl",
            copy=True,
        )
        .add_local_file(
            str(ROOT / RUN_RELATIVE / "candidate_masks.zip"),
            f"{WORK}/{RUN_RELATIVE}/candidate_masks.zip",
            copy=True,
        )
        .add_local_dir(
            str(ROOT / RUN_RELATIVE / "source_frames"),
            f"{WORK}/{RUN_RELATIVE}/source_frames",
            copy=True,
        )
    )
    app = modal.App("reva-phase3b-gate1")

    @app.function(image=image, gpu=GPU, cpu=CPUS, memory=MEMORY_GIB * 1024, timeout=MAX_SECONDS, serialized=True)
    def siglip2_gate1_rank() -> dict[str, bytes]:
        return _run(WORK)

    return app, siglip2_gate1_rank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="run CPU-only validation and skip Modal")
    mode.add_argument("--launch", action="store_true", help="launch the bounded Modal A100 job")
    parser.add_argument("--work-root", help="repository/runtime root (defaults to this repository)")
    args = parser.parse_args()
    work_root = Path(args.work_root or ROOT).resolve()
    preflight = _preflight(work_root)
    if not args.launch:
        print(json.dumps(preflight, indent=2))
        return 0
    if preflight["errors"]:
        raise RuntimeError("Gate 1 preflight failed: " + "; ".join(preflight["errors"]))
    run_root = work_root / RUN_RELATIVE
    output_names = (
        "execution_manifest.json",
        "modal_report.json",
        "rankings.json",
        "random_baseline.json",
        "runtime.json",
        "vram.json",
        "storage.json",
    )
    existing = [name for name in output_names if (run_root / name).exists()]
    if existing:
        raise RuntimeError("refusing to overwrite existing Gate 1 outputs: " + ", ".join(existing))
    # Programmatic Modal calls must run inside the app context.  Keep this
    # branch behind --launch so preflight never imports Modal or contacts it.
    import modal

    app, function = _build_modal_app()
    with modal.enable_output():
        with app.run():
            result = function.remote()
    run_root.mkdir(parents=True, exist_ok=True)
    for name, contents in result.items():
        destination = run_root / name
        if destination.exists():
            raise RuntimeError(f"refusing to overwrite existing Gate 1 output: {destination}")
        destination.write_bytes(contents)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
