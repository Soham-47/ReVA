"""Bounded Modal/HPC SigLIP2 crop ranking for the Phase 3A two-video smoke.

This job is intentionally separate from the existing answer-scoring jobs. It
consumes only ``runs/phase3a_retrieval_smoke`` and a locally staged, immutable
SigLIP2 snapshot. It does not load Qwen, SAM, depth, or 3D models.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import resource
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN_RELATIVE = os.environ.get("PHASE3A_RUN", "runs/phase3a_retrieval_smoke")
MODEL_RELATIVE = os.environ.get(
    "SIGLIP2_DIR", "hf_cache/siglip2-base-patch16-224-022b6f7"
)
ATTESTATION = os.environ.get("PHASE3A_ATTESTATION")
HPC_MAX_WALL_MINUTES = 30
HPC_PARTITION = "gpu_h200_8"
HPC_GPU = "H200"
HPC_CPUS = 4
HPC_MEMORY_GIB = 32


def _safe_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(word in str(key).casefold() for word in ("token", "secret", "password"))
            or _secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_secret_key(item) for item in value)
    return False


def _validate_modal_gate(plan: Mapping[str, Any], attestation: Mapping[str, Any]) -> None:
    required = (
        "modal_credentials_available",
        "modal_runtime_available",
    )
    if _secret_key(attestation):
        raise RuntimeError("attestation contains credential-like data; do not persist credentials")
    missing = [key for key in required if attestation.get(key) is not True]
    if missing:
        raise RuntimeError("Modal preflight missing: " + ", ".join(missing))
    budget = plan["modal_budget"]
    if budget.get("gpu") != "A100" or int(budget.get("cpu")) != 8 or int(budget.get("memory_gib")) != 32:
        raise RuntimeError("Modal operational resources must remain A100, 8 vCPU, and 32 GiB")
    if int(budget.get("max_gpu_minutes")) > 30 or int(budget.get("max_wall_minutes")) > 60:
        raise RuntimeError("Modal operational time limits exceed the approved smoke scope")


class _Budget:
    def __init__(self, plan: Mapping[str, Any], *, backend: str = "modal"):
        budget = plan["modal_budget"]
        self.backend = backend
        self.started = time.monotonic()
        configured_wall_minutes = int(budget["max_wall_minutes"])
        self.max_wall_minutes = min(
            configured_wall_minutes,
            HPC_MAX_WALL_MINUTES if backend == "hpc" else configured_wall_minutes,
        )
        self.max_wall_seconds = float(self.max_wall_minutes) * 60.0
        # One GPU is allocated for the whole job. A conservative wall limit
        # also bounds GPU-minutes; no local API exposes billed seconds.
        self.max_gpu_seconds = float(budget["max_gpu_minutes"]) * 60.0

    def check(self) -> None:
        elapsed = time.monotonic() - self.started
        if elapsed >= min(self.max_wall_seconds, self.max_gpu_seconds):
            raise TimeoutError(f"Phase 3A {self.backend} wall/GPU-minute guard reached")

    def summary(self) -> dict[str, Any]:
        elapsed = time.monotonic() - self.started
        return {
            "wall_seconds": round(elapsed, 3),
            "local_gpu_seconds_upper_bound": round(elapsed, 3),
            "local_time_guard_status": "passed" if elapsed < min(self.max_wall_seconds, self.max_gpu_seconds) else "stopped",
        }


def _load_candidates(work: Path, run_root: Path, manifest: Mapping[str, Any], budget: _Budget) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decode direct SAM3 masks and make source-coordinate crop records."""

    import numpy as np
    from PIL import Image

    import sys

    sys.path.insert(0, str(work / "code"))
    import common as C
    import replicate_sam3 as R

    source_policies = manifest["source_policies"]
    source_manifest = C.load_manifest(work)
    source_frames: dict[str, dict[int, Any]] = {}
    failures: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for record_index, record in enumerate(manifest.get("records", [])):
        budget.check()
        if record.get("arm") == "tile-probe" or record.get("status") != "complete":
            continue
        qa_id = str(record["qa_id"])
        policy = source_policies[qa_id]
        video_path = str(policy["video_path"])
        if qa_id not in source_frames:
            indices = [int(index) for index in policy["source_frame_indices"]]
            frames = C.read_frames(video_path, indices, work)
            source_frames[qa_id] = dict(zip(indices, frames))
        zip_path = run_root / str(record["zip"])
        try:
            entries = R.read_mask_zip(zip_path.read_bytes())
        except (OSError, ValueError, RuntimeError) as exc:
            failures.append({"qa_id": qa_id, "record_index": record_index, "failure": type(exc).__name__})
            continue
        info = source_manifest.get(video_path)
        if info is None:
            raise RuntimeError(f"source manifest is missing {video_path}")
        width, height = int(info["width"]), int(info["height"])
        source_indices = [int(index) for index in policy["source_frame_indices"]]
        for mask_index, entry in enumerate(entries):
            budget.check()
            packed_index = int(entry["frame_index"])
            if not 0 <= packed_index < len(source_indices):
                failures.append({"qa_id": qa_id, "record_index": record_index, "mask_index": mask_index, "failure": "packed_frame_out_of_range"})
                continue
            source_index = source_indices[packed_index]
            frame = source_frames[qa_id][source_index]
            with Image.open(BytesIO(entry["payload"])) as image:
                mask = np.asarray(image.convert("L")) > 0
            if mask.shape != (height, width) or frame.shape[:2] != (height, width):
                failures.append({"qa_id": qa_id, "record_index": record_index, "mask_index": mask_index, "failure": "mask_source_shape_mismatch"})
                continue
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                failures.append({"qa_id": qa_id, "record_index": record_index, "mask_index": mask_index, "failure": "empty_mask"})
                continue
            left, right = int(xs.min()), int(xs.max()) + 1
            top, bottom = int(ys.min()), int(ys.max()) + 1
            pad_x = max(1, int(round((right - left) * 0.10)))
            pad_y = max(1, int(round((bottom - top) * 0.10)))
            crop_left, crop_right = max(0, left - pad_x), min(width, right + pad_x)
            crop_top, crop_bottom = max(0, top - pad_y), min(height, bottom + pad_y)
            box_crop = frame[crop_top:crop_bottom, crop_left:crop_right].copy()
            masked_crop = box_crop.copy()
            local_mask = mask[crop_top:crop_bottom, crop_left:crop_right]
            masked_crop[~local_mask] = 0
            candidates.append(
                {
                    "candidate_id": f"{qa_id}-call{record_index:02d}-frame{packed_index:02d}-mask{mask_index:04d}",
                    "qa_id": qa_id,
                    "arm": str(record["arm"]),
                    "proposal_prompt": str(record["prompt"]),
                    "packed_frame_index": packed_index,
                    "source_frame_index": source_index,
                    "timestamp": float(policy["source_timestamps"][packed_index]),
                    "source_box_pixels": {"left": left, "top": top, "right": right, "bottom": bottom},
                    "crop_box_pixels": {"left": crop_left, "top": crop_top, "right": crop_right, "bottom": crop_bottom},
                    "mask_area_pixels": int(mask.sum()),
                    "native_score": None,
                    "native_score_status": "unavailable",
                    "box_crop": box_crop,
                    "masked_crop": masked_crop,
                }
            )
    return candidates, failures


def _stage_source_videos(
    work: Path,
    manifest: Mapping[str, Any],
    budget: _Budget,
    *,
    allow_download: bool = True,
) -> None:
    """Fetch only the two selected source clips at the pinned dataset revision."""

    budget.check()
    paths = sorted({str(item["video_path"]) for item in manifest["source_policies"].values()})
    dataset_root = work / "data" / "ReVA"
    dataset_root.mkdir(parents=True, exist_ok=True)
    missing = [path for path in paths if not (dataset_root / path).is_file()]
    if not missing:
        return
    if not allow_download:
        raise RuntimeError(
            "HPC retrieval requires the two selected source videos to be staged locally; "
            f"missing: {missing}"
        )
    from huggingface_hub import snapshot_download

    import sys

    sys.path.insert(0, str(work / "code"))
    import common as C

    snapshot_download(
        repo_id=C.DATASET_REPO,
        repo_type="dataset",
        revision=C.DATASET_REVISION,
        local_dir=str(dataset_root),
        allow_patterns=paths,
        max_workers=2,
    )
    missing = [path for path in paths if not (dataset_root / path).is_file()]
    if missing:
        raise RuntimeError(f"selected source videos are missing after pinned download: {missing}")


def _candidate_public(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in candidate.items() if key not in {"box_crop", "masked_crop"}}


def _rankings(
    plan: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    image_features: Mapping[str, list[list[float]]],
    text_features: Mapping[str, list[float]],
    budget: _Budget,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np

    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_key[(str(candidate["qa_id"]), str(candidate["arm"]))].append(index)
    result: list[dict[str, Any]] = []
    examples: dict[str, Any] = {}
    for video in plan["videos"]:
        qa_id = str(video["qa_id"])
        for phrase in video["question_phrases"]:
            query = str(phrase).casefold()
            if query not in text_features:
                continue
            for arm in ("fixed-vocabulary", "question-conditioned"):
                pool = by_key.get((qa_id, arm), [])
                if not pool:
                    result.append({"qa_id": qa_id, "phrase": query, "arm": arm, "status": "no_candidates"})
                    continue
                for style in ("padded_rgb_box", "masked_rgb"):
                    budget.check()
                    features = np.asarray([image_features[style][index] for index in pool], dtype=np.float32)
                    query_feature = np.asarray(text_features[query], dtype=np.float32)
                    scores = features @ query_feature
                    order = sorted(range(len(pool)), key=lambda item: (-float(scores[item]), str(candidates[pool[item]]["candidate_id"])))
                    ranked = [pool[item] for item in order]
                    # Use a stable seed; Python's built-in hash is process-randomized.
                    seed = sum((position + 1) * ord(char) for position, char in enumerate(f"{qa_id}|{arm}|{query}|{style}")) & 0xFFFFFFFF
                    rng = random.Random(seed)
                    random_top5 = [
                        [_candidate_public(candidates[pool[item]]) for item in rng.sample(range(len(pool)), min(5, len(pool)))]
                        for _ in range(int(plan["siglip"]["random_draws"]))
                    ]
                    row = {
                        "qa_id": qa_id,
                        "phrase": query,
                        "arm": arm,
                        "crop_style": style,
                        "candidate_count": len(pool),
                        "similarity": "cosine_on_l2_normalized_features",
                        "siglip_top5": [{**_candidate_public(candidates[index]), "similarity": float(scores[order[position]])} for position, index in enumerate(ranked[:5])],
                        "random_seed": seed,
                        "random_draws": int(plan["siglip"]["random_draws"]),
                        "random_top5": random_top5,
                        "native_score_ranking": {"status": "unavailable", "reason": "SAM3 return_zip did not provide attributable per-mask scores"},
                        "manual_labels": {"status": "pending_blind_annotation", "semantic_category": None, "correct_instance": None, "frame_match": None, "cross_frame_identity": None, "mask_iou": None, "target_coverage": None},
                    }
                    result.append(row)
                    examples[f"{qa_id}|{arm}|{query}|{style}"] = row["siglip_top5"]
    return result, examples


def _validate_hpc_gate(plan: Mapping[str, Any]) -> None:
    budget = plan["modal_budget"]
    if int(budget.get("memory_gib")) != HPC_MEMORY_GIB:
        raise RuntimeError(f"HPC operational memory must remain {HPC_MEMORY_GIB} GiB")
    if int(budget.get("max_gpu_minutes")) > 30:
        raise RuntimeError("HPC operational GPU limit exceeds the approved 30-minute smoke scope")


def _validate_hpc_device(torch_module: Any) -> str:
    if not torch_module.cuda.is_available():
        raise RuntimeError("Phase 3A HPC smoke requires the allocated H200 CUDA device")
    gpu_name = str(torch_module.cuda.get_device_name(0))
    if HPC_GPU.casefold() not in gpu_name.casefold():
        raise RuntimeError(f"Phase 3A HPC smoke requires an H200 device; allocated {gpu_name}")
    return gpu_name


def _run(work: str = WORK, *, backend: str = "modal") -> dict[str, bytes]:
    if backend not in {"modal", "hpc"}:
        raise ValueError(f"unsupported Phase 3A retrieval backend: {backend}")
    work_path = Path(work)
    code_path = work_path / "code"
    import sys

    sys.path.insert(0, str(code_path))
    import build_feature_cache as B
    import common as C
    import phase3a_retrieval as P

    run_root = work_path / RUN_RELATIVE
    plan = P.load_plan(work_path)
    manifest = _safe_json(run_root / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("qwen_scoring") != "not-run":
        raise RuntimeError("Phase 3A Replicate manifest is incomplete or includes forbidden Qwen work")
    if int(manifest.get("replicate_attempts", 0)) > int(plan["replicate_budget"]["max_attempts"]):
        raise RuntimeError("Replicate attempt cap is exceeded")
    if backend == "modal":
        # The local attestation is mounted at this stable in-container path. Its
        # contents are non-secret operator evidence; provider enforcement is not
        # independently observable from inside the function.
        attestation_path = work_path / "tasks" / "phase3a_modal_attestation.json"
        attestation = _safe_json(attestation_path)
        _validate_modal_gate(plan, attestation)
    else:
        _validate_hpc_gate(plan)
    model_dir = work_path / MODEL_RELATIVE
    B.validate_local_model_dir(model_dir)
    budget = _Budget(plan, backend=backend)
    import torch

    gpu_name = _validate_hpc_device(torch) if backend == "hpc" else None
    if backend == "modal" and not torch.cuda.is_available():
        raise RuntimeError("Phase 3A Modal smoke requires the allocated A100 CUDA device")
    torch.cuda.reset_peak_memory_stats()
    _stage_source_videos(work_path, manifest, budget, allow_download=backend == "modal")
    candidates, failures = _load_candidates(work_path, run_root, manifest, budget)
    if not candidates:
        raise RuntimeError("no non-empty direct SAM3 masks were available for SigLIP2")
    torch_module, processor, model = B._load_local_encoder(model_dir, "cuda")
    budget.check()
    image_features: dict[str, list[list[float]]] = {}
    for style in ("padded_rgb_box", "masked_rgb"):
        budget.check()
        image_features[style] = B.encode_images(
            processor,
            model,
            [candidate["box_crop"] if style == "padded_rgb_box" else candidate["masked_crop"] for candidate in candidates],
            torch=torch_module,
            device="cuda",
            batch_size=16,
        )
    phrases = sorted({str(item).casefold() for video in plan["videos"] for item in video["question_phrases"]})
    text_rows = B.encode_queries(processor, model, phrases, torch=torch_module, device="cuda", batch_size=16)
    text_features = dict(zip(phrases, text_rows))
    rankings, examples = _rankings(plan, candidates, image_features, text_features, budget)
    budget_summary = budget.summary()
    peak_vram = int(torch.cuda.max_memory_allocated())
    peak_vram_reserved = int(torch.cuda.max_memory_reserved())
    gpu_name = gpu_name or torch.cuda.get_device_name(0)
    feature_bytes = sum(len(rows) * (len(rows[0]) if rows else 0) * 4 for rows in image_features.values())
    report = {
        "schema_version": 1,
        "status": "complete",
        "backend": "Modal" if backend == "modal" else "HPC",
        "scope": "Phase 3A two-video retrieval smoke",
        "plan": plan,
        "candidate_count": len(candidates),
        "candidate_counts_by_arm": {arm: sum(candidate["arm"] == arm for candidate in candidates) for arm in ("fixed-vocabulary", "question-conditioned")},
        "feature_cache": {"styles": list(image_features), "vector_count": len(candidates) * 2, "estimated_float32_bytes": feature_bytes, "persistent_volume": False},
        "rankings": rankings,
        "examples": examples,
        "failures": failures,
        "native_score_status": "unavailable unless provider returned attributable per-mask scores",
        "manual_audit": {"status": "pending_blind_annotation", "semantic_category_metrics": None, "correct_instance_metrics": None, "frame_level_metrics": None, "cross_frame_identity_metrics": None, "boundary_iou_metrics": None},
        "runtime": {**budget_summary, "gpu_device": gpu_name, "gpu_count": int(torch.cuda.device_count()), "peak_cuda_memory_allocated_bytes": peak_vram, "peak_cuda_memory_reserved_bytes": peak_vram_reserved, "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024},
        "hardware": {"actual_gpu": gpu_name, "gpu_count": int(torch.cuda.device_count()), "allocated_cpus": HPC_CPUS if backend == "hpc" else 8, "allocated_memory_gib": HPC_MEMORY_GIB},
        "budget": {"gpu": HPC_GPU if backend == "hpc" else "A100", "cpu": HPC_CPUS if backend == "hpc" else 8, "memory_gib": HPC_MEMORY_GIB, "max_gpu_minutes": 30, "max_wall_minutes": budget.max_wall_minutes, "monetary_cap_status": "waived_by_user_for_two_video_smoke; provider charges are not locally bounded", "actual_cost_usd": None, "cost_status": "provider billing is not exposed to this function; local code cannot independently verify account charges"},
        "qwen_scoring": "not-run",
        "depth_3d": "not-run",
    }
    report_name = "modal_report.json" if backend == "modal" else "hpc_report.json"
    return {report_name: (json.dumps(report, indent=2) + "\n").encode(), "rankings.json": (json.dumps(rankings, indent=2) + "\n").encode()}


def _build_modal_app() -> Any:
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.14.1",
            "torchvision==0.29.1",
            "transformers==5.19.0",
            "accelerate==1.15.0",
            "huggingface-hub==2.2.0",
            "av==19.0.1",
            "numpy",
            "pillow",
        )
        .env({"WORK": WORK, "HF_HOME": f"{WORK}/hf_cache"})
        .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
        .add_local_file(str(ROOT / "tasks" / "phase3a_retrieval_smoke_plan.json"), f"{WORK}/tasks/phase3a_retrieval_smoke_plan.json", copy=True)
        .add_local_file(str(ROOT / "data" / "manifest.json"), f"{WORK}/data/manifest.json", copy=True)
    )
    run_root = ROOT / RUN_RELATIVE
    if run_root.is_dir():
        image = image.add_local_dir(str(run_root), f"{WORK}/{RUN_RELATIVE}", copy=True)
    model_dir = ROOT / MODEL_RELATIVE
    if model_dir.is_dir():
        image = image.add_local_dir(str(model_dir), f"{WORK}/{MODEL_RELATIVE}", copy=True)
    if ATTESTATION and Path(ATTESTATION).is_file():
        image = image.add_local_file(str(Path(ATTESTATION)), f"{WORK}/tasks/phase3a_modal_attestation.json", copy=True)
    app = modal.App("reva-phase3a-retrieval-smoke")

    @app.function(gpu="A100", cpu=8, memory=32768, timeout=60 * 60, serialized=True)
    def siglip2_retrieval_smoke() -> dict[str, bytes]:
        return _run(WORK)

    return app, siglip2_retrieval_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="validate the local plan and report missing future artifacts")
    parser.add_argument("--backend", choices=("modal", "hpc"), default="modal")
    parser.add_argument("--work-root", help="runtime tree for the HPC backend (defaults to WORK)")
    args = parser.parse_args()
    import sys

    sys.path.insert(0, str(ROOT / "code"))
    import phase3a_retrieval as P

    work_root = Path(
        args.work_root or (os.environ.get("WORK") if args.backend == "hpc" else None) or ROOT
    ).resolve()
    plan = P.load_plan(work_root)
    P.build_call_plan(plan)
    if args.dry_run:
        requested_resources = dict(plan["modal_budget"])
        if args.backend == "hpc":
            requested_resources.update(
                {
                    "gpu": HPC_GPU,
                    "cpu": HPC_CPUS,
                    "memory_gib": HPC_MEMORY_GIB,
                    "partition": HPC_PARTITION,
                }
            )
            requested_resources["max_wall_minutes"] = min(
                int(requested_resources["max_wall_minutes"]), HPC_MAX_WALL_MINUTES
            )
        print(json.dumps({"status": "blocked", "backend": "HPC" if args.backend == "hpc" else "Modal", "reason": "provider launch intentionally skipped", "run_manifest_present": (work_root / RUN_RELATIVE / "manifest.json").is_file(), "model_snapshot_present": (work_root / MODEL_RELATIVE).is_dir(), "attestation_present": bool(ATTESTATION and Path(ATTESTATION).is_file()) if args.backend == "modal" else False, "requested_resources": requested_resources}, indent=2))
        return 0
    if args.backend == "hpc":
        _validate_hpc_gate(plan)
        destination = work_root / RUN_RELATIVE
        expected = ("hpc_report.json", "rankings.json")
        existing = sorted(name for name in expected if (destination / name).exists())
        if existing:
            raise RuntimeError(f"refusing to overwrite existing Phase 3A HPC artifacts: {', '.join(existing)}")
        result = _run(str(work_root), backend="hpc")
        existing = sorted(name for name in result if (destination / name).exists())
        if existing:
            raise RuntimeError(f"refusing to overwrite existing Phase 3A HPC artifacts: {', '.join(existing)}")
        destination.mkdir(parents=True, exist_ok=True)
        for name, contents in result.items():
            (destination / name).write_bytes(contents)
        return 0
    run_root = ROOT / RUN_RELATIVE
    run_manifest = _safe_json(run_root / "manifest.json")
    if run_manifest.get("status") != "complete" or run_manifest.get("qwen_scoring") != "not-run":
        raise RuntimeError("refusing Modal launch: the bounded Replicate smoke is incomplete or includes forbidden Qwen work")
    if int(run_manifest.get("replicate_attempts", 0)) > int(plan["replicate_budget"]["max_attempts"]):
        raise RuntimeError("refusing Modal launch: the Replicate attempt cap is exceeded")
    if not (ROOT / MODEL_RELATIVE).is_dir():
        raise RuntimeError("refusing Modal launch: the pinned local SigLIP2 snapshot is missing")
    if not ATTESTATION or not Path(ATTESTATION).is_file():
        raise RuntimeError("refusing Modal launch: provide a non-secret Modal readiness attestation")
    _validate_modal_gate(plan, _safe_json(Path(ATTESTATION)))
    app, function = _build_modal_app()
    result = function.remote()
    destination = ROOT / RUN_RELATIVE
    destination.mkdir(parents=True, exist_ok=True)
    for name, contents in result.items():
        (destination / name).write_bytes(contents)
    return 0


try:
    app, siglip2_retrieval_smoke = _build_modal_app()
except (ImportError, FileNotFoundError):
    app = None
    siglip2_retrieval_smoke = None


if __name__ == "__main__":
    raise SystemExit(main())
