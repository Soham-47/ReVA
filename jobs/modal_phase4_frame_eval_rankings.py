"""Cache-only SigLIP2 score reconstruction for the frozen Phase 4 frame packet.

This is a recovery job for missing intermediate scores. It does not call
Replicate or any inference API and does not modify the original Phase 4 files.
Run without ``--launch`` for the local preflight only.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import unicodedata


ROOT = Path(__file__).resolve().parents[1]
WORK = "/root/ReVA"
RUN_RELATIVE = "runs/phase4_qcsrr"
SOURCE_MANIFEST_RELATIVE = f"{RUN_RELATIVE}/source_manifest.json"
SELECTION_RELATIVE = f"{RUN_RELATIVE}/annotations/fixed_frame_eval/selection_manifest.json"
CANDIDATES_RELATIVE = f"{RUN_RELATIVE}/results_resumed-004/candidates.json.gz"
RANKINGS_RELATIVE = f"{RUN_RELATIVE}/results_resumed-004/rankings.json.gz"
EXECUTION_PLAN_RELATIVE = f"{RUN_RELATIVE}/execution_plan.json"
SOURCE_VIDEOS_RELATIVE = f"{RUN_RELATIVE}/source_videos"
OUTPUT_RELATIVE = f"{RUN_RELATIVE}/annotations/fixed_frame_eval/score_reconstruction"
MODEL_DIR = "/root/siglip2"
MODEL_ID = "google/siglip2-base-patch16-224"
MODEL_REVISION = "022b6f71160ffb0169ca4709e2d7e25be659598a"
PACKED_FRAME_INDICES = (0, 13)
EXPECTED_CANDIDATES = 470
MAX_SECONDS = 180.0
FUNCTION_TIMEOUT_SECONDS = 240
GPU = "A100-40GB"
CPU = 4
MEMORY_GIB = 16

# The approved Phase 4 ranking report: 3,689 candidates in 85.788 s for
# $0.06337610 on one A100-SXM4-40GB. The estimate below is deliberately a
# transparent proportional estimate for the 470 frozen candidates.
BASELINE = {
    "candidate_count": 3689,
    "runtime_seconds": 85.788,
    "modal_cost_usd": 0.06337610,
    "gpu_cost_usd": 0.05492675,
    "cpu_cost_usd": 0.00495850,
    "memory_cost_usd": 0.00334958,
    "egress_cost_usd": 0.00014127,
}


class BudgetStop(RuntimeError):
    """Raised before the isolated job can exceed its wall-time budget."""


class Budget:
    def __init__(self, maximum_seconds: float) -> None:
        self.started = time.monotonic()
        self.maximum_seconds = float(maximum_seconds)

    def check(self) -> None:
        if time.monotonic() - self.started >= self.maximum_seconds:
            raise BudgetStop(f"score reconstruction exceeded {self.maximum_seconds:.1f}s")

    def elapsed(self) -> float:
        return time.monotonic() - self.started


def _json(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, (dict, list)):
        raise ValueError(f"expected JSON object or list: {path}")
    return value


def _gzip_json(path: Path):
    with gzip.open(path, "rt") as stream:
        return json.load(stream)


def _norm(text: str) -> str:
    """Match the original Phase 4 NFC, case-fold, whitespace normalization."""

    return " ".join(unicodedata.normalize("NFC", str(text)).casefold().split())


def _candidate_record_digest(row: dict) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _estimate(candidate_count: int = EXPECTED_CANDIDATES) -> dict:
    ratio = float(candidate_count) / float(BASELINE["candidate_count"])
    return {
        "basis": "linear candidate-count scaling from the approved A100 Phase 4 report",
        "baseline": dict(BASELINE),
        "candidate_count": int(candidate_count),
        "scaling_ratio": ratio,
        "expected_runtime_seconds": round(BASELINE["runtime_seconds"] * ratio, 3),
        "expected_modal_cost_usd": round(BASELINE["modal_cost_usd"] * ratio, 6),
        "expected_cost_breakdown_usd": {
            key: round(float(value) * ratio, 6)
            for key, value in BASELINE.items()
            if key.endswith("_cost_usd")
        },
        "hard_limits": {
            "gpu": GPU,
            "cpu": CPU,
            "memory_gib": MEMORY_GIB,
            "soft_stop_seconds": MAX_SECONDS,
            "function_timeout_seconds": FUNCTION_TIMEOUT_SECONDS,
            "same_rate_hard_timeout_cost_ceiling_usd": round(
                BASELINE["modal_cost_usd"]
                / BASELINE["runtime_seconds"]
                * FUNCTION_TIMEOUT_SECONDS,
                6,
            ),
        },
    }


def _load_inputs(root: Path) -> dict:
    return {
        "source": _json(root / SOURCE_MANIFEST_RELATIVE),
        "selection_manifest": _json(root / SELECTION_RELATIVE),
        "candidates": _gzip_json(root / CANDIDATES_RELATIVE),
        "rankings": _gzip_json(root / RANKINGS_RELATIVE),
        "plan": _json(root / EXECUTION_PLAN_RELATIVE),
        "selection": _json(root / "tasks/qcsrr_final_benchmark_selection.json"),
    }


def _query_specs(inputs: dict) -> list[dict]:
    questions = inputs["selection"]["questions"]
    fixed = list(inputs["plan"]["fixed_vocabulary_by_question_order"])
    if len(questions) != 12 or len(fixed) != 12:
        raise ValueError("Phase 4 query selection must contain 12 questions and 12 fixed prompts")
    specs = []
    for question_order, question in enumerate(questions):
        qa_id = str(question["qa_id"])
        phrases = [str(item).strip() for item in question["entity_prompt_phrases"]]
        for phrase_index, phrase in enumerate(phrases):
            specs.append(
                {
                    "qa_id": qa_id,
                    "arm": "question-conditioned",
                    "phrase": _norm(phrase),
                    "phrase_role": "target" if phrase_index == 0 else "reference",
                }
            )
        specs.append(
            {
                "qa_id": qa_id,
                "arm": "fixed-vocabulary",
                "phrase": _norm(fixed[question_order]),
                "phrase_role": "fixed_control",
            }
        )
    return specs


def _archived_scores(rankings: list[dict], selected_ids: set[str]) -> dict:
    scores: dict[tuple[str, str, str, str], dict] = {}
    for row in rankings:
        for branch in ("R0", "R1", "R2"):
            for item in row.get(branch, {}).get("top5", []):
                candidate_id = str(item.get("candidate_id", ""))
                if candidate_id not in selected_ids:
                    continue
                key = (str(row["qa_id"]), str(row["arm"]), _norm(row["phrase"]), candidate_id)
                score = float(item["similarity"])
                existing = scores.get(key)
                if existing is None:
                    scores[key] = {"score": score, "branches": [branch]}
                else:
                    if not math.isclose(existing["score"], score, rel_tol=0.0, abs_tol=1e-12):
                        raise AssertionError(f"archived score disagreement for {key}")
                    existing["branches"].append(branch)
    for value in scores.values():
        value["branches"] = sorted(set(value["branches"]))
    return scores


def _validate_inputs(root: Path, inputs: dict) -> dict:
    frozen = inputs["selection_manifest"]
    rows = frozen.get("selected_candidates")
    if frozen.get("status") != "frozen" or not isinstance(rows, list):
        raise ValueError("fixed-frame selection manifest is not frozen")
    assert frozen["selection_rule"]["packed_frame_indices"] == list(PACKED_FRAME_INDICES)
    assert len(rows) == EXPECTED_CANDIDATES
    candidates = inputs["candidates"]
    by_id = {str(row["candidate_id"]): row for row in candidates}
    assert len(by_id) == len(candidates)
    selected_ids = set()
    for row in rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id in selected_ids:
            raise AssertionError(f"duplicate frozen candidate {candidate_id}")
        selected_ids.add(candidate_id)
        candidate = by_id.get(candidate_id)
        if candidate is None:
            raise AssertionError(f"frozen candidate missing from source JSON: {candidate_id}")
        assert int(candidate["packed_frame_index"]) in PACKED_FRAME_INDICES
        assert str(candidate["arm"]) == str(row["arm"])
        assert int(candidate["source_frame_index"]) == int(row["source_frame_index"])
        assert str(candidate["mask_sha256"]) == str(row["mask_sha256"])
        assert _candidate_record_digest(candidate) == str(row["candidate_record_sha256"])
        source_record = inputs["source"][str(row["qa_id"])]
        packed = int(row["packed_frame_index"])
        assert int(row["source_frame_index"]) == int(source_record["source_frame_indices"][packed])
        assert math.isclose(
            float(row["timestamp_s"]),
            float(source_record["source_timestamps_s"][packed]),
            rel_tol=0.0,
            abs_tol=1e-6,
        )

    source_video_paths = {}
    for qa_id, record in inputs["source"].items():
        video_path = Path(str(record["video_path"]))
        if video_path.is_absolute() or ".." in video_path.parts:
            raise ValueError(f"unsafe source video path for {qa_id}")
        path = root / SOURCE_VIDEOS_RELATIVE / video_path
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(path.stat().st_size) != int(record["file_size_bytes"]):
            raise ValueError(f"source video size mismatch for {qa_id}")
        if _sha256(path) != str(record["video_sha256"]):
            raise ValueError(f"source video digest mismatch for {qa_id}")
        source_video_paths[qa_id] = str(path)

    archived = _archived_scores(inputs["rankings"], selected_ids)
    archived_ids = {key[-1] for key in archived}
    score_entries = sum(
        len(row.get(branch, {}).get("top5", []))
        for row in inputs["rankings"]
        for branch in ("R0", "R1", "R2")
    )
    return {
        "selected_rows": rows,
        "candidate_by_id": by_id,
        "selected_ids": selected_ids,
        "source_video_paths": source_video_paths,
        "query_specs": _query_specs(inputs),
        "archived_scores": archived,
        "archived_summary": {
            "ranking_rows": len(inputs["rankings"]),
            "top5_score_entries": score_entries,
            "unique_candidate_ids_with_archived_scores": len(
                {item["candidate_id"] for row in inputs["rankings"] for branch in ("R0", "R1", "R2") for item in row.get(branch, {}).get("top5", [])}
            ),
            "frozen_candidate_ids_with_archived_scores": len(archived_ids),
            "frozen_query_candidate_score_keys": len(archived),
            "full_per_candidate_scores_present": False,
        },
    }


def _preflight(root: Path) -> dict:
    inputs = _load_inputs(root)
    state = _validate_inputs(root, inputs)
    assert len(state["selected_rows"]) == EXPECTED_CANDIDATES
    assert len(state["query_specs"]) == 36
    assert state["archived_summary"]["full_per_candidate_scores_present"] is False
    return {
        "status": "ready",
        "purpose": "cache-only score reconstruction for missing intermediate data",
        "external_inference_api_calls": False,
        "frozen_candidate_count": len(state["selected_rows"]),
        "packed_frame_indices": list(PACKED_FRAME_INDICES),
        "proposal_arm_counts": {
            arm: sum(row["arm"] == arm for row in state["selected_rows"])
            for arm in ("fixed-vocabulary", "question-conditioned")
        },
        "query_count": len(state["query_specs"]),
        "archived_rankings": state["archived_summary"],
        "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION},
        "estimate": _estimate(),
        "no_modal_launch": True,
    }


def _read_source_frames(path: Path, indices: list[int], budget: Budget) -> dict[int, object]:
    import av

    wanted = set(int(index) for index in indices)
    result = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for source_index, frame in enumerate(container.decode(stream)):
            budget.check()
            if source_index in wanted:
                result[source_index] = frame.to_ndarray(format="rgb24")
    if set(result) != wanted:
        raise RuntimeError(f"source frame decode did not return exact indices for {path}")
    return result


def _execute(work: str = WORK) -> dict[str, bytes]:
    import sys
    import numpy as np

    work_root = Path(work)
    budget = Budget(MAX_SECONDS)
    inputs = _load_inputs(work_root)
    state = _validate_inputs(work_root, inputs)
    rows = sorted(
        state["selected_rows"],
        key=lambda row: (str(row["qa_id"]), int(row["packed_frame_index"]), str(row["candidate_id"])),
    )
    by_id = state["candidate_by_id"]

    sys.path.insert(0, str(work_root / "code"))
    import build_feature_cache as B

    assert B.SIGLIP2_MODEL_REVISION == MODEL_REVISION
    model_root = B.validate_local_model_dir(Path(MODEL_DIR))
    torch, processor, model = B._load_local_encoder(model_root, "cuda")
    budget.check()

    source_frames = {}
    for qa_id, source_path in state["source_video_paths"].items():
        packed = inputs["source"][qa_id]["source_frame_indices"]
        needed = sorted({int(packed[index]) for index in PACKED_FRAME_INDICES})
        source_frames[qa_id] = _read_source_frames(Path(source_path), needed, budget)

    crops = []
    for frozen in rows:
        candidate = by_id[str(frozen["candidate_id"])]
        assert candidate["crop_box_pixels"]
        box = candidate["crop_box_pixels"]
        frame = source_frames[str(frozen["qa_id"])][int(frozen["source_frame_index"])]
        left, top, right, bottom = (int(box[key]) for key in ("left", "top", "right", "bottom"))
        if not (0 <= left < right <= frame.shape[1] and 0 <= top < bottom <= frame.shape[0]):
            raise AssertionError(f"invalid frozen crop box for {frozen['candidate_id']}")
        crop = frame[top:bottom, left:right].copy()
        if crop.size == 0:
            raise AssertionError(f"empty frozen crop for {frozen['candidate_id']}")
        crops.append(crop)

    image_features = B.encode_images(
        processor.image_processor,
        model,
        crops,
        torch=torch,
        device="cuda",
        batch_size=32,
        input_data_format="channels_last",
    )
    budget.check()
    phrases = sorted({spec["phrase"] for spec in state["query_specs"]})
    text_features = dict(
        zip(
            phrases,
            B.encode_queries(processor, model, phrases, torch=torch, device="cuda", batch_size=32),
        )
    )

    by_pool: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_pool[(str(row["qa_id"]), str(row["arm"]), int(row["packed_frame_index"]))].append(index)

    query_results = []
    reproduced_scores = {}
    for spec in state["query_specs"]:
        for packed_frame_index in PACKED_FRAME_INDICES:
            pool = by_pool.get((spec["qa_id"], spec["arm"], packed_frame_index), [])
            scored = []
            query = text_features[spec["phrase"]]
            for index in pool:
                score = float(
                    np.dot(
                        np.asarray(image_features[index], dtype=np.float32),
                        np.asarray(query, dtype=np.float32),
                    )
                )
                candidate_id = str(rows[index]["candidate_id"])
                key = (spec["qa_id"], spec["arm"], spec["phrase"], candidate_id)
                assert key not in reproduced_scores
                reproduced_scores[key] = score
                scored.append((score, candidate_id, index))
            scored.sort(key=lambda item: (-item[0], item[1]))
            ranked_candidates = [
                {
                    "rank": rank,
                    "candidate_id": candidate_id,
                    "alias": str(rows[index]["alias"]),
                    "packed_frame_index": int(rows[index]["packed_frame_index"]),
                    "source_frame_index": int(rows[index]["source_frame_index"]),
                    "timestamp_s": float(rows[index]["timestamp_s"]),
                    "score": float(score),
                }
                for rank, (score, candidate_id, index) in enumerate(scored, start=1)
            ]
            query_results.append(
                {
                    "qa_id": spec["qa_id"],
                    "arm": spec["arm"],
                    "packed_frame_index": packed_frame_index,
                    "phrase": spec["phrase"],
                    "phrase_role": spec["phrase_role"],
                    "candidate_count": len(ranked_candidates),
                    "status": "scored" if ranked_candidates else "no_candidates",
                    "ranked_candidates": ranked_candidates,
                }
            )
        budget.check()

    tolerance = 1e-6
    validation_matches = []
    for key, archived in sorted(state["archived_scores"].items()):
        reproduced = reproduced_scores.get(key)
        if reproduced is None:
            continue
        validation_matches.append(
            {
                "qa_id": key[0],
                "arm": key[1],
                "phrase": key[2],
                "candidate_id": key[3],
                "archived_score": float(archived["score"]),
                "reproduced_score": float(reproduced),
                "absolute_error": abs(float(reproduced) - float(archived["score"])),
                "archived_branches": archived["branches"],
            }
        )
    max_error = max((item["absolute_error"] for item in validation_matches), default=0.0)
    validation = {
        "archived_score_keys": len(state["archived_scores"]),
        "reproduced_score_keys": len(reproduced_scores),
        "matched_score_keys": len(validation_matches),
        "matched_candidate_ids": len({item["candidate_id"] for item in validation_matches}),
        "tolerance": tolerance,
        "max_absolute_error": max_error,
        "all_matches_within_tolerance": max_error <= tolerance,
        "matches": validation_matches,
    }
    assert len(rows) == EXPECTED_CANDIDATES
    assert sum(item["candidate_count"] for item in query_results) == len(reproduced_scores)
    assert validation["all_matches_within_tolerance"]

    report = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "score_reconstruction_missing_intermediate_data",
        "external_inference_api_calls": False,
        "source_results": "runs/phase4_qcsrr/results_resumed-004",
        "selection_manifest": SELECTION_RELATIVE,
        "candidate_count": len(rows),
        "query_pool_count": len(query_results),
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "image_preprocess": "build_feature_cache.encode_images with channels_last RGB crops",
            "text_normalization": "NFC, casefold, collapse whitespace",
            "text_preprocess": "build_feature_cache.encode_queries padding=max_length truncation=True",
        },
        "runtime": {
            "wall_seconds": round(budget.elapsed(), 3),
            "soft_stop_seconds": MAX_SECONDS,
            "function_timeout_seconds": FUNCTION_TIMEOUT_SECONDS,
            "gpu": GPU,
            "cpu": CPU,
            "memory_gib": MEMORY_GIB,
        },
        "estimate": _estimate(),
        "archived_score_validation": validation,
    }
    payload = {
        "schema_version": 1,
        "status": "complete",
        "purpose": "score_reconstruction_missing_intermediate_data",
        "model": report["model"],
        "candidate_count": len(rows),
        "query_results": query_results,
        "archived_score_validation": validation,
    }
    return {
        "score_reconstruction.json": (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(),
        "modal_report.json": (json.dumps(report, indent=2, sort_keys=True) + "\n").encode(),
    }


def _remote_entrypoint() -> dict[str, bytes]:
    import importlib.util

    path = "/root/modal_phase4_frame_eval_rankings.py"
    spec = importlib.util.spec_from_file_location("reva_phase4_frame_eval_rankings", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("remote score reconstruction module is missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._execute("/root/ReVA")


def _build_modal_app():
    import modal

    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.14.1",
            "torchvision==0.29.1",
            "transformers==5.19.0",
            "accelerate==1.15.0",
            "huggingface-hub==2.2.0",
            "av==18.1.0",
            "numpy",
            "pillow",
        )
        .run_commands(
            "python -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/siglip2-base-patch16-224', revision='022b6f71160ffb0169ca4709e2d7e25be659598a', local_dir='/root/siglip2', allow_patterns=['config.json','model.safetensors','preprocessor_config.json','special_tokens_map.json','tokenizer.json','tokenizer.model','tokenizer_config.json'])\"",
            "python -c \"from pathlib import Path; Path('/root/siglip2/.reva_encoder_revision').write_text('022b6f71160ffb0169ca4709e2d7e25be659598a\\n')\"",
        )
        .env({"WORK": WORK, "HF_HOME": f"{WORK}/hf_cache"})
        .add_local_dir(str(ROOT / "code"), f"{WORK}/code", copy=True)
        .add_local_file(str(ROOT / SOURCE_MANIFEST_RELATIVE), f"{WORK}/{SOURCE_MANIFEST_RELATIVE}", copy=True)
        .add_local_file(str(ROOT / SELECTION_RELATIVE), f"{WORK}/{SELECTION_RELATIVE}", copy=True)
        .add_local_file(str(ROOT / CANDIDATES_RELATIVE), f"{WORK}/{CANDIDATES_RELATIVE}", copy=True)
        .add_local_file(str(ROOT / RANKINGS_RELATIVE), f"{WORK}/{RANKINGS_RELATIVE}", copy=True)
        .add_local_file(str(ROOT / EXECUTION_PLAN_RELATIVE), f"{WORK}/{EXECUTION_PLAN_RELATIVE}", copy=True)
        .add_local_file(
            str(ROOT / "tasks/qcsrr_final_benchmark_selection.json"),
            f"{WORK}/tasks/qcsrr_final_benchmark_selection.json",
            copy=True,
        )
        .add_local_dir(str(ROOT / SOURCE_VIDEOS_RELATIVE), f"{WORK}/{SOURCE_VIDEOS_RELATIVE}", copy=True)
        .add_local_file(str(Path(__file__).resolve()), "/root/modal_phase4_frame_eval_rankings.py", copy=True)
    )
    app = modal.App(
        "reva-phase4-frame-eval-score-reconstruction",
        tags={"experiment": "phase4-qcsrr", "purpose": "score-reconstruction", "cache-only": "true"},
    )

    @app.function(
        image=image,
        gpu=GPU,
        cpu=CPU,
        memory=MEMORY_GIB * 1024,
        timeout=FUNCTION_TIMEOUT_SECONDS,
        serialized=True,
    )
    def reconstruct_scores() -> dict[str, bytes]:
        return _remote_entrypoint()

    return app, reconstruct_scores


def _launch(root: Path) -> int:
    import modal

    app, function = _build_modal_app()
    with modal.enable_output():
        with app.run():
            outputs = function.remote()
    destination = root / OUTPUT_RELATIVE
    destination.mkdir(parents=True, exist_ok=True)
    for name, data in outputs.items():
        path = destination / name
        if path.exists():
            raise RuntimeError(f"refusing to overwrite score reconstruction output: {path}")
        path.write_bytes(data)
    print(json.dumps({"status": "complete", "output_dir": str(destination), "files": sorted(outputs)}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="run the local preflight without launching Modal")
    parser.add_argument("--launch", action="store_true", help="launch the isolated Modal reconstruction job")
    args = parser.parse_args()
    if args.dry_run and args.launch:
        parser.error("--dry-run and --launch are mutually exclusive")
    if not args.launch:
        print(json.dumps(_preflight(ROOT), indent=2, sort_keys=True))
        return 0
    return _launch(ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
