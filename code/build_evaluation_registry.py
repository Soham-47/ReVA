"""Build results/registry.csv, results/evaluations.csv, and comparison tables.

Reads existing summary JSON and report-backed records only. Does not re-score
models or modify raw prediction files.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLIT_N = {"valQ": 595, "valT": 1059, "val": 2000}

REGISTRY_FIELDS = [
    "exp_number",
    "experiment_id",
    "family",
    "phase_label",
    "title",
    "description",
    "parent_experiment_id",
    "baseline_experiment_id",
    "code_paths",
    "config_paths",
    "artifact_paths",
    "execution_status",
    "reproducibility_status",
    "decision_status",
    "readme_path",
]

EVAL_FIELDS = [
    "evaluation_id",
    "experiment_id",
    "metric_name",
    "split",
    "denominator",
    "numerator",
    "value",
    "value_is_fraction",
    "comparison_group",
    "baseline_experiment_id",
    "source_artifact",
    "evaluation_kind",
    "notes",
]


def _load_summary(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def _from_acc_summary(
    experiment_id: str,
    split: str,
    summary_path: Path,
    comparison_group: str,
    baseline_id: str,
    kind: str = "full_split_tta",
    notes: str = "",
) -> dict:
    payload = _load_summary(summary_path)
    n = SPLIT_N[split]
    acc = float(payload["acc"])
    num = int(round(acc * n))
    return {
        "evaluation_id": f"{experiment_id}:{split}:overall_tta",
        "experiment_id": experiment_id,
        "metric_name": "overall_accuracy_tta",
        "split": split,
        "denominator": n,
        "numerator": num,
        "value": acc,
        "value_is_fraction": "true",
        "comparison_group": comparison_group,
        "baseline_experiment_id": baseline_id,
        "source_artifact": str(summary_path.relative_to(ROOT)),
        "evaluation_kind": kind,
        "notes": notes,
    }


def _manual_eval(row: dict) -> dict:
    return {field: row.get(field, "") for field in EVAL_FIELDS}


# Canonical experiment index order (EXP-001 … EXP-030).
REGISTRY_ORDER = [
    "reva-stage-a-legacy",
    "reva-error-review-baseline",
    "reva-endpoint-frames",
    "reva-adaptive-siglip2-frames",
    "reva-stage-b-planned",
    "reva-prior-private-test",
    "reva-phase2-temporal-t0",
    "reva-phase2-temporal-coarsefine",
    "reva-phase2-verified-window",
    "reva-temporal-train-audit",
    "reva-grounding-dino-pilot",
    "reva-sam2-segmentation-pilot",
    "reva-sam3-pilot",
    "reva-phase3-explicit-geometry",
    "reva-phase3a-retrieval-smoke",
    "reva-phase3b-gates",
    "reva-phase4-qcsrr",
    "reva-phase5-crop-e0",
    "reva-phase5-crop-ablation",
    "reva-phase6-a0",
    "reva-phase6-a1",
    "reva-phase6-a2",
    "reva-phase6-a3",
    "reva-phase6-a4",
    "reva-phase7a-manifest",
    "reva-phase7b-source-evidence",
    "reva-phase7c-replicate-tracking",
    "reva-phase8-qwen38-h200",
    "reva-phase8-test-package-407542",
    "reva-phase8-test-user-reported",
    "reva-reference-air-temporal-audit",
    "reva-visual-prompt-2x2",
    "reva-visual-prompt-tg-geometry-bar",
    "reva-visual-prompt-trend-bar",
    "reva-spatial-contract",
    "reva-reze-cross-tg",
]

KEEP_CODE_PREFIX = (
    "code/train.py",
    "code/score.py",
    "code/evaluate.py",
    "code/common.py",
    "code/download_data.py",
    "code/build_manifest.py",
    "code/make_splits.py",
    "code/make_submission.py",
    "code/smoke.py",
    "code/diagnose_gpu.py",
    "code/build_evaluation_registry.py",
    "code/configs/stageA.yaml",
    "code/configs/stageB.yaml",
    "jobs/pipeline.sh",
    "jobs/train.sbatch",
    "jobs/score.sbatch",
    "jobs/submit_stageB.sh",
    "jobs/wait_child.sh",
    "runs/",
    "results/",
    "archive/",
)


def archive_code_paths(paths: str) -> str:
    out = []
    for raw in (paths or "").split(";"):
        path = raw.strip()
        if not path:
            continue
        if path.startswith(KEEP_CODE_PREFIX) or path.startswith("archive/"):
            out.append(path)
        elif path.startswith("code/") or path.startswith("jobs/"):
            out.append("archive/implementation/" + path)
        else:
            out.append(path)
    return ";".join(out)


def finalize_registry(raw: list[dict]) -> list[dict]:
    by_id = {row["experiment_id"]: row for row in raw}
    missing = set(REGISTRY_ORDER) - set(by_id)
    extra = set(by_id) - set(REGISTRY_ORDER)
    if missing or extra:
        raise ValueError(f"registry order mismatch: missing={missing} extra={extra}")
    finalized: list[dict] = []
    for index, experiment_id in enumerate(REGISTRY_ORDER, start=1):
        row = dict(by_id[experiment_id])
        row.pop("readme_path", None)
        number = f"{index:03d}"
        slug = experiment_id.removeprefix("reva-")
        row["exp_number"] = number
        row["code_paths"] = archive_code_paths(row.get("code_paths") or "")
        if row.get("config_paths", "").startswith("code/configs/") and not row["config_paths"].startswith(
            ("code/configs/stageA.yaml", "code/configs/stageB.yaml")
        ):
            row["config_paths"] = archive_code_paths(row["config_paths"])
        row["readme_path"] = f"experiments/EXP-{number}/experiment.md"
        finalized.append(row)
    return finalized


def _experiment_narrative_path(row: dict) -> Path | None:
    slug = row["experiment_id"].removeprefix("reva-")
    legacy = ROOT / "docs" / "experiments" / f"{row['exp_number']}-{slug}.md"
    if legacy.is_file():
        return legacy
    return None


def _narrative_body(row: dict) -> str:
    path = _experiment_narrative_path(row)
    if path:
        text = path.read_text(encoding="utf-8", errors="replace")
        if text.startswith("#"):
            text = text.split("\n", 1)[-1].lstrip()
        return text
    return f"## Summary\n\n{row['description']}\n"


def write_experiment_bundle(rows: list[dict]) -> None:
    """JSON index + pointer docs. Canonical writeups live in experiments/EXP-NNN/."""
    bundle = []
    for row in rows:
        entry = dict(row)
        entry["narrative_markdown"] = _narrative_body(row)
        bundle.append(entry)

    json_path = RESULTS / "experiments.json"
    json_path.write_text(json.dumps({"experiments": bundle}, indent=2) + "\n", encoding="utf-8")

    index_lines = [
        "# Experiments",
        "",
        "One folder per experiment. Canonical writeup: `experiments/EXP-NNN/experiment.md`.",
        "Schema: [`TEMPLATE.md`](TEMPLATE.md). Metrics: [`../results/evaluations.csv`](../results/evaluations.csv).",
        "",
        "**Shared architecture:** 16-frame Qwen3-VL-8B LoRA r16, four-view TTA, letter-logit scoring.",
        "",
        "| EXP | ID | Family | Decision | Doc |",
        "| ---: | --- | --- | --- | --- |",
    ]
    for row in rows:
        n = row["exp_number"]
        index_lines.append(
            f"| [{n}](EXP-{n}/experiment.md) | `{row['experiment_id']}` | `{row['family']}` | "
            f"`{row['decision_status']}` | [EXP-{n}](EXP-{n}/experiment.md) |"
        )
    index_lines.append("")
    (ROOT / "experiments" / "INDEX.md").write_text("\n".join(index_lines), encoding="utf-8")

    pointer = [
        "# Experiments",
        "",
        "Canonical comparable writeups: [`experiments/INDEX.md`](../experiments/INDEX.md).",
        "Each experiment is `experiments/EXP-NNN/experiment.md` (shared architecture + what changed).",
        "",
        "Machine-readable: [`results/experiments.json`](../results/experiments.json), "
        "[`results/registry.csv`](../results/registry.csv), "
        "[`results/evaluations.csv`](../results/evaluations.csv).",
        "",
        "| EXP | ID | Title | Decision |",
        "| ---: | --- | --- | --- |",
    ]
    for row in rows:
        pointer.append(
            f"| [{row['exp_number']}](../experiments/EXP-{row['exp_number']}/experiment.md) | "
            f"`{row['experiment_id']}` | {row['title']} | `{row['decision_status']}` |"
        )
    pointer.append("")
    (ROOT / "docs" / "EXPERIMENTS.md").write_text("\n".join(pointer) + "\n", encoding="utf-8")


def registry_rows() -> list[dict]:
    """Curated experiment registry reconciled with local artifacts (2026-10-23)."""
    rows = [
        {
            "experiment_id": "reva-stage-a-legacy",
            "family": "baseline_sampling",
            "phase_label": "Stage A",
            "title": "Frozen Stage A Qwen3-VL-8B legacy 16-frame baseline",
            "description": "Language LoRA r16 on Qwen3-VL-8B-Instruct; 16 legacy frames; offsets 0/0.5; two option rotations.",
            "parent_experiment_id": "",
            "baseline_experiment_id": "",
            "code_paths": "code/train.py;code/score.py;code/evaluate.py;code/configs/stageA.yaml",
            "config_paths": "code/configs/stageA.yaml",
            "artifact_paths": "runs/stageA_q3vl8b_r16/legacy/",
            "execution_status": "completed",
            "reproducibility_status": "scores_and_adapter_local",
            "decision_status": "active_reference",
        },
        {
            "experiment_id": "reva-endpoint-frames",
            "family": "baseline_sampling",
            "phase_label": "endpoint ablation",
            "title": "Endpoint-inclusive 16-frame sampling",
            "description": "Same Stage A adapter; frame indices include clip endpoints instead of legacy uniform sampling.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/score.py;jobs/score_endpoints.sbatch;jobs/modal_score_endpoints.py",
            "config_paths": "",
            "artifact_paths": "runs/stageA_q3vl8b_r16/scores_valQ_endpoints_summary.json",
            "execution_status": "completed",
            "reproducibility_status": "summaries_local",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-adaptive-siglip2-frames",
            "family": "baseline_sampling",
            "phase_label": "adaptive frames",
            "title": "SigLIP2 query-adaptive frame selection",
            "description": "Offline SigLIP2 caches select 16 frames per question; same scorer and adapter as Stage A.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/build_feature_cache.py;code/score.py;jobs/adaptive_cache.sbatch;jobs/adaptive_score.sbatch",
            "config_paths": "",
            "artifact_paths": "results/error_review/adaptive_temporal_geometry_audit.md",
            "execution_status": "completed",
            "reproducibility_status": "audit_report_local",
            "decision_status": "closed_no_promotion",
        },
        {
            "experiment_id": "reva-phase2-temporal-coarsefine",
            "family": "temporal",
            "phase_label": "Phase 2",
            "title": "Coarse-to-fine temporal sampling (T1) and timestamp context (T2)",
            "description": "Temporal-task-only rescoring with coarse-to-fine frame localization or explicit timestamp context.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-phase2-temporal-t0",
            "code_paths": "code/score_temporal.py;code/evaluate_temporal.py;jobs/modal_phase2_temporal.py",
            "config_paths": "",
            "artifact_paths": "runs/phase2_temporal_coarsefine/phase2_report.md",
            "execution_status": "completed",
            "reproducibility_status": "report_local;paired_json_local_only",
            "decision_status": "gate_failed",
        },
        {
            "experiment_id": "reva-phase2-temporal-t0",
            "family": "temporal",
            "phase_label": "Phase 2 T0",
            "title": "Phase 2 temporal legacy reference (T0)",
            "description": "Temporal subset of Stage A legacy predictions used as Phase 2 reference arm.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/score_temporal.py",
            "config_paths": "",
            "artifact_paths": "runs/phase2_temporal_coarsefine/phase2_report.md",
            "execution_status": "completed",
            "reproducibility_status": "embedded_in_phase2_report",
            "decision_status": "reference_arm",
        },
        {
            "experiment_id": "reva-phase2-verified-window",
            "family": "temporal",
            "phase_label": "Phase 2 verified window",
            "title": "Reviewed-window temporal pilot (T1R)",
            "description": "24 selected questions with human-reviewed temporal windows; diagnostic only.",
            "parent_experiment_id": "reva-phase2-temporal-coarsefine",
            "baseline_experiment_id": "reva-phase2-temporal-t0",
            "code_paths": "code/score_temporal_verified.py;code/evaluate_temporal_verified.py;jobs/modal_phase2_verified_window.py",
            "config_paths": "",
            "artifact_paths": "runs/phase2_temporal_verified_window/review_report.md",
            "execution_status": "completed",
            "reproducibility_status": "selected_cohort_pilot",
            "decision_status": "inconclusive_pilot",
        },
        {
            "experiment_id": "reva-temporal-train-audit",
            "family": "temporal",
            "phase_label": "temporal data audit",
            "title": "Train-only temporal supervision feasibility audit",
            "description": "100-question audit of whether train labels support targeted temporal fine-tuning.",
            "parent_experiment_id": "",
            "baseline_experiment_id": "",
            "code_paths": "jobs/build_temporal_training_review_material.py;jobs/temporal_training_audit_material.sbatch",
            "config_paths": "",
            "artifact_paths": "results/temporal_training_data_audit/phase1_report.md",
            "execution_status": "completed",
            "reproducibility_status": "report_local",
            "decision_status": "gate_failed_no_training",
        },
        {
            "experiment_id": "reva-grounding-dino-pilot",
            "family": "grounding_geometry",
            "phase_label": "grounding pilot",
            "title": "GroundingDINO box-context pilot (16 questions per split)",
            "description": "Detector boxes as extra visual context on fixed 16-question pilots.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/grounding.py;code/build_grounding_cache.py;jobs/modal_grounding_pilot.py",
            "config_paths": "",
            "artifact_paths": "runs/grounding_pilot_stageA_modal/;runs/segmentation_pilot_stageA_modal/comparison_report.md",
            "execution_status": "completed",
            "reproducibility_status": "local_score_jsonl",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-sam2-segmentation-pilot",
            "family": "grounding_geometry",
            "phase_label": "SAM2 pilot",
            "title": "SAM 2 mask overlay pilot (32 questions)",
            "description": "GroundingDINO plus SAM2 masks on selected geometry-heavy questions.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/segmentation.py;code/build_segmentation_cache.py;jobs/modal_segmentation_pilot.py",
            "config_paths": "",
            "artifact_paths": "runs/segmentation_pilot_stageA_modal/comparison_report.md",
            "execution_status": "completed",
            "reproducibility_status": "report_and_summary_local",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-sam3-pilot",
            "family": "grounding_geometry",
            "phase_label": "SAM3 pilot",
            "title": "SAM 3 text-prompt segmentation pilots",
            "description": "Replicate SAM3 mask context on 32-question and 17-question selected cohorts.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/replicate_sam3.py;code/sam3_overlay.py;jobs/replicate_sam3_pilot.py;jobs/modal_sam3_score_pilot.py",
            "config_paths": "",
            "artifact_paths": "runs/sam3_pilot_stageA_replicate/comparison_report.md",
            "execution_status": "completed",
            "reproducibility_status": "pilot_reports_local",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-phase3-explicit-geometry",
            "family": "grounding_geometry",
            "phase_label": "Phase 3 explicit geometry",
            "title": "Explicit geometry G0/G1/G2 (30 questions per split)",
            "description": "Selected geometry errors plus controls with box or mask context.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/grounding.py;code/replicate_sam3.py;runs/phase3_explicit_geometry/",
            "config_paths": "",
            "artifact_paths": "runs/phase3_explicit_geometry/full/phase3_report.md",
            "execution_status": "completed",
            "reproducibility_status": "report_local;stale_status_json",
            "decision_status": "closed_no_promotion",
        },
        {
            "experiment_id": "reva-phase3a-retrieval-smoke",
            "family": "grounding_geometry",
            "phase_label": "Phase 3A",
            "title": "SAM3 proposal + SigLIP2 retrieval smoke (2 videos)",
            "description": "Category-level retrieval diagnostic without Qwen rescoring.",
            "parent_experiment_id": "",
            "baseline_experiment_id": "",
            "code_paths": "code/phase3a_retrieval.py;jobs/hpc_phase3a_retrieval_smoke.sbatch",
            "config_paths": "",
            "artifact_paths": "runs/phase3a_retrieval_smoke/review/manual_audit/diagnostic_report.md",
            "execution_status": "completed",
            "reproducibility_status": "smoke_local",
            "decision_status": "exploratory",
        },
        {
            "experiment_id": "reva-phase3b-gates",
            "family": "grounding_geometry",
            "phase_label": "Phase 3B",
            "title": "Retrieval annotation gates 1–2",
            "description": "Frozen SigLIP2 rankings with manual annotation feasibility checks.",
            "parent_experiment_id": "reva-phase3a-retrieval-smoke",
            "baseline_experiment_id": "",
            "code_paths": "jobs/modal_phase3b_gate1.py;runs/phase3b_gate1/",
            "config_paths": "",
            "artifact_paths": "runs/phase3b_gate1/scientific_evaluation_2026-10-18.md",
            "execution_status": "completed",
            "reproducibility_status": "gate_reports_local",
            "decision_status": "no_qwen_promotion",
        },
        {
            "experiment_id": "reva-phase4-qcsrr",
            "family": "grounding_geometry",
            "phase_label": "Phase 4",
            "title": "QCSRR fixed-frame ranking gate",
            "description": "Frozen frame-pool Hit@k diagnostics for vocabulary vs question-conditioned retrieval.",
            "parent_experiment_id": "reva-phase3b-gates",
            "baseline_experiment_id": "",
            "code_paths": "jobs/modal_phase4_qcsrr.py;jobs/modal_phase4_frame_eval_rankings.py",
            "config_paths": "",
            "artifact_paths": "runs/phase4_qcsrr/PHASE4_FINAL_REPORT.md",
            "execution_status": "completed",
            "reproducibility_status": "report_local",
            "decision_status": "led_to_phase5_only",
        },
        {
            "experiment_id": "reva-phase5-crop-ablation",
            "family": "crop_ablation",
            "phase_label": "Phase 5",
            "title": "Frozen crop-evidence ablation (12-question cohort)",
            "description": "Video-only vs random crop vs retrieved crop on selected training diagnostic cohort.",
            "parent_experiment_id": "reva-phase4-qcsrr",
            "baseline_experiment_id": "reva-phase5-crop-e0",
            "code_paths": "jobs/phase5_prepare_crop_ablation.py;jobs/phase5_qwen_crop_ablation_hpc.py;jobs/phase5_analyze_crop_ablation.py",
            "config_paths": "",
            "artifact_paths": "runs/phase5_crop_ablation/PHASE5_FINAL_REPORT.md",
            "execution_status": "completed",
            "reproducibility_status": "selected_cohort_local",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-phase5-crop-e0",
            "family": "crop_ablation",
            "phase_label": "Phase 5 E0",
            "title": "Phase 5 video-only arm",
            "description": "Video-only baseline within Phase 5 cohort.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "",
            "code_paths": "jobs/phase5_qwen_crop_ablation_hpc.py",
            "config_paths": "",
            "artifact_paths": "runs/phase5_crop_ablation/PHASE5_FINAL_REPORT.md",
            "execution_status": "completed",
            "reproducibility_status": "report_local",
            "decision_status": "reference_arm",
        },
        {
            "experiment_id": "reva-phase6-a0",
            "family": "visual_adaptation",
            "phase_label": "Phase 6 A0",
            "title": "Visual adaptation reference (frozen Stage A)",
            "description": "Copied Stage A scores used as A0 reference for Phase 6 paired analysis.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/phase6b_analysis.py;runs/phase6_visual_adaptation_pilot/paired_report.md",
            "config_paths": "",
            "artifact_paths": "runs/phase6_visual_adaptation_pilot/scores/A0/",
            "execution_status": "completed",
            "reproducibility_status": "byte_identical_to_stage_a",
            "decision_status": "reference_arm",
        },
        {
            "experiment_id": "reva-phase6-a1",
            "family": "visual_adaptation",
            "phase_label": "Phase 6 A1",
            "title": "Language-only continuation LoRA",
            "description": "200-step language LoRA continuation from Stage A.",
            "parent_experiment_id": "reva-phase6-a0",
            "baseline_experiment_id": "reva-phase6-a0",
            "code_paths": "code/train.py;code/configs/phase6_A1.yaml;jobs/phase6_train.sbatch",
            "config_paths": "code/configs/phase6_A1.yaml",
            "artifact_paths": "runs/phase6_visual_adaptation_pilot/scores/A1/",
            "execution_status": "completed",
            "reproducibility_status": "summaries_local",
            "decision_status": "regressed_vs_a0",
        },
        {
            "experiment_id": "reva-phase6-a2",
            "family": "visual_adaptation",
            "phase_label": "Phase 6 A2",
            "title": "Language + merger LoRA",
            "description": "Language LoRA plus merger adaptation; screened against A1.",
            "parent_experiment_id": "reva-phase6-a1",
            "baseline_experiment_id": "reva-phase6-a0",
            "code_paths": "code/train.py;code/configs/phase6_A2.yaml;jobs/phase6_train.sbatch",
            "config_paths": "code/configs/phase6_A2.yaml",
            "artifact_paths": "runs/phase6_visual_adaptation_pilot/scores/A2/",
            "execution_status": "completed",
            "reproducibility_status": "summaries_local",
            "decision_status": "below_screening_gate",
        },
        {
            "experiment_id": "reva-phase6-a3",
            "family": "visual_adaptation",
            "phase_label": "Phase 6 A3",
            "title": "Language + upper vision-block LoRA",
            "description": "Language plus vision blocks 23–26 attention LoRA.",
            "parent_experiment_id": "reva-phase6-a2",
            "baseline_experiment_id": "reva-phase6-a0",
            "code_paths": "code/train.py;code/configs/phase6_A3.yaml;jobs/phase6_train.sbatch",
            "config_paths": "code/configs/phase6_A3.yaml",
            "artifact_paths": "runs/phase6_A3_language_plus_vision_from_stageA/phase6b_final_report.md",
            "execution_status": "completed",
            "reproducibility_status": "report_local",
            "decision_status": "regressed_vs_a0",
        },
        {
            "experiment_id": "reva-phase6-a4",
            "family": "visual_adaptation",
            "phase_label": "Phase 6 A4",
            "title": "Language + vision + merger LoRA",
            "description": "Joint language, vision-block, and merger adaptation (200 steps).",
            "parent_experiment_id": "reva-phase6-a3",
            "baseline_experiment_id": "reva-phase6-a0",
            "code_paths": "code/train.py;code/configs/phase6_A4.yaml;jobs/phase6_train.sbatch",
            "config_paths": "code/configs/phase6_A4.yaml",
            "artifact_paths": "runs/phase6_A4_language_plus_vision_and_merger_from_stageA/phase6b_final_report.md",
            "execution_status": "completed",
            "reproducibility_status": "report_local",
            "decision_status": "regressed_vs_a0",
        },
        {
            "experiment_id": "reva-phase7a-manifest",
            "family": "tracking",
            "phase_label": "Phase 7A",
            "title": "Canonical frame manifest",
            "description": "Frame index and timestamp provenance for 1,047 videos.",
            "parent_experiment_id": "",
            "baseline_experiment_id": "",
            "code_paths": "code/phase7_manifest.py",
            "config_paths": "",
            "artifact_paths": "runs/phase7A_feasibility/",
            "execution_status": "completed",
            "reproducibility_status": "manifest_local_only",
            "decision_status": "infrastructure",
        },
        {
            "experiment_id": "reva-phase7b-source-evidence",
            "family": "tracking",
            "phase_label": "Phase 7B",
            "title": "Source-evidence cache validation",
            "description": "24 reviewed cases checking automatic proposals against source video.",
            "parent_experiment_id": "reva-phase7a-manifest",
            "baseline_experiment_id": "",
            "code_paths": "code/phase7b_source_evidence.py;code/validate_phase7_cache.py",
            "config_paths": "",
            "artifact_paths": "runs/phase7B_cache_validation/report.md",
            "execution_status": "completed",
            "reproducibility_status": "report_local",
            "decision_status": "gate_failed",
        },
        {
            "experiment_id": "reva-phase7c-replicate-tracking",
            "family": "tracking",
            "phase_label": "Phase 7C",
            "title": "SAM3 / GroundingDINO / SAM2 tracking feasibility",
            "description": "Two-video Replicate diagnostic for mask continuity and re-anchoring.",
            "parent_experiment_id": "reva-phase7b-source-evidence",
            "baseline_experiment_id": "",
            "code_paths": "code/phase7c_replicate_runner.py;code/phase7c_replicate_diagnostics.py",
            "config_paths": "",
            "artifact_paths": "runs/phase7C_replicate_smoke/report.md",
            "execution_status": "completed",
            "reproducibility_status": "diagnostic_local",
            "decision_status": "gate_failed",
        },
        {
            "experiment_id": "reva-phase8-qwen38-h200",
            "family": "alternative_backbones",
            "phase_label": "Phase 8",
            "title": "Qwen3.8-27B Stage-A-style scoring (partial valT)",
            "description": "Four-view legacy-frame scoring with Qwen3.8-27B; complete valQ and partial valT in local checkout.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/phase8_scorer.py;code/phase8_analysis.py;code/phase8_adapters.py;jobs/phase8_score.sbatch",
            "config_paths": "code/configs/stageA_qwen38_a100.yaml",
            "artifact_paths": "runs/phase8_qwen38_27b_r16_h200_checkpoint1641/",
            "execution_status": "partial",
            "reproducibility_status": "valQ_complete_valT_prefix",
            "decision_status": "pending_full_comparison",
        },
        {
            "experiment_id": "reva-phase8-test-package-407542",
            "family": "alternative_backbones",
            "phase_label": "Phase 8 test",
            "title": "H100 test submission package (job 407542)",
            "description": "Validated 4,000-answer ZIP; not uploaded; lineage not fully traced locally.",
            "parent_experiment_id": "reva-phase8-qwen38-h200",
            "baseline_experiment_id": "reva-prior-private-test",
            "code_paths": "code/make_submission.py;jobs/phase8_score.sbatch",
            "config_paths": "",
            "artifact_paths": "submissions/phase8_qwen38_h100_407542/submission.zip",
            "execution_status": "prepared_not_submitted",
            "reproducibility_status": "zip_local_no_score_jsonl",
            "decision_status": "pending_provenance",
        },
        {
            "experiment_id": "reva-prior-private-test",
            "family": "baseline_sampling",
            "phase_label": "prior submission",
            "title": "Prior private-test submission (Stage A lineage)",
            "description": "README-reported 3,358/4,000 (83.95%); no local prediction file in checkout.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "",
            "code_paths": "code/make_submission.py",
            "config_paths": "",
            "artifact_paths": "README.md",
            "execution_status": "user_reported",
            "reproducibility_status": "no_local_predictions",
            "decision_status": "historical_reference",
        },
        {
            "experiment_id": "reva-phase8-test-user-reported",
            "family": "alternative_backbones",
            "phase_label": "Phase 8 test",
            "title": "User-reported 4,000-pair test metrics (83.75%)",
            "description": "Aggregate metrics supplied without local score JSONL; comparability to prior test unverified.",
            "parent_experiment_id": "reva-phase8-test-package-407542",
            "baseline_experiment_id": "reva-prior-private-test",
            "code_paths": "",
            "config_paths": "",
            "artifact_paths": "results/evaluations.csv",
            "execution_status": "user_reported",
            "reproducibility_status": "no_local_predictions",
            "decision_status": "pending_provenance",
        },
        {
            "experiment_id": "reva-stage-b-planned",
            "family": "baseline_sampling",
            "phase_label": "Stage B",
            "title": "Stage B full-train recipe (planned)",
            "description": "Documented gate and config; no completed local run artifacts.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "code/train.py;code/configs/stageB.yaml;jobs/submit_stageB.sh",
            "config_paths": "code/configs/stageB.yaml",
            "artifact_paths": "",
            "execution_status": "not_executed_locally",
            "reproducibility_status": "recipe_only",
            "decision_status": "planned",
        },
        {
            "experiment_id": "reva-error-review-baseline",
            "family": "baseline_sampling",
            "phase_label": "Phase 0 audit",
            "title": "Stage A error review and task slices",
            "description": "Full-split error taxonomy for valQ, valT, and official val.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "",
            "code_paths": "code/evaluate.py",
            "config_paths": "",
            "artifact_paths": "results/error_review/REVIEW.md",
            "execution_status": "completed",
            "reproducibility_status": "audit_local",
            "decision_status": "diagnostic",
        },
        {
            "experiment_id": "reva-reference-air-temporal-audit",
            "family": "temporal",
            "phase_label": "Temporal Grounding audit",
            "title": "Reference A.I.R. selector on Temporal Grounding",
            "description": "Reference A.I.R. selector with the ReVA Qwen3-VL-8B adapter scored 211/320 Temporal Grounding val items (65.94%), below U16 at 235/320 (73.44%); the video-cluster interval for the −7.50 pp paired difference was [−12.46, −2.68].",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "runs/air_temporal_audit/run_temporal_audit.py;runs/air_temporal_audit/analyze_temporal_audit.py",
            "config_paths": "runs/air_temporal_audit/config.json",
            "artifact_paths": "runs/air_temporal_audit/report.md;runs/air_temporal_audit/official_air_qwen3/scores_temporal.jsonl",
            "execution_status": "completed",
            "reproducibility_status": "scores_and_manifest_local;gpu_instance_deleted",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-visual-prompt-2x2",
            "family": "training_free_overlay",
            "phase_label": "visual prompt 2×2",
            "title": "Progress-bar and ORB-motion overlays (60-item valT cohort)",
            "description": "Frozen Stage A 2×2 on an error-enriched 60-item valT set: progress bar, motion sentence, both.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "archive/implementation/code/visual_prompts.py;archive/implementation/code/analyze_visual_prompt_pilot.py",
            "config_paths": "",
            "artifact_paths": "runs/visual_prompt_pilot/",
            "execution_status": "completed",
            "reproducibility_status": "scores_and_report_local",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-visual-prompt-tg-geometry-bar",
            "family": "training_free_overlay",
            "phase_label": "progress bar TG+geometry",
            "title": "Progress bar on all Temporal Grounding and Geometric Relation items",
            "description": "Progress-bar overlay on the full TG+geometry slices of valQ and valT.",
            "parent_experiment_id": "reva-visual-prompt-2x2",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "archive/implementation/code/visual_prompts.py;archive/implementation/jobs/brev_score_tg_geometry_bar.sh",
            "config_paths": "",
            "artifact_paths": "runs/visual_prompt_tg_geometry/",
            "execution_status": "completed",
            "reproducibility_status": "scores_and_report_local",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-visual-prompt-trend-bar",
            "family": "training_free_overlay",
            "phase_label": "progress bar trend",
            "title": "Progress bar on Trend and Pattern",
            "description": "Progress-bar overlay on all Trend and Pattern items in valQ and valT.",
            "parent_experiment_id": "reva-visual-prompt-2x2",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "archive/implementation/code/visual_prompts.py;archive/implementation/jobs/brev_score_trend_bar.sh",
            "config_paths": "",
            "artifact_paths": "runs/visual_prompt_trend_pattern/",
            "execution_status": "completed",
            "reproducibility_status": "scores_and_report_local",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-spatial-contract",
            "family": "training_free_prompt",
            "phase_label": "spatial contract",
            "title": "Spatial-contract prompt on TG and spatial tasks",
            "description": "Prompt-only spatial contract on Temporal Grounding plus Geometric Relation, Perspective, and Structural Layout.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "archive/implementation/code/spatial_contract.py;archive/implementation/jobs/brev_score_spatial_reze.sh",
            "config_paths": "",
            "artifact_paths": "runs/spatial_contract/",
            "execution_status": "completed",
            "reproducibility_status": "scores_and_report_local",
            "decision_status": "not_promoted",
        },
        {
            "experiment_id": "reva-reze-cross-tg",
            "family": "training_free_prompt",
            "phase_label": "REZE + CROSS",
            "title": "CROSS prompt and REZE evidence fusion on Temporal Grounding",
            "description": "Option-order CROSS plus REZE yes/no fusion on Temporal Grounding; selected α=0 so fusion did not beat letter logits.",
            "parent_experiment_id": "reva-stage-a-legacy",
            "baseline_experiment_id": "reva-stage-a-legacy",
            "code_paths": "archive/implementation/code/reze.py;archive/implementation/code/analyze_reze.py",
            "config_paths": "",
            "artifact_paths": "runs/reze_tg/",
            "execution_status": "completed",
            "reproducibility_status": "scores_and_report_local",
            "decision_status": "not_promoted",
        },
    ]
    return rows


def evaluation_rows() -> list[dict]:
    rows: list[dict] = []

    # Stage A legacy summaries (verified JSON)
    for split in ("valQ", "valT", "val"):
        rows.append(
            _from_acc_summary(
                "reva-stage-a-legacy",
                split,
                ROOT / f"runs/stageA_q3vl8b_r16/legacy/scores_{split}_summary.json",
                "stage_a_full_split",
                "",
            )
        )

    # Endpoint sampling
    for split in ("valQ", "valT", "val"):
        path = ROOT / f"runs/stageA_q3vl8b_r16/scores_{split}_endpoints_summary.json"
        if path.exists():
            rows.append(
                _from_acc_summary(
                    "reva-endpoint-frames",
                    split,
                    path,
                    "stage_a_full_split",
                    "reva-stage-a-legacy",
                )
            )

    # Adaptive — official val from audit (not summary JSON)
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-adaptive-siglip2-frames:val:overall_tta",
                "experiment_id": "reva-adaptive-siglip2-frames",
                "metric_name": "overall_accuracy_tta",
                "split": "val",
                "denominator": 2000,
                "numerator": 1745,
                "value": 0.8725,
                "value_is_fraction": "true",
                "comparison_group": "stage_a_full_split",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "results/error_review/adaptive_temporal_geometry_audit.md",
                "evaluation_kind": "full_split_tta",
                "notes": "From paired audit vs legacy baseline on official val.",
            }
        )
    )

    # Phase 6 A0–A2 from summary JSON
    arm_map = {
        "A0": "reva-phase6-a0",
        "A1": "reva-phase6-a1",
        "A2": "reva-phase6-a2",
    }
    for arm, exp_id in arm_map.items():
        for split in ("valQ", "valT"):
            path = (
                ROOT
                / f"runs/phase6_visual_adaptation_pilot/scores/{arm}/scores_{split}_summary.json"
            )
            rows.append(
                _from_acc_summary(
                    exp_id,
                    split,
                    path,
                    "phase6_visual_adaptation",
                    "reva-phase6-a0",
                )
            )

    # Phase 6 A3/A4 from reports (verified in phase6b_final_report.md)
    phase6_reported = {
        "reva-phase6-a3": {
            "artifact": "runs/phase6_A3_language_plus_vision_from_stageA/phase6b_final_report.md",
            "splits": {"valQ": (495, 595), "valT": (858, 1059)},
        },
        "reva-phase6-a4": {
            "artifact": "runs/phase6_A4_language_plus_vision_and_merger_from_stageA/phase6b_final_report.md",
            "splits": {"valQ": (500, 595), "valT": (863, 1059)},
        },
    }
    for exp_id, payload in phase6_reported.items():
        for split, (num, den) in payload["splits"].items():
            rows.append(
                _manual_eval(
                    {
                        "evaluation_id": f"{exp_id}:{split}:overall_tta",
                        "experiment_id": exp_id,
                        "metric_name": "overall_accuracy_tta",
                        "split": split,
                        "denominator": den,
                        "numerator": num,
                        "value": num / den,
                        "value_is_fraction": "true",
                        "comparison_group": "phase6_visual_adaptation",
                        "baseline_experiment_id": "reva-phase6-a0",
                        "source_artifact": payload["artifact"],
                        "evaluation_kind": "full_split_tta",
                        "notes": "From phase6b_final_report.md tables.",
                    }
                )
            )

    # Phase 2 temporal task-only counts from ledger
    temporal_rows = [
        ("reva-phase2-temporal-t0", "valQ", 51, 69),
        ("reva-phase2-temporal-t0", "valT", 99, 133),
        ("reva-phase2-temporal-coarsefine", "valQ", 49, 69),
        ("reva-phase2-temporal-coarsefine", "valT", 98, 133),
    ]
    for exp_id, split, num, den in temporal_rows:
        rows.append(
            _manual_eval(
                {
                    "evaluation_id": f"{exp_id}:{split}:temporal_task_only",
                    "experiment_id": exp_id,
                    "metric_name": "temporal_grounding_task_accuracy",
                    "split": split,
                    "denominator": den,
                    "numerator": num,
                    "value": num / den,
                    "value_is_fraction": "true",
                    "comparison_group": "phase2_temporal_task_only",
                    "baseline_experiment_id": "reva-phase2-temporal-t0",
                    "source_artifact": "runs/phase2_temporal_coarsefine/phase2_report.md",
                    "evaluation_kind": "task_subset",
                    "notes": "T1 coarse-to-fine arm; T0 is separate rows.",
                }
            )
        )

    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-phase2-verified-window:pilot24:overall",
                "experiment_id": "reva-phase2-verified-window",
                "metric_name": "overall_accuracy_selected",
                "split": "pilot24",
                "denominator": 24,
                "numerator": 13,
                "value": 13 / 24,
                "value_is_fraction": "true",
                "comparison_group": "phase2_selected_pilot",
                "baseline_experiment_id": "reva-phase2-temporal-t0",
                "source_artifact": "runs/phase2_temporal_verified_window/review_report.md",
                "evaluation_kind": "selected_pilot",
                "notes": "Not comparable to full-split metrics.",
            }
        )
    )

    # Phase 8 valQ complete — use ledger-verified 506/595; recompute from jsonl if possible
    qwen38_valq = ROOT / "runs/phase8_qwen38_27b_r16_h200_checkpoint1641/scores_valQ.jsonl"
    if qwen38_valq.exists():
        # Count TTA correctness using same rule as evaluate.py (majority over views)
        import sys

        gold_path = ROOT / "data/splits/valQ.json"
        if gold_path.exists():
            gold = {r["qa_id"]: r for r in json.loads(gold_path.read_text())["QA"]}
            correct = 0
            total = 0
            for line in qwen38_valq.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                qa_id = row["qa_id"]
                if qa_id not in gold:
                    continue
                total += 1
                if row.get("pred") == gold[qa_id]["correct_answer"]:
                    correct += 1
            if total == 595:
                rows.append(
                    _manual_eval(
                        {
                            "evaluation_id": "reva-phase8-qwen38-h200:valQ:overall_tta",
                            "experiment_id": "reva-phase8-qwen38-h200",
                            "metric_name": "overall_accuracy_tta",
                            "split": "valQ",
                            "denominator": total,
                            "numerator": correct,
                            "value": correct / total,
                            "value_is_fraction": "true",
                            "comparison_group": "phase8_vs_stage_a",
                            "baseline_experiment_id": "reva-stage-a-legacy",
                            "source_artifact": str(qwen38_valq.relative_to(ROOT)),
                            "evaluation_kind": "full_split_tta",
                            "notes": "Recomputed from local JSONL with split gold.",
                        }
                    )
                )

    # Partial valT prefix 558/658 from ledger if gold available
    qwen38_valt = ROOT / "runs/phase8_qwen38_27b_r16_h200_checkpoint1641/scores_valT.jsonl"
    if qwen38_valt.exists() and (ROOT / "data/splits/valT.json").exists():
        import sys

        sys.path.insert(0, str(ROOT / "code"))
        gold = {r["qa_id"]: r for r in json.loads((ROOT / "data/splits/valT.json").read_text())["QA"]}
        correct = 0
        n = 0
        for line in qwen38_valt.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            qa_id = row["qa_id"]
            if qa_id not in gold:
                continue
            n += 1
            if row.get("pred") == gold[qa_id]["correct_answer"]:
                correct += 1
        if n:
            rows.append(
                _manual_eval(
                    {
                        "evaluation_id": "reva-phase8-qwen38-h200:valT:overall_tta_prefix",
                        "experiment_id": "reva-phase8-qwen38-h200",
                        "metric_name": "overall_accuracy_tta",
                        "split": "valT",
                        "denominator": n,
                        "numerator": correct,
                        "value": correct / n if n else "",
                        "value_is_fraction": "true",
                        "comparison_group": "phase8_vs_stage_a_prefix",
                        "baseline_experiment_id": "reva-stage-a-legacy",
                        "source_artifact": str(qwen38_valt.relative_to(ROOT)),
                        "evaluation_kind": "partial_split",
                        "notes": f"Only {n}/1059 valT rows present locally; not comparable to full valT.",
                    }
                )
            )

    # User-reported / prior test (not verified locally)
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-prior-private-test:test:overall",
                "experiment_id": "reva-prior-private-test",
                "metric_name": "overall_accuracy",
                "split": "test",
                "denominator": 4000,
                "numerator": 3358,
                "value": 0.8395,
                "value_is_fraction": "true",
                "comparison_group": "private_test",
                "baseline_experiment_id": "",
                "source_artifact": "README.md",
                "evaluation_kind": "leaderboard_reported",
                "notes": "No local prediction file.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-phase8-test-user-reported:test:overall",
                "experiment_id": "reva-phase8-test-user-reported",
                "metric_name": "overall_accuracy",
                "split": "test",
                "denominator": 4000,
                "numerator": 3350,
                "value": 0.8375,
                "value_is_fraction": "true",
                "comparison_group": "private_test",
                "baseline_experiment_id": "reva-prior-private-test",
                "source_artifact": "results/evaluations.csv",
                "evaluation_kind": "user_reported",
                "notes": "Not verified against local artifacts; may differ by 8 answers if same test set.",
            }
        )
    )

    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-reference-air-temporal-audit:val_temporal_grounding:accuracy",
                "experiment_id": "reva-reference-air-temporal-audit",
                "metric_name": "temporal_grounding_task_accuracy",
                "split": "val_temporal_grounding",
                "denominator": 320,
                "numerator": 211,
                "value": 211 / 320,
                "value_is_fraction": "true",
                "comparison_group": "reference_air_temporal_audit",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/air_temporal_audit/report.md",
                "evaluation_kind": "task_subset",
                "notes": "Offset 0 with two option rotations; adapted reference selector scored with ReVA Qwen3-VL-8B adapter. Paired arms and video-cluster intervals are in the report.",
            }
        )
    )

    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-visual-prompt-2x2:valT-phase0-60:bar",
                "experiment_id": "reva-visual-prompt-2x2",
                "metric_name": "overall_accuracy_selected",
                "split": "valT-phase0-60",
                "denominator": 60,
                "numerator": 25,
                "value": 25 / 60,
                "value_is_fraction": "true",
                "comparison_group": "visual_prompt_2x2",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/visual_prompt_pilot/report.json",
                "evaluation_kind": "selected_pilot",
                "notes": "Progress-bar arm. Cohort is error-enriched; baseline on same IDs is 20/60. Not a full-split metric.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-visual-prompt-tg-geometry-bar:valQ:tg_geometry",
                "experiment_id": "reva-visual-prompt-tg-geometry-bar",
                "metric_name": "task_subset_accuracy",
                "split": "valQ",
                "denominator": 179,
                "numerator": 137,
                "value": 137 / 179,
                "value_is_fraction": "true",
                "comparison_group": "visual_prompt_tg_geometry",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/visual_prompt_tg_geometry/valQ_report.json",
                "evaluation_kind": "task_subset",
                "notes": "All Temporal Grounding + Geometric Relation items. Baseline 138/179.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-visual-prompt-tg-geometry-bar:valT:tg_geometry",
                "experiment_id": "reva-visual-prompt-tg-geometry-bar",
                "metric_name": "task_subset_accuracy",
                "split": "valT",
                "denominator": 304,
                "numerator": 235,
                "value": 235 / 304,
                "value_is_fraction": "true",
                "comparison_group": "visual_prompt_tg_geometry",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/visual_prompt_tg_geometry/valT_report.json",
                "evaluation_kind": "task_subset",
                "notes": "All Temporal Grounding + Geometric Relation items. Baseline 233/304.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-visual-prompt-trend-bar:valQ:trend",
                "experiment_id": "reva-visual-prompt-trend-bar",
                "metric_name": "task_subset_accuracy",
                "split": "valQ",
                "denominator": 63,
                "numerator": 52,
                "value": 52 / 63,
                "value_is_fraction": "true",
                "comparison_group": "visual_prompt_trend",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/visual_prompt_trend_pattern/valQ_report.json",
                "evaluation_kind": "task_subset",
                "notes": "Trend and Pattern. Baseline 51/63.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-visual-prompt-trend-bar:valT:trend",
                "experiment_id": "reva-visual-prompt-trend-bar",
                "metric_name": "task_subset_accuracy",
                "split": "valT",
                "denominator": 110,
                "numerator": 92,
                "value": 92 / 110,
                "value_is_fraction": "true",
                "comparison_group": "visual_prompt_trend",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/visual_prompt_trend_pattern/valT_report.json",
                "evaluation_kind": "task_subset",
                "notes": "Trend and Pattern. Baseline 93/110.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-spatial-contract:valQ:four_task",
                "experiment_id": "reva-spatial-contract",
                "metric_name": "task_subset_accuracy",
                "split": "valQ",
                "denominator": 342,
                "numerator": 272,
                "value": 272 / 342,
                "value_is_fraction": "true",
                "comparison_group": "spatial_contract",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/spatial_contract/valQ_report.json",
                "evaluation_kind": "task_subset",
                "notes": "TG + three spatial tasks. Baseline 276/342.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-spatial-contract:valT:four_task",
                "experiment_id": "reva-spatial-contract",
                "metric_name": "task_subset_accuracy",
                "split": "valT",
                "denominator": 600,
                "numerator": 489,
                "value": 489 / 600,
                "value_is_fraction": "true",
                "comparison_group": "spatial_contract",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/spatial_contract/valT_report.json",
                "evaluation_kind": "task_subset",
                "notes": "TG + three spatial tasks. Baseline 487/600.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-reze-cross-tg:valQ:temporal_grounding",
                "experiment_id": "reva-reze-cross-tg",
                "metric_name": "temporal_grounding_task_accuracy",
                "split": "valQ",
                "denominator": 69,
                "numerator": 49,
                "value": 49 / 69,
                "value_is_fraction": "true",
                "comparison_group": "reze_cross_tg",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/reze_tg/report.json",
                "evaluation_kind": "task_subset",
                "notes": "Selected α=0 (letter logits / CROSS). Baseline 51/69.",
            }
        )
    )
    rows.append(
        _manual_eval(
            {
                "evaluation_id": "reva-reze-cross-tg:valT:temporal_grounding",
                "experiment_id": "reva-reze-cross-tg",
                "metric_name": "temporal_grounding_task_accuracy",
                "split": "valT",
                "denominator": 133,
                "numerator": 100,
                "value": 100 / 133,
                "value_is_fraction": "true",
                "comparison_group": "reze_cross_tg",
                "baseline_experiment_id": "reva-stage-a-legacy",
                "source_artifact": "runs/reze_tg/report.json",
                "evaluation_kind": "task_subset",
                "notes": "Selected α=0. Baseline 99/133.",
            }
        )
    )

    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _pct(num: int, den: int) -> str:
    if not den:
        return "—"
    return f"{num}/{den} ({100 * num / den:.2f}%)"


def write_comparisons(evaluations: list[dict]) -> None:
    out_dir = RESULTS / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in evaluations:
        if row.get("metric_name") == "overall_accuracy_tta" and row.get("evaluation_kind") in {
            "full_split_tta",
            "partial_split",
        }:
            by_group[row["comparison_group"]].append(row)

    registry = {r["experiment_id"]: r for r in finalize_registry(registry_rows())}

    def experiment_label(exp_id: str) -> str:
        meta = registry.get(exp_id, {})
        number = meta.get("exp_number") or "?"
        label = meta.get("phase_label") or exp_id
        return f"EXP-{number} {label} (`{exp_id}`)"

    lines = ["# Overall evaluation comparison", "", "Generated by `code/build_evaluation_registry.py`. Do not edit by hand.", ""]
    for group, rows in sorted(by_group.items()):
        lines.append(f"## {group}")
        lines.append("")
        lines.append("| Experiment | split | Result | Kind | Source |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in sorted(rows, key=lambda r: (r["split"], r["experiment_id"])):
            num, den = int(row["numerator"]), int(row["denominator"])
            lines.append(
                f"| {experiment_label(row['experiment_id'])} | {row['split']} | {_pct(num, den)} | {row['evaluation_kind']} | `{row['source_artifact']}` |"
            )
        lines.append("")

    # Phase 6 focused table
    phase6 = [r for r in evaluations if r.get("comparison_group") == "phase6_visual_adaptation"]
    if phase6:
        lines.append("## Phase 6 visual adaptation (valQ / valT)")
        lines.append("")
        arms = sorted({r["experiment_id"] for r in phase6})
        lines.append("| Arm | valQ | valT |")
        lines.append("| --- | --- | --- |")
        for arm in arms:
            valq = next((r for r in phase6 if r["experiment_id"] == arm and r["split"] == "valQ"), None)
            valt = next((r for r in phase6 if r["experiment_id"] == arm and r["split"] == "valT"), None)
            q = _pct(int(valq["numerator"]), int(valq["denominator"])) if valq else "—"
            t = _pct(int(valt["numerator"]), int(valt["denominator"])) if valt else "—"
            lines.append(f"| {experiment_label(arm)} | {q} | {t} |")
        lines.append("")

    (out_dir / "overall.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    reg = finalize_registry(registry_rows())
    write_experiment_bundle(reg)
    ev = evaluation_rows()
    write_csv(RESULTS / "registry.csv", REGISTRY_FIELDS, reg)
    write_csv(RESULTS / "evaluations.csv", EVAL_FIELDS, ev)
    write_comparisons(ev)
    print(f"Wrote {len(reg)} registry rows and {len(ev)} evaluation rows under {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
