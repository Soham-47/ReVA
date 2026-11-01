---
exp_number: "009"
experiment_id: "reva-phase2-verified-window"
family: "temporal"
phase_label: "Phase 2 verified window"
title: "Reviewed-window temporal pilot (T1R)"
decision_status: "inconclusive_pilot"
execution_status: "completed"
reproducibility_status: "selected_cohort_pilot"
parent_experiment_id: "reva-phase2-temporal-coarsefine"
baseline_experiment_id: "reva-phase2-temporal-t0"
---

# EXP-009: Reviewed-window temporal pilot (T1R)

## Decision

`inconclusive_pilot`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Human-reviewed temporal windows on 24 selected questions (T1R). Diagnostic cohort, not the production sampler.

## Method

24 selected questions with human-reviewed temporal windows; diagnostic only.

Protocol notes: family `temporal`; phase `Phase 2 verified window`; parent `reva-phase2-temporal-coarsefine`; compare-to `reva-phase2-temporal-t0`. Execution `completed`; reproducibility `selected_cohort_pilot`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| pilot24 | overall_accuracy_selected | 13/24 | — | `selected_pilot` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase2_temporal_verified_window/review_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/score_temporal_verified.py;archive/implementation/code/evaluate_temporal_verified.py;archive/implementation/jobs/modal_phase2_verified_window.py`
- Config: `—`
