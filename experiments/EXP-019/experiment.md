---
exp_number: "019"
experiment_id: "reva-phase5-crop-ablation"
family: "crop_ablation"
phase_label: "Phase 5"
title: "Frozen crop-evidence ablation (12-question cohort)"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "selected_cohort_local"
parent_experiment_id: "reva-phase4-qcsrr"
baseline_experiment_id: "reva-phase5-crop-e0"
---

# EXP-019: Frozen crop-evidence ablation (12-question cohort)

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Adds random or retrieved crop evidence versus EXP-018 on a 12-question diagnostic cohort.

## Method

Video-only vs random crop vs retrieved crop on selected training diagnostic cohort.

Protocol notes: family `crop_ablation`; phase `Phase 5`; parent `reva-phase4-qcsrr`; compare-to `reva-phase5-crop-e0`. Execution `completed`; reproducibility `selected_cohort_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase5_crop_ablation/PHASE5_FINAL_REPORT.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/jobs/phase5_prepare_crop_ablation.py;archive/implementation/jobs/phase5_qwen_crop_ablation_hpc.py;archive/implementation/jobs/phase5_analyze_crop_ablation.py`
- Config: `—`
