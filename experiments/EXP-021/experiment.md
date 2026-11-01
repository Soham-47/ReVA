---
exp_number: "021"
experiment_id: "reva-phase6-a1"
family: "visual_adaptation"
phase_label: "Phase 6 A1"
title: "Language-only continuation LoRA"
decision_status: "regressed_vs_a0"
execution_status: "completed"
reproducibility_status: "summaries_local"
parent_experiment_id: "reva-phase6-a0"
baseline_experiment_id: "reva-phase6-a0"
---

# EXP-021: Language-only continuation LoRA

## Decision

`regressed_vs_a0`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Training: 200-step language LoRA continuation from Stage A. Inference protocol unchanged.

## Method

200-step language LoRA continuation from Stage A.

Protocol notes: family `visual_adaptation`; phase `Phase 6 A1`; parent `reva-phase6-a0`; compare-to `reva-phase6-a0`. Execution `completed`; reproducibility `summaries_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | overall_accuracy_tta | 490/595 | 498/595 | `full_split_tta` |
| valT | overall_accuracy_tta | 854/1059 | 885/1059 | `full_split_tta` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase6_visual_adaptation_pilot/scores/A1/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/train.py;archive/implementation/code/configs/phase6_A1.yaml;archive/implementation/jobs/phase6_train.sbatch`
- Config: `code/configs/phase6_A1.yaml`
