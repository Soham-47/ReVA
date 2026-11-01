---
exp_number: "022"
experiment_id: "reva-phase6-a2"
family: "visual_adaptation"
phase_label: "Phase 6 A2"
title: "Language + merger LoRA"
decision_status: "below_screening_gate"
execution_status: "completed"
reproducibility_status: "summaries_local"
parent_experiment_id: "reva-phase6-a1"
baseline_experiment_id: "reva-phase6-a0"
---

# EXP-022: Language + merger LoRA

## Decision

`below_screening_gate`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Training: language LoRA plus merger adaptation. Inference protocol unchanged.

## Method

Language LoRA plus merger adaptation; screened against A1.

Protocol notes: family `visual_adaptation`; phase `Phase 6 A2`; parent `reva-phase6-a1`; compare-to `reva-phase6-a0`. Execution `completed`; reproducibility `summaries_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | overall_accuracy_tta | 499/595 | 498/595 | `full_split_tta` |
| valT | overall_accuracy_tta | 871/1059 | 885/1059 | `full_split_tta` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase6_visual_adaptation_pilot/scores/A2/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/train.py;archive/implementation/code/configs/phase6_A2.yaml;archive/implementation/jobs/phase6_train.sbatch`
- Config: `code/configs/phase6_A2.yaml`
