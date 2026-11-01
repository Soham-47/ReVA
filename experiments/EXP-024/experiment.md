---
exp_number: "024"
experiment_id: "reva-phase6-a4"
family: "visual_adaptation"
phase_label: "Phase 6 A4"
title: "Language + vision + merger LoRA"
decision_status: "regressed_vs_a0"
execution_status: "completed"
reproducibility_status: "report_local"
parent_experiment_id: "reva-phase6-a3"
baseline_experiment_id: "reva-phase6-a0"
---

# EXP-024: Language + vision + merger LoRA

## Decision

`regressed_vs_a0`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Training: joint language, vision-block, and merger LoRA (200 steps). Inference protocol unchanged.

## Method

Joint language, vision-block, and merger adaptation (200 steps).

Protocol notes: family `visual_adaptation`; phase `Phase 6 A4`; parent `reva-phase6-a3`; compare-to `reva-phase6-a0`. Execution `completed`; reproducibility `report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | overall_accuracy_tta | 500/595 | 498/595 | `full_split_tta` |
| valT | overall_accuracy_tta | 863/1059 | 885/1059 | `full_split_tta` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase6_A4_language_plus_vision_and_merger_from_stageA/phase6b_final_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/train.py;archive/implementation/code/configs/phase6_A4.yaml;archive/implementation/jobs/phase6_train.sbatch`
- Config: `code/configs/phase6_A4.yaml`
