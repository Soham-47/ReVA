---
exp_number: "020"
experiment_id: "reva-phase6-a0"
family: "visual_adaptation"
phase_label: "Phase 6 A0"
title: "Visual adaptation reference (frozen Stage A)"
decision_status: "reference_arm"
execution_status: "completed"
reproducibility_status: "byte_identical_to_stage_a"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-020: Visual adaptation reference (frozen Stage A)

## Decision

`reference_arm`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Copied EXP-001 scores as Phase 6 A0. Architecture identical to the shared path.

## Method

Copied Stage A scores used as A0 reference for Phase 6 paired analysis.

Protocol notes: family `visual_adaptation`; phase `Phase 6 A0`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `byte_identical_to_stage_a`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | overall_accuracy_tta | 498/595 | 498/595 | `full_split_tta` |
| valT | overall_accuracy_tta | 885/1059 | 885/1059 | `full_split_tta` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase6_visual_adaptation_pilot/scores/A0/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/phase6b_analysis.py;runs/phase6_visual_adaptation_pilot/paired_report.md`
- Config: `—`
