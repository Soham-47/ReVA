---
exp_number: "007"
experiment_id: "reva-phase2-temporal-t0"
family: "temporal"
phase_label: "Phase 2 T0"
title: "Phase 2 temporal legacy reference (T0)"
decision_status: "reference_arm"
execution_status: "completed"
reproducibility_status: "embedded_in_phase2_report"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-007: Phase 2 temporal legacy reference (T0)

## Decision

`reference_arm`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Subset evaluation of EXP-001 on Temporal Grounding items (Phase 2 T0). Same predictions as the shared path.

## Method

Temporal subset of Stage A legacy predictions used as Phase 2 reference arm.

Protocol notes: family `temporal`; phase `Phase 2 T0`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `embedded_in_phase2_report`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | temporal_grounding_task_accuracy | 51/69 | 51/69 | `task_subset` |
| valT | temporal_grounding_task_accuracy | 99/133 | 99/133 | `task_subset` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase2_temporal_coarsefine/phase2_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/score_temporal.py`
- Config: `—`
