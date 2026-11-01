---
exp_number: "010"
experiment_id: "reva-temporal-train-audit"
family: "temporal"
phase_label: "temporal data audit"
title: "Train-only temporal supervision feasibility audit"
decision_status: "gate_failed_no_training"
execution_status: "completed"
reproducibility_status: "report_local"
parent_experiment_id: ""
baseline_experiment_id: ""
---

# EXP-010: Train-only temporal supervision feasibility audit

## Decision

`gate_failed_no_training`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Data audit only: whether train labels support temporal fine-tuning. No scoring architecture change.

## Method

100-question audit of whether train labels support targeted temporal fine-tuning.

Protocol notes: family `temporal`; phase `temporal data audit`; parent `none`; compare-to `none`. Execution `completed`; reproducibility `report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`results/temporal_training_data_audit/phase1_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/jobs/build_temporal_training_review_material.py;archive/implementation/jobs/temporal_training_audit_material.sbatch`
- Config: `—`
