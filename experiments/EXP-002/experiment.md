---
exp_number: "002"
experiment_id: "reva-error-review-baseline"
family: "baseline_sampling"
phase_label: "Phase 0 audit"
title: "Stage A error review and task slices"
decision_status: "diagnostic"
execution_status: "completed"
reproducibility_status: "audit_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: ""
---

# EXP-002: Stage A error review and task slices

## Decision

`diagnostic`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Evaluation-only: full-split error taxonomy and task slices on EXP-001 scores. No model, sampler, or prompt change.

## Method

Full-split error taxonomy for valQ, valT, and official val.

Protocol notes: family `baseline_sampling`; phase `Phase 0 audit`; parent `reva-stage-a-legacy`; compare-to `none`. Execution `completed`; reproducibility `audit_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`results/error_review/REVIEW.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/evaluate.py`
- Config: `—`
