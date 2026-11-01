---
exp_number: "006"
experiment_id: "reva-prior-private-test"
family: "baseline_sampling"
phase_label: "prior submission"
title: "Prior private-test submission (Stage A lineage)"
decision_status: "historical_reference"
execution_status: "user_reported"
reproducibility_status: "no_local_predictions"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: ""
---

# EXP-006: Prior private-test submission (Stage A lineage)

## Decision

`historical_reference`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Packaging only: prior private-test submission from Stage A lineage. No inference-time architecture change in-repo.

## Method

README-reported 3,358/4,000 (83.95%); no local prediction file in checkout.

Protocol notes: family `baseline_sampling`; phase `prior submission`; parent `reva-stage-a-legacy`; compare-to `none`. Execution `user_reported`; reproducibility `no_local_predictions`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| test | overall_accuracy | 3358/4000 | — | `leaderboard_reported` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`README.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/make_submission.py`
- Config: `—`
