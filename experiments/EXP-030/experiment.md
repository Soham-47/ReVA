---
exp_number: "030"
experiment_id: "reva-phase8-test-user-reported"
family: "alternative_backbones"
phase_label: "Phase 8 test"
title: "User-reported 4,000-pair test metrics (83.75%)"
decision_status: "pending_provenance"
execution_status: "user_reported"
reproducibility_status: "no_local_predictions"
parent_experiment_id: "reva-phase8-test-package-407542"
baseline_experiment_id: "reva-prior-private-test"
---

# EXP-030: User-reported 4,000-pair test metrics (83.75%)

## Decision

`pending_provenance`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

User-reported test aggregate for the Phase 8 line. No local score JSONL.

## Method

Aggregate metrics supplied without local score JSONL; comparability to prior test unverified.

Protocol notes: family `alternative_backbones`; phase `Phase 8 test`; parent `reva-phase8-test-package-407542`; compare-to `reva-prior-private-test`. Execution `user_reported`; reproducibility `no_local_predictions`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| test | overall_accuracy | 3350/4000 | 3358/4000 | `user_reported` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`results/evaluations.csv`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `—`
- Config: `—`
