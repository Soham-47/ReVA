---
exp_number: "004"
experiment_id: "reva-adaptive-siglip2-frames"
family: "baseline_sampling"
phase_label: "adaptive frames"
title: "SigLIP2 query-adaptive frame selection"
decision_status: "closed_no_promotion"
execution_status: "completed"
reproducibility_status: "audit_report_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-004: SigLIP2 query-adaptive frame selection

## Decision

`closed_no_promotion`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Replaces uniform frames with offline SigLIP2 query-adaptive selection of 16 frames. Same adapter and scorer.

## Method

Offline SigLIP2 caches select 16 frames per question; same scorer and adapter as Stage A.

Protocol notes: family `baseline_sampling`; phase `adaptive frames`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `audit_report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| val | overall_accuracy_tta | 1745/2000 | 1757/2000 | `full_split_tta` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`results/error_review/adaptive_temporal_geometry_audit.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/build_feature_cache.py;code/score.py;archive/implementation/jobs/adaptive_cache.sbatch;archive/implementation/jobs/adaptive_score.sbatch`
- Config: `—`
