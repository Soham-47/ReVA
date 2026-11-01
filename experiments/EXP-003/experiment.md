---
exp_number: "003"
experiment_id: "reva-endpoint-frames"
family: "baseline_sampling"
phase_label: "endpoint ablation"
title: "Endpoint-inclusive 16-frame sampling"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "summaries_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-003: Endpoint-inclusive 16-frame sampling

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Sampler only: include clip endpoints in the 16-frame grid. Same adapter, TTA, and letter scoring as EXP-001.

## Method

Same Stage A adapter; frame indices include clip endpoints instead of legacy uniform sampling.

Protocol notes: family `baseline_sampling`; phase `endpoint ablation`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `summaries_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | overall_accuracy_tta | 495/595 | 498/595 | `full_split_tta` |
| valT | overall_accuracy_tta | 890/1059 | 885/1059 | `full_split_tta` |
| val | overall_accuracy_tta | 1755/2000 | 1757/2000 | `full_split_tta` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/stageA_q3vl8b_r16/scores_valQ_endpoints_summary.json`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/score.py;archive/implementation/jobs/score_endpoints.sbatch;archive/implementation/jobs/modal_score_endpoints.py`
- Config: `—`
