---
exp_number: "001"
experiment_id: "reva-stage-a-legacy"
family: "baseline_sampling"
phase_label: "Stage A"
title: "Frozen Stage A Qwen3-VL-8B legacy 16-frame baseline"
decision_status: "active_reference"
execution_status: "completed"
reproducibility_status: "scores_and_adapter_local"
parent_experiment_id: ""
baseline_experiment_id: ""
---

# EXP-001: Frozen Stage A Qwen3-VL-8B legacy 16-frame baseline

## Decision

`active_reference`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

This experiment **is** the shared architecture. Frozen language LoRA r16, 16 legacy frames, four-view TTA, letter-logit scoring. No overlay, no alternate sampler, no alternate backbone.

## Method

Language LoRA r16 on Qwen3-VL-8B-Instruct; 16 legacy frames; offsets 0/0.5; two option rotations.

Protocol notes: family `baseline_sampling`; phase `Stage A`; parent `none`; compare-to `none`. Execution `completed`; reproducibility `scores_and_adapter_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | overall_accuracy_tta | 498/595 | — | `full_split_tta` |
| valT | overall_accuracy_tta | 885/1059 | — | `full_split_tta` |
| val | overall_accuracy_tta | 1757/2000 | — | `full_split_tta` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/stageA_q3vl8b_r16/legacy/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/train.py;code/score.py;code/evaluate.py;code/configs/stageA.yaml`
- Config: `code/configs/stageA.yaml`
