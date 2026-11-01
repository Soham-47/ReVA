---
exp_number: "005"
experiment_id: "reva-stage-b-planned"
family: "baseline_sampling"
phase_label: "Stage B"
title: "Stage B full-train recipe (planned)"
decision_status: "planned"
execution_status: "not_executed_locally"
reproducibility_status: "recipe_only"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-005: Stage B full-train recipe (planned)

## Decision

`planned`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Training recipe only: Stage B config and Slurm submit helper. Architecture at inference is unchanged; no local run.

## Method

Documented gate and config; no completed local run artifacts.

Protocol notes: family `baseline_sampling`; phase `Stage B`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `not_executed_locally`; reproducibility `recipe_only`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`—`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/train.py;code/configs/stageB.yaml;jobs/submit_stageB.sh`
- Config: `code/configs/stageB.yaml`
