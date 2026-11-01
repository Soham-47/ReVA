---
exp_number: "035"
experiment_id: "reva-spatial-contract"
family: "training_free_prompt"
phase_label: "spatial contract"
title: "Spatial-contract prompt on TG and spatial tasks"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_report_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-035: Spatial-contract prompt on TG and spatial tasks

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Prompt-only spatial contract (no pixel overlay) on Temporal Grounding plus three spatial tasks, valQ and valT.

## Method

Prompt prefix/suffix spatial contract on valQ (342) and valT (600) items covering Temporal Grounding, Geometric Relation, Perspective and Viewpoint, and Structural Layout. No pixel overlay.

Protocol notes: family `training_free_prompt`; phase `spatial contract`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `scores_and_report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ four-task | overall | 272/342 | 276/342 | `task_subset` |
| valQ | Temporal Grounding | 49/69 | 51/69 | `task_subset` |
| valT four-task | overall | 489/600 | 487/600 | `task_subset` |
| valT | Temporal Grounding | 100/133 | 99/133 | `task_subset` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/spatial_contract/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/spatial_contract.py;archive/implementation/jobs/brev_score_spatial_reze.sh`
- Config: `—`
