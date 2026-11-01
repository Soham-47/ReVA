---
exp_number: "033"
experiment_id: "reva-visual-prompt-tg-geometry-bar"
family: "training_free_overlay"
phase_label: "progress bar TG+geometry"
title: "Progress bar on all Temporal Grounding and Geometric Relation items"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_report_local"
parent_experiment_id: "reva-visual-prompt-2x2"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-033: Progress bar on all Temporal Grounding and Geometric Relation items

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Progress-bar overlay on **all** Temporal Grounding + Geometric Relation items in valQ and valT (not the error-enriched 60).

## Method

All valQ (179) and valT (304) Temporal Grounding + Geometric Relation items. Progress bar only. Same Stage A adapter and four-view TTA.

Protocol notes: family `training_free_overlay`; phase `progress bar TG+geometry`; parent `reva-visual-prompt-2x2`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `scores_and_report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ TG+geometry | overall | 137/179 | 138/179 | `task_subset` |
| valQ | Temporal Grounding | 50/69 | 51/69 | `task_subset` |
| valQ | Geometric Relation | 87/110 | 87/110 | `task_subset` |
| valT TG+geometry | overall | 235/304 | 233/304 | `task_subset` |
| valT | Temporal Grounding | 100/133 | 99/133 | `task_subset` |
| valT | Geometric Relation | 135/171 | 134/171 | `task_subset` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/visual_prompt_tg_geometry/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/visual_prompts.py;archive/implementation/jobs/brev_score_tg_geometry_bar.sh`
- Config: `—`
