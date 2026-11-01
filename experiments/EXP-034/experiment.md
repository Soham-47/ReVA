---
exp_number: "034"
experiment_id: "reva-visual-prompt-trend-bar"
family: "training_free_overlay"
phase_label: "progress bar trend"
title: "Progress bar on Trend and Pattern"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_report_local"
parent_experiment_id: "reva-visual-prompt-2x2"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-034: Progress bar on Trend and Pattern

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Progress-bar overlay on **all** Trend and Pattern items in valQ and valT.

## Method

All valQ (63) and valT (110) Trend and Pattern items. Progress bar only. Same Stage A adapter and four-view TTA.

Protocol notes: family `training_free_overlay`; phase `progress bar trend`; parent `reva-visual-prompt-2x2`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `scores_and_report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | Trend and Pattern | 52/63 | 51/63 | `task_subset` |
| valT | Trend and Pattern | 92/110 | 93/110 | `task_subset` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/visual_prompt_trend_pattern/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/visual_prompts.py;archive/implementation/jobs/brev_score_trend_bar.sh`
- Config: `—`
