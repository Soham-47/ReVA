---
exp_number: "036"
experiment_id: "reva-reze-cross-tg"
family: "training_free_prompt"
phase_label: "REZE + CROSS"
title: "CROSS prompt and REZE evidence fusion on Temporal Grounding"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_report_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-036: CROSS prompt and REZE evidence fusion on Temporal Grounding

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

CROSS option-order prompt plus REZE yes/no evidence fusion on Temporal Grounding only. Best fusion weight was α=0 (letter logits only).

## Method

Temporal Grounding only (valQ 69, valT 133). CROSS keeps option order; REZE adds per-frame yes/no evidence fused as letter_logit + α·evidence. α chosen on valQ grid {0,0.25,0.5,1,2,4}; selected α=0.

Protocol notes: family `training_free_prompt`; phase `REZE + CROSS`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `scores_and_report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | Temporal Grounding letter/CROSS | 49/69 | 51/69 | `task_subset` |
| valT | Temporal Grounding letter/CROSS | 100/133 | 99/133 | `task_subset` |
| valQ | REZE α=0 (selected) | 49/69 | 51/69 | `task_subset` |
| valT | REZE α=0 | 100/133 | 99/133 | `task_subset` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/reze_tg/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/reze.py;archive/implementation/code/analyze_reze.py`
- Config: `—`
