---
exp_number: "032"
experiment_id: "reva-visual-prompt-2x2"
family: "training_free_overlay"
phase_label: "visual prompt 2×2"
title: "Progress-bar and ORB-motion overlays (60-item valT cohort)"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_report_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-032: Progress-bar and ORB-motion overlays (60-item valT cohort)

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Training-free pixel overlays on frozen Stage A: progress bar, ORB motion sentence, and both, on a 60-item error-enriched valT cohort.

## Method

Cohort is 60 valT items from the phase-0 temporal/geometry review (20 TG errors, 10 TG controls, 20 geometry errors, 10 geometry controls). Baseline on this set is 20/60 by construction. Frame budget, offsets, rotations, and letter scoring match EXP-001. Overlay modules: `archive/implementation/code/visual_prompts.py`.

Protocol notes: family `training_free_overlay`; phase `visual prompt 2×2`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `scores_and_report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valT-phase0-60 | overall | 25/60 bar; 23/60 motion; 23/60 both | 20/60 | `selected_pilot` |
| valT-phase0-60 | Temporal Grounding | 13/30 bar | 10/30 | `selected_pilot` |
| valT-phase0-60 | Geometric Relation | 12/30 bar | 10/30 | `selected_pilot` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/visual_prompt_pilot/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/visual_prompts.py;archive/implementation/code/analyze_visual_prompt_pilot.py`
- Config: `—`
