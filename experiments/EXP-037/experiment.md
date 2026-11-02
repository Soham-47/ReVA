---
exp_number: "037"
experiment_id: "reva-cotr-tg-burnin"
family: "training_free_overlay"
phase_label: "CoTR timestamp burn-in"
title: "CoTR visual time anchors on Temporal Grounding"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_report_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-037: CoTR visual time anchors on Temporal Grounding

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

## What this experiment changed

CoTR-style **top-right** `mm:ss` + `[t s]` glyphs burned into each sampled frame (distinct from EXP-032 bottom progress bar and EXP-008 text timestamps).

## Method

Temporal Grounding cohort only (`results/visual_prompt_pilot/reze_tg_ids.json`): valQ 69, valT 133. Scorer: `archive/implementation/code/score_stageA_pilots.py --cotr-timestamp-burnin`. Run on Brev A100 (`reva-temporal-tg-exp`).

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | Temporal Grounding | 48/69 | 51/69 | `task_subset` |
| valT | Temporal Grounding | 100/133 | 99/133 | `task_subset` |

Paired net delta: valQ **−3**, valT **+1**.

## Artifacts

`runs/cotr_tg_burnin/` (JSONL + `*_report.json`)

## Code

- `archive/implementation/code/visual_prompts.py` (`overlay_cotr_timestamp`)
- `archive/implementation/code/score_stageA_pilots.py`
- `archive/implementation/jobs/brev_score_cotr_tg.sh`
