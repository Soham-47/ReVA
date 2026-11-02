---
exp_number: "038"
experiment_id: "reva-patk-tta-reweight"
family: "training_free_postprocess"
phase_label: "PATK view reweight"
title: "Phase-aligned TTA view reweighting on Temporal Grounding"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_report_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-038: Phase-aligned TTA view reweighting on Temporal Grounding

## Decision

`not_promoted`

## Shared architecture

Unchanged Stage A inference; post-hoc reweight of the four TTA views by inverse distance from sampled frame times to option time anchors.

## What this experiment changed

CPU-only PATK-style fusion over existing `views` in Stage A JSONL (no rescore, no new frames).

## Method

Same TG cohort as EXP-037. Script: `archive/implementation/code/patk_reweight.py`. Weights: `1 / (1 + min_t min_anchor |t − anchor|)` per view.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | Temporal Grounding | 51/69 | 51/69 | `task_subset` |
| valT | Temporal Grounding | 99/133 | 99/133 | `task_subset` |

No change vs letter-logit mean on either split.

## Artifacts

`runs/patk_tg/` (`*_patk.jsonl`, `*.report.json`)

## Code

- `archive/implementation/code/patk_reweight.py`
