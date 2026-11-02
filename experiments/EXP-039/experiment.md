---
exp_number: "039"
experiment_id: "reva-dse-vtg-brev"
family: "training_free_decoder"
phase_label: "DSE-VTG gated fusion"
title: "Qwen3-VL-Embedding MSF + gated letter fusion on TG"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_report_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-039: DSE-VTG gated fusion (Brev)

## Decision

`not_promoted`

## Method

`Qwen/Qwen3-VL-Embedding-8B` at ~3 fps, MSF over frame similarities, light query-TTA (two paraphrases). Option energy via REZE parser; fuse into Stage A letter logits only when margin < 0.12.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | Temporal Grounding | 51/69 | 51/69 | `task_subset` |
| valT | Temporal Grounding | 100/133 | 99/133 | `task_subset` |

Paired net delta: valQ **0**, valT **+1**.

## Artifacts

`runs/dse_vtg_brev/`

## Code

`archive/implementation/code/dse_vtg.py`, `dse_vtg_embedder.py`, `score_dse_vtg.py`, `jobs/brev_score_dse_tg.sh`
