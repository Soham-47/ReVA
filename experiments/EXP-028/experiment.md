---
exp_number: "028"
experiment_id: "reva-phase8-qwen38-h200"
family: "alternative_backbones"
phase_label: "Phase 8"
title: "Qwen3.8-27B Stage-A-style scoring (partial valT)"
decision_status: "pending_full_comparison"
execution_status: "partial"
reproducibility_status: "valQ_complete_valT_prefix"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-028: Qwen3.8-27B Stage-A-style scoring (partial valT)

## Decision

`pending_full_comparison`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Backbone swap: Qwen3.8-27B with the same four-view legacy-frame letter scoring. valT locally incomplete.

## Method

Four-view legacy-frame scoring with Qwen3.8-27B; complete valQ and partial valT in local checkout.

Protocol notes: family `alternative_backbones`; phase `Phase 8`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `partial`; reproducibility `valQ_complete_valT_prefix`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | overall_accuracy_tta | 506/595 | 498/595 | `full_split_tta` |
| valT | overall_accuracy_tta | 558/658 | 885/1059 | `partial_split` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase8_qwen38_27b_r16_h200_checkpoint1641/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/phase8_scorer.py;archive/implementation/code/phase8_analysis.py;archive/implementation/code/phase8_adapters.py;archive/implementation/jobs/phase8_score.sbatch`
- Config: `code/configs/stageA_qwen38_a100.yaml`
