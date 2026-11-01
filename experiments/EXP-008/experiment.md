---
exp_number: "008"
experiment_id: "reva-phase2-temporal-coarsefine"
family: "temporal"
phase_label: "Phase 2"
title: "Coarse-to-fine temporal sampling (T1) and timestamp context (T2)"
decision_status: "gate_failed"
execution_status: "completed"
reproducibility_status: "report_local;paired_json_local_only"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-phase2-temporal-t0"
---

# EXP-008: Coarse-to-fine temporal sampling (T1) and timestamp context (T2)

## Decision

`gate_failed`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Inference sampler/prompt: coarse-to-fine temporal localization (T1) and timestamp context (T2) on the temporal-task subset.

## Method

Temporal-task-only rescoring with coarse-to-fine frame localization or explicit timestamp context.

Protocol notes: family `temporal`; phase `Phase 2`; parent `reva-stage-a-legacy`; compare-to `reva-phase2-temporal-t0`. Execution `completed`; reproducibility `report_local;paired_json_local_only`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| valQ | temporal_grounding_task_accuracy | 49/69 | 51/69 | `task_subset` |
| valT | temporal_grounding_task_accuracy | 98/133 | 99/133 | `task_subset` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase2_temporal_coarsefine/phase2_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/score_temporal.py;archive/implementation/code/evaluate_temporal.py;archive/implementation/jobs/modal_phase2_temporal.py`
- Config: `—`
