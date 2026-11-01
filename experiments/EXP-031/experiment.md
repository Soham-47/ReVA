---
exp_number: "031"
experiment_id: "reva-reference-air-temporal-audit"
family: "temporal"
phase_label: "Temporal Grounding audit"
title: "Reference A.I.R. selector on Temporal Grounding"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "scores_and_manifest_local;gpu_instance_deleted"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-031: Reference A.I.R. selector on Temporal Grounding

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Replaces uniform 16-frame selection with the reference A.I.R. selector on Temporal Grounding val (320 items); Qwen adapter unchanged.

## Method

Reference A.I.R. selector with the ReVA Qwen3-VL-8B adapter scored 211/320 Temporal Grounding val items (65.94%), below U16 at 235/320 (73.44%); the video-cluster interval for the −7.50 pp paired difference was [−12.46, −2.68].

Protocol notes: family `temporal`; phase `Temporal Grounding audit`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `scores_and_manifest_local;gpu_instance_deleted`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| val_temporal_grounding | temporal_grounding_task_accuracy | 211/320 | — | `task_subset` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/air_temporal_audit/report.md;runs/air_temporal_audit/official_air_qwen3/scores_temporal.jsonl`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `runs/air_temporal_audit/run_temporal_audit.py;runs/air_temporal_audit/analyze_temporal_audit.py`
- Config: `runs/air_temporal_audit/config.json`
