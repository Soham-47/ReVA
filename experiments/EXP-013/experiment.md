---
exp_number: "013"
experiment_id: "reva-sam3-pilot"
family: "grounding_geometry"
phase_label: "SAM3 pilot"
title: "SAM 3 text-prompt segmentation pilots"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "pilot_reports_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-013: SAM 3 text-prompt segmentation pilots

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Adds SAM 3 text-prompt mask context on selected 32- and 17-question cohorts.

## Method

Replicate SAM3 mask context on 32-question and 17-question selected cohorts.

Protocol notes: family `grounding_geometry`; phase `SAM3 pilot`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `pilot_reports_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/sam3_pilot_stageA_replicate/comparison_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/replicate_sam3.py;archive/implementation/code/sam3_overlay.py;archive/implementation/jobs/replicate_sam3_pilot.py;archive/implementation/jobs/modal_sam3_score_pilot.py`
- Config: `—`
