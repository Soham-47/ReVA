---
exp_number: "012"
experiment_id: "reva-sam2-segmentation-pilot"
family: "grounding_geometry"
phase_label: "SAM2 pilot"
title: "SAM 2 mask overlay pilot (32 questions)"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "report_and_summary_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-012: SAM 2 mask overlay pilot (32 questions)

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Adds SAM 2 mask overlays (via GroundingDINO) on 32 geometry-heavy questions.

## Method

GroundingDINO plus SAM2 masks on selected geometry-heavy questions.

Protocol notes: family `grounding_geometry`; phase `SAM2 pilot`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `report_and_summary_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/segmentation_pilot_stageA_modal/comparison_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/segmentation.py;archive/implementation/code/build_segmentation_cache.py;archive/implementation/jobs/modal_segmentation_pilot.py`
- Config: `—`
