---
exp_number: "011"
experiment_id: "reva-grounding-dino-pilot"
family: "grounding_geometry"
phase_label: "grounding pilot"
title: "GroundingDINO box-context pilot (16 questions per split)"
decision_status: "not_promoted"
execution_status: "completed"
reproducibility_status: "local_score_jsonl"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-011: GroundingDINO box-context pilot (16 questions per split)

## Decision

`not_promoted`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Adds GroundingDINO boxes as extra visual context on a 16-question-per-split pilot. Core 16-frame TTA otherwise unchanged.

## Method

Detector boxes as extra visual context on fixed 16-question pilots.

Protocol notes: family `grounding_geometry`; phase `grounding pilot`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `local_score_jsonl`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/grounding_pilot_stageA_modal/;runs/segmentation_pilot_stageA_modal/comparison_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/grounding.py;archive/implementation/code/build_grounding_cache.py;archive/implementation/jobs/modal_grounding_pilot.py`
- Config: `—`
