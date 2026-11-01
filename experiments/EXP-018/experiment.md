---
exp_number: "018"
experiment_id: "reva-phase5-crop-e0"
family: "crop_ablation"
phase_label: "Phase 5 E0"
title: "Phase 5 video-only arm"
decision_status: "reference_arm"
execution_status: "completed"
reproducibility_status: "report_local"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: ""
---

# EXP-018: Phase 5 video-only arm

## Decision

`reference_arm`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Phase 5 video-only arm on the crop-ablation cohort. Shared Qwen path without crop evidence.

## Method

Video-only baseline within Phase 5 cohort.

Protocol notes: family `crop_ablation`; phase `Phase 5 E0`; parent `reva-stage-a-legacy`; compare-to `none`. Execution `completed`; reproducibility `report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase5_crop_ablation/PHASE5_FINAL_REPORT.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/jobs/phase5_qwen_crop_ablation_hpc.py`
- Config: `—`
