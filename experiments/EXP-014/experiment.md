---
exp_number: "014"
experiment_id: "reva-phase3-explicit-geometry"
family: "grounding_geometry"
phase_label: "Phase 3 explicit geometry"
title: "Explicit geometry G0/G1/G2 (30 questions per split)"
decision_status: "closed_no_promotion"
execution_status: "completed"
reproducibility_status: "report_local;stale_status_json"
parent_experiment_id: "reva-stage-a-legacy"
baseline_experiment_id: "reva-stage-a-legacy"
---

# EXP-014: Explicit geometry G0/G1/G2 (30 questions per split)

## Decision

`closed_no_promotion`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Selected geometry errors plus controls with explicit box or mask context (G0/G1/G2).

## Method

Selected geometry errors plus controls with box or mask context.

Protocol notes: family `grounding_geometry`; phase `Phase 3 explicit geometry`; parent `reva-stage-a-legacy`; compare-to `reva-stage-a-legacy`. Execution `completed`; reproducibility `report_local;stale_status_json`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase3_explicit_geometry/full/phase3_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/grounding.py;archive/implementation/code/replicate_sam3.py;runs/phase3_explicit_geometry/`
- Config: `—`
