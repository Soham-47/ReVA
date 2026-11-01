---
exp_number: "025"
experiment_id: "reva-phase7a-manifest"
family: "tracking"
phase_label: "Phase 7A"
title: "Canonical frame manifest"
decision_status: "infrastructure"
execution_status: "completed"
reproducibility_status: "manifest_local_only"
parent_experiment_id: ""
baseline_experiment_id: ""
---

# EXP-025: Canonical frame manifest

## Decision

`infrastructure`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Infrastructure: canonical frame index/timestamp manifest for 1,047 videos. Not a scorer change.

## Method

Frame index and timestamp provenance for 1,047 videos.

Protocol notes: family `tracking`; phase `Phase 7A`; parent `none`; compare-to `none`. Execution `completed`; reproducibility `manifest_local_only`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase7A_feasibility/`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/phase7_manifest.py`
- Config: `—`
