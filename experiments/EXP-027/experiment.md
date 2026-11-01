---
exp_number: "027"
experiment_id: "reva-phase7c-replicate-tracking"
family: "tracking"
phase_label: "Phase 7C"
title: "SAM3 / GroundingDINO / SAM2 tracking feasibility"
decision_status: "gate_failed"
execution_status: "completed"
reproducibility_status: "diagnostic_local"
parent_experiment_id: "reva-phase7b-source-evidence"
baseline_experiment_id: ""
---

# EXP-027: SAM3 / GroundingDINO / SAM2 tracking feasibility

## Decision

`gate_failed`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Two-video Replicate diagnostic of SAM3 / GroundingDINO / SAM2 tracking continuity.

## Method

Two-video Replicate diagnostic for mask continuity and re-anchoring.

Protocol notes: family `tracking`; phase `Phase 7C`; parent `reva-phase7b-source-evidence`; compare-to `none`. Execution `completed`; reproducibility `diagnostic_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase7C_replicate_smoke/report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/phase7c_replicate_runner.py;archive/implementation/code/phase7c_replicate_diagnostics.py`
- Config: `—`
