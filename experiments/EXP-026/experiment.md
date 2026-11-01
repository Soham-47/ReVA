---
exp_number: "026"
experiment_id: "reva-phase7b-source-evidence"
family: "tracking"
phase_label: "Phase 7B"
title: "Source-evidence cache validation"
decision_status: "gate_failed"
execution_status: "completed"
reproducibility_status: "report_local"
parent_experiment_id: "reva-phase7a-manifest"
baseline_experiment_id: ""
---

# EXP-026: Source-evidence cache validation

## Decision

`gate_failed`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Validation of automatic source-evidence proposals on 24 reviewed cases. Gate, not a production overlay.

## Method

24 reviewed cases checking automatic proposals against source video.

Protocol notes: family `tracking`; phase `Phase 7B`; parent `reva-phase7a-manifest`; compare-to `none`. Execution `completed`; reproducibility `report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase7B_cache_validation/report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/phase7b_source_evidence.py;archive/implementation/code/validate_phase7_cache.py`
- Config: `—`
