---
exp_number: "017"
experiment_id: "reva-phase4-qcsrr"
family: "grounding_geometry"
phase_label: "Phase 4"
title: "QCSRR fixed-frame ranking gate"
decision_status: "led_to_phase5_only"
execution_status: "completed"
reproducibility_status: "report_local"
parent_experiment_id: "reva-phase3b-gates"
baseline_experiment_id: ""
---

# EXP-017: QCSRR fixed-frame ranking gate

## Decision

`led_to_phase5_only`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

QCSRR fixed-frame ranking Hit@k. Retrieval diagnostic, not letter-logit VideoQA.

## Method

Frozen frame-pool Hit@k diagnostics for vocabulary vs question-conditioned retrieval.

Protocol notes: family `grounding_geometry`; phase `Phase 4`; parent `reva-phase3b-gates`; compare-to `none`. Execution `completed`; reproducibility `report_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase4_qcsrr/PHASE4_FINAL_REPORT.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/jobs/modal_phase4_qcsrr.py;archive/implementation/jobs/modal_phase4_frame_eval_rankings.py`
- Config: `—`
