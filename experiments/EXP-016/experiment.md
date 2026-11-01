---
exp_number: "016"
experiment_id: "reva-phase3b-gates"
family: "grounding_geometry"
phase_label: "Phase 3B"
title: "Retrieval annotation gates 1–2"
decision_status: "no_qwen_promotion"
execution_status: "completed"
reproducibility_status: "gate_reports_local"
parent_experiment_id: "reva-phase3a-retrieval-smoke"
baseline_experiment_id: ""
---

# EXP-016: Retrieval annotation gates 1–2

## Decision

`no_qwen_promotion`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Frozen SigLIP2 ranking gates with manual annotation checks. No Qwen rescoring.

## Method

Frozen SigLIP2 rankings with manual annotation feasibility checks.

Protocol notes: family `grounding_geometry`; phase `Phase 3B`; parent `reva-phase3a-retrieval-smoke`; compare-to `none`. Execution `completed`; reproducibility `gate_reports_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase3b_gate1/scientific_evaluation_2026-10-18.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/jobs/modal_phase3b_gate1.py;runs/phase3b_gate1/`
- Config: `—`
