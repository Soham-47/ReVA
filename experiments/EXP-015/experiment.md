---
exp_number: "015"
experiment_id: "reva-phase3a-retrieval-smoke"
family: "grounding_geometry"
phase_label: "Phase 3A"
title: "SAM3 proposal + SigLIP2 retrieval smoke (2 videos)"
decision_status: "exploratory"
execution_status: "completed"
reproducibility_status: "smoke_local"
parent_experiment_id: ""
baseline_experiment_id: ""
---

# EXP-015: SAM3 proposal + SigLIP2 retrieval smoke (2 videos)

## Decision

`exploratory`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

No Qwen path: SAM3 proposal + SigLIP2 retrieval smoke on 2 videos.

## Method

Category-level retrieval diagnostic without Qwen rescoring.

Protocol notes: family `grounding_geometry`; phase `Phase 3A`; parent `none`; compare-to `none`. Execution `completed`; reproducibility `smoke_local`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`runs/phase3a_retrieval_smoke/review/manual_audit/diagnostic_report.md`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `archive/implementation/code/phase3a_retrieval.py;archive/implementation/jobs/hpc_phase3a_retrieval_smoke.sbatch`
- Config: `—`
