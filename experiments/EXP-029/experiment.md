---
exp_number: "029"
experiment_id: "reva-phase8-test-package-407542"
family: "alternative_backbones"
phase_label: "Phase 8 test"
title: "H100 test submission package (job 407542)"
decision_status: "pending_provenance"
execution_status: "prepared_not_submitted"
reproducibility_status: "zip_local_no_score_jsonl"
parent_experiment_id: "reva-phase8-qwen38-h200"
baseline_experiment_id: "reva-prior-private-test"
---

# EXP-029: H100 test submission package (job 407542)

## Decision

`pending_provenance`

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

Submission ZIP from the Phase 8 line (job 407542). Not an architecture experiment.

## Method

Validated 4,000-answer ZIP; not uploaded; lineage not fully traced locally.

Protocol notes: family `alternative_backbones`; phase `Phase 8 test`; parent `reva-phase8-qwen38-h200`; compare-to `reva-prior-private-test`. Execution `prepared_not_submitted`; reproducibility `zip_local_no_score_jsonl`.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |
| — | — | no local metric row | — | `unscored` |

Always quote numerator/denominator. Do not mix valQ, valT, val, test, or selected pilots.

## Artifacts

`submissions/phase8_qwen38_h100_407542/submission.zip`

Score JSONL under `runs/` is never overwritten.

## Code

- Active core (every future Stage A/B run): `code/train.py`, `code/score.py`, `code/evaluate.py`, `code/common.py`, `code/configs/stageA.yaml`
- This experiment: `code/make_submission.py;archive/implementation/jobs/phase8_score.sbatch`
- Config: `—`
