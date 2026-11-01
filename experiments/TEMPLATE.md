---
exp_number: "000"
experiment_id: example
family: example
phase_label: example
title: Example
decision_status: example
execution_status: example
reproducibility_status: example
parent_experiment_id: ""
baseline_experiment_id: reva-stage-a-legacy
---

# EXP-000: Title

## Decision

One of: `active_reference` | `not_promoted` | `gate_failed` | `diagnostic` | `planned`.

## Shared architecture

UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16 → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax.

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py`.

## What this experiment changed

One paragraph: the single lever vs the parent/baseline (sampler, prompt, overlay, weights, backbone, or evaluation-only).

## Method

Protocol, cohort, and gate.

## Results

| Split | Metric | This EXP | Baseline | Kind |
| --- | --- | --- | --- | --- |

Always quote numerator/denominator. Do not mix valQ, valT, val, and test.

## Artifacts

Paths under `runs/` (never overwrite score JSONL).

## Code

Active `code/` vs `archive/implementation/` paths used to produce the artifacts.
