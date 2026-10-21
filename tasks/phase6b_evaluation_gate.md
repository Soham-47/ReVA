# Phase 6B A3/A4 evaluation gate

**Date recorded:** 2026-10-19 (Asia/Kolkata)  
**Arms:** A0, A1, A2, A3, A4  
**Primary comparison:** each candidate is compared with frozen A0 on the same questions and the same four-view protocol.

## Predeclared decision rule

- **Primary metric:** overall four-view (`offsets=[0.0, 0.5]`, `rots=2`) accuracy on `valT`.
- **Screening pass:** the candidate point estimate must exceed A0 valT accuracy. A positive point estimate alone is a screening signal.
- **Strong evidence:** the paired video-cluster 95% bootstrap interval for the valT accuracy delta must exclude zero. The interval is reported for uncertainty and does not change the screening rule.
- **No evidence to proceed:** a candidate that does not exceed A0 on the primary point estimate provides no evidence to proceed with vision LoRA.

## Secondary outcomes

Report the same four-view accuracy, single-view accuracy, gold probability, and gold margin for:

- `valQ` overall;
- the `Geometric Relation` and `Spatial Reasoning` slices;
- the `Temporal Grounding` and `Temporal Understanding` slices;
- `Object and Land Cover Recognition` slices;
- counting, when a labeled counting task exists in the split.

Task and category names are taken from the split metadata. Empty or absent slices are reported as absent rather than inferred.

## Guardrail and paired diagnostics

- **Category guard:** no top-level category with `n >= 50` may fall more than 1 percentage point versus A0, matching the historical Phase 6 tolerance. Report every category delta, including categories below the guard sample size.
- Report every candidate versus A0 fix and regression count, unchanged correct/wrong counts, and per-question transitions.
- Report candidate versus A1 and A2 transitions as secondary context, without changing the primary gate.
- Report video-cluster 95% intervals, task/category/source slices, gold probability and margin deltas, and per-question diagnostics.

## Compute and limitations

Record Slurm IDs, elapsed time, allocated GPU, peak memory/utilization and any available `TRESUsageInMax`/`TRESUsageInAve` fields for training and scoring. Report accounting data when available; do not invent a dollar cost when the cluster exposes no billing rate.

Interpretation remains descriptive. A paired transition or score change does not establish causation, and valQ/valT are development evaluations rather than held-out test generalization.

## Execution status

A4 was initially cancelled by the user at 16/200 optimizer steps before complete training or evaluation. At that time it was excluded from the scored-arm analysis; no resubmission was permitted without a new user instruction.

### Later user-authorized completion (2026-10-19)

The user subsequently requested “Try a4 full end to end,” explicitly authorizing completion. The original attempt had no resumable trainer checkpoint, so A4 was trained from step 0 under the unchanged configuration, then scored on both splits. Training job 403239 completed 200/200 steps on H200 `gpunode7`; scoring job 403248 completed all 595 valQ and 1,059 valT questions in merged-adapter mode. The predeclared gates and evaluation definitions above were not changed. Results and resource usage are in `runs/phase6_A4_language_plus_vision_and_merger_from_stageA/`.
