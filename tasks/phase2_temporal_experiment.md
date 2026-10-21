# Approved Phase 2: Question-Aware Temporal Evidence

**Status:** Approved by the user on 2026-10-17.  
**Scope:** Complete the planned Phase 0 error/control review, then run the bounded inference-only Phase 2 comparison.  
**Reference:** `/home/soham/Downloads/ReVA_Geometric_Temporal_Improvement_Plan.md`.

## Frozen reference

- Stage A Qwen3-VL-8B-Instruct LoRA adapter: `runs/stageA_q3vl8b_r16/final_adapter`.
- Baseline: `runs/stageA_q3vl8b_r16/legacy/scores_valQ.jsonl` and `scores_valT.jsonl`.
- Model, adapter, dataset revision, question text, choices, four-view aggregation and labels remain fixed.
- New outputs go under `runs/phase2_temporal_coarsefine/`; no old outputs are overwritten.

## Phase 0 review gate

Review 20 baseline errors and 10 correct controls for each target task from `valT`: Temporal Grounding and Geometric Relation. Selection is uniform pseudo-random within the error/control pools, without replacement, with seeds recorded in `results/phase0_temporal_geometry_review/selection.json`.

For each selected item, classify the visible evidence and the likely failure source using the plan's T1–T7 and G1–G7 categories. Mark uncertain cases as unresolved. Do not change labels or exclude difficult cases from the Phase 2 score set.

## Phase 2 arms

| Arm | Frames/evidence | Comparison |
|---|---|---|
| T0 | Existing legacy sampler; 16 frames at each offset (0 and 0.5); two option rotations | Existing paired baseline |
| T1 | For each offset, a coarse global scan of 8 frames followed by 8 frames around a question-relevant candidate event window; final VLM input remains 16 frames. Use legacy sampling if localization confidence is low. Keep the answer prompt unchanged. | T1 vs T0 tests frame selection |
| T2 | Reuse T1's exact selected frames and add a compact timestamp evidence block with supported event brackets and uncertainty. | T2 vs T1 tests explicit timestamp evidence |

The same frozen Stage A model proposes windows from the video and question only. It does not receive answer choices or the gold answer during localization. No segmentation or external vision model is used.

## Question set

Score every Temporal Grounding question: 69 on `valQ` and 133 on `valT`, 202 paired items across 135 unique videos. Classify event forms before analysis (first appearance, disappearance/exit, duration/continuous visibility, event/camera onset or ordering); report localization coverage and fallback rate. T0 predictions are already saved.

## Evaluation and gate

- Report T0/T1/T2 exact-match accuracy separately on `valQ` and `valT`, plus pooled counts.
- For each pair, report fixes, regressions, wrong-to-wrong changes, unchanged predictions, all four answer probabilities, and per-question evidence.
- Recompute split-level overall accuracy by combining each new Temporal Grounding arm with unchanged baseline predictions for non-target tasks.
- Report paired uncertainty and keep validation reuse explicit. Review every changed answer against the source frames; separate suspected label/clip ambiguity.
- A variant is a Phase 2 candidate only if it gains at least 3 net Temporal Grounding answers on `valT`, is non-negative on `valQ`, and reduces overall accuracy by no more than 1 percentage point on either split. Otherwise report inconclusive/no-go. This is a pilot gate, not a claim of statistical proof.

## Resources and outputs

- Phase 0 staging: one bounded Modal CPU job to fetch and render only selected review clips; no GPU and no inference.
- Phase 2: one Modal A100 job using the existing 8-CPU/64-GB host-memory setup, with a 12-hour cap. No fine-tuning, SAM 3, Replicate calls, or HPC jobs.
- Approximate Phase 2 workload: 404 coarse localization calls, 1,616 answer-scoring forwards, and 16 final frames per offset per question. Run an eight-question timing/IO tranche first and report measured throughput before the full scoring pass.
- Save contact sheets, review classifications, selector timestamps, selected frame indices, prediction JSONL files, manifests, and paired reports under the new Phase 0 and Phase 2 directories.
