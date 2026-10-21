# Implementation Plan: Evidence-Grounded ReVA Inference

> **Status note (2026-10-21):** This file is the historical design record. The
> adaptive-frame arm is closed without promotion; later segmentation, retrieval,
> crop, visual-adaptation, and tracking results are recorded in the
> [experiment ledger](../docs/experiment_ledger.md). The `.git` directory in
> this checkout is empty, so no since-commit boundary is asserted here.

## Goal

Test two inference-time changes against the existing Qwen3-VL-8B LoRA system: (1) query-guided frame selection while retaining coverage of the whole video, and (2) object grounding with compact detections and tracks. Do not retrain the model in this experiment. Keep the current scorer and predictions intact; write new outputs and caches under `$WORK` with separate manifests.

## Current pipeline and constraints

- `code/common.py` provides fixed 16-frame sampling (`legacy` and `endpoints`), video decoding, and prompt/model-input construction.
- `code/score.py` scores answer letters from model logits with two temporal offsets and two option rotations (four Qwen forwards per question by default). It already records a manifest and rejects mismatched/resumed runs.
- `code/evaluate.py` reports overall, task, category, and source accuracy, and writes item-level predictions/errors.
- `code/configs/stageA.yaml` trains on `fit_A`; `valQ` and `valT` are held out from that adapter and are therefore appropriate development splits for an inference ablation. Stage B trains on all train questions, so its `valQ`/`valT` scores must not be used as held-out evidence.
- The scorer currently has no detector, tracker, frame-ranking model, or grounding-context schema. Adding one means caching a separately versioned vision model under `$WORK` and measuring its runtime and false detections.
- Preserve the current official-val and Codabench results as reference results. Do not use the hidden test score to tune the wrapper.

## Proposed design

### A. Query-guided frame selection

Build a bounded candidate-frame pool for each unique video and cache frozen image features once. For each question, rank candidate frames against the question's named entities or concise object phrases. Select 16 frames total: retain evenly spaced temporal anchors, then fill the remaining slots with high-relevance frames subject to temporal spacing. For motion/change questions, require coverage across the clip and preserve useful frame pairs; do not let a similarity score collapse the sample to one time region.

The VLM still receives at most 16 frames and uses the existing two offsets and two option rotations. This isolates frame choice without increasing Qwen forward count. Store selected source-frame indices and timestamps in each score record. The selector, feature-model revision, candidate rate/cap, and selection parameters must be recorded in the manifest.

### B. Object grounding and tracking

Run one pinned open-vocabulary detector on cached candidate frames, once per video, and cache class/phrase, confidence, bounding box, frame index, and timestamp. Associate detections across adjacent frames with a lightweight tracker so the scorer can distinguish a moving object from camera motion. Use question entities to select relevant tracks and pass a short, bounded, structured context beside the original video frames; do not replace the visual evidence with detector text.

Use image-relative coordinates explicitly (left/right/top/bottom in the displayed frame). Treat detector misses and false positives as fallible evidence. Do not infer metric depth or physical distance from 2D boxes alone; add a depth estimator only in a later experiment if error review shows that depth questions warrant it.

## Evaluation design

1. Save the existing Stage A `valQ` and `valT` predictions as immutable baselines. Use the Stage A adapter for development because it did not train on those splits.
2. Run a deterministic, task-stratified pilot containing temporal, geometric/spatial, object-count/location, and randomly sampled control questions. Compare four arms: baseline; adaptive frames only; grounding only; both together. Keep model revision, adapter, prompt, offsets, rotations, frame count, and answer scorer fixed.
3. Inspect selected frames/tracks and false detections on a small hand-checked sample before spending on a full split. Reject an arm that systematically misses the questioned object, breaks temporal coverage, or adds misleading context.
4. If an arm helps the target tasks without materially hurting controls, score the complete held-out `valQ` and `valT` with the Stage A adapter. Compare per-question changes and use paired bootstrap confidence intervals. Report each split and task separately; do not claim a general gain from the enriched pilot alone.
5. Freeze the method and run one final comparison on official validation for the intended Stage B adapter. Use that as a final check, not another tuning set. Do not submit to Codabench until the validation decision is documented.

### Phase 2 closure

The adaptive-frame audit is complete and the arm is not promoted. On `val`,
adaptive frames scored 1,745/2,000 (87.25%) versus 1,757/2,000 (87.85%) for
the canonical baseline. Temporal grounding fell from 75.94% to 74.06%; the
geometric task gained one answer (90.00% to 90.50%), which is too small and
unstable to use as a gate. The detailed paired audit is in
`results/error_review/adaptive_temporal_geometry_audit.md`. Subsequent grounding
and segmentation diagnostics used baseline frames; adaptive frames remain
excluded because no later promotion gate passed.

### Suggested advancement rule

Advance an arm only if the full held-out evaluation improves the targeted task(s), its paired overall accuracy is not materially worse, and the improvement is not driven by one split or a few examples. Set numerical margins before running the full ablation. A positive pilot alone is not a promotion criterion.

## Implementation sequence

### Phase 1: Freeze interfaces and baselines

- Define a versioned selector/grounding manifest and a separate output path; never reuse the existing `scores_*.jsonl` paths.
- Add pure functions for candidate selection, temporal diversity, entity/track filtering, and bounded grounding-text formatting.
- Add deterministic synthetic tests for frame coverage, no duplicate/invalid indices, stable tie-breaking, box coordinate normalization, and output-size limits.

### Checkpoint 1

- Existing `score.py` defaults and score manifests remain unchanged.
- The new path can be invoked with selector and grounding both disabled and reproduces the old score inputs exactly.

### Phase 2: Adaptive frame selector

- Add candidate-feature caching and a query-conditioned, temporally diverse 16-frame selection mode.
- Verify cached features are keyed by video checksum, encoder revision, candidate policy, and image preprocessing.
- Run the stratified pilot for adaptive-only versus baseline; inspect selected frames and error repairs/regressions.

### Checkpoint 2

- Frame budget and Qwen forward count match baseline.
- Full held-out `valQ`/`valT` is run only if the pilot passes the predeclared quality/runtime gate.

### Phase 3: Object grounding and tracking

- Pin and cache one detector; measure per-clip processing cost and detection quality before integrating scoring.
- Use `IDEA-Research/grounding-dino-tiny` at HF commit `a2bb814dd30d776dcf7e30523b00659f4f141c71` through the native Transformers zero-shot object-detection API. Stage the roughly 689 MB snapshot explicitly; no builder command downloads weights.
- Validate every required staged file against the immutable HF tree metadata and safetensors SHA-256; the local revision marker records provenance but cannot make modified files valid.
- Cache detections and tracks by video checksum, detector revision, normalized prompt, preprocessing, thresholds, baseline frame policy, and tracker settings.
- Associate only adjacent sampled frames with deterministic same-label greedy IoU matching. Preserve confidence, source timestamps, and normalized 2D boxes; mark single-observation tracks uncertain.
- Pass compact question-relevant tracks in a dedicated prompt field while retaining the original video frames.
- Run the grounding-only arm on the checked-in 16-question-per-split stratified pilot in `tasks/phase3_grounding_pilot_ids.json`. Its bounded prompt covers recurring courts, fields, parking lots, pools, barns, and boats; colors, fine-grained classes, and unseen synonyms remain possible misses. The recorded pilot scored 12/16 on `valQ` and 11/16 on `valT`, exactly matching the same-ID Stage A baseline; no promotion decision is supported without the missing detector-quality/runtime audit. The combined adaptive-plus-grounding arm remains deferred.

### Checkpoint 3

- Hand-check representative detections, object associations, camera-motion edge cases, and false-positive cases.
- Confirm no spatial claim is presented as 3D/depth when only image coordinates were measured.

### Phase 4: Full comparison and report

- Score complete Stage A `valQ` and `valT` for any arm that passes the pilot.
- Produce overall/task tables, per-question changed-answer counts, paired confidence intervals, runtime, cache size, and representative failures.
- Freeze one candidate and evaluate it once on official validation with the Stage B adapter. Promote only if the agreed rule is met.

## Files likely to change

- `code/common.py`: reusable temporal-diversity and grounded-input helpers, only if these remain generic.
- `code/score.py`: opt-in selector/grounding arguments, versioned manifests, per-question evidence provenance.
- New `code/grounding.py` (or similarly small module): detector cache, track association, and compact context construction.
- `tests/test_pure_logic.py`: deterministic selection, schema, coordinate, and bounds checks.
- `README.md`: document the optional inference ablation and exact reproduction commands after implementation is approved.
- New Slurm wrapper only if the existing `jobs/score.sbatch` cannot run the optional path without changing its stable behavior.

## Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Query/frame similarity misses small or fast objects | No benefit or worse temporal score | Keep uniform anchors and temporal diversity; hand-check candidate selections; compare against baseline. |
| Detector false positives or missed small vehicles | Misleading context may reduce factual accuracy | Preserve raw frames, confidence-filter detections, inspect false positives, and test grounding-only separately. |
| Camera motion is mistaken for object motion | Wrong tracking context for drone footage | Store camera-relative and frame-relative motion separately; test on panning/forward-flight cases; omit uncertain tracks. |
| New model/cache increases setup and inference cost | HPC quota or storage pressure | Precompute once per clip; cap candidate frames; report actual runtime and cache size; stop if it exceeds the agreed budget. |
| Tuning repeatedly on official validation inflates the result | Misleading estimate of generalization | Use Stage A `valQ`/`valT` for development, then freeze and use official validation once. |
| Detector geometry is mistaken for true 3D geometry | Incorrect near/far conclusions | Treat boxes as 2D image coordinates; no physical depth claims without a separately validated depth model. |

## Open decisions before implementation

- The detector choice, revision, integrity manifest, and pilot ID manifest are fixed above. Confirm license, storage, Transformers/PyTorch compatibility, and the offline snapshot path on Sharanga before staging.
- Agree on maximum candidate frames per clip, disk-cache cap, and acceptable added preprocessing time.
- Set numeric promotion margins before running the complete ablation.

## Recorded status after the plan

The plan's intended phases have since produced these documented states:

| Area | Current state | Record |
| --- | --- | --- |
| Adaptive frame selection | Complete diagnostic; not promoted after `val` fell from 87.85% to 87.25% | [`adaptive_temporal_geometry_audit.md`](../results/error_review/adaptive_temporal_geometry_audit.md) |
| Grounding-only pilot | Complete selected diagnostic: 12/16 `valQ` and 11/16 `valT`, tied with same-ID Stage A; no promotion | `grounding_pilot_stageA_modal/` (local-only artifact: `../runs/grounding_pilot_stageA_modal/`), [`experiment_ledger.md`](../docs/experiment_ledger.md) |
| Segmentation and direct SAM 3 pilots | Selected-pilot diagnostics complete; no broad gain claim | [`experiment_ledger.md`](../docs/experiment_ledger.md) |
| Phase 7 cache/evidence gate | Complete and failed downstream evidence gate because no correct, independently assessable target track was available | [`phase7B_cache_validation/report.md`](../runs/phase7B_cache_validation/report.md) |
| Current reference | Frozen Stage A legacy scorer and adapter remain the comparison baseline | [`stageA_q3vl8b_r16/legacy/`](../runs/stageA_q3vl8b_r16/legacy/) |

Read the ledger before treating any unchecked design item above as an active
next step; plans describe intended workflows and preserve historical decisions.
