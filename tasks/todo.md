# Implementation checklist: Evidence-Grounded ReVA Inference

> **Status note (2026-10-21):** This checklist preserves the original
> implementation plan. Use the [experiment ledger](../docs/experiment_ledger.md)
> for verified run state and caveats. Adaptive-frame selection is closed
> without promotion; the selected Grounding DINO pilot tied the same-ID Stage A
> baseline; Phase 7 evidence validation failed its downstream gate. Phase 8 has
> complete Qwen3.8 `valQ`, partial `valT`, and a validated H100 test ZIP from job
> `407542`, but no complete paired `valT` comparison. The user reported a new
> 83.75% result over 4,000 QA pairs; its score artifact and adapter lineage are
> not in this checkout. The ZIP has not been uploaded to Codabench. The checkout
> has no usable Git history, so no commit-relative status is implied.

## Phase 1 — freeze interfaces and baselines

- [ ] Confirm model licenses, hardware compatibility, and offline cache path for the frame encoder and detector.
- [x] Fix candidate-frame density/cap, frame-cache size, and added preprocessing budget for the recorded adaptive selector.
- [x] Define separate output directory and complete manifest fields for selector and grounding versions.
- [x] Add pure helpers and deterministic tests without changing existing scorer defaults.
- [x] Confirm disabled options reproduce existing input construction and preserve the recorded Stage A baseline outputs.

## Phase 2 — adaptive frame selection

- [x] Cache candidate-frame features once per unique video.
- [x] Select 16 frames using uniform temporal anchors plus query-relevant frames with a spacing constraint.
- [x] Preserve broad time coverage and paired observations for motion/change questions.
- [x] Run baseline and adaptive-only on the same deterministic stratified pilot.
- [x] Inspect selections, error repairs/regressions, runtime, and cache size.
- [x] Close the adaptive audit: it scored 1,745/2,000 on `val` versus 1,757/2,000 for baseline and is not promoted.
- [x] Run the held-out audit documented in `results/error_review/adaptive_temporal_geometry_audit.md`.

## Phase 3 — object grounding and tracking

- [x] Add an offline-only Grounding DINO detector cache with a pinned model revision and separate cache identity.
- [x] Validate the staged detector snapshot against immutable HF file digests; the revision marker is provenance only.
- [x] Cache detector outputs once per unique baseline frame and reuse complete per-video caches.
- [x] Associate detections into deterministic adjacent-frame tracks and preserve confidence, timestamps, and normalized 2D boxes.
- [x] Add bounded, question-relevant, explicitly uncertain track context while retaining original video frames.
- [x] Expand the bounded default prompt to cover the pilot's courts, fields, parking lots, pools, barns, and boats; record its vocabulary limits.
- [x] Freeze the deterministic 16-question-per-split stratified pilot in `tasks/phase3_grounding_pilot_ids.json`.
- [x] Run grounding-only on deterministic pilot QA/video IDs; score JSONL and manifests are present for 16 IDs per split and record the pinned detector revision.
- [ ] Hand-check object misses, false positives, track identity, and camera-motion cases; no standalone detector-quality audit is recorded for this pilot.
- [x] Compare grounding-only against the same-ID baseline: `valQ` 12/16 and `valT` 11/16, tied on both selected splits.
- [ ] Record the pilot's added runtime and cache size; these fields are absent from the grounding score manifests.
- [ ] Run combined adaptive-plus-grounding only if grounding passes the pilot gate.
- [ ] Run full held-out splits only if the pilot passes the agreed gate.

### Phase 3 pilot gate

- Target task accuracy improves on the fixed pilot.
- Paired overall accuracy does not materially decline and controls show no systematic regressions.
- Detector misses, false positives, and camera-motion associations are acceptable after hand review.
- No context or conclusion treats normalized image boxes as physical depth or 3D distance.

## Phase 4 — comparison and report

- [x] Compare per-question predictions, overall accuracy, and preselected task metrics for the completed pilot and adaptation records.
- [ ] Compute paired bootstrap intervals and report split-level results separately.
- [ ] Freeze the candidate before one official-validation evaluation with the intended adapter.
- [x] Document runtime, storage, failure cases, and promotion decisions for the completed records in the experiment ledger.

## Current status summary

| Workstream | Status | Evidence |
| --- | --- | --- |
| Adaptive frames | Complete; not promoted | [`results/error_review/adaptive_temporal_geometry_audit.md`](../results/error_review/adaptive_temporal_geometry_audit.md) |
| Grounding/segmentation pilots | Diagnostics exist; no broad promotion | [`docs/experiment_ledger.md`](../docs/experiment_ledger.md) |
| Crop evidence ablation | Complete; retrieved crops tied video-only baseline on 12-question diagnostic | [`runs/phase5_crop_ablation/PHASE5_FINAL_REPORT.md`](../runs/phase5_crop_ablation/PHASE5_FINAL_REPORT.md) |
| A1–A4 visual adaptation | Complete; no-go on the recorded `valT` gate | [`runs/phase6_A4_language_plus_vision_and_merger_from_stageA/phase6b_final_report.md`](../runs/phase6_A4_language_plus_vision_and_merger_from_stageA/phase6b_final_report.md) |
| Phase 7B target evidence | Complete; downstream evidence gate failed | [`runs/phase7B_cache_validation/report.md`](../runs/phase7B_cache_validation/report.md) |
| Phase 8 backbone comparison and test package | Qwen3.8 `valQ` is complete (506/595, 85.04%) and `valT` partial (558/658, 84.80%); no complete paired `valT` comparison or Qwen3-VL-32B output. Job `407497` status is an old snapshot, not live. H100 test ZIP from job `407542` is validated locally but not submitted. User-reported 4,000-pair accuracy is 83.75%; its underlying score artifact and model lineage still need linking. The ZIP and raw scores remain local-only. | [`docs/experiment_ledger.md`](../docs/experiment_ledger.md), local score path `runs/phase8_qwen38_27b_r16_h200_checkpoint1641/`, local ZIP `submissions/phase8_qwen38_h100_407542/submission.zip` |

Unchecked items above should be resumed only after the relevant gate and
artifact requirements in the ledger have been met.
