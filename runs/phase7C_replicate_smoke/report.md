# Phase 7C — Replicate segmentation and tracking feasibility report

## Scope

Two 180-frame, 640×360, 30 fps videos were processed: `train_000047` (archway appearance) and `train_001138` (marked white van). The fixed-vocabulary controls were `vehicle`, `building`, and `road`. The run tested the existing merged-mask endpoint, an ID-preserving SAM 3 endpoint, short-clip text-prompt variants, Grounding DINO automatic boxes, and SAM 2 tracking with both one initial click and seven automatically generated same-ID re-anchors. It did not evaluate a broad cohort or downstream Qwen answers.

## Findings

| Case | Result | Interpretation |
|---|---|---|
| Archway, question-conditioned ID-preserving SAM 3 | One persistent label from frame 104 to 179 (3.47–5.97 s), best mask 4,240 px (1.84% of frame), box `[207,10,383,92]`. | Mask covers the archway, but begins 14 frames / 0.47 s after the source review's first recognizable frame at 3.0 s. The Chinese characters cannot be read at 640×360. |
| Marked white van, question-conditioned SAM 3 | One 641 px mask at frame 49 (1.63 s), box `[452,349,529,359]`; visually the wrong vehicle/edge fragment. | The actual marked van is already visible in source frame 0, so the “first appears” event is left-censored and cannot be timed from this clip. The target instance was missed. |
| White-van text sensitivity, first 12 frames | Exact phrase: 0 objects. `white van`: 1 track, but it masks another small vehicle (best box `[403,137,442,156]`). `white vehicle with blue side markings`: 0 objects. | Wording changes do not recover the target. All three archives contain 12 valid 16-bit masks mapped to source frames 0–11 at 30 fps. |
| Automatic Grounding DINO seed at frame 0 | Four detections, including a `white van` box `[88,274,163,358]` at confidence 0.555 covering the marked van; two detections overlap that same target region. | Automatic box localization is feasible, but duplicate proposals need deduplication. No manual coordinates or QA answer selected the box. |
| Grounding DINO box → SAM 3 video point seed | Full-video output again contains only the frame-49 mask. It is pixel-identical (IoU 1.0) to the text-only merged-mask output. | This pinned video-wrapper path did not usefully propagate the automatic seed. The schema advertises `visual_prompt`, but runtime behavior did not improve the target mask. |
| Grounding DINO → SAM 2 multi-frame re-anchoring | Seven positive points at frames 0, 3, 6, 9, 12, 15, and 18 were generated automatically by nearest-center association of question-conditioned DINO boxes. SAM 2 returned 180 binary masks at 640×360 / 30 fps. Nonempty masks occur on frames 0–9, 12, 15, and 18. | Re-anchoring restores nonempty masks at frames 9, 12, 15, and 18; the single-click track had a 139 px edge fragment at frame 9 and was empty afterward. The new masks are 1,700 px at frame 9, 2,034 px at 12, 1,769 px at 15, and 752 px at 18. However, frames 10–11 remain empty, and the same physical instance is not independently verified across the gaps. This is evidence of improved sparse mask recovery, not validated continuous identity tracking. |

The source-review first-visible bracket for the archway is frames 84–90 (2.8–3.0 s); the question-conditioned track starts at frame 104. The van's first-visible time is censored at frame 0, so no first-appearance accuracy is reported.

The fixed `building` prompt masks a broad adjacent building region rather than the archway. The fixed `vehicle` prompt creates many fragments on the archway clip and failed at the provider on the van clip for both SAM 3 endpoints.

## Frame mapping and mask validation

Fourteen available archives from the original 16-request batch now pass the updated offline validator. The two archived merged-mask question outputs are sparse: archway masks are present only on frames 106–179; van masks only on frame 49. Their overlay videos have 180 frames at 30 fps, with 640×360 masks and an overlay padded to 640×368. The indices can be joined to the canonical source timeline, but the merged-mask endpoint supplies no explicit source timestamps, so its absolute timestamp mapping remains unverified at the endpoint level. The ID-preserving endpoint's 180 labels and `objects.json` agree with the source frame count and fps.

The original execution manifest remains unchanged; recovered raw archives and the post-hoc validation ledger are preserved separately.

## SAM 2 mask validation and frame mapping

All 180 SAM 2 outputs validate as 640×360 binary PNG masks with values `{0,255}`. Output index `i` maps to source frame `i` at `i/30` seconds; this matches the full 180-frame input and is consistent with the initial click and sampled source overlays. The multiseed request reports `predict_time` 15.34 s and `total_time` 33.95 s. The two empty frames (10–11) and the shrinking edge masks at frames 15–18 prevent reliable first/last-visibility estimates. The original source review confirms the target is already visible at frame 0, so “first appears” remains left-censored.

The multi-seed clicks and all detector boxes are inference-time automatic outputs derived from the question and source frames. No human target coordinates, QA answer, or manually corrected box was used to generate them. Human interpretation was used only after inference to review mask placement. The source review did not independently establish cross-frame identity, so the report does not count the later masks as verified target-track observations.

## Provider use and limits

Across the Phase 7C run artifacts, there are 32 unique prediction IDs: 27 provider successes and 5 provider failures. The manifests record 38 SDK create invocations, plus one extra local call blocked before reaching Replicate. There were three pre-provider network failures and four provider error responses without prediction IDs (one 429, one 500, one 422, and one earlier error whose status was not retained; its request used the same invalid video field). Provider metrics across the 32 IDs sum to 904.6 s `predict_time` and 1,304.3 s `total_time`; these are provider metrics, not wall-clock totals. The account endpoint was blocked by Cloudflare in this execution context, so the actual invoice and credits could not be checked. Replicate currently lists SAM 2 at approximately $0.026 per run; that is a planning estimate, not an actual charge. The records contain no per-run GPU model or VRAM telemetry. All inference in this Phase 7C diagnostic ran through Replicate; no HPC, Modal, H200, or local GPU inference was used.

## Decision

The SAM 3 video wrapper remains unsuitable as the instance-aware tracking path in this diagnostic: the marked vehicle is missed and a DINO-derived point did not change the wrapper’s merged mask. Grounding DINO followed by SAM 2 with repeated automatic positive points is more promising: it restores masks at several later anchor frames after the single-click track collapses. The result still has two missing frames and unverified identity through the gaps, so it does not establish reliable visibility or tracking. Keep this as a small feasibility result; do not run Qwen or DA3 on it. A next experiment should use a cleaner, uncensored target with independent identity annotations and compare DINO+SAM2 re-anchoring against the single-click baseline before integrating geometric code. The pinned SAM 2 schema documents same-object-ID points across frames and the `input_video` field; the runtime result is reported against that exact pinned endpoint.

## Artifacts

- Original request ledger: `manifest.json`
- Recovered archive validation: `diagnostics/recovered_output_validation.json`
- Provider counts and metrics: `execution_summary.json`
- Archway and van frame sheets: `diagnostics/train_000047_sam3_video_contact_sheet.jpg`, `diagnostics/train_001138_sam3_video_contact_sheet.jpg`
- Prompt comparison: `automatic_box_seed/text_prompt_variants_contact_sheet.jpg`
- Automatic detector visualization: `automatic_box_seed/result_image.png`
- Sparse detector re-anchoring boxes: `automatic_box_seed/temporal_dino_reanchoring_contact_sheet.jpg`
- SAM 2 multi-seed masks and comparison: `automatic_box_seed/sam2_multiseed_masks/`, `automatic_box_seed/sam2_multiseed_validation.json`, `automatic_box_seed/sam2_multiseed_comparison_contact_sheet.jpg`
- Multi-seed request manifest: `automatic_box_seed/sam2_multiseed_manifest.json`
- Reconciled provider usage: `diagnostics/provider_usage_reconciled.json`
- Automatic box and SAM 3 seed manifests: `automatic_box_seed/manifest.json`, `automatic_box_seed/sam3_video_seed_manifest.json`

Pinned endpoint schema: [SAM 2 video API](https://replicate.com/meta/sam-2-video/versions/33432afdfc06a10da6b4018932893d39b0159f838b6d11dd1236dff85cc5ec1d/api) and [SAM 2 model page, runtime/cost listing](https://replicate.com/meta/sam-2-video); [SAM 3 video API](https://replicate.com/lucataco/sam3-video/versions/408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269/api), [Grounding DINO API](https://replicate.com/adirik/grounding-dino/versions/efd10a8ddc57ea28773327e881ce95e20cc1d734c589f7dd01d2036921ed78aa/api). Current public wrapper source, which may differ from the pinned deployment: [visual-prompt path](https://github.com/lucataco/cog-sam3-video/blob/main/predict.py#L975-L1050).
