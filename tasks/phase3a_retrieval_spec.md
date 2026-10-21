# Phase 3A — QCSRR retrieval-quality pilot

**Status:** Phase 3A design approved; implementation/execution approval is limited to the two-video smoke. Execution remains gated: no rotated Replicate credential is available to the local runtime. The user waived monetary caps for this smoke only; the app-level limits remain 30 Replicate attempts (including up to 5 retries), one Modal A100 job capped at 30 GPU-minutes/60 wall-clock minutes, and no persistent volume. Modal and Replicate SDKs were installed in an isolated `/tmp` environment; no model/API call, Modal job, model-checkpoint download, or scoring has been started.  
**Goal:** test candidate proposal coverage and SigLIP2 region ranking on the existing stratified pilot, before any Stage A/Qwen answer ablation.

## 1. Approved design choices

- Use the existing `tasks/phase3_grounding_pilot_ids.json` unchanged: 16 questions from `valQ` and 16 from `valT`, selected by the recorded seed/strata. The 32 questions cover temporal grounding, geometric relation, object/count/location, and controls. They map to 30 unique videos (15 per split); reuse proposals for repeated clips.
- Keep the five already studied cases (`train_013068`, `train_012502`, `train_009393`, `train_009869`, `train_010397`) tagged as diagnostic case studies. Report their per-case observations separately and exclude them from any generalization-style aggregate. Their inclusion in the fixed ID list does not make them an unbiased sample.
- Keep Stage A frozen. For retrieval, use the exact 16-frame `legacy` sample at offset 0 and retain source-frame indices/timestamps. This is a retrieval-only stage, so no answer-view scoring runs here. If a later approved QA experiment proceeds, retain the original 16-frame Stage A sampling, both offsets, both option rotations, and all four scoring views.
- Use the pinned `google/siglip2-base-patch16-224` snapshot and score both padded RGB-box and mask-composited crops against question-only entity phrases. Do not use answer choices or gold answers in decomposition, proposal prompts, similarity text, crop selection, annotation, or evidence.
- Compare direct question-conditioned SAM 3 video prompts with a fixed, question-independent vocabulary bank using the same exact sampled frames, candidate post-processing, SigLIP2, and annotations.
- Limit any later Qwen evidence to four selected crops per question plus concise, timestamped uncertainty text. Do not score Qwen in Phase 3A.
- Use Modal for bounded GPU feature extraction and image/crop processing. Use Replicate's pinned SAM 3 video model for proposal calls. A separate call/cost gate is listed in §8.

## 2. Models and proposal arms

### SAM 3 video endpoint

Candidate endpoint: Replicate `lucataco/sam3-video`, pin the immutable version shown in the model example (`408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269`) in every run manifest. The current public page states an approximate `$0.0017` per run / 2 second typical runtime; cost varies with inputs. Recheck the pinned version/schema before execution and stop if it is unavailable or materially different. Do not silently switch to a floating `latest` version or a SAM 3.1 endpoint.

Each submitted video will contain exactly the same 16 sampled images used by the canonical retrieval view, in their original order. Preserve a sidecar mapping from packed-frame index to original frame index and timestamp; do not derive event times from the repacked clip's artificial frame rate. Use `return_zip=true` so evaluation receives frame masks rather than only a visual overlay. Preserve genuine model scores if returned. The existing ReVA SAM 3 cache writes `confidence: 1.0`, which is bookkeeping and must not be treated as confidence.

### Arm A — question-conditioned proposals

Apply the deterministic extractor to the question text only. Emit entity phrases with their original spans, optional attributes (such as `black`), and one reasoning operation (for example `distance comparison` or `first appearance`). Prompt SAM 3 video once for each distinct `(video checksum, extracted phrase, model version, frame policy)` key. Cache duplicate video/phrase calls across questions.

No LM/VLM proposes additional objects in this arm. If the deterministic extractor fails its manual audit, stop and bring a revised extraction design for approval instead of letting another model expand prompts implicitly.

### Arm B — fixed-vocabulary dense-like proposals

Use this fixed, question-independent bank on all 30 unique videos:

`vehicle`, `building`, `road`, `river`, `water`, `field`, `court`, `boat`, `tree`, `pool`.

This is a controlled concept-prompt bank, not prompt-free segmentation and not full SSA. It provides reusable broad proposals; SigLIP2 ranks the returned region crops against each question's entity phrase. The bank is frozen before inference. It deliberately may over-segment broad categories such as `field`; broad grass masks are measured as failure cases rather than relabeled as soccer/baseball fields.

### SigLIP2 ranking

Encode every retained proposal crop in two forms: (1) padded RGB box crop and (2) RGB crop with pixels outside the mask suppressed, with original color retained inside the mask. Encode lower-case, concise question-derived entity phrases with the pinned text encoder. L2-normalize features and rank using cosine similarity. Compare both crop forms against random selection from the exact same per-arm candidate pool.

Treat cosine similarity as a ranking score, never a detection probability. The prompt/semantic label attached to a SAM 3 mask is only the requested concept, not a verified ground-truth label. The manual audit supplies semantic-label correctness.

Compare SigLIP2 ranking with two baselines on the same candidate pool: (1) random candidate rankings (1,000 deterministic draws) and (2) SAM 3's returned native per-mask/proposal scores, wherever the response supplies usable scores. Report native-score availability and score semantics first; do not substitute the ReVA cache's bookkeeping `confidence: 1.0`, infer scores from mask order, or compare native scores across prompts unless the response schema supports that interpretation. If native scores are absent or not attributable to individual masks, mark this comparison unavailable. Keep proposal coverage separate from every ranking comparison.

## 3. Resolution and boundary handling

- Decode and preserve source-native pixels for the 16 selected frames. Do not globally enlarge frames with interpolation and describe the result as higher resolution; interpolation adds no image detail.
- During the one-to-two-video smoke, compare the normal source-native input against an overlapping tile input on one low-resolution/small-object case if the endpoint accepts it. Use tiles cut from the original pixels and map returned masks back to the original frame coordinates. Tiles should overlap by 25% so objects on tile borders remain represented. This is a feasibility probe, not a new arm in the 32-question ranking comparison.
- Report mask boundary quality separately from proposal coverage: for a predeclared subset of up to 20 visible targets spanning small vehicles, oblique buildings, and field boundaries, annotators draw a reference polygon and we report mask IoU and target-area coverage. Do not gate based on an upscaled image or a resized preview.
- If tiles do not fit the current endpoint or fail to improve boundary measurements, record that result. Do not add another segmentation model, train a head, or switch API providers without approval. A future approved boundary-refinement test may compare high-resolution tiled SAM 3 image proposals with another suitable promptable segmenter on the same polygons.

## 4. Manual annotation protocol

### Blinding and order

1. Extract question entities and operations from text alone. Before proposal outputs are shown, a reviewer checks whether every required entity, attribute, and operation is captured. Do not show the gold answer, options, baseline prediction, QCSRR ranks, similarity values, or arm identity.
2. For each distinct video/entity query, review all 16 original sampled frames in temporal order at native resolution. Record whether the target is visible, the target instance identity as described by the question, visible source-frame indices, and any ambiguity. For geometry items, identify each required entity and frames where all operands can be compared together.
3. Then inspect proposal masks over their source frames in randomized order with score/rank hidden. Mark each as correct target instance, same class/wrong instance, related region, background/overbroad, or miss. For every frame, annotate region-to-target matching independently. For multi-frame queries, separately verify whether masks in different frames depict the same physical object; record identity continuity, switches, fragmentation, occlusion/re-entry, and unverified cases. A SAM 3 track ID is a proposed association, not proof of identity.
4. Draw tight reference boxes for all target instances. Draw reference polygons for the boundary subset. A second reviewer independently labels 20% of entity queries (sampled with a fixed hash order); adjudicate disagreements and report agreement.
5. Freeze annotations and the per-arm candidate pools before computing ranking metrics. Keep annotator corrections in the audit record; do not feed correction labels or gold answers back into prompts, retrieval, or Qwen.

### Matching rules

- A proposal is a **coverage hit** only if the annotator judges it to be the requested semantic entity and the correct instance, and its mask covers at least 50% of the manually marked target region. Report mask IoU continuously as a separate boundary measure.
- Report **semantic-category retrieval** separately: a candidate counts if its manually verified category matches the requested entity, even when it is another instance. Report **correct-instance retrieval** only when the candidate matches the particular question-referenced object and meets the target-region coverage rule above. Queries with no uniquely identifiable instance remain eligible for category metrics but are excluded from instance metrics and counted as ambiguous.
- **Frame-level region matching** is evaluated per source frame and does not establish an object's identity across frames. Report **cross-frame identity** only on the subset with human-verified associations; give its denominator and report identity switches/fragmentation. Never merge these measures into one recall number.
- A **wrong-instance** mask of the right class is not a target hit. If wording cannot identify a single intended instance from video context, label the query ambiguous and exclude it from target-instance recall while reporting its count.
- For multi-entity geometry, report both per-entity results and **joint co-visible coverage**: all required entities have valid proposals in at least one common sampled frame. Also report joint top-5 retrieval, requiring every required entity to have a correct target instance in its top five at a compatible frame.
- For temporal entities, evaluate coverage over the full 16-frame sequence and note that first/last visibility is bracketed by the sample; do not invent finer timing than the sampled frames support.

## 5. Metrics and pre-registered gates

Report per proposal arm, crop style, split, task stratum, and diagnostic/non-diagnostic status. The primary unit is a manually assessable question-entity query; cluster intervals by unique video so repeated questions on the same video do not count as independent samples. Report semantic-category and correct-instance results separately. The five diagnostic cases remain separate from the main descriptive aggregate.

| Metric | Definition | Proposed gate before any Qwen scoring |
|---|---|---:|
| Decomposition audit | Correct required entity phrases, attributes, and operation against blinded human review | At least 95% entity coverage and 90% operation agreement; otherwise stop and revise extractor with approval |
| Proposal recall | Fraction of assessable entity queries with a correct-instance coverage hit anywhere among that arm's proposals | At least 0.80 overall, with no split below 0.70 |
| Semantic-category proposal recall | Fraction with any proposal of the correct manually verified category, regardless of instance | Report; no independent promotion gate |
| Joint proposal coverage | Fraction of assessable multi-entity geometry questions with every required entity proposed in one common frame | At least 0.65 |
| Recall@1 / Recall@5 | Fraction of entity queries whose correct target instance appears in the top K SigLIP2-ranked candidates | Recall@5 at least 0.75 |
| Precision@5 | Correct-target candidates among the five ranked candidates (when five exist); report actual K when fewer exist | At least 0.40 |
| Ranking over random | Paired absolute Recall@5 difference against random selection from the same candidate set, averaged over 1,000 deterministic random rankings | At least +0.15, and the lower bound of a 90% video-cluster bootstrap interval must be above 0 |
| Native SAM 3 score ranking | Same-pool semantic and correct-instance ranking using returned SAM 3 per-mask scores where usable | Compare with SigLIP2 and random; unavailable if valid per-mask scores are not returned |
| Joint top-5 retrieval | Multi-entity geometry questions where all required entities appear in top 5 at a compatible frame | At least 0.60 |
| Semantic label accuracy | Correct semantic class among auditable masks, judged manually (not prompt string equality) | At least 0.80 overall; report per-class counts and confusion categories |
| Boundary quality | IoU and target-region coverage for up to 20 preselected polygon cases | Report median, quartiles, and examples; descriptive only in this pilot |

Calculate proposal recall before ranking so a low ranking score is not mistaken for a proposal miss. Calculate ranking against the exact same proposal set for RGB-box and masked crops, native SAM 3 scores, and random baselines. Freeze thresholds above before viewing any QCSRR or QA outcome. All numerical gates are **provisional screening thresholds**, not validated cutoffs or claims of statistical significance. The 32-question cohort is selected and stratified for pilot coverage, includes repeated videos and five known diagnostic cases, and is not a probability sample; results cannot establish population performance or broad generalization. Report counts and uncertainty alongside every rate, separate diagnostic cases, and treat a gate pass only as evidence to propose the next experiment.

If both arms pass, report their paired coverage/ranking and cost trade-off; do not automatically promote one. If only one passes, present the result and request approval before using it in a Qwen experiment. If neither passes, stop before Qwen scoring.

## 6. Proposed smoke test

Run only under the approved two-video scope, after confirming the rotated Replicate credential and endpoint/model compatibility. Monetary caps are waived for this smoke only; the operational limits below remain mandatory, and actual provider billing must be reported when available. Local code cannot guarantee an account-level spend cap. Select two videos from the fixed pilot by predeclared criteria: one small/oblique object or building, one field-boundary or temporal class. The IDs will be recorded before API calls and selected without examining QCSRR results. The first video is the compatibility probe; the second checks repeatability and a different scene type.

Smoke actions (fixed call allocation, with at most 5 retry attempts inside the 30-attempt total):

1. On the first video, call all ten fixed prompts and up to four distinct question-derived phrases. On the second video, call three relevant fixed prompts chosen by a rule recorded before inference and up to four question-derived phrases. Request masks and record input/output frame counts, API duration, billed cost, failures, and mask count.
2. On the first video only, attempt the 25%-overlap native-pixel tile path for one preselected question phrase; it is four tile-sequence calls. The maximum planned smoke use is 25 predictions (10 + 4 + 3 + 4 + 4); the remaining five attempts are retry headroom. Cap at 30 total paid attempts. Stop the tile branch if its input format or returned coordinate mapping cannot be validated.
3. On Modal, load the already pinned SigLIP2 snapshot and embed smoke masks in batches for both crop styles. Compare SigLIP2, random ranking, and usable SAM 3 native scores on shared candidate sets. No new model weights, downloads from an unpinned source, or Qwen model load.
4. Confirm exact sampled-frame mapping, nonempty mask parsing, output schema, and crop-to-source-coordinate overlays. Inspect GPU peak memory, wall time, image batch size, feature-cache size, and cold-start cost.

The frozen smoke manifest currently plans 20 initial calls: video `train_010397` uses ten fixed prompts plus its two extracted question phrases; video `train_009869` uses three relevant fixed prompts plus its one extracted question phrase; the first video adds four tile-probe calls. The earlier 25-call maximum assumed up to four question phrases on each video (21 direct calls) plus four tile calls. This manifest uses only phrases actually extracted for the selected questions. Five retries remain within the 30-attempt hard attempt count; they are not part of the planned 20 calls. Both proposal arms are represented on both videos. Both SigLIP2 crop styles and the random baseline share their proposal pools. Native SAM 3 score ranking is attempted only if the response provides per-mask scores; the current `return_zip` parser sees mask PNGs without attributable scores, so it must report N/A if the schema remains that way.

### Modal smoke resources

- One Modal A100, 8 vCPU, 32 GiB RAM.
- Maximum 30 GPU-minutes and 60 wall-clock minutes for the smoke, whichever limit is reached first. No persistent volume is required.
- Monetary cap is waived for this two-video smoke only. Local code cannot guarantee provider billing or an account-level spend limit; report actual billed cost when available. The A100 resource and time limits remain mandatory operational limits.

### Replicate smoke budget

- Maximum 30 prediction attempts, including retries; at most 5 attempts are retry headroom. The public page estimate is only a planning reference because per-input charges vary. Monetary cap is waived for this two-video smoke only; the 30-attempt operational limit remains mandatory and actual billed cost must be reconciled.
- Replicate bills by model/hardware runtime and the model page says cost varies by input. Local code cannot guarantee provider billing or an account-level spend limit.

Smoke outcomes are a report to the user and include actual mask quality, source-frame mapping, category and instance retrieval examples, per-frame matches versus any human-verified cross-frame identity, failures, runtime, peak memory, and measured provider cost. If compatibility, coordinate mapping, credentials, operational limits, or throughput fails, stop and request a revised setup; do not silently switch to native Modal SAM 3 inference or another Replicate version.

## 7. Full 32-question Phase 3A resource ceiling

This estimate is for the approved proposal/retrieval benchmark only; it excludes Qwen answer ablations and any depth, multi-view, or tracking experiment.

- There are 30 unique videos. The fixed bank entails 10 × 30 = 300 video/prompt calls.
- Question-conditioned prompts are expected to require roughly 64–96 unique video/entity calls after cache reuse (estimate assumes 2–3 phrases per question; actual count is generated after blinded decomposition audit and must be reported before these calls).
- Estimated primary calls: 364–396. Allow up to 20% retry headroom, capped at 480 paid attempts total. At the current public page estimate, 480 attempts would be about `$0.816`; the previously proposed **$13.00** remains a conservative planning ceiling only and must be reconsidered using observed smoke billing before any separately approved full pilot. If an account-level cap cannot be verified or actual run cost trends above an approved cap, pause before exceeding it.
- Submit only the 16 sampled frames to the segmentation endpoint. Store output masks/features under a new Phase 3A artifact path; cap retained pilot outputs at 5 GiB and stop if returned archives exceed that cap. The 5 GiB cap is a guardrail; mask zip size is not known before the smoke.
- Modal feature extraction after the smoke: budget one H100 job up to 60 GPU-minutes and 90 wall-clock minutes, with 8 vCPU / 32 GiB RAM and a **$5.00 hard stop**. Use batches and temporary storage; report actual candidate count, two-crop feature bytes, and total bill. If the smoke shows the full pilot will exceed this envelope, stop and present a revised estimate.
- Replicate API usage for the full pilot begins only after the smoke report is reviewed and the user approves continuation. A separate approval is required before any Qwen scoring or broader validation run.

The Replicate model page was rechecked on 2026-10-17 and currently estimates about `$0.0017` per run and 2 seconds typical runtime for `lucataco/sam3-video`; this is not a billing guarantee for these video inputs. Use provider billing records for actual costs. Modal pricing and GPU compatibility references are linked in §9; the smoke uses A100 while the deferred full-pilot estimate retains H100.

## 8. Output report and stop rules

The Phase 3A report must include:

- ID manifest, source split checksums, unique video count, pinned model versions, prompt-bank version, crop preprocessing, exact frame mapping, cache keys, and API attempts/billing.
- Question-only decomposition audit, with entity/operation omissions and corrections counted.
- Proposal recall before ranking; correct-instance vs same-class/wrong-instance; joint co-visible coverage; Recall@1/5; precision@5; semantic class correctness/confusions; mask IoU/target coverage for boundary subset.
- SigLIP2 versus random ranking on exactly matched proposal sets, with paired video-cluster intervals; padded RGB and masked RGB results side by side.
- Fixed-vocabulary versus question-conditioned coverage, ranking, calls, time, masks, storage, Modal peak VRAM/runtime, failed jobs, and edge cases.
- Diagnostic case sheets separate from primary non-diagnostic aggregate; no answer score or gold-label comparison in the retrieval benchmark.

Stop before Qwen scoring if any pre-registered gate fails, the smoke output mapping fails validation, the 480-attempt / `$13` Replicate cap is reached, the Modal cap is reached, the 5 GiB artifact cap is reached, or an input would need an unapproved model/version/design change. No full validation, training, submission, depth-model run, or separate tracking experiment is included.

## 9. Deferred spatial-intelligence router

After retrieval is reviewed, the future architecture can route question types conditionally. Nothing below is part of Phase 3A and no weights are selected for a run here.

| Question type | Candidate path after separate approval | Evidence / caution |
|---|---|---|
| Simple image-plane relation | Existing mask/box geometry on same-frame observations | Report left/right, overlap, containment, and visible orientation only when confidence and co-visibility are adequate. |
| Depth-aware monocular geometry | Compare Depth Anything 3 and current MoGe family (the Microsoft repository now lists MoGe-3 as well as MoGe-2) on manually checked ReVA scenes | DA3 supports monocular and multi-view geometry/pose; MoGe exposes depth/point maps and metric scale for certain checkpoints. Metric-scale outputs still need validation on aerial imagery and should not be treated as surveyed ground truth. |
| Multi-view geometry/camera pose | VGGT or a pose-aware DA3 variant on temporally separated views | These models estimate camera/scene geometry from multiple views. Camera-motion, repeated texture, parallax, and low-texture rooftops may make estimates unreliable. Validate 3D point maps/poses before computing distances. |
| Temporal identity/visibility | SAM 3 video tracking with manually verified identities; separate temporal experiment | Track identity and visibility intervals require track-switch, occlusion, re-entry, and camera-motion audits. This retrieval benchmark records masks/timestamps but does not evaluate track correctness. |

A question-only router may identify requested operation (`image-plane`, `depth/distance`, `multi-view`, `temporal`) and request the corresponding evidence; it must abstain when the needed model evidence is unreliable. Preserve this DA3/VGGT-capable spatial-intelligence router in the architectural roadmap only: it is not implemented or run in Phase 3A. A future Qwen study should compare G0–G4 exactly as approved by the user, with G4 restricted to questions genuinely requiring 3D. No 3D outputs are allowed to override unvalidated retrieval or create answer-derived regions.

The original SSA-engine is not run. If later approved, a compact SSA-inspired variant would keep SAM 3/video as its mask source and test SigLIP2 as the region-label/ranking branch. Replacing SSA's old SAM mask branch and BLIP/CLIPSeg/CLIP labeling stack with current models is a separate experiment; it should be justified by proposal recall and semantic-label failures from Phase 3A. Do not add an LM/VLM proposal generator until the question-only extractor has been audited and a specific unresolved proposal failure is documented.

## 10. Sources and local evidence

- Replicate `lucataco/sam3-video` model page, current price/runtime estimate and version example: [model page](https://replicate.com/lucataco/sam3-video).
- Replicate billing behavior and variable runtime billing: [billing documentation](https://replicate.com/docs/topics/billing).
- Modal A100 smoke resources and deferred H100 estimate: [Modal pricing](https://modal.com/pricing); GPU types and compatibility notes: [Modal GPU guide](https://modal.com/docs/guide/gpu).
- Official SAM 3 image/video capabilities and version context: [Meta SAM 3 repository](https://github.com/facebookresearch/sam3); [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).
- SigLIP2 model: [Google SigLIP2 base checkpoint](https://huggingface.co/google/siglip2-base-patch16-224); [SigLIP2 paper](https://arxiv.org/abs/2502.14786).
- Deferred depth/multi-view candidates: [Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3); [MoGe repository (MoGe-2 and MoGe-3)](https://github.com/microsoft/moge); [VGGT](https://github.com/facebookresearch/vggt).
- SSA reference only (not implemented in Phase 3A): [Semantic-Segment-Anything repository](https://github.com/fudan-zvg/Semantic-Segment-Anything).
- Existing pilot selection and local artifacts: `tasks/phase3_grounding_pilot_ids.json`, `runs/sam3_pilot_stageA_replicate/qualitative_review.md`, `runs/sam3_pilot_stageA_replicate/comparison_report.md`, `runs/sam3_remaining17_stageA_replicate/comparison_report_32.md`, `runs/phase3_explicit_geometry/full/phase3_report.md`, `code/build_feature_cache.py`, `code/score.py`, and `code/sam3_overlay.py`.
