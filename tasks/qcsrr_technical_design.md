# QCSRR technical design for ReVA

**Status:** proposal for research approval; no implementation or jobs started.  
**Scope:** frozen Stage A Qwen3-VL-8B-Instruct; existing `valQ` and `valT`; no training or changes to the baseline.

## Executive summary

This design proposes a small retrieval-quality pilot before considering the full dense semantic segmentation stack. The smallest experiment that tests the central idea is: keep the same 16 baseline frames; generate candidate regions; rank them against question-derived entity phrases with the already pinned SigLIP2 base model; pass a small, diverse set of crops and timestamps beside the original frames to the frozen Qwen3-VL model. Measure whether relevant regions are retrieved before interpreting any answer change.

SAM 3 is a plausible region proposal and video-instance tool: it accepts short concept prompts and returns masks, boxes, scores, and instance IDs, including video propagation. It is not a prompt-free dense semantic segmenter. A fixed class-prompt bank could approximate dense scene coverage, while a prompt made from entities extracted from the question is question-conditioned. The original SSA architecture is useful design precedent for attaching labels to masks, but its branches rely on the original SAM and separate semantic models; its code is not a drop-in SAM 3 / SAM 3 Video pipeline. SigLIP2 can rank region crops against text, but its score is a retrieval signal, not a calibrated probability or proof that a crop contains the object.

This is a proposal, not a decision. Model-choice, proposal strategy, temporal handling, pilot selection, and GPU/API use remain approval decisions in §10.

## 1. Repository and prior-artifact inspection

The checkout describes Stage A as Qwen3-VL-8B-Instruct with a rank-16 LoRA adapter, 16 frames, two temporal offsets and two option rotations. The requested experiment can add a separate evidence path while leaving baseline scoring and artifacts untouched. The repository already pins `google/siglip2-base-patch16-224` and has offline feature-cache code, but that cache embeds sampled full frames and whole question/options; it does not currently encode mask crops or perform region retrieval.

Relevant local evidence:

- `runs/sam3_pilot_stageA_replicate/qualitative_review.md`: 15 selected questions (8 geometry, 7 temporal), 5,224 masks in overlay caches. SAM 3 overlay scored 10/15 vs baseline 9/15; geometry 6/8 vs 5/8; temporal stayed 4/7. It corrected one geometry answer, with no reported regressions in that comparison, but the pilot is selected and small. The soccer-field prompt selected broad grass, baseball-field masks spilled into neighboring areas, and black-car masks selected multiple parked cars.
- `runs/sam3_remaining17_stageA_replicate/comparison_report.md` and `comparison_report_32.md`: on the supplemental selected 17, all arms scored 14/17; SAM 3 made two fixes and two regressions versus baseline. Across the combined selected 32, direct overlays scored 24/32 vs baseline 23/32, with 3 fixes and 2 regressions. Geometry rose 5/8→6/8, temporal 4/8→5/8, and object/land-cover recognition fell 6/8→5/8. These selected cases remain diagnostic, not a generalization estimate.
- `runs/phase3_explicit_geometry/full/phase3_report.md`: the error-enriched geometry pilot had no changed answers. Only one relation pair was deemed reliable in valQ and none in valT; evidence coverage was too low to judge geometry text as a remedy.
- `results/error_review/adaptive_temporal_geometry_audit.md`: adaptive frame selection decreased held-out temporal accuracy (237/320 vs 243/320 on `val`) despite using question-aware frame similarity. QCSRR must test region relevance rather than assume that temporal/frame relevance will help.
- `runs/phase2_temporal_coarsefine/phase2_report.md`: denser question-aware temporal sampling also failed its predeclared gate and decreased temporal answers on both validation partitions.
- `code/build_feature_cache.py`: SigLIP2 snapshot is pinned; each video cache stores normalized image and text embeddings. The current design would need region-crop features added as a new, separately versioned artifact.
- `runs/phase3_explicit_geometry/geometry_relations.py`: existing geometry evidence is deliberately restricted to same-frame normalized 2D relations. Continue that limit; centroids and pixel distances do not establish physical distances or depth.

The workspace's `.git` entry is present but unavailable as a readable Git directory in this environment, so Git status/diff could not be inspected. File-level repository and experiment artifacts were inspected directly.

## 2. Proposed end-to-end architecture

The first implementation should be an offline, cacheable evidence builder plus an inference-time evidence adapter. It should not alter Stage A weights, prompt structure for the baseline arm, sampling, scoring, or existing results. Inputs to region generation and retrieval must exclude gold answers and answer-derived locations.

| Component | Purpose | Inputs | Outputs | Why it may help / key limit |
|---|---|---|---|---|
| 1. Fixed frame sampler | Hold visual evidence comparable across arms | Video plus Stage A frame policy | Same 16 frame indices and timestamps used by baseline | Controls sampling as a confound. Sparse frames limit timing precision and may miss small targets. |
| 2. Question decomposition | Turn wording into region queries and reasoning type | Question text only; optionally answer choices only if explicitly approved later | Entity phrases (e.g. `barn`, `river`, `road`), attributes (e.g. `black`), operation (`distance comparison`, `first appearance`) | Makes retrieval target explicit. Start with a transparent question-only extractor or human-checked annotations; do not feed the answer key, gold answer, or expected locations. |
| 3. Region proposals | Find candidate masks on the sampled frames | Frames; fixed class prompt bank or question-derived phrases | Mask, box, frame/time, proposal prompt, model score/quality metadata | Provides localized candidates. SAM 3 directly supports concept prompts; it does not by itself provide exhaustive, prompt-free semantic masks. |
| 4. Region features and labels | Describe candidates in a common image/text embedding space | RGB crop, optional masked crop, text query or a controlled label bank | Normalized image/text vectors; similarity; optional semantic label and alternatives | SigLIP2 can rank relevance cheaply after one-time feature extraction. Similarity is not calibrated confidence and may miss fine-grained aerial attributes. |
| 5. Deduplication and retrieval | Prevent many similar masks from flooding evidence | Candidate boxes/masks, query similarities, track IDs where available | Top-K diverse regions per entity, with scores and alternatives | Keeps evidence compact. NMS/IoU and cross-frame track grouping must avoid deleting adjacent distinct cars or duplicate proposals of one field. |
| 6. Geometry/temporal evidence | Compute conservative observations from retrieved regions | Retrieved masks/boxes and timestamps | Same-frame relative direction, overlap/containment, image-plane distance comparisons, possible visibility interval | Adds structured evidence. Do not claim world distance, depth, or reliable object identity from a 2D box or independent frame matches. |
| 7. Evidence integration | Let Qwen3-VL compare localized detail while retaining the full scene | Original 16 frames, original question/options, limited retrieved crops/overlays and short evidence text | Existing A-D scores | Region crops could make small structures and relations easier to inspect. Extra images also consume vision tokens and can distract; random-crop control is essential. |
| 8. Audit and evaluation | Separate retrieval quality from QA outcomes | Frozen outputs and blinded manual region judgments | Recall@K, precision, label correctness, relation/interval checks, paired answer changes, time/memory/storage | Distinguishes a retrieval failure from an evidence-integration failure. |

Suggested candidate record: video checksum; source-frame index and timestamp; proposal source/prompt; mask encoding; box; area; model mask/detection scores (unaltered); crop policy; semantic label candidates; SigLIP2 similarities; dedup/track identity; and a versioned cache key. Retain alternatives when top scores are close; do not turn a cosine score into an object probability.

### Dense proposals versus question-conditioned proposals

These are different experiments, not interchangeable implementations:

1. **Fixed-vocabulary/dense-like proposals:** run SAM 3 on the same frames with a small, fixed aerial-scene prompt bank (buildings, vehicles, roads, water, vegetation, fields, courts, etc.) independent of the question; deduplicate; label/rank candidate crops using SigLIP2. This tests reusable scene coverage but still depends on chosen vocabulary and is not truly category-free.
2. **Direct question-conditioned proposals:** extract target/reference entities from the question only and prompt SAM 3 with each phrase. Retrieve/deduplicate those masks and optionally use SigLIP2 to rank within each query. This is likely cheaper and can find categories omitted from a fixed bank, but uses the segmentation model itself to condition proposals and makes the contribution of SigLIP2 less isolated.

For a clean comparison, share the same frames, crop policy, evidence budget, and frozen Qwen scorer. Report SAM 3 mask generation separately from SigLIP2 ranking. An additional mask-free text-to-box detector is outside this initial proposal.

## 3. Actual components and models under consideration

- **Existing frozen answerer:** the current Stage A adapter/checkpoint and Qwen3-VL-8B-Instruct scorer; preserve all baseline settings and run each arm on identical IDs and video inputs.
- **Region proposal candidate:** local Meta SAM 3.1 image/video code and weights, subject to environment compatibility and the user's GPU/access approval. The official repository currently lists Python 3.12+, PyTorch 2.7+, CUDA 12.6+, and authenticated checkpoint access; this is a separate compatibility path from the existing Stage A setup. The prior SAM3 ReVA path used a remote API for direct masks, not a local SAM3 Video integration.
- **Embedding candidate:** pinned `google/siglip2-base-patch16-224` already described in README and referenced in `code/build_feature_cache.py` (768-dimensional features in the existing cache; local weight file about 1.50 GB). Use one text query per entity phrase and per-region features. Cache query-independent region features by video/model/crop policy, and query text features by phrase. Compare RGB box crops with mask-composited crops before choosing one.
- **SSA reference:** Semantic-Segment-Anything (SSA) combines a SAM mask branch with a pixelwise semantic segmentation branch and votes over each mask's pixels to attach a category. SSA-engine adds closed-set categories, caption-derived candidates, CLIP proposal filtering, and an open-vocabulary segmentor. Its public implementation documents original SAM checkpoints and multiple models; adopting all of it would add substantial runtime and compatibility work. Reuse the conceptual separation (mask generation vs semantics), not its code wholesale.
- **Optional temporal candidate:** SAM 3/3.1 video prompting and its memory-based tracker can provide identities across frames. Confirm actual compatibility, throughput, and identity stability on ReVA clips first. A video track is evidence with uncertainty, not ground truth identity.

## 4. SSA, SAM 3, and SigLIP2 roles

### SSA

SSA answers “what class is this mask?” by combining boundaries from SAM with coarse per-pixel semantic predictions and mask-level voting. SSA-engine generates open-vocabulary labels through caption-derived candidate words and CLIP filtering, with an open-vocabulary semantic segmentor as a final selector. For ReVA, its main lesson is to separate region proposals from region labels. The original recipe is not a direct SAM 3 adapter: it expects the SAM-era branch and semantic-model interfaces, uses an image segmentation setup, and introduces models whose ADE20K/COCO categories may not fit aerial footage. A robust ReVA adaptation would require model/version testing and evaluation, so it is not part of the first pilot unless approved.

### SAM 3 / SAM 3.1

SAM 3 performs promptable concept segmentation: short noun phrases or visual prompts can return masks for matching instances; official code also offers a video predictor and identities. This makes it a useful candidate generator for “soccer field” or “black car,” and a plausible path to temporal instance evidence. It does not expose SSA's pixelwise semantic-voting branch, nor should we assume it produces an exhaustive generic object proposal set without prompts. It can return broad false matches, miss small/oblique/occluded objects, or split/merge repeated scene regions. Its published benchmark evidence is not a ReVA aerial-video retrieval evaluation.

### SigLIP2

SigLIP2 is the retrieval/ranking model, not the segmentation model. Encode normalized crops and short query phrases; cosine similarity supplies an ordering. Start from the repository's pinned base model for reproducibility and low additional model size. Compare short noun phrases plus relevant visual attributes (for example “black car”) with basic labels (“car”) separately. Compare masked RGB crop and padded RGB box crop, since removing surroundings may hurt recognition of context-dependent aerial categories. No aerial-domain validation currently establishes that SigLIP2 is best; include a hand-checked retrieval benchmark before considering other encoders. The existing full-frame cache is a starting point but does not cover region crops.

## 5. Example walkthroughs

These examples show the proposed data flow; they do not claim a retrieved region, correct answer, or accuracy improvement.

### Geometry: `train_013068`, barn / river / road

1. Read only the question wording. Extract `barn`, `river`, `road`; operation is a relative-distance comparison. Do not use the answer options to select a region or prompt.
2. Use the baseline's 16 timestamped frames. For question-conditioned proposals, prompt the segmenter for all three phrases; for the dense-like arm, use the fixed bank and let text-image similarity select those entities.
3. Save each mask, box, timestamp, score and query similarity. Retain multiple plausible candidates if the scene contains multiple barns or river-like features. Deduplicate high-overlap masks within one frame, but do not merge adjacent distinct objects simply because their boxes are near.
4. Compare barn-to-river and barn-to-road geometry within the same frames. Report normalized image-plane centroid/boundary separation and the frames that support each comparison. If reference regions are not reliably identified, label the result insufficient/ambiguous and send only crops/alternatives, not an asserted relation.
5. Provide original scene frames plus a small number of labeled crops and a short statement such as “image-plane separation in sampled frame 8: barn–river smaller/larger than barn–road; uncertain region matches.” Qwen3-VL retains global context and decides among choices.

Pixel separation does not establish physical distance: camera tilt, height, perspective, occlusion, curved roads/rivers, and inconsistent ground planes can reverse or distort apparent distances. A convincing 2D result only addresses what appears in the frame.

### Temporal: `train_009393`, first appearance of soccer field

1. Extract `soccer field` and operation “first identifiable appearance” from the question text only.
2. Search the baseline's sampled frames for candidates. Compare phrase-to-region matches and inspect masks over time. If using SAM 3 Video, prompt the concept and propagate its instance identity; also test for re-detection across the clip because an incorrectly initialized track can remain wrong.
3. Record the earliest sampled frame with a specific, identifiable soccer-field candidate and the next prior frame with no supported candidate. Preserve the bracket, scores, and alternative field candidates; do not call generic grass a soccer field. If target-specific evidence is absent, mark the interval unresolved.
4. Give Qwen3-VL the original scene frames and a compact timestamped evidence strip, not only the mask overlay. The reported first-appearance time is bounded by sampling density and any additional search frames.

With 16 frames, the temporal bracket is coarse; a candidate interval narrower than the frame spacing is not supported. Dense video propagation may add runtime and memory, while independent per-frame text prompts alone do not prove the masks refer to the same field instance.

## 6. Difference from the prior SAM 3 overlay pipeline

The prior SAM 3 arm appended direct text-prompt masks/overlays to the same baseline frames. It did not encode candidate region crops with SigLIP2, rank regions by question similarity, cap output by per-entity diversity, or independently validate region retrieval. It therefore tested “do these prompt masks help when shown to Qwen?” and not the central QCSRR claim “does question-conditioned retrieval select useful local evidence?”

The existing ReVA SAM 3 overlay cache assigns each returned mask `confidence: 1.0`; that is a cache bookkeeping value, not a calibrated model confidence. QCSRR should preserve genuine source scores where available and mark absent/unvalidated confidence as unavailable.

The observed qualitative issues also motivate retrieval and uncertainty handling: broad grass instead of a field, neighboring fields bleeding into baseball-field masks, and numerous cars matched for “black car.” Retrieval ranking and deduplication might help prioritize better candidates, but cannot repair missing or semantically incorrect proposals. The small overlay results are mixed and selected; they support testing a different pipeline, not assuming it will win.

The previously tested SigLIP2 adaptive sampler ranked whole frames and declined temporal accuracy on the held-out evaluation. Region-crop retrieval is a new use of SigLIP2, but that prior result is reason to gate on manual retrieval quality before spending on large answer runs.

## 7. Ablations and evaluation

Use the same frozen adapter, ID set, video checksums, baseline frame indices, option scoring, decoding and answer aggregation for every arm. Store each arm under a new run/output prefix. Do not use gold answers to create question decomposition, proposals, rankings, evidence text, or Qwen inputs.

| Arm | Difference from E0 | Purpose |
|---|---|---|
| E0 | Frozen Stage A baseline | Reference. |
| E1 | Baseline + random region crops matched to the number, size and time distribution of E2 crops | Measures whether any extra crop detail helps. |
| E2 | Baseline + top-K question-retrieved crops with labels/timestamps and uncertainty | Tests region retrieval + localized visual evidence. |
| E3 | E2 + conservative same-frame 2D relations | Tests explicit geometry while keeping crop evidence fixed. No world-distance/depth claim. |
| E4 | E2 + candidate track/visibility evidence | Tests temporal identity and timing. Keep geometry text out. |

Also compare **fixed-vocabulary dense-like region proposals** with **direct question-conditioned proposals** using the same retrieval and Qwen arms. If budget is tight, first compare both retrieval paths without Qwen scoring and carry only one proposal path into E1–E4.

### Pilot selection

The five user-supplied diagnostic questions (`train_013068`, `train_012502`, `train_009393`, `train_009869`, `train_010397`) are useful qualitative cases, not an unbiased estimate. The existing `tasks/phase3_grounding_pilot_ids.json` provides a deterministic 32-question template across valQ/valT, with temporal, geometric, object/count and control strata; it includes several of the diagnostics. A proposed first answer pilot is 24–32 questions from these partitions, selected by fixed task/entity criteria and deterministic hash order, while ensuring some Stage A correct and incorrect examples occur in both target tasks. Record the rule before seeing QCSRR outputs; report this sample as a diagnostic pilot, never as split-wide accuracy. Alternatively, use the existing 32-ID list unchanged. This selection choice needs approval.

### Retrieval quality before answer quality

Manually annotate a small fixed subset of question-entity/frame pairs, blinded to QCSRR rank and predicted answer. For each query, mark relevant masks/crops, the target instance or ambiguity, and visibility interval where applicable. Report:

- Recall@1/5/10 and precision@K; count a retrieval hit only against the manually checked target instance. For masks, report IoU/coverage when an annotator can draw a reference mask; otherwise record human semantic relevance separately.
- Semantic label accuracy and common confusion classes. Preserve an “ambiguous / not identifiable” label rather than forcing correctness.
- For geometry, manually judged target/reference identity and relation correctness, separated into image-plane direction, overlap/containment, orientation, and distance comparison. Do not use the four-way answer key as a substitute for region-level labels.
- For temporal questions, target identity consistency, first/last visibility interval, interval error in seconds, and track switches/misses.
- For all QA arms, accuracy by task/split, exact paired fixes, regressions, wrong-to-wrong changes, and confidence/probability changes. Use matched per-question comparisons and paired uncertainty intervals; a small pilot is descriptive, not a promotion claim.
- Runtime by stage, per-video amortized and per-question cost, peak VRAM, CPU/RAM, mask-cache size, crop-feature-cache size, proposal count, and number of Qwen forwards.

Retrieval-quality gates should be set before the pilot; a reasonable decision point is to halt before answer scoring if entity-level Recall@5 is poor or if “target object” retrieval is mostly broad background. The precise threshold is an approval decision.

## 8. Compute, storage, and implementation cost

Known quantities: baseline Stage A scores four views per question (two temporal offsets × two option orders) with 16 frames per view. The pinned SigLIP2 base checkpoint is about 1.50 GB on disk, with 86M parameters and 768-dimensional features. One float32 feature vector is about 3 KB (about 1.5 KB in float16); for example, 100 candidate regions × 16 frames × one crop representation is roughly 4.7 MiB of raw float32 features per video, excluding JSON, masks, metadata, and duplicate crop styles. RLE mask storage varies greatly with region shape and resolution.

Additional compute has two components:

- **Offline proposal and feature work:** SAM 3 image inference once for each sampled frame and prompt set, plus one SigLIP2 image-feature pass per crop. Query-independent crop features can be reused across questions for a video; text features can be reused by identical query phrase. Question-conditioned SAM 3 masks require new proposal inference for each unique entity prompt unless the same video/query phrase cache already exists. Video propagation must be timed separately.
- **Online answer scoring:** the same four Qwen forwards per question as E0, but additional region images/overlays can increase preprocessing, vision tokens, VRAM and latency. Keep the evidence budget fixed and measure, rather than assume, the incremental cost.

For context only, the prior remote SAM 3 overlay pilot reports 28 direct-prompt calls and roughly `$0.27` for its final 15-question invocation; the report estimates about `$0.56` across earlier smoke/retry calls. The supplemental 17-question run reports 45 attempts and approximately `$0.81`. These are historical remote-API estimates, not a cost estimate for a local SAM 3.1 run or the proposed SigLIP2/Qwen pipeline.

The SSA repository reports its own older single-A6000 reference measurements: 1.149 seconds/image for close-set SSA and 33.333 seconds/image for SSA-engine; open-vocabulary SSA-engine with SAM used about 28–31 GB GPU memory. These figures are not directly transferable to ReVA or current hardware, but they support keeping full SSA-engine out of the first pilot and measuring any reduced adaptation separately.

The repository documents an H200 validation/training environment, but the existing SAM3 local environment/version and region-crop throughput have not been measured here. The official SAM3 README's current Python/PyTorch/CUDA requirements may require a separate environment. No defensible per-video seconds or peak-VRAM figure exists yet. After approval, use one short compatibility smoke and 1–2 representative clips to measure peak VRAM, masks/frame, processing time, and cache growth before approving a larger run. Avoid simultaneous SAM3 and Qwen loads if they exceed available memory; sequential stages with disk caches are the likely low-risk layout.

Approximate storage formula for raw float32 region features: `N_videos × regions_per_video × 768 × 4 bytes`; add masks/crops separately. Store masks as RLE rather than bitmaps when supported and avoid archiving duplicate full-resolution crops. Exact bytes and speeds are pilot measurements, not promises.

## 9. Limitations, risks, removable components

- **Domain shift:** public SAM/SigLIP evidence does not establish reliable small-object, oblique building, field-boundary, road, water or vehicle-color retrieval in these ReVA clips.
- **Proposal ceiling:** no embedding ranker can retrieve a region that the proposal model missed. Dense prompts may miss uncommon classes; question prompts may fail on paraphrases, relational referring phrases or one-instance attributes.
- **Fine-grained text grounding:** “black car,” “barn,” or “soccer field” can have weak image-text separation in small, low-resolution aerial crops. Region context, upscaling and higher-resolution encoders are alternatives to evaluate, not assumptions.
- **Crop-context trade-off:** masked crops may remove useful surroundings; box crops retain distracting context. Compare both on retrieval labels first.
- **Duplicate domination:** many masks, overlapping frames, or multiple cars of a class can consume top-K slots. Use per-frame overlap deduplication, per-track cap, query-level coverage and include a small number of alternatives; set thresholds from annotated examples.
- **Temporal identity:** SAM3 track IDs can drift, merge or disappear under camera motion/occlusion; do not equate re-identification by class with same-object identity. Use uncertainty and manual track audit.
- **2D geometry:** apparent distances and relative positions vary with perspective, camera translation/rotation and ground-plane assumptions. Only same-frame image-plane statements are admissible in this proposal.
- **Prompt/answer leakage:** derive entity phrases from the question only for the main arm. Do not use correct options/gold label, pre-known answer timing, or gold geometry to generate candidates or context. Freeze manual annotation before comparing rank lists.
- **Integration cost:** Qwen's original frames already provide scene context. More crops may add token cost and distraction. E1 random crop control measures this; if retrieval does not improve judged Recall@K, remove semantic labeling and the Qwen evidence arm.
- **Remove without losing the core hypothesis:** full SSA-engine captioning and open-vocabulary segmentation; global dense class labels; multi-model ensemble; full-frame adaptive sampling; depth/3D inference; fine-tuning; and broad all-video caching. The core test needs proposals, query-region similarity, deduplication, bounded crops, and independent evaluation.

## 10. Decisions approved; next phase gated

The user's approval supersedes the choices previously listed here: Phase 3A compares question-conditioned with fixed-vocabulary SAM 3 proposals on the existing stratified 32-question list; uses pinned SigLIP2 with both padded-box and masked crops; applies deterministic question-only decomposition with manual audit; keeps Stage A frozen and preserves its 16-frame/four-view protocol for any later answer experiment; caps later Qwen evidence at four crops; and does retrieval evaluation before answer scoring. Use Modal for bounded compute and Replicate for SAM 3 model calls.

Do not implement the complete SSA-engine or the conditional depth/multi-view router now. Do not run any GPU/API jobs until the user approves the finalized Phase 3A specification. The current executable proposal and caps are in [phase3a_retrieval_spec.md](phase3a_retrieval_spec.md).

## 11. Minimal falsifiable pilot

**Claim to test:** On manually annotated ReVA question-entity pairs, question-conditioned SAM 3 proposals plus SigLIP2 ranking retrieve the named target region more often in the top 5 than a random candidate baseline. Any later claim about whether crops help frozen Qwen3-VL requires a separate approved paired answer experiment.

**Procedure:** use the approved existing 32-ID stratified list unchanged; tag the five supplied diagnostics and interpret them separately. Generate candidates on the fixed baseline frames, cache them, and blind-manually annotate target relevance/identity before examining rank order. Compare dense-like and question-conditioned proposals on retrieval only. If a path passes the pre-registered gates, present its report and wait for approval before scoring any Qwen arm.

**Falsification / stop conditions:** the retrieval hypothesis is not supported if (a) relevant target regions are not consistently present among proposals; (b) SigLIP2 ranking does not beat random top-K on manually checked examples; or (c) compute exceeds the approved cap. No Qwen accuracy claim is part of this pilot. Do not proceed to Qwen scoring, full `valQ`/`valT`, fine-tuning, or the spatial router without reviewing Phase 3A and obtaining the user's next approval.

## Sources

- SSA public implementation and documented branches / SSA-engine: [Fudan Zhang Vision Group, Semantic-Segment-Anything README](https://github.com/fudan-zvg/Semantic-Segment-Anything/blob/main/README.md).
- SAM 3 official inference and repository requirements: [Meta SAM 3 repository](https://github.com/facebookresearch/sam3); [SAM 3 paper](https://arxiv.org/abs/2511.16719).
- SAM 3.1 update / video tracking and efficiency results: [Meta SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md).
- SAM 3 video inference implementation: [official `sam3_video_inference.py`](https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_video_inference.py).
- SigLIP2 model card and available model configuration: [Google SigLIP2 base patch16 224](https://huggingface.co/google/siglip2-base-patch16-224).
- SigLIP2 capabilities/benchmark framing: [Tschannen et al., SigLIP 2](https://arxiv.org/abs/2502.14786).
- Repository sources inspected: `README.md`, `code/build_feature_cache.py`, `code/score.py`, `runs/sam3_pilot_stageA_replicate/qualitative_review.md`, `runs/sam3_pilot_stageA_replicate/comparison_report.md`, `runs/sam3_remaining17_stageA_replicate/comparison_report.md`, `runs/phase3_explicit_geometry/full/phase3_report.md`, `runs/phase2_temporal_coarsefine/phase2_report.md`, `results/error_review/adaptive_temporal_geometry_audit.md`, `tasks/phase3_grounding_pilot_ids.json`, and `runs/phase3_explicit_geometry/geometry_relations.py`.
