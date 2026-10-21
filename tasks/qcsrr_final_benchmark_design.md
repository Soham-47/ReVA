# Final ReVA QCSRR Retrieval Benchmark — Proposal

**Status:** design for review only. No segmentation calls, SigLIP2 embedding job, or Qwen scoring has run for this cohort. Phase 3B Gate 2 artifacts remain unchanged.

## Decision this benchmark is meant to support

Use one small, video-held-out-from-prior-work train cohort to decide whether question-conditioned region retrieval can locate the regions named by the question, preserve spatial constraints, and associate regions over time. This is a screening study, not an architecture-level performance estimate. The next confirmation set would use distinct validation/test videos only after this design is frozen.

## Cohort and deterministic selection

The proposed cohort has **12 questions from 12 distinct videos**, all from the pinned ReVA train split at dataset revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`:

| Selection stratum | Questions | Coverage |
|---|---:|---|
| Geometric relations | 4 | Multiple named regions and relative position or distance operations |
| Temporal visibility | 4 | First appearance or visible duration |
| Scene-region retrieval | 2 | A structure/field context and a land-cover region |
| Ambiguity probes | 2 | A bare singular class reference and an unqualified group reference |

The ambiguity tags are probes based on wording only. Reviewers will decide from the source video whether a unique instance is actually identifiable; no instance label is presumed.

Selection used only the question ID, video path, the dataset question-task taxonomy, and question wording. The pre-selection scan found 1,754 prior train question IDs and excluded every train video whose question ID or exact video path appeared in existing `tasks/`, `results/`, or `runs/` artifacts, including the two Gate 2 videos. This conservative scan excluded 604 video paths; 441 distinct train videos remained eligible. The newly written selection/design files are excluded from this prior-exposure scan. Within each predeclared text/task stratum, rows were ordered by SHA-256 of `ReVA-QCSRR-final-v1|<qa_id>`. The first rows were selected subject to one question per video. Options, correct answers, baseline predictions/correctness, source-video appearance, and model rankings were not used.

The complete list of question text, paths, hashes, and question-only decompositions is recorded in qcsrr_final_benchmark_selection.json (local-only artifact: `/home/soham/ReVA/tasks/qcsrr_final_benchmark_selection.json`). The selected video distribution is Hawk_UAV 7, ERA_Tra 2, VisDrone 2, and UAVDT 1.

### Selected questions and question-only decomposition

| QA ID | Video | Target and attributes | Reference entities | Spatial constraint / operation | Temporal operation |
|---|---|---|---|---|---|
| `train_014154` | `Hawk_UAV/BE/DJI_0338_02.mp4` | commercial building; commercial | apartment complex | Measure left/right; the question does not supply the direction | — |
| `train_005918` | `ERA_Select/NonEvent/NonEvent_132.mp4` | large green space; large, green | roundabout; surrounding roads | Compare target orientation with the road network | — |
| `train_011679` | `Hawk_UAV/BE/DJI_0377_01.mp4` | AMC theater | Cinemark building; parking lot | Compare both distances from the AMC theater | — |
| `train_012957` | `Hawk_UAV/BE/DJI_0387_01.mp4` | vehicles | intersection; traffic lights; white building with green roof | Compare vehicle-to-anchor distances within one frame | — |
| `train_008503` | `Hawk_UAV/IL/DJI_0062_d4_01.mp4` | large white-roofed building | — | — | First visible sample |
| `train_008163` | `Hawk_UAV/IL/DJI_0037_d4_02.mp4` | construction site | — | — | First-to-last visible interval |
| `train_001478` | `VisDrone/uav0000266_04830_v_01.mp4` | bus; white and red | — | — | Visible interval |
| `train_010200` | `Hawk_UAV/PH/DJI_0250_d4_01.mp4` | baseball field | — | — | First visible sample |
| `train_003224` | `ERA_Select/Baseball/Baseball_027.mp4` | structure; type unspecified by the question | baseball field | Structure on the field's left side | — |
| `train_002092` | `VisDrone/uav0000300_00000_v_03.mp4` | land-cover region | central median strip; road | Region inside the road median | — |
| `train_012936` | `Hawk_UAV/BE/DJI_0370_06.mp4` | motorcycle; no distinguishing attribute in wording | — | — | Disappearance time; ambiguity probe |
| `train_014264` | `UAVDT/M0301.mp4` | vehicles as a group; no unique vehicle specified | highway | Vehicle regions on the highway; group-level evidence | Repeated movement pattern; ambiguity probe |

## Question decomposition and retrieval arms

Keep the current noun-phrase method as R0. Do not feed the full question to SigLIP2 and assume that cosine similarity resolves relations. Each question is represented before inference as target entity, visual attributes, reference entities, spatial constraint, geometric operation, and temporal operation. The entity phrase proposals in the JSON are the only text prompts; relation words stay in the structured fields.

All three arms use the same automatic candidate pool and the pinned checkpoint `google/siglip2-base-patch16-224` at revision `022b6f71160ffb0169ca4709e2d7e25be659598a`:

- **R0 — noun-phrase SigLIP2:** rank each question's retained candidate regions by cosine similarity to the target noun phrase. Rank references separately when the question needs them. No spatial or temporal reranking.
- **R1 — spatial/relational reranking:** keep the R0 cosine scores, then enumerate same-frame target/reference candidate tuples. Apply only the answer-neutral constraints stated by the question. For constraints such as “left side” or “within the median,” rank tuples lexicographically by constraint satisfaction, summed cosine, then stable candidate ID. For questions asking *which side* or *which distance is greater*, do not invent the answer as a constraint: retrieve all named regions, compute the requested relation from candidate geometry, and score that relation against the blind source-video annotation.
- **R2 — temporal association:** add a deterministic one-to-one adjacent-frame association over the R1 candidate masks. A link is eligible when mask IoU is at least 0.25 or normalized centroid displacement is at most 0.03 of the frame diagonal and component area ratio is within `[0.5, 2.0]`. Choose the maximum-total-IoU eligible assignment with candidate-ID tie breaks. Break tracks when no eligible link exists; do not force identity through occlusion, crossings, or camera motion. R2 identity metrics apply only where the source-video reviewers independently verify identity.

The R2 thresholds are frozen before annotation and inference. No learned tracker or new model is introduced. The existing Gate 2 masked-crop values remain a historical reference only; the new primary representation is padded RGB.

## Candidate generation and crops

1. Use the pinned SAM 3 video endpoint already used for Phase 3A, with one text prompt per entity occurrence listed in the selection JSON. There are **24 primary predictions** over 22 distinct normalized phrases, each on the same 16-frame, offset-0 legacy sample for its source video. Preserve the original source-frame index, FPS, timestamp, dimensions, and packed-frame-to-source-frame mapping. Do not use visual/oracle prompts, answer text, human boxes, or target labels.
2. Decode every prompt's frame-level union mask. Keep the union mask as a contextual candidate. For the question's target phrase only, split the union into 8-connected components and retain at most the two largest nonempty components per frame. Record the full pre-cap component count and every component omitted by the cap; this cap is part of the tested automatic proposal generator, not a manual target selection.
3. Remove only exact duplicate binary masks from the same source frame, preserving all prompt provenance. The planned pool is at most 384 union masks plus 384 target-component proposals, or **768 candidate regions before exact de-duplication**. If the endpoint returns a different mask layout, source mapping fails, or the output would exceed the bound, stop and revise the plan before ranking.
4. Create one RGB context crop per region from its source-resolution bounding box with 10% padding on each axis, clipped to the frame. This reuses the existing Phase 3A crop padding. Store original masks/boxes and candidate provenance alongside crops. Do not create masked RGB embeddings in this benchmark.
5. R0, R1, and R2 receive identical candidate IDs and crops per question/frame. Existing fixed-vocabulary and masked-crop Gate 2 results remain frozen and are not mixed into the new candidate pool.

## SigLIP2 text preprocessing check

A CPU-only check was run in the existing local tokenizer environment against the pinned snapshot marker. Under Transformers 5.19.0, `SiglipProcessor` / `GemmaTokenizer` processed the 22 distinct selected target/reference phrases (24 question-entity occurrences) after explicit Unicode NFC normalization, `casefold()`, and whitespace collapse. With the same arguments used by the ranking path (`padding="max_length", truncation=True`), the default call produced 64-token rows and matched explicit `max_length=64` IDs byte-for-byte. The longest phrase had 7 non-pad tokens including EOS; none was truncated. This agrees with the official [SigLIP2 64-token convention](https://huggingface.co/docs/transformers/model_doc/siglip2).

The tokenizer object itself did not lowercase text, despite the checkpoint's lower-case training convention, so `casefold()` is explicit in the plan. The processor returned no `attention_mask`; the existing Gate 2 review records that the ranking path also passed no mask and pools the final padded position. Keep this behavior fixed for R0/R1/R2 comparability and preserve the checkpoint, but report it as a remaining implementation limitation. The BOS/EOS configuration warning was reproduced in the CPU-only tokenizer setup; actual tokenizer IDs use PAD=0/EOS=1 and the pinned model config reports a 256,000-entry text vocabulary. No image or text embeddings were rerun.

## Annotation protocol and labels

Freeze the cohort, full question text, decomposition, sample indices, crop rules, thresholds, and evaluation code before any model ranking is shown.

1. **Source-video truth first:** two reviewers independently watch each original video with the exact question only. Do not show options, correct answers, baseline output, SAM 3 masks, SigLIP2 rankings, or predicted answers. On each of the 16 sampled source frames, mark target/reference visibility and source-resolution boxes; trace a mask only where its boundary is visually defensible. Record target identity, occlusion, visibility, ambiguity, and confidence. If a question does not specify a unique instance, label instance identity “not identifiable” rather than inventing one.
2. **Blind candidate labels:** after source labels are frozen, present every retained candidate in randomized order with anonymous IDs, the original question, and the relevant entity phrase. Hide rank, cosine, proposal prompt provenance, and R0/R1/R2 arm. Label correct category, correct instance (when assessable), useful contextual region, over-broad/background, merged/fragmented mask, wrong instance, and ambiguity. Reviewers may see source timestamp and frame but not model score.
3. **Second review:** independently double-label a hash-selected 25% of source entity/frame annotations and candidate rows, stratified by question and candidate type. Double-review every ambiguity-probe identity decision. Reconcile disagreements in a recorded adjudication field; unresolved cases remain ambiguous/unassessable.
4. **Boundary and context:** report mask IoU, target-pixel coverage, proposal precision, and box IoU where a stable source annotation exists. For scene regions, separately report the blinded usefulness label, target coverage, crop background fraction, and whether all named context is retained. Do not treat a useful contextual box as a pixel-perfect mask.

Manual annotations are evaluation-only. No annotation, target coordinate, instance ID, or adjudication enters prompt generation, candidate filtering, or ranking.

## Metrics and matched baselines

Report per-frame, per-question/video, and task-stratum results. The independent unit is the **video/question**; the 16 frames are repeated observations, not 16 independent examples.

- Category Recall@1/5 and correct-instance Hit@1/5 for assessable target entities. Report proposal coverage before ranking, ranking conditional on a correct proposal existing, and failures caused by proposal generation, component truncation, embedding rank, geometry, or identity association.
- Query/video Hit@1/5 ranks candidate-time regions over the full sampled video. Per-frame Hit@1/5 is secondary. Report pool size, positives, exact duplicates, and repeated-track duplicates so repeated frames cannot silently inflate the score.
- Joint geometric coverage is a hit only when every required entity is present in a same-frame tuple and the requested relation computed from candidate geometry agrees with the blind source annotation. Report entity coverage and relation correctness separately.
- Temporal validity requires the correct track where identity is assessable and the requested visibility/event evidence on the sampled timeline. First/last visibility must be within one sample interval of the source annotation. Do not interpolate sub-sample event times. Report ambiguous identity and resolution-limited cases explicitly.
- Compare all arms with 1,000 deterministic random rankings from the exact same retained candidate pools. Because all retained candidates are annotated, also report the exact without-replacement random Hit@K probability from the number of positive candidates and pool size. If a pool has five or fewer candidates, report its full size and do not use its Hit@5 as evidence of nontrivial ranking.
- Use paired question/video-level bootstrap intervals, stratified by the predeclared cohort strata. No frame-level significance test and no claim that this cohort estimates broad ReVA performance.

## Predeclared decision rule

Proceed to a *proposal* for a small paired frozen-Qwen3-VL ablation only if all conditions hold:

1. At least 8/12 questions are source-video assessable, including at least 3 geometric and 3 temporal questions, and at least 6 have an assessable unique target instance. Keep every unassessable case in the report.
2. R0 category Hit@5 has at least a 0.15 absolute lift over its exact matched-pool random expectation, with the lower bound of the paired video-level 95% bootstrap interval above zero.
3. R1 correct-instance Hit@5 is at least 0.70 among assessable unique-instance targets and at least 0.15 above matched-pool random. R1 also has correct joint geometry in at least 3 of the 4 geometric questions and improves joint geometry over R0 in at least one question without using an answer-derived direction.
4. R2 temporal validity is correct in at least 3 of 4 assessable core temporal questions and improves at least one temporal question over R1. If cross-frame identity cannot be independently verified, this condition is not met; it is not replaced with a category-only result.

If a minimum assessable denominator is not met, any gate fails, or its confidence interval remains inconclusive, do not score Qwen. Recommend ending this QCSRR path or gathering a cleaner diagnostic cohort before considering targeted mLLM training or separately validated spatial/temporal evidence.

Even if the rule passes, the 12-video result only justifies requesting a small paired frozen-Qwen ablation. A separate video-held-out confirmation is still needed before any architecture claim.

## Proposed compute and cost

| Work | Planned resources | Expected use | Cost estimate |
|---|---|---:|---:|
| SAM 3 proposals | Pinned `lucataco/sam3-video` version `408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269`; Replicate L40S endpoint | 24 primary predictions; up to 5 aggregate retries (29 attempts maximum), 16 sampled frames per prompt | About $0.04–$0.06 using the current model-page estimate of ~$0.0016/run and a rounded $0.002/run planning rate. The price varies with input; it is not a provider-enforced cap. [Replicate model page](https://replicate.com/lucataco/sam3-video) |
| SigLIP2 ranking | One Modal A100-40GB job, 4 vCPU, 16 GiB RAM | Up to 768 padded crops and 24 text phrases; 1,000 matched-pool random rankings; 10-minute hard function timeout | Expected 2–4 minutes, about $0.08–$0.16. At the full 10-minute resource timeout, listed GPU/CPU/memory rates imply about $0.39, excluding image-build and egress charges. [Modal pricing](https://modal.com/pricing) |
| **Total** | One segmentation provider + one ranking job | No Qwen, training, depth, or 3D calls | Expected about **$0.12–$0.22**; resource-based estimate about **$0.45** at the proposed maxima, excluding possible image-build/egress charges |

The previous Gate 1 A100 job encoded 2,319 candidates in 44.7 seconds and measured 1.78 GiB peak allocated CUDA memory. For this smaller pool, plan for 2–4 GiB peak CUDA allocation, 16 GiB host RAM, and 2–4 minutes including the ranking stage; retain a 10-minute stop. These are estimates, not guarantees. Modal's published rates bill GPU, CPU, memory, and application load time; the estimate is not a workspace spend ceiling. Replicate's model-page price is input-dependent. Actual provider billing must be reconciled after any approved run.

Only 3 referenced source videos are currently present under the local `data/ReVA` tree, and **none of the 12 selected clips is staged locally**. Staging these 12 source clips from the same pinned dataset revision is a prerequisite before annotation. The existing downloader can fetch only selected paths. No provider call or data staging is part of this design submission.

## Approval boundary

This document and its selection JSON are the complete proposal. No paid inference, data download, candidate annotation, Qwen scoring, model training, DA3/VGGT, or 32-question pilot has been launched. Review this exact cohort and budget before authorizing execution.
