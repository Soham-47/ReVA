# Phase 4 QCSRR benchmark — execution report

**Status: incomplete because Replicate throttled the approved run.** The 12-video source preparation and blind source reviews completed. The full proposal, ranking, geometric, temporal, fixed-vocabulary, and retrieval-metric comparison did not. The 41 approved Replicate create attempts are exhausted; no more model calls were made.

## Executive result

- The cohort contains 12 questions from 12 distinct, previously unexamined training videos. The frozen selection used the specified deterministic hash rule and did not use answers, baseline correctness, video appearance, or model rankings.
- All 12 videos were staged and checked. Each has 16 original-frame samples (192 frames total); source frame indices and timestamps come from the source manifest and source FPS. Independent primary and second source-only reviews were frozen before model outputs were inspected.
- Replicate planned 24 question-conditioned SAM 3 calls and 12 fixed-vocabulary controls, plus five shared retries. It used 41 create attempts. Five attempts created prediction IDs, including one duplicate commercial-building prediction; 36 requests were rejected with HTTP 429 before a prediction ID was created. Four unique question-conditioned prompts produced usable outputs. No fixed-vocabulary control produced masks.
- Usable masks and candidates cover only three videos: `train_014154` (20 candidates), `train_005918` (48), and `train_012957` (3). Nine videos have no candidates. The retained 71 rows are question-conditioned only. The candidate policy omitted 277 connected components beyond its per-frame cap and removed one exact duplicate.
- One qualitative R0 result is encouraging: SigLIP2 ranks a correct commercial-building component first among the incomplete candidates for `train_014154`. This is one inspected example, not a valid Hit@1 estimate. R1 gives the correct right-of relation for that candidate pair but does not change the ranking order. R2 attaches a two-frame association to the target candidates; temporal accuracy is not evaluated.
- **No aggregate Hit@1/5, random-baseline, joint-geometry, temporal-validity, or fixed-vocabulary comparison is scientifically reportable. Do not proceed to frozen-Qwen scoring on this run.**

## Cohort and source review

| Stratum | Questions |
|---|---:|
| Geometric | 4 |
| Temporal | 4 |
| Scene region | 2 |
| Ambiguity probes | 2 |
| **Total** | **12 questions / 12 videos** |

The cohort spans Hawk_UAV (7), ERA_Tra (2), VisDrone (2), and UAVDT (1). Source-only review covered the same 16 sampled frames for each video. The primary and independent second reviewers recorded target visibility, entity assessability, relations, temporal censoring, ambiguity, and identity limits before viewing predictions or retrieval rankings. The adjudicated source labels are in `annotations/source_review_adjudication.json`; original reviews remain preserved alongside it.

Notable limits in the frozen annotations include: the near-circular green roundabout island does not support a unique “orientation” comparison; the `train_012957` vehicle-group distance question is not stable enough to score; the bus interval in `train_001478` is right-censored at clip end; and the small motorcycle in `train_012936` is not a reliable unique-instance target. These remain unassessable rather than being forced into positive labels.

## Execution and proposal coverage

The pinned SAM 3 endpoint was `lucataco/sam3-video`, version `408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269`. Its returned ZIP masks do not include attributable per-mask native scores.

The provider error stated that prediction creation was limited to six requests per minute with burst one while account credit was below $5. The execution sent requests too rapidly for that limit. The manifest's run-wide attempt counter is 41; the resumed manifest stores 31 of the 36 429 error records, while five earlier failures are represented in the exhausted call-2 retry history. Five prediction IDs were created, of which two are duplicate executions of the same `commercial building` prompt. Four unique logical prompts succeeded: `commercial building`, `apartment complex`, `green space`, and `traffic lights`.

Created prediction IDs: `6b4y2255rhrmt0d19sybv125qm`, `ghznf33qahrmy0d19syrv6213r` (duplicate commercial-building prompt), `z2a9knzn4nrmr0d19t39tpvjd8`, `eadkka7h1drmy0d19t9v0ysczm`, and `4pmg9fgrzxrmy0d19ta8nk6ysg`.

| Video | Question-derived prompt results | Fixed control | Retained pool | Diagnostic |
|---|---|---|---:|---|
| `train_014154` | Commercial building: masks on 2/16 frames; apartment complex: 16/16 | Building: no output | 20 | Target found only in late frames; apartment prompt also covers multiple nearby structures. |
| `train_005918` | Green space: union masks on 16/16; roundabout and surrounding roads: no outputs | Vegetation: no output | 48 | Central island is included, but unrelated lawns/green strips are also masked; question’s orientation operation is unassessable. |
| `train_011679` | AMC theater, Cinemark building, parking lot: no outputs | Building: no output | 0 | No proposal evidence. |
| `train_012957` | Vehicles, intersection, white building: no outputs; traffic lights: masks on 3/16 | Vehicle: no output | 3 | Only tiny signal-head regions; target group and relation are absent. |
| `train_008503` | Building: no output | Building: no output | 0 | No proposal evidence. |
| `train_008163` | Construction site: no output | Construction site: no output | 0 | No proposal evidence. |
| `train_001478` | Bus: no output | Vehicle: no output | 0 | No proposal evidence for the reviewed late-appearing bus. |
| `train_010200` | Baseball field: no output | Sports field: no output | 0 | No proposal evidence. |
| `train_003224` | Structure and baseball field: no outputs | Building: no output | 0 | No proposal evidence. |
| `train_002092` | Land-cover region, median strip, road: no outputs | Vegetation: no output | 0 | No proposal evidence. |
| `train_012936` | Motorcycle: no output | Vehicle: no output | 0 | No proposal evidence for a framing-limited small target. |
| `train_014264` | Vehicles and highway: no outputs | Vehicle: no output | 0 | No proposal evidence. |

“No output” for throttled or skipped calls is an API execution failure, not a SAM 3 proposal miss. Only the four successful prompts support proposal-quality observations. The 71 retained candidates came from 37 union rows and 35 target-component rows before deduplication; 277 additional connected components were omitted by the predeclared maximum of two target components per prompt/frame. Thus even the three nonempty pools are capped partial pools.

### Source-frame and mask checks

Returned mask archives were checked for packed-frame indices, source-frame mapping, 640×360 source dimensions, and binary values (0/255). For the commercial-building example, packed frame 14 maps to source frame 210 at 7.007 s; packed frame 15 maps to source frame 225 at 7.5075 s (source 29.97003 fps). The target mask has 7,454 pixels at frame 210. At frame 225, the commercial-building prompt returns a union with two 8-connected components: 15,608 pixels over the apartment-complex side and 8,576 pixels over the target building. Component splitting isolates the target component, but the target prompt only returns it in 2 of 16 sampled frames despite source review finding the building visible throughout.

The green-space union appears in all 16 sampled frames, but combines the central roundabout island with several unrelated vegetation patches. The traffic-light masks occupy only 89–113 pixels each, about 0.04–0.05% of a 640×360 frame. These observations indicate sparse and/or overbroad proposal quality on the few calls that ran; they do not quantify IoU because dense boundary annotations were not made.

## Retrieval approaches

| Approach | What ran | What can be concluded |
|---|---|---|
| **R0 — noun-phrase SigLIP2** | Padded RGB contextual crops, pinned `google/siglip2-base-patch16-224` revision `022b6f71160ffb0169ca4709e2d7e25be659598a`; 71 candidates across three videos. | A single inspected commercial-building target candidate ranks first. Incomplete/capped candidate labels prevent valid Recall@K or random-baseline results. |
| **R1 — spatial/relational** | Deterministic geometry attached to available question-conditioned candidates; relation calculations use candidate coordinates/masks and the frozen question-only decomposition. | On `train_014154`, the top target candidate is right of the same-frame apartment-complex candidate, matching the blind source review. R1 leaves R0’s candidate order unchanged. Other relations lack required proposals. No R1 gain can be estimated. |
| **R2 — temporal association** | Adjacent-frame association metadata on available target candidates; 19 tracks and 131 links across the partial pool using the approved IoU/centroid/area rules. | The commercial-building component candidates at source frames 210 and 225 are associated into a length-2 track. This is a plausible link, not a temporal-identity accuracy result. None of the four temporal-question target prompts returned masks, so R2 temporal validity is untested. |

SigLIP2’s R0 commercial-building top five were: (1) correct target component at source frame 225, cosine 0.099722; (2) correct target region at frame 210, 0.096491; (3) broad frame-225 union containing both target and neighboring structures, 0.090074; (4) apartment-complex distractor component, 0.077725; (5) apartment-complex distractor union, 0.074195. This example shows both useful instance ranking and a proposal/over-merging problem. No candidate-complete manual labels exist, so it must not be reported as benchmark Recall@1.

Only padded RGB crops were encoded in this run. No new masked-crop comparison was performed. The fixed-vocabulary arm has zero candidates, so question-conditioned versus fixed-vocabulary proposal quality is unavailable. Native SAM 3 scores are also unavailable from these ZIP outputs. Although 1,000 matched-pool random top-five draws were saved for ranking rows, no relevant-candidate labels exist to score those draws or compute exact expected random Recall@K.

## SigLIP2 preprocessing audit

The run uses the pinned checkpoint and `AutoProcessor`/`AutoModel` from that local pinned snapshot. The crop pipeline passes RGB contextual crops through the processor; text and image features are L2-normalized before cosine ranking. Question phrases are normalized with Unicode NFC, case-folding, and whitespace collapse.

Text encoding uses `padding="max_length"` and `truncation=True`, but does not explicitly pass `max_length=64`; therefore, the effective length depends on the pinned tokenizer configuration. The run did not save actual token IDs, attention masks, sequence lengths, or special-token IDs, and the local audit environment did not contain the staged tokenizer files. The intended 64-token convention is **not independently verified** here. This is an evidence gap, not a demonstrated embedding bug; no embeddings were rerun to investigate it.

## Runtime, hardware, API use, and cost

| Component | Runtime / memory | Cost status |
|---|---|---|
| Replicate SAM 3 predictions | Five prediction IDs; provider-reported aggregate prediction time about 14.94 s (one was a duplicate prompt). The 36 429 create rejections had no prediction IDs. | Actual invoice amount not available. At the prior approximate $0.0017 per successful prediction, five predictions imply roughly **$0.0085**, not an invoice. The runner’s `$0.0697` estimate incorrectly multiplied the per-prediction estimate by all 41 HTTP attempts and is not valid. |
| Modal no-retry continuation (`ap-zFKh7OgHGn2483v6YmgEQJ`) | 30.61 s wall; peak RSS 3.59 GB; no CUDA device used. | Latest app charge not reconciled. |
| Modal cached SigLIP2 postprocess (`ap-ExOORN7YjBfariOfqjRUMj`) | 22.756 s wall on one NVIDIA A100-SXM4-40GB; peak allocated CUDA memory 1.78 GB, reserved 1.99 GB; peak RSS 5.98 GB. | Latest app charge not reconciled. |

The prior Modal billing reconciliation for six preparation apps was $0.04079229, including $0.02800474 of A100 resource charges; that amount excludes the two apps above. Two attempts to retrieve the latest Modal billing report hung and were stopped, so the total Phase 4 Modal charge remains unknown. No actual provider spend is claimed from application-level counters.

## Research decision

This run does **not** establish whether QCSRR improves retrieval across videos or whether R1/R2 improve ReVA. It provides one positive instance-ranking example, and shows sparse coverage (commercial-building target 2/16; signal reference 3/16), a broad green-space union, and a clear endpoint-throttling failure. The fixed-vocabulary control, nine videos, most references, and every temporal target are unobserved.

Do not proceed to frozen-Qwen evidence integration or promote the architecture. Completing the approved cohort requires a new call allowance for the 32 logical prompts with no successful output, plus a fresh bounded retry reserve. A resumed runner should honor the provider’s burst/requests-per-minute limit, poll provider retry timing, and reuse the five existing prediction outputs. No new calls were made after the approved 41-attempt cap was reached.

Qwen scoring, training, depth/3D models, Codabench, and the 32-question pilot were not run.

## Visual audit artifacts

*[Image omitted from public mirror: `results_sparse_masks/partial_prompt_mask_examples.png`]*

*[Image omitted from public mirror: `results_continued/postprocess/partial_ranked_commercial_building_top5.png`]*

*[Image omitted from public mirror: `results_continued/partial_green_space_masks.png`]*

*[Image omitted from public mirror: `results_continued/partial_traffic_light_masks.png`]*
