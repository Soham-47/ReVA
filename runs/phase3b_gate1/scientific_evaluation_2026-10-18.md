# Phase 3B Gate 1 — Scientific evaluation

**Status: diagnostic review complete; metric audit incomplete.** No model or API inference was run for this review. The frozen rankings provide plausible semantic examples, but the candidate-level annotation template remains blank for all 2,319 candidates. Category Recall@1/5, correct-instance Recall@1/5, and matched-random comparisons therefore cannot be reported as scientific results.

## 1. SigLIP2 implementation check

- **Checkpoint:** `google/siglip2-base-patch16-224`, commit `022b6f71160ffb0169ca4709e2d7e25be659598a`. The execution manifest records this revision. The Modal job verified the staged model files against the pinned snapshot hashes before loading them.
- **Image preprocessing:** RGB; resize to 224×224 with resample value 2 (PIL bilinear); rescale by 1/255; normalize each channel with mean/std 0.5, giving values in [-1, 1]. This matches the pinned [processor configuration](https://huggingface.co/google/siglip2-base-patch16-224/blob/022b6f71160ffb0169ca4709e2d7e25be659598a/preprocessor_config.json).
- **Features:** `get_image_features` and `get_text_features`; pooled output is converted to float and L2-normalized before cosine ranking. This part is consistent with cosine similarity on normalized features.
- **Text:** the three stored phrases are already lowercase. The run calls the checkpoint processor with `padding="max_length"` and `truncation=True`, but it does **not** pass `max_length=64` and did not save the resulting token IDs or attention-mask shape. The pinned preprocessing file identifies `SiglipProcessor`; the current [Transformers SigLIP2 guidance](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md) specifies 64 tokens for the trained padding/truncation convention, while the pinned [tokenizer configuration](https://huggingface.co/google/siglip2-base-patch16-224/blob/022b6f71160ffb0169ca4709e2d7e25be659598a/tokenizer_config.json) uses `GemmaTokenizer` and a sentinel-sized `model_max_length`. Because the exact effective text length used by Transformers 5.19.0 is not recorded, I cannot verify that the run used length 64. Treat rankings as exploratory until this one preprocessing detail is checked. The existing tokenizer output may have been padded dynamically; that is not confirmed from the artifacts.
- **Special-token warning:** the reported `49406/49407` versus `vocabulary size 32000` warning matches a Transformers configuration-validation warning documented for SigLIP2: the loaded text vocabulary is 256,000 entries, so those numeric IDs are in range. The tokenizer’s own Gemma special tokens are IDs 0/1/2. The warning alone is not evidence that the text embeddings used out-of-range token IDs or were corrupted; it is consistent with an incorrect default-config comparison in Transformers ([issue #47612](https://github.com/huggingface/transformers/issues/47612)). Exact run-time token IDs were not saved, so the warning cannot be fully cleared from logs alone.

No checkpoint change or inference rerun was made. Before relying on the ranking as a final benchmark, record the runtime processor/tokenizer class, effective sequence length, and a tokenization fingerprint for these exact phrases; the simplest future call should explicitly set `max_length=64`.

## 2. Source-video relevance review

The videos and sampled frames are 640×360 at approximately 29.97 fps. Source review was possible for all 32 sampled frames. Rank-free sheets hide rank and proposal metadata, but prior exposure to some ranked examples means this is **not a fully blinded audit**. The existing CSV has 2,319 blank candidate rows; none of the candidate-level labels were filled.

| Query | Source-frame finding | Instance/identity judgment |
|---|---|---|
| Black car (`train_010397`) | Cars are visible in all 16 samples. A dark parked vehicle is visible in the lower-right area, among several other dark and moving cars. | The question-referenced vehicle cannot be uniquely identified. No cross-frame identity or target visibility labels are defensible from these samples. Do not score exact-instance retrieval. |
| Parking lot (`train_010397`) | Parking rows and paved parking context are visible in all 16 samples. | The lot boundary blends into access roads and a paved plaza. No unique region polygon or candidate relevance labels are available. |
| Baseball field (`train_009869`) | A baseball/softball diamond is clear in samples 0–7, timestamps 0.000–2.603 s. Similar dirt/sports-field regions make samples 8–13 ambiguous; samples 14–15 do not clearly show the same baseball field. | Same-field identity across frames is unverified. The late samples must not be assigned the target instance merely because they contain green or tan field-like areas. |

The complete frame-level judgment record is in independent_source_review_2026-10-18.json (local-only artifact: `annotations/independent_source_review_2026-10-18.json`). These are source-scene judgments only, not candidate labels.

## 3. Frozen candidate pools and ranking examples

| Video / arm | Candidate count | Pool composition |
|---|---:|---|
| `train_010397`, question-conditioned | 769 | 721 connected components, 32 original masks, 16 context regions; sources include black-car and parking-lot proposals |
| `train_010397`, fixed vocabulary | 828 | 812 connected components, 16 original masks; vehicle proposals only |
| `train_009869`, question-conditioned | 84 | 45 connected components, 24 context regions, 15 original masks |
| `train_009869`, fixed vocabulary | 638 | 168 connected components, 454 context regions, 16 original masks |

The fixed-vocabulary pool for `train_010397` contains **no parking-lot prompt**; its parking-lot comparison ranks vehicle proposals. It is not a like-for-like parking-lot proposal arm. SigLIP2 and the saved random draws use the same candidate rows within each arm; arm pool sizes differ. Across the combined artifact there are 59 exact-mask duplicate groups (59 extra rows). Within evaluated `(video, arm)` pools, the only duplicates are two extra rows in the question-conditioned baseball-field pool. Any row-based metrics must disclose these duplicates.

The frozen job has 204 ranking groups (192 per-frame, 12 video-global) and 102 random-baseline groups, each with 1,000 saved draws. Native SAM 3 proposal scores are unavailable in the binary mask archives.

### Video-global top-one examples

Crop fraction is the input crop rectangle area divided by the 640×360 frame area. Scores are cosine similarities, not calibrated probabilities.

| Query | Arm / crop | Score | Selected source time | Crop fraction | Visual diagnosis |
|---|---|---:|---:|---:|---|
| Black car | Question-conditioned / padded RGB | 0.1248 | 0.000 s | 0.33% | Small isolated vehicle-like component, but its box is left of the requested lower-right area; instance match is unverified. |
| Black car | Question-conditioned / masked RGB | 0.1139 | 2.002 s | 0.29% | Small vehicle-like component; no identity proof. |
| Black car | Fixed / padded RGB | 0.1224 | 1.502 s | 0.15% | Vehicle component in another part of the scene. |
| Black car | Fixed / masked RGB | 0.1047 | 2.503 s | 0.30% | Vehicle component; no identity proof. |
| Parking lot | Question-conditioned / padded RGB | 0.1534 | 2.002 s | 12.55% | Plausible contextual parking-row region. |
| Parking lot | Question-conditioned / masked RGB | 0.1132 | 3.003 s | 100.0% | Near-whole-frame region; high background and weak localization. |
| Parking lot | Fixed / padded RGB | 0.1549 | 7.007 s | 3.79% | Vehicle component from a pool with no parking-lot proposals. |
| Parking lot | Fixed / masked RGB | 0.1107 | 2.503 s | 4.02% | Vehicle component, not a lot-boundary proposal. |
| Baseball field | Question-conditioned / padded RGB | 0.1523 | 2.236 s | 77.81% | Contains a field, but the crop is broad (original unsplit proposal). |
| Baseball field | Question-conditioned / masked RGB | 0.1479 | 2.603 s | 93.91% | Near-whole-frame context region. |
| Baseball field | Fixed / padded RGB | 0.1532 | 2.236 s | 100.0% | Whole-frame context proposal. |
| Baseball field | Fixed / masked RGB | 0.1501 | 2.603 s | 100.0% | Whole-frame context proposal. |

Across per-frame top-one rows, padded RGB has higher median cosine than masked RGB in every arm, but that is not an accuracy result:

| Query | Question-conditioned median cosine (padded / masked) | Fixed median cosine (padded / masked) | Median top-one crop area fraction (Q padded / Q masked / F padded / F masked) |
|---|---:|---:|---:|
| Black car | 0.1165 / 0.1029 | 0.1155 / 0.0990 | 0.33% / 0.26% / 0.14% / 1.05% |
| Parking lot | 0.1512 / 0.1094 | 0.1515 / 0.1013 | 3.49% / 100.0% / 1.86% / 3.99% |
| Baseball field | 0.1471 / 0.1444 | 0.1454 / 0.1445 | 51.2% / 61.3% / 49.9% / 65.2% |

For parking and baseball, contextual RGB helps preserve scene information, but field crops frequently occupy half to nearly all of the source frame. The report’s crop-area fraction is **not** a target-background fraction; no target polygons were annotated, so exact target/background proportion cannot be calculated.

The contact sheets show the main examples: global top-one source frames (local-only artifact: `global_top1_source_frames.png`), [rank-free candidate sheets](annotations/blinded_review/README.md), and [fixed-layout ranked top-five sheets](annotations/ranked_top5_review/README.md).

## 4. Retrieval metrics and random baseline

| Metric | Status | Reason |
|---|---|---|
| Category Recall@1/5, per frame | Not reported | Candidate relevance labels are absent; only the top-one examples above received qualitative visual inspection. |
| Category Recall@1/5, video/query level | Not reported | Global top-one examples are not a full relevance-labeled top-five audit. |
| Correct-instance Recall@1/5 | N/A for these examples | Black-car identity and baseball-field identity are unresolved; parking-lot boundaries are not uniquely defined. |
| Padded RGB vs masked RGB accuracy | Not computed | Similarity magnitudes and ranks exist, but candidate-level relevance labels do not. |
| Question-conditioned vs fixed-vocabulary accuracy | Not computed | Candidate labels are missing and the parking-lot fixed pool has no parking-lot proposal. |
| SigLIP2 vs random | Not computed | The 102 saved simulations contain candidate IDs but no relevance labels. |
| Native SAM 3 score comparison | Unavailable | Saved masks are binary, without attributable native scores. |

For a pool with `N` candidates and `R` labeled relevant candidates, the exact random probability of retrieving at least one relevant candidate in top `K` is `1 − C(N−R,K)/C(N,K)`. Expected set-recall (`number of relevant rows returned / R`) for uniformly sampled top `K` is `K/N`. Here `N` is known, but `R` is not, so neither per-pool random hit probability nor its simulation cross-check can be computed. The saved 1,000-draw random runs alone do not establish performance.

The 16 frames in a clip are temporally correlated and represent only one video per query. They are not 16 independent video samples; no confidence interval or generalization claim is warranted.

## 5. Scientific decision

- **Component splitting:** It creates many small, often vehicle-shaped proposals (588 black-car-source components and 812 fixed-vehicle components in this video). This is useful candidate recovery, but correct target-instance coverage is unmeasured. Small fragments and repeated/multiple-car regions remain visible in the proposal sheets. The target cannot be verified independently, so do not report these as true instance recoveries.
- **SigLIP2 ranking:** The global examples are semantically plausible for cars and fields, and the parking-lot question-conditioned padded crop is a useful contextual example. This does not show that SigLIP2 ranks the right question-referenced instance. For black car, the global top-one boxes do not consistently honor the lower-right clue; for baseball, top-one proposals are often very broad.
- **Crop choice:** Padded RGB is the better provisional representation for scene context and yields higher cosine values. Masked crops are less informative for lot context, and the masked parking top-one is almost the whole frame. No retrieval-accuracy preference can be established without annotations.
- **Scene-region representation:** Context boxes can carry parking-lot and field context, but field boxes are too broad for useful boundaries. A tighter scene-region proposal is needed before claiming localization.
- **Remaining proposal failures:** Car unions/components do not establish the right vehicle identity. Parking-lot boundaries merge with roads/plaza; the fixed pool lacks a lot prompt. Field candidates include broad background and late-frame false positives; target field identity and boundaries are uncertain.
- **Resolution / new models:** The originals are only 640×360. Upscaling cannot recover missing source detail. Tiling could use those pixels more effectively at the model input, but would require a new inference run. Reliable black-car identity likely needs instance localization plus tracking; precise baseball boundaries may need higher-effective-resolution or scene-specific proposals. None was tested here.
- **Proceed to more independent videos?** **Not yet.** The evidence is qualitative, candidate relevance is 0/2,319 labeled, exact-instance assessability is unresolved, and the effective text sequence length is not recorded. First freeze/verify the tokenizer length and complete a candidate-level annotation on a predeclared subset or the full pools. Do not use these results as a Phase 3A pilot gate.

## 6. Execution and cost record

The approved ranking job ran on **one Modal NVIDIA A100-SXM4-40GB**. Wall time was 44.668 s including model download and load; the recorded GPU-seconds upper bound is also 44.668 s. Peak CUDA allocation was 1.780 GB (1.999 GB reserved). The function was allocated 4 CPUs and 16 GiB system memory. Inputs staged 19.37 MB and the model snapshot was 1.539 GB. The retained provider billing report sums to **$0.04431426** across CPU, memory, A100, and network egress. No new resource use occurred during this source/ranking review.

## Artifacts

- [This evaluation report](scientific_evaluation_2026-10-18.md)
- Independent source-frame review (local-only artifact: `annotations/independent_source_review_2026-10-18.json`)
- Blank candidate annotation template (local-only artifact: `annotations/blinded_review/annotation_template.csv`)
- [Rank-free review sheets and blinding note](annotations/blinded_review/README.md)
- [Ranked top-five sheets with corrected label layout](annotations/ranked_top5_review/README.md)
- Frozen rankings (local-only artifact: `rankings.json`), random draws (local-only artifact: `random_baseline.json`), execution manifest (local-only artifact: `execution_manifest.json`), and provider billing report (local-only artifact: `modal_billing_report_2026-10-18.json`)
