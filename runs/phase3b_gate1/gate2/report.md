# Phase 3B Gate 2 — Minimal Retrieval Validity Benchmark

## Decision

The frozen SigLIP2 rankings show a useful, narrow signal for **parking-scene context in padded RGB crops**. They do not establish black-car instance retrieval. In the scored endpoint pools, baseball evidence comes from one clear frame and often uses broad crops; a separate global top-five inspection finds more clear intermediate frames but has incomplete labels. These two videos and four scored frames are not enough to justify testing retrieved crops with frozen Qwen3-VL. Use a small, cleaner diagnostic cohort with identifiable targets and explicit temporal labels first.

No embeddings, model inference, Qwen scoring, or paid jobs were run for this review.

## Tokenization and implementation check

The pinned snapshot is `google/siglip2-base-patch16-224`, revision `022b6f71160ffb0169ca4709e2d7e25be659598a`; the cached files load as `SiglipProcessor` with `GemmaTokenizer` under Transformers 5.19.0. The checkpoint config reports `model_type=siglip`.

The tokenizer backend pads on the right to length 64 and truncates at 64. Default processor output and explicit `max_length=64` are byte-identical. The official [SigLIP2 preprocessing convention](https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/siglip2.md) uses fixed padding/truncation to 64.

| Input | Active token IDs, including EOS | Active length | Total length |
|---|---|---:|---:|
| `black car` | `[6944, 1226, 1]` | 3 | 64 |
| `parking lot` | `[74811, 2940, 1]` | 3 | 64 |
| `baseball field` | `[111021, 2725, 1]` | 3 | 64 |
| Full black-car question | Recorded in `tokenization_cpu.json` | 24 | 64 |
| Full baseball question | Recorded in `tokenization_cpu.json` | 17 | 64 |

The processor returned only `input_ids`; there was no `attention_mask` tensor in its output or Gate 1 call. An inferred non-padding mask is saved for inspection, but it was not passed to the model. With the pinned Transformers 5.19.0 SigLIP text forward path, the missing mask means bidirectional attention over the fixed 64 positions, and the model pools the final position, which is padding. This is the fixed-length SigLIP path; it is not an explicit non-pad mask.

The inferred binary masks are `[1,1,1,0×61]` for each short entity phrase, `[1×24,0×40]` for the full black-car question, and `[1×17,0×47]` for the full baseball question. These are derived for inspection from PAD ID 0, not actual model inputs; the actual passed `attention_mask` was absent (`None`).

Tokenizer special IDs are BOS=2, EOS=1, PAD=0, while the loaded text config defaults to BOS=49406, EOS=49407, PAD=1. The warnings are a real metadata mismatch, but the exact lowercase Gate 1 strings were tokenized with IDs inside the 256k vocabulary, and this text forward path does not use its configured BOS/EOS IDs to select the pooling position. No material mismatch was found for the three lowercase phrases, so embeddings were not rerun. The `GemmaTokenizer` call did **not** lowercase `BLACK CAR`; it tokenizes differently from `black car`, despite `do_lower_case: true` in config. Gate 1's query strings were already lowercase. Future full-question or uppercase queries need explicit verified lowercasing. The CPU check did not run a model forward pass.

## Sample and labels

The frame rule was packed frames 0 and 15 per video, chosen before looking at per-frame ranks:

| Video | Packed frame | Source frame | Timestamp | Size |
|---|---:|---:|---:|---|
| `train_010397` | 0 | 0 | 0.0000 s | 640×360 |
| `train_010397` | 15 | 225 | 7.5075 s | 640×360 |
| `train_009869` | 0 | 0 | 0.0000 s | 640×360 |
| `train_009869` | 15 | 168 | 5.6056 s | 640×360 |

All 281 candidate IDs in these frame pools were labeled against the requested entities: 482 candidate/entity rows. The full questions were recorded before target-identifiability decisions:

- `train_010397`: “At what time does the black car appear in the lower right section of the parking lot and when does it disappear?”
- `train_009869`: “From which timestamp to which timestamp is the baseball field continuously visible in the video?”

Black-car **instance identity is unassessable**. The question supplies a spatial phrase, but multiple dark cars and moving vehicles occur in the scene; the sampled views do not establish one target car or its cross-frame identity. Instance Recall@K is therefore not calculated. Car category values below use a declared prompt/component/area/luminance proxy, not objective target labels.

Parking-lot scores use a declared local-context rubric, not a manually defined lot boundary. `Useful context`, `overbroad`, and `ambiguous/partial` are reported separately. Baseball relevance uses a manually approximated frame-0 diamond polygon and mask/box coverage thresholds; these are coverage proxies, not mask IoU. The frame-15 baseball target is absent. There was one reviewer, no second-rater adjudication, and strict blinding was impossible because earlier turns exposed ranked examples. Treat the labels and metrics as provisional diagnostics.

The pools contain 281 IDs but 277 unique mask/crop evidence items: four exact duplicates occur across the two proposal arms for `train_010397`; none repeats within a single arm/frame pool or in a selected top-five list. The 1,000 matched random rankings use each same candidate pool. In incomplete-label groups they are comparisons against definite positives only; no global random baseline is calculated.

## Retrieval results

Recall@K below is the fraction of all definite positive candidates in the pool found in the top K. `Hit@5` is also provided where it helps interpret a pool with many positives. Each row is one frame from one of two videos; these are descriptive observations, not independent video samples.

### Black-car category proxy

| Frame / arm | Pool; positive / ambiguous | Padded RGB R@1 / R@5 | Masked RGB R@1 / R@5 | Matched random R@1 / R@5 | Hit@5: SigLIP / random |
|---|---:|---:|---:|---:|---:|
| 010397 f0, question-conditioned | 51; 36 / 15 | 0.028 / 0.139 | 0.028 / 0.139 | 0.020 / 0.097 | 1.00 / 0.997 |
| 010397 f0, fixed-vocabulary | 53; 34 / 6 | 0.029 / 0.147 | 0.029 / 0.147 | 0.019 / 0.096 | 1.00 / 0.996 |
| 010397 f15, question-conditioned | 47; 33 / 14 | 0.030 / 0.121 | 0.030 / 0.152 | 0.021 / 0.107 | 1.00 / 0.997 |
| 010397 f15, fixed-vocabulary | 50; 30 / 7 | 0.033 / 0.100 | 0.033 / 0.133 | 0.020 / 0.099 | 1.00 / 0.990 |

The ranking only modestly exceeds random on this class proxy, while random already has a 99–99.7% chance of putting some positive in the top five because 30–36 candidates are positive. This does not demonstrate retrieval of the referenced car. Padded and masked rankings tie at frame 0; masked crops are slightly better at frame 15, with no consistent overall winner.

For example, the question-conditioned padded top-1 at frame 0 is candidate `P0196` (similarity 0.12481; box x=162–186, y=182–205). The representation query was only `black car`; it contained no lower-right or parking-lot relation. The rank is not evidence that this is the question's target.

### Proposal misses versus ranking errors

- **Black car:** both proposal arms contain many dark vehicle-like components. The correct-instance label is unassessable, so candidate generation and ranking cannot be separated for the target car. The class-only rank is not a substitute for an instance match.
- **Parking lot:** useful local regions exist in every selected pool. The padded question-conditioned top five at frame 0 contains five rubric-positive context regions; its masked top five contains none. This points to loss of scene context in masked crops, not a missing proposal in that frame.
- **Baseball field:** correct frame-level proposals exist at frame 0 in both arms. The fixed-vocabulary padded ranking retrieves all four definite target-coverage positives; the masked ranking retrieves three. At the later endpoint the target is absent, so the empty question-conditioned pool is not a miss of a visible field; the 28 fixed proposals are false positives. The ambiguous intermediate interval is a source-labeling limit, not evidence that another model is required.

The padded question-conditioned parking top five at frame 0 are `P0090` (0.14951), `P0072` (0.14279), `P0044` (0.12933), `P0017` (0.12573), and `P0011` (0.12129); all five are useful under the declared rubric. For baseball, the fixed-vocabulary padded top five begin `P0218` (0.14435; full-frame union), `P0192` (0.14312), `P0198` (0.14273), `P0132` (0.14196; partial/ambiguous), and `P0319` (0.14002). Four are definite target-region positives, but the whole-frame and broad-box candidates show that scene identification does not imply a precise field mask.

### Parking-lot scene context

This table counts only `useful_context` candidates as relevant. Broad candidates are listed separately rather than credited as useful regions.

| Frame / arm | Pool; useful / broad / ambiguous | Padded RGB R@1 / R@5 | Masked RGB R@1 / R@5 | Matched random R@1 / R@5 | Hit@5: padded / masked / random |
|---|---:|---:|---:|---:|---:|
| 010397 f0, question-conditioned | 51; 7 / 3 / 14 | 0.143 / 0.714 | 0.000 / 0.000 | 0.018 / 0.101 | 1.00 / 0.00 / 0.552 |
| 010397 f0, fixed-vocabulary | 53; 6 / 1 / 12 | 0.167 / 0.667 | 0.167 / 0.500 | 0.019 / 0.101 | 1.00 / 1.00 / 0.495 |
| 010397 f15, question-conditioned | 47; 10 / 3 / 13 | 0.100 / 0.500 | 0.000 / 0.100 | 0.021 / 0.106 | 1.00 / 0.20 / 0.707 |
| 010397 f15, fixed-vocabulary | 50; 7 / 1 / 12 | 0.143 / 0.714 | 0.143 / 0.429 | 0.023 / 0.097 | 1.00 / 1.00 / 0.521 |

Padded boxes consistently rank more useful scene context than masked crops under this rubric, and the padded top five beat the matched random top-five recall in all four pools. Masking often removes the surrounding pavement/stalls that makes a crop useful for a scene-level query. The pool label remains a size-based proxy because there is no agreed boundary for the lot.

### Baseball-field target region

| Frame / arm | Pool; positive / ambiguous | Padded RGB R@1 / R@5 | Masked RGB R@1 / R@5 | Matched random R@1 / R@5 | Hit@5: SigLIP / random |
|---|---:|---:|---:|---:|---:|
| 009869 f0, question-conditioned | 6; 4 / 0 | 0.250 / 1.000 | 0.250 / 1.000 | 0.164 / 0.834 (exact: 0.167 / 0.833) | 1.00 / 1.00 |
| 009869 f0, fixed-vocabulary | 46; 4 / 4 | 0.250 / 1.000 | 0.250 / 0.750 | 0.023 / 0.109 | 1.00 / 1.00 |

The target frame-level diamond region is covered by four candidates in each arm; the fixed pool also has four partial/ambiguous candidates. In the six-candidate question-conditioned pool, random retrieval already reaches top-five recall 0.833 and hit@5=1, so the top-five result is not informative. In the 46-candidate fixed pool, the padded ranking finds all four definite candidates while random finds about 0.11 of them on average. However, several top results are near-whole-frame or very broad context regions. The approximate polygon gives target coverage, not precise mask IoU. At f15 the field is not visible: question-conditioned pool size is zero, and all 28 fixed-vocabulary proposals are false positives for that frame. No score is defined for an absent target.

### Global top-five union (qualitative only)

The four frozen ranking variants contributed 60 top-five mentions, representing 53 unique candidates: 19 black-car, 20 parking-lot, and 14 baseball-field candidates. Global candidate-level labels are in `global_top5_qualitative_review.csv`; no global Recall or random baseline is computed. Ten baseball candidates come from packed frames 6–7, where a diamond is clear; four candidates from frames 8–10 come from the ambiguous similar-sports-field portion of the video. The black-car union contains object-like vehicle components, but none can be assigned to the referenced instance. Parking results include local parked-car/pavement crops and several overbroad views.

## Question decomposition

The current `black car` query loses important wording from the exact question: “lower right section,” the relation “of the parking lot,” and the appearance/disappearance timing request. A future question-only representation should preserve attributes and relations separately, such as `target=car`, `attribute=black`, `located_in=parking lot`, `spatial_anchor=lower-right section`, and `temporal_operation=first/last visibility`. The system can use these constraints against detected scene regions and frame locations without reading gold answers or manually supplied target coordinates. The frozen ranking experiment remains unchanged.

The baseball query likewise reduces “continuously visible” to `baseball field`; that loses the interval/continuity requirement. The endpoint benchmark cannot validate that temporal operation.

## Research decision

1. **Better than random on assessable categories?** Yes for padded parking-context ranking under the rubric, and for fixed-vocabulary baseball padded ranking on the one clear frame. Black-car category ranking is only slightly above random and has a near-certain random top-five hit because the pool contains many positive class proxies. The question-conditioned baseball pool is too positive-heavy for a useful top-five comparison.
2. **Specific objects or general class?** Only general car/category evidence. The black-car target instance and cross-frame identity are not assessable. A single baseball diamond is assessable in frame 0, but identity across the video and the continuous interval are not.
3. **Which crops?** Padded RGB is the clear choice for scene context. Masked crops are mixed for small-object categories and lose parking context; fixed-vocabulary baseball ranking also drops from R@5=1.0 to 0.75 when masked.
4. **Does decomposition discard critical reference information?** Yes. It drops the black car's location/scene relation and the temporal event; it also drops “continuously visible” for baseball.
5. **Are these videos enough?** No. They are useful diagnostic cases, not a valid architecture benchmark: one instance is ambiguous, a scene boundary is undefined, the baseball target changes visibility and then becomes ambiguous, and there are only two source videos.
6. **Enough evidence for frozen Qwen3-VL crop testing?** No. The study validates a possible padded scene-context signal, not target-instance or temporal retrieval. First assemble a small cohort with objectively identifiable instances, clear source-frame visibility, and independent relevance annotations.

## Artifacts

- Candidate-level labels: `candidate_relevance_annotations.csv`
- Per-frame rankings and matched random results: `per_frame_retrieval_metrics.csv`
- Global top-five qualitative review: `global_top5_qualitative_review.csv`
- Tokenizer IDs and inferred non-pad masks: `tokenization_cpu.json`
- Tokenization implementation notes: `siglip_implementation_review.md`
- Selection, rubric, and blinding limits: `annotation_protocol.md`
- Frozen visual sheets: `blind_sheets/`

