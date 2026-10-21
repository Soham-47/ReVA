# Phase 3B Gate 2 — minimal retrieval validity benchmark

## Exact questions recorded before target identifiability labels

- `train_010397`: “At what time does the black car appear in the lower right section of the parking lot and when does it disappear?”
- `train_009869`: “From which timestamp to which timestamp is the baseball field continuously visible in the video?”

## Frame selection

Use packed frame 0 and packed frame 15 for each video. This rule was fixed before reading any SigLIP2 frame rankings. These are 0 s and 7.5075 s for `train_010397`, and 0 s and 5.6056 s for `train_009869`. Source coordinates remain 640×360. This is a frame-level endpoint audit, not a continuous-visibility annotation.

## Pools and complete labeling scope

Retain every Gate 1 candidate in the selected frame and video/arm pool, after the existing image-only filters. This yields 281 unique candidate IDs (104, 97, 52, and 28 by the four video/frame groups). For `train_010397`, label every candidate against both question-derived entities (`black car`, `parking lot`); for `train_009869`, label every candidate against `baseball field` (482 candidate/entity judgments). Candidate labels are evaluation-only and are never supplied to retrieval.

The existing proposal pools are unchanged: question-conditioned SAM 3 candidates and fixed-vocabulary candidates. Padded RGB and masked RGB use the same frozen candidate IDs, labels and question phrase; only the stored ranking differs by crop representation. Exact-mask duplicates remain separate candidate IDs for the primary ranking metric and are reported as duplicate evidence.

## Label definitions

- **Car category / black attribute:** the CSV uses a reproducible *operational proxy*, not a second-rater manual ground truth. A proposal is treated as object-level only when it is a connected component from the `black car` or `vehicle` prompt with mask area ≥50 pixels. Median masked-pixel luminance ≤85 is labeled dark, ≥120 not-dark, and between ambiguous. The proxy can misclassify shadows, roof material, or tiny aerial vehicles at 640×360. Broad/unsplit masks are not credited as object-level black-car proposals. A category-positive candidate is not the question-referenced instance.
- **Question-referenced car instance:** unassessable in both sampled frames. The exact wording supplies “lower right section,” but multiple dark cars and moving vehicles occur there and across the scene; these frames do not establish a unique object or cross-frame identity. Do not score instance Recall@K for this query.
- **Parking-lot category/context:** the candidate CSV’s `context_label` is also an operational proxy. A connected-component/context-region proposal with mask area ≥500 pixels and proposal-box area ≤45% of frame is `useful_context`; an original unsplit mask or box covering ≥50% of the frame is `overbroad`; a connected component under 250 pixels is not useful; other cases are ambiguous/partial. The broader category label credits both useful and overbroad proposals as containing parking-lot scene content. No precise lot polygon or unique region instance is asserted. These labels do not support exact boundary IoU.
- **Baseball field:** frame 0 contains one visually clear baseball/softball diamond. An approximate, manually traced polygon is stored in `evaluation_targets.json` and used only for evaluation. It is not an inference input. Coverage is the fraction of polygon pixels covered by a candidate mask or proposal box, not mask IoU. Coverage ≥0.50 is positive, <0.10 negative, and [0.10, 0.50) ambiguous/partial. The mask and box are scored separately. Contextual crops can be relevant while the mask boundary remains poor. At frame 15 the target is absent: the question-conditioned pool is empty and fixed-vocabulary proposals are false positives, not misses of a visible target.
- **Correct instance for the field:** assessable in frame 0 as the only clear diamond. This does not establish cross-frame identity or the answer to “continuously visible.”

These operational labels were assigned to all selected candidate rows, but they are not equivalent to an independent visual audit of all semantics. Black-car category labels depend on prompt/type/area/luminance rules; parking-context labels depend on size rules; the baseball target polygon is approximate and hand drawn. One reviewer prepared the rubric and reviewed rank-free visual sheets after prior exposure to ranked examples. There was no independent second annotator or adjudication. Treat metric values as provisional diagnostics, not validated ground truth.

## Ranking and random comparison

For each selected frame, query, proposal arm and crop style, calculate Recall@1 and Recall@5 as the number of relevant candidates retrieved in top K divided by all relevant candidate IDs in that frozen pool. Also report the number of positive, negative and ambiguous candidates. Report query-level/video-level top-5 unions descriptively only. For complete binary labels, compare the saved 1,000 matched-pool random rankings and the hypergeometric expected random Recall@K. Where ambiguous labels remain, report matched-random simulations only as provisional bounds/checks and do not call them exact. Do not estimate a global random baseline from the partially labeled global top-five union.

Duplicates are not removed from the frozen candidate pool; in addition to candidate-ID metrics, report unique evidence counts and whether top-K slots repeat an identical mask/crop. Candidate-ID Recall can therefore overstate the number of unique visual regions retrieved.

## Blinding and limits

The contact sheets hide rank, score, arm and proposal prompt. Strict blinding was nevertheless compromised: prior Gate 1 review and this task history exposed several ranked examples before annotation. One reviewer reviewed source frames and rank-free sheets; several candidate labels are generated by deterministic proxies rather than independent manual semantic judgments. The two source videos are diagnostic cases, not independent video-level samples; two endpoints cannot validate temporal continuity, tracking, or generalization.
