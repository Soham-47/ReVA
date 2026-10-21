# Phase 4 QCSRR — derived Gate evaluation

**Status: complete from frozen artifacts.** This report joins the 470 source-only candidate labels to the rank-independent fixed-frame manifest and cached SigLIP2 score reconstruction. It replays frozen R1 geometry ordering from candidate metadata and archived prompt scores, and reads archived R2 track metadata. No model, provider, inference, or Qwen call was made.

## Decision

The fixed-frame Gate is **quantitatively scoreable for the labeled candidate pools**, with explicit visibility and source-review eligibility limits. Use the results for this 470-row fixed-frame packet only. They do not establish unlabeled-video retrieval or temporal event accuracy.

Category retrieval is measurable on source-visible, source-assessable query pools. The matched random baseline is computed exactly from each pool's `n` positive candidates and `r` candidates; it is not a global 1/r approximation. R1 can change ordering through target-provenance/component filtering and frozen geometry ordering; the report separates those effects. R2 cannot improve ranking by design and remains descriptive except for the source-reviewed bus interval.

## Frozen inputs and validation

| Artifact | Check |
|---|---|
| Fixed-frame manifest | 470 aliases, packed frames 0 and 13, both arms; 290 fixed-vocabulary and 180 question-conditioned rows |
| Candidate labels | 470 unique aliases, all required fields, source-only category/instance/crop labels |
| Cached score reconstruction | 72 query pools, 821 score observations, 470 unique aliases; archived score validation max absolute error <= 1e-6 |
| Frozen candidate metadata | Every manifest candidate ID, frame index, source index, and mask digest matched |
| Frozen R0/R1/R2 archive | 36 ranking rows preserved for the full 3,689-candidate experiment; used descriptively and for R2 track metadata |

The score reconstruction's 72 pools are the two selected packed frames for each retained prompt/arm combination. The same candidate can occur in several phrase pools because a mask may carry multiple prompt provenance entries. Metrics below are query-pool metrics; they are not independent-video estimates.

## Label denominators and eligibility

The 470 frozen aliases have category visibility counts `{'absent': 6, 'ambiguous': 12, 'visible': 452}` and singular target visibility counts `{'absent': 6, 'ambiguous': 38, 'visible': 426}`. Candidate category labels are `{'ambiguous': 54, 'negative': 46, 'positive': 370}`; correct-instance labels are `{'ambiguous': 89, 'negative': 105, 'not_assessable': 203, 'positive': 73}`. Ambiguous and `not_assessable` rows remain counted and excluded from resolved-positive/negative retrieval denominators.

The materialized source-review slot table has 24 slots (12 questions × packed frames 0 and 13): 19 category-eligible slots per arm and 11 singular-instance-eligible slots per arm. Empty slots are retained. For a phrase-role query, source eligibility requires the relevant source-review assessability and visible category visibility for category metrics or visible target visibility for singular-instance metrics at that source timestamp. A positive candidate is a row whose role-matched label is `positive`; a role-mismatched positive for another entity is a resolved negative for this phrase. Empty pools and pools without a positive proposal count as Hit misses; candidate Recall is undefined when the positive count `n` is zero. The exact formulas are recorded in `metrics.json`.

## Category retrieval on the fixed-frame packet

| Arm | Selected aliases | Source-eligible target/control pools | Pools with a positive proposal | Category Hit@1 | Category Hit@5 |
|---|---:|---:|---:|---|---|
| fixed-vocabulary | 290 | 19 | 17 | Hit@1 12/19 (63.2%); macro Recall@1 17.4%; exact random Hit@1 mean 59.6%; conditional Hit@1 12/17 (70.6%) | Hit@5 16/19 (84.2%); macro Recall@5 45.8%; exact random Hit@5 mean 82.3%; conditional Hit@5 16/17 (94.1%) |
| question-conditioned | 180 | 19 | 17 | Hit@1 14/19 (73.7%); macro Recall@1 47.1%; exact random Hit@1 mean 65.6%; conditional Hit@1 14/17 (82.4%) | Hit@5 17/19 (89.5%); macro Recall@5 71.7%; exact random Hit@5 mean 85.5%; conditional Hit@5 17/17 (100.0%) |

The full target/control aggregate is also in `metrics.json`, including category and correct-instance metrics, macro and micro candidate Recall@K, exact random Hit@K from `n/r`, and conditional-on-proposal values. The compact formula is:

* `Hit@K = 1` when at least one positive candidate occurs in the top K.
* `candidate Recall@K = (# positive candidates in top K) / (# positive candidates in the same pool)`.
* `exact random Hit@K = 1 - C(r-n,K) / C(r,K)` for `n` positives and pool size `r`.
* Expected random candidate Recall@K is `K/r` for `K <= r` and `n > 0`.

The Hit denominator includes all 19 category-eligible frame slots per arm, including the fixed-arm construction-site proposal miss at packed frame 0 and the fixed-arm/QC baseball-field proposal misses at packed frame 13. These are frame-level query-pool metrics. A separate two-frame union is reported below with one denominator unit per QA/video; it is not a video-global temporal result.

### Two-frame query-level R0 union

For this diagnostic, the candidate pools from packed frames 0 and 13 are concatenated per QA/arm/phrase, sorted by cached R0 score descending with deterministic `candidate_id` tie-breaks, and evaluated as one union pool. The denominator is the number of independent QA/video IDs with at least one eligible selected frame. The matched random baseline uses the same union pool size `r` and positive count `n`: `1 - C(r-n,K) / C(r,K)`. This is a two-frame query-level union only; it does not claim visibility, identity continuity, or video-global retrieval.

| Arm | Category-eligible videos | Category Hit@1 vs random | Category Hit@5 vs random |
|---|---:|---|---|
| fixed-vocabulary | 11 | 7/11 (63.6%) vs random 65.5% | 10/11 (90.9%) vs random 92.9% |
| question-conditioned | 11 | 11/11 (100.0%) vs random 68.1% | 11/11 (100.0%) vs random 94.2% |

| Arm | Instance-eligible videos | Instance Hit@1 vs random | Instance Hit@5 vs random |
|---|---:|---|---|
| fixed-vocabulary | 7 | 3/7 (42.9%) vs random 29.7% | 5/7 (71.4%) vs random 76.8% |
| question-conditioned | 7 | 4/7 (57.1%) vs random 36.6% | 6/7 (85.7%) vs random 72.7% |

The QC-vs-fixed arm comparison changes both the proposal prompt/bank and the text query phrase: the fixed-vocabulary arm uses its fixed vocabulary/bank, while the question-conditioned arm uses question-derived phrases. The comparison is therefore not a pure proposal-generation ablation; proposal generation and query wording/embedding both change.

### Descriptive two-frame summary by benchmark video family

The table below reuses the two-frame union rows and groups them by the benchmark `dataset_name` video family crossed with the frozen selection-stratum family. `temporal_motion` combines the `temporal`, `ambiguity_probe_singular`, and `ambiguity_probe_group` strata. Each cell is `Hit@K vs matched-random Hit@K`; proposal coverage is covered eligible videos / eligible videos. Category and instance eligible-video denominators can differ.

| Video family | Family | QA IDs | Arm | Category n | Category proposal coverage | Category Hit@1 vs random | Category Hit@5 vs random | Instance n | Instance proposal coverage | Instance Hit@1 vs random | Instance Hit@5 vs random |
|---|---|---|---|---:|---:|---|---|---:|---:|---|---|
| ERA_Tra | geometry | train_005918 | fixed-vocabulary | 1 | 1/1 | 0/1 (0.0%) vs 16.7% | 0/1 (0.0%) vs 61.6% | 1 | 1/1 | 0/1 (0.0%) vs 16.7% | 0/1 (0.0%) vs 61.6% |
| ERA_Tra | geometry | train_005918 | question-conditioned | 1 | 1/1 | 1/1 (100.0%) vs 20.0% | 1/1 (100.0%) vs 69.1% | 1 | 1/1 | 1/1 (100.0%) vs 20.0% | 1/1 (100.0%) vs 69.1% |
| ERA_Tra | scene_region | train_003224 | fixed-vocabulary | 1 | 1/1 | 1/1 (100.0%) vs 100.0% | 1/1 (100.0%) vs 100.0% | 0 | 0/0 | 0/0 (n/a) vs n/a | 0/0 (n/a) vs n/a |
| ERA_Tra | scene_region | train_003224 | question-conditioned | 1 | 1/1 | 1/1 (100.0%) vs 100.0% | 1/1 (100.0%) vs 100.0% | 0 | 0/0 | 0/0 (n/a) vs n/a | 0/0 (n/a) vs n/a |
| Hawk_UAV | geometry | train_011679, train_012957, train_014154 | fixed-vocabulary | 3 | 3/3 | 2/3 (66.7%) vs 71.6% | 3/3 (100.0%) vs 96.3% | 2 | 2/2 | 1/2 (50.0%) vs 55.4% | 2/2 (100.0%) vs 94.5% |
| Hawk_UAV | geometry | train_011679, train_012957, train_014154 | question-conditioned | 3 | 3/3 | 3/3 (100.0%) vs 82.4% | 3/3 (100.0%) vs 100.0% | 2 | 1/2 | 1/2 (50.0%) vs 25.0% | 1/2 (50.0%) vs 50.0% |
| Hawk_UAV | temporal_motion | train_008163, train_008503, train_010200, train_012936 | fixed-vocabulary | 3 | 3/3 | 3/3 (100.0%) vs 74.1% | 3/3 (100.0%) vs 94.4% | 2 | 2/2 | 2/2 (100.0%) vs 18.1% | 2/2 (100.0%) vs 69.8% |
| Hawk_UAV | temporal_motion | train_008163, train_008503, train_010200, train_012936 | question-conditioned | 3 | 3/3 | 3/3 (100.0%) vs 46.3% | 3/3 (100.0%) vs 88.9% | 2 | 2/2 | 1/2 (50.0%) vs 18.1% | 2/2 (100.0%) vs 69.8% |
| UAVDT | temporal_motion | train_014264 | fixed-vocabulary | 1 | 1/1 | 1/1 (100.0%) vs 90.5% | 1/1 (100.0%) vs 100.0% | 0 | 0/0 | 0/0 (n/a) vs n/a | 0/0 (n/a) vs n/a |
| UAVDT | temporal_motion | train_014264 | question-conditioned | 1 | 1/1 | 1/1 (100.0%) vs 68.2% | 1/1 (100.0%) vs 99.9% | 0 | 0/0 | 0/0 (n/a) vs n/a | 0/0 (n/a) vs n/a |
| VisDrone | temporal_motion | train_001478 | fixed-vocabulary | 1 | 1/1 | 0/1 (0.0%) vs 35.7% | 1/1 (100.0%) vs 93.7% | 1 | 1/1 | 0/1 (0.0%) vs 28.6% | 1/1 (100.0%) vs 87.4% |
| VisDrone | temporal_motion | train_001478 | question-conditioned | 1 | 1/1 | 1/1 (100.0%) vs 100.0% | 1/1 (100.0%) vs 100.0% | 1 | 1/1 | 1/1 (100.0%) vs 100.0% | 1/1 (100.0%) vs 100.0% |
| VisDrone | scene_region | train_002092 | fixed-vocabulary | 1 | 1/1 | 0/1 (0.0%) vs 40.6% | 1/1 (100.0%) vs 94.2% | 1 | 1/1 | 0/1 (0.0%) vs 15.6% | 0/1 (0.0%) vs 59.9% |
| VisDrone | scene_region | train_002092 | question-conditioned | 1 | 1/1 | 1/1 (100.0%) vs 75.0% | 1/1 (100.0%) vs 100.0% | 1 | 1/1 | 0/1 (0.0%) vs 50.0% | 1/1 (100.0%) vs 100.0% |

These family rows are descriptive, with small denominators often equal to one or three videos. They should not be treated as independent estimates or causal task effects; source dataset families and task types are crossed, and the temporal/motion row intentionally pools temporal grounding with motion/ambiguity probes.

### Singular correct-instance retrieval

The singular-instance denominator is stricter: it has 11 eligible frame slots per arm after excluding absent, uncertain, multi-instance, group, and source-unassessable target cases. Candidate Recall is reported only for pools with at least one resolved positive instance candidate; the Hit denominator still includes eligible empty or proposal-miss pools.

| Arm | Selected aliases | Instance-eligible target/control pools | Pools with a positive proposal | Instance Hit@1 | Instance Hit@5 |
|---|---:|---:|---:|---|---|
| fixed-vocabulary | 290 | 11 | 11 | Hit@1 6/11 (54.5%); macro Recall@1 22.0%; exact random Hit@1 mean 36.2%; conditional Hit@1 6/11 (54.5%) | Hit@5 9/11 (81.8%); macro Recall@5 48.4%; exact random Hit@5 mean 76.8%; conditional Hit@5 9/11 (81.8%) |
| question-conditioned | 180 | 11 | 9 | Hit@1 5/11 (45.5%); macro Recall@1 41.1%; exact random Hit@1 mean 34.2%; conditional Hit@1 5/9 (55.6%) | Hit@5 9/11 (81.8%); macro Recall@5 77.6%; exact random Hit@5 mean 71.1%; conditional Hit@5 9/9 (100.0%) |

### Arm, crop-score, pool, duplicate, and proposal findings

The fixed-vocabulary arm has 290 selected aliases and the question-conditioned arm has 180. Across both arms, 56 exact mask duplicate groups are shared across arms (112 rows, 56 extra rows beyond unique masks); there are no within-arm duplicate mask groups and no duplicate candidate IDs. Selected rows contain 417 component proposal entries and 53 union proposal entries. Candidate score summaries by source label state and arm are stored under `arms.score_and_pool_summary` in `metrics.json`.

| Arm | Total target/control score observations | Mean / median pool size | Positive crop scores | Negative crop scores | Ambiguous crop scores |
|---|---:|---:|---|---|---|
| fixed-vocabulary | 290 | 12.083333 / 10.0 | 184 obs; mean 0.07111; median 0.069323 | 59 obs; mean 0.079177; median 0.080834 | 47 obs; mean 0.074269; median 0.07519 |
| question-conditioned | 180 | 7.5 / 2.0 | 119 obs; mean 0.077287; median 0.075566 | 54 obs; mean 0.075512; median 0.074642 | 7 obs; mean 0.075938; median 0.079156 |

The score columns summarize cached crop-level score observations for target/control phrase pools; they are descriptive and do not replace the ranked Hit/Recall metrics.

Proposal coverage is reported separately from ranking: a pool with zero positive labels is a proposal miss/unsupported pool for this packet, while the conditional-on-proposal ranking is computed only after at least one positive exists. This avoids treating absent proposals as a SigLIP2 ranking failure.

## Frozen R1 replay and paired changes

R1 was reconstructed from the frozen geometry implementation: target candidates are selected by prompt provenance, non-broad scene targets prefer component proposals when available, same-frame question-conditioned references exclude the target candidate itself and are selected by their candidate-frame cached scores, and geometry failures are sorted after satisfied candidates before the unchanged R0 remainder. Cross-frame fallback references are retained only to reproduce the frozen ordering and are invalid for spatial evidence.

- Replayed target/control query rows: **48**.
- Top-1 ordering changes: **3**; top-5 changes: **14**.
- Changes attributable to target-provenance/component filtering: top-1 **3**, top-5 **13**.
- Additional changes from geometry ordering: top-1 **0**, top-5 **1**.
- Paired category and correct-instance top-1 state changes remain zero in the source-only labels, but that does not imply that candidate selection/order was unchanged. These are post-hoc source-only label comparisons; labels were not used in inference or reranking.

### Geometry co-occurrence and relation diagnostics

| QA | Source relation assessable | Targets with all same-frame refs | Same-frame refs / fallback refs | Raw / valid spatial rows | Raw relation outputs | Identity-conditioned accuracy | AMC/Cinemark raw ties |
|---|---|---:|---:|---:|---|---|---:|
| `train_002092` | yes | 32 | 64 / 0 | 32 / 32 | {'False': 2, 'True': 30} | unscorable | n/a |
| `train_003224` | yes | 0 | 0 / 0 | 0 / 0 | {} | unscorable | n/a |
| `train_005918` | no | 89 | 178 / 0 | 0 / 0 | {} | unscorable | n/a |
| `train_011679` | yes | 19 | 38 / 0 | 19 / 19 | {'equal': 6, 'first_closer': 5, 'second_closer': 8} | unscorable | 6 |
| `train_012957` | no | 0 | 213 / 73 | 0 / 0 | {} | unscorable | n/a |
| `train_014154` | yes | 22 | 22 / 0 | 22 / 22 | {'left': 6, 'right': 16} | unscorable | n/a |

The relation function outputs are raw box-geometry diagnostics. Identity-conditioned correctness is scored only when both the frozen R1 target and every same-frame reference satisfy explicit role, positive correct-instance, complete-mask, and useful-crop checks. `train_003224` has no reference proposal; `train_005918` orientation is source-review unassessable; `train_012957` is invalid for stable group/tiny-anchor relation scoring. For `train_011679`, AMC/Cinemark edge gaps can tie at `[0, 0]` even when center distances differ; ties are reported as failures of that edge proxy, not semantic relation correctness.

Frozen R1 identity-pair audit:

- `train_014154`: 0 verified frozen R1 target-reference pairs out of 22 same-frame pair rows; 0 correct and 0 incorrect (n/a). All frozen R1 same-frame references are the QC broad crops FVE-93ffe0557c84/FVE-35b7b4ce37f7, each correct-instance ambiguous and overbroad. Positive fixed-arm reference-labeled rows exist, but R1 did not select them as its reference pool and they cannot substitute for the frozen QC references.
- `train_002092`: 0 verified frozen R1 target-reference pairs out of 32 same-frame pair rows; 0 correct and 0 incorrect (n/a). Frozen R1 references are generic target/context crops: frame 0 selects target-labeled vegetation rows, and frame 13 includes a context/reference road/median crop whose instance is negative plus a target-labeled road crop. No selected reference has a verified named reference identity.

The `train_014154` fixed-arm labels do contain positive commercial-building and apartment-complex rows, but the frozen R1 replay selects the question-conditioned reference pool: its only same-frame references are the broad, correct-instance-ambiguous multi-building crops. The positive fixed-arm reference rows cannot replace those frozen references. For `train_002092`, the frozen references are target-labeled vegetation/road crops or a negative context/reference crop, so the named median-strip reference is not verified. Therefore neither QA contributes a defensible pair-level relation-accuracy denominator. `train_002092`'s raw relation output still uses the permissive any-positive-bbox-IoU containment proxy.

## Frozen R2 temporal diagnostics

R2 attaches track metadata to unchanged R1 ordering, so R2 has no rank-improvement metric. The table reports the top-eight stored tracks per archived target/control ranking row against source-reviewed visibility; misses and switches are descriptive because the archive does not expose a gold track or all association links.

| QA | Source-review limit | Best stored tracks by arm | Interpretation |
|---|---|---|---|
| `train_001478` | right_censored_at_clip_end | question-conditioned:[12, 13, 14, 15]; fixed-vocabulary:[2] | The bus track covers ambiguous packed 12 and definitive visible packed 13-15; the end is right-censored at the final sample. |
| `train_008163` | framing_bounded | question-conditioned:[0, 1, 2, 3, 4, 5, 6]; fixed-vocabulary:[0, 1, 2, 3, 4, 5, 6] | R2 stores only top-eight tracks per ranking row; misses are descriptive, and track IDs are not gold identity labels. |
| `train_008503` | left_censored_at_clip_start | question-conditioned:[0, 1, 2, 3, 4, 5]; fixed-vocabulary:[0, 1, 2, 3, 4, 5] | Primary review marks packed 0-15 visible, while second review confirms p0 partly cropped and p1 visible but reports p10/p12/p15 not visible; frozen tracks provide clip-start presence only, so continuity and duration are unscorable. |
| `train_010200` | left_censored_at_clip_start_and_multi_instance | question-conditioned:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]; fixed-vocabulary:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] | R2 stores only top-eight tracks per ranking row; misses are descriptive, and track IDs are not gold identity labels. |
| `train_012936` | camera/framing_confounded | question-conditioned:[5]; fixed-vocabulary:[13, 14] | R2 stores only top-eight tracks per ranking row; misses are descriptive, and track IDs are not gold identity labels. |

The bus is the only focused temporal case with an objectively usable identity interval: no identifiable target bus is source-verified at packed 0–11, packed 12 is ambiguous/partly occluded, and packed 13–15 are definitive visible frames. The QC track spans packed 12–15, covers all definitive visible frames, shows no observed switch, and remains right-censored at the final sample. `r2.descriptive_track_counts` records visible/ambiguous/absent-overlap, second-review not-visible overlap, both-overlap (a descriptive switch/censoring proxy), and no-known-overlap counts for every other stored track; those counts are not identity scores. The motorcycle is not scoreable at 5–12 source pixels and its QC tracks fragment into single frames. Construction-site tracks overlap the source-visible window but represent a framing-bounded region. Baseball-field tracks overlap category visibility, while multiple diamonds prevent singular identity scoring. The large white-roof building in `train_008503` is left-censored at clip start: the primary review marks packed 0–15 visible, while the second review confirms p0 (partly cropped) and p1 but reports p10/p12/p15 not visible and did not exhaustively review the other samples. Its frozen p0–5 top track and short fragments therefore establish only descriptive clip-start presence; continuity and exact duration are unscorable.

## Existing full-run top-5 context

The archived full run contains 3,689 candidates and 36 R0/R1/R2 ranking rows. Its R0 and R1 top-5 entries are included in `global_top5_descriptive` in `metrics.json` for context only. Specific archived R1 diagnostics: `train_014154` top rows report `right`, consistent with the source relation, though mask identity remains separate; `train_011679` top-five rows report `equal` edge gaps `[0, 0]` while center distances are approximately `[35, 175]` pixels, the source relation expects first-closer, and `satisfied` remains true so geometry does not filter the order; `train_003224` lacks reference proposals; `train_005918` is unassessable; `train_012957` is not assessable; and `train_002092` uses the permissive IoU>0 containment proxy. They use the full experiment pools, not the 470-row fixed-frame denominator, and do not change the Gate metrics above.

Representative visual artifacts remain available in fixed-frame contact sheets (local-only artifact: `/home/soham/ReVA/runs/phase4_qcsrr/annotations/fixed_frame_eval/contact_sheets/page-001.jpg`) and the archived ranking contact sheets (local-only artifact: `/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_014154.jpg`), bus sheet (local-only artifact: `/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_001478.jpg`), AMC/Cinemark sheet (local-only artifact: `/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_011679.jpg`), motorcycle sheet (local-only artifact: `/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_012936.jpg`), and baseball sheet (local-only artifact: `/home/soham/ReVA/runs/phase4_qcsrr/annotations/ranking_contact_sheets/train_010200.jpg`).

## Limitations and decision

- The frozen packet covers only packed frames 0 and 13. It supports frame-level and two-frame query-pool diagnostics, not video-global temporal accuracy.
- Source labels are role-aware but not dense segmentation IoU or answer correctness labels. Ambiguous, overbroad, fragment, and `not_assessable` rows remain visible in denominators and are not forced positive.
- The 56 cross-arm exact mask duplicates make raw arm totals non-independent. Arm comparisons report pool sizes and duplicate structure explicitly.
- R1 geometry uses answer-neutral candidate metadata and source-review relation descriptions. Cross-frame fallback is excluded from valid spatial evidence, and distinct-reference identity labels are unavailable, so relation correctness is not scored. Paired correctness changes are descriptive evaluation outputs, not inference-time gold labels.
- R2 track metadata is truncated to top eight tracks per archived ranking row; only the bus's definitive packed-13–15 interval receives an objectively assessable identity-interval comparison, with packed 12 ambiguous and the end right-censored. The `train_008503` building has conflicting frame visibility reviews and is scored only for clip-start presence; no continuous identity or exact duration is claimed. Other misses/switches are descriptive.

**Decision: retain this as a completed fixed-frame Gate diagnostic and do not promote a broad QCSRR retrieval claim.** The packet shows a useful category-retrieval signal, but it does not establish reliable instance retrieval or spatial/temporal reasoning. A small paired frozen-Qwen crop-evidence diagnostic is justified as a downstream utility test, not as validation of QCSRR: compare no-crop, retrieved-crop, and matched-random-crop conditions on only objectively assessable questions; keep proposal misses in the denominator; freeze Qwen; and report paired question-level changes. Do not expand beyond that diagnostic unless it demonstrates a reproducible benefit.

Earlier execution context is preserved in [PHASE4_PRE_RECOVERY_REPORT.md](/home/soham/ReVA/runs/phase4_qcsrr/PHASE4_PRE_RECOVERY_REPORT.md). The reproducible evaluator is [analyze_gate_evaluation.py](/home/soham/ReVA/runs/phase4_qcsrr/annotations/fixed_frame_eval/analyze_gate_evaluation.py), and its machine-readable output is metrics.json (local-only artifact: `/home/soham/ReVA/runs/phase4_qcsrr/annotations/fixed_frame_eval/metrics.json`).
