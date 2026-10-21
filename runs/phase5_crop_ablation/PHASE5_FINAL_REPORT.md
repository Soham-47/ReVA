# Phase 5 — Frozen Qwen3-VL crop-evidence ablation

## Result

On the 12-question training diagnostic cohort, E0 scored **11/12**, E1 **10/12**, and E2 **11/12**. E2 matched E0; retrieved crops did not improve accuracy over the video-only baseline. E1 was one answer lower.

| Arm | Correct | Accuracy | Mean p(gold) | Mean gold margin |
|---|---:|---:|---:|---:|
| E0 | 11/12 | 91.7% | 0.876 | +0.790 |
| E1 | 10/12 | 83.3% | 0.835 | +0.723 |
| E2 | 11/12 | 91.7% | 0.848 | +0.750 |

## Paired changes

| Comparison | Wrong→correct | Correct→wrong | Interpretation |
|---|---|---|---|
| E0 → E1 | none | train_011679 | fixes 0; regressions 1 |
| E0 → E2 | none | none | fixes 0; regressions 0 |
| E1 → E2 | train_011679 | none | fixes 1; regressions 0 |

## Per-question scores

| QA ID | Task | Gold | E0 | E1 | E2 |
|---|---|---:|---|---|---|
| train_014154 | Geometric Relation | A | A ✓; p(gold) 0.997; margin +0.995 | A ✓; p(gold) 0.993; margin +0.990 | A ✓; p(gold) 0.995; margin +0.993 |
| train_005918 | Geometric Relation | D | D ✓; p(gold) 0.998; margin +0.998 | D ✓; p(gold) 0.996; margin +0.994 | D ✓; p(gold) 0.996; margin +0.994 |
| train_011679 | Geometric Relation | A | A ✓; p(gold) 0.791; margin +0.648 | C ✗; p(gold) 0.443; margin -0.013 | A ✓; p(gold) 0.566; margin +0.250 |
| train_012957 | Geometric Relation | A | A ✓; p(gold) 0.976; margin +0.966 | A ✓; p(gold) 0.966; margin +0.952 | A ✓; p(gold) 0.968; margin +0.954 |
| train_008503 | Temporal Grounding | D | D ✓; p(gold) 0.989; margin +0.986 | D ✓; p(gold) 0.972; margin +0.962 | D ✓; p(gold) 0.983; margin +0.977 |
| train_008163 | Temporal Grounding | B | B ✓; p(gold) 0.980; margin +0.962 | B ✓; p(gold) 0.970; margin +0.946 | B ✓; p(gold) 0.973; margin +0.951 |
| train_001478 | Temporal Grounding | D | D ✓; p(gold) 0.457; margin +0.113 | D ✓; p(gold) 0.383; margin +0.050 | D ✓; p(gold) 0.398; margin +0.080 |
| train_010200 | Temporal Grounding | B | B ✓; p(gold) 0.968; margin +0.952 | B ✓; p(gold) 0.934; margin +0.904 | B ✓; p(gold) 0.936; margin +0.907 |
| train_003224 | Object and Land Cover Recognition | C | C ✓; p(gold) 1.000; margin +0.999 | C ✓; p(gold) 0.997; margin +0.996 | C ✓; p(gold) 0.997; margin +0.996 |
| train_002092 | Object and Land Cover Recognition | C | C ✓; p(gold) 0.999; margin +0.999 | C ✓; p(gold) 0.998; margin +0.996 | C ✓; p(gold) 0.998; margin +0.997 |
| train_012936 | Temporal Grounding | A | B ✗; p(gold) 0.365; margin -0.138 | B ✗; p(gold) 0.385; margin -0.086 | B ✗; p(gold) 0.382; margin -0.085 |
| train_014264 | Trend and Pattern | C | C ✓; p(gold) 0.996; margin +0.994 | C ✓; p(gold) 0.986; margin +0.981 | C ✓; p(gold) 0.989; margin +0.985 |

## Task strata

| Task | Questions | E0 | E1 | E2 |
|---|---:|---:|---:|---:|
| Geometric Relation | 4 | 4/4 | 3/4 | 4/4 |
| Temporal Grounding | 5 | 4/5 | 4/5 | 4/5 |
| Object and Land Cover Recognition | 2 | 2/2 | 2/2 | 2/2 |
| Trend and Pattern | 1 | 1/1 | 1/1 | 1/1 |

## Source-video assessability

Independent Phase 4 source reviews were completed before model calls. Official QA correctness is still reported for all 12; these notes identify where visual ground truth is assessable:

| QA ID | Category | Instance | Geometry | Temporal | Source review note |
|---|---|---|---|---|---|
| train_014154 | True | True | True | None | Two building regions are visible across the sampled view. The multi-unit complex and separate commercial-looking building can be visually distinguished; source geometry supports a left/right comparison. Candidate-level boundary and identity labels remain to be reviewed blinded. |
| train_005918 | True | True | False | None | The central island and road network are visible. The island is approximately rounded, so no stable principal orientation can be assigned from source geometry; orientation comparison is unassessable. |
| train_011679 | True | True | True | None | AMC signage and adjacent theater/parking areas are source-visible. The named theater landmarks are distinguishable; pairwise distance comparison is assessable at scene level. |
| train_012957 | True | False | False | None | Vehicle group, junction and traffic-control structures are visible. I could not uniquely verify the described white building with green roof in the sampled source frames; the joint distance comparison is unassessable. |
| train_008503 | True | True | None | True | The large white-roofed building is already partly in frame at the first sampled frame (source index 0, t=0). First visibility is left-censored at clip start; the answer-neutral sampled result is present at t=0. |
| train_008163 | True | True | None | False | A large construction/cleared parcel is visible in early samples and remains partly present as the camera view changes. Its exact last visible sample is not defensible from the current source-only contact review; temporal interval is unresolved, not interpolated. |
| train_001478 | True | False | None | False | The frames contain bus-like vehicles, but the visible bright-green bus conflicts with the question’s white-and-red description and more than one vehicle enters/exits. The question-referenced bus identity and duration cannot be independently fixed from this review. |
| train_010200 | True | False | None | False | A baseball complex with multiple diamonds is already visible at source sample 0. Category and earliest sampled presence are clear, but the singular field is not uniquely specified among the visible diamonds; instance-level temporal target is ambiguous. |
| train_003224 | True | False | False | None | One baseball field and several field-side structures are visible. “Structure on the left side” narrows the region, but the structure type/unique referent is not sufficiently clear for a confident instance label from these frames. |
| train_002092 | True | True | True | None | A planted/vegetated median strip is visible with the road throughout the sampled sequence. Scene-region retrieval and containment are assessable; pixel-level boundaries are not treated as the target label. |
| train_012936 | False | False | None | False | Several small road users/vehicles are visible, but no unique motorcycle is defensibly identifiable at 640x360 with no distinguishing attribute in the question. Identity and disappearance timing are unassessable. |
| train_014264 | True | False | False | False | Vehicles and highway are visible at group level. Individual vehicle identity across samples is not reliably verifiable at this scale, and the question does not single out one vehicle; do not count per-object tracking as assessable. |

At the operation level, source review marked 2/4 geometric questions assessable (`train_014154`, `train_011679`): E0/E2 answer both correctly; E1 gets one of two because of the `train_011679` regression. Only `train_008503` has assessable sampled temporal presence, and it is left-censored at clip start; all arms choose the sampled answer. The median scene-region question `train_002092` is assessable and all arms are correct. Other temporal references and the structure/field or group-motion cases retain the source-review ambiguity shown above.

## Crop evidence and interpretation

The paired crop sheet shows all supplied random/retrieved regions and their source timestamps alongside answer outcomes. There were 23 crop slots across 12 questions; 18 E1/E2 pairs use the same source timestamp and the remaining pairs use the nearest available frame. The paired crops match dimensions; no random/retrieved pair reused the same candidate ID, while 2 pairs were pixel-identical due to different proposals yielding the same crop. All 92 E1 and 92 E2 crop-to-source-frame timestamp records matched the Phase 4 source map, and the offset-view membership flags matched the Stage A sample indices. The source timestamps remain explicitly labeled even when they do not occur in the offset-specific 16-frame sample.

The 12 per-question question-conditioned pools contained 1,497 candidate rows total (4–552 per question). Three entity prompts produced no prompt-specific proposals: `intersection` in `train_012957`, `baseball field` in `train_003224`, and `land-cover region` in `train_002092`. The shared pool still supplied fallback crops for all three; the target-specific land-cover proposal was absent for `train_002092`.

Manual inspection of the crop sheet is post hoc and not blinded: some retrieved regions are useful scene context (roundabout, mall, median, baseball field, highway), while several small-object crops (bus, motorcycle, some vehicles) are tiny and are not independently identifiable from the crop alone. The original 16 sampled video frames remain present in E1/E2. The cohort therefore tests whether crops help the frozen model, not whether each crop is a valid target label.

E2 fixed the E1 regression on `train_011679` (AMC/Cinemark/parking distance), but did not fix any E0 error. E0 and E2 were both wrong only on `train_012936`, whose source review says no unique motorcycle is identifiable at 640×360. E1 additionally regressed `train_011679`. Across all questions, mean gold probability and mean gold margin were lower in both crop arms than E0, indicating no evidence of a general confidence benefit.

## Runtime, memory and tokens

- Model: `Qwen/Qwen3-VL-8B-Instruct` revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` with the frozen Stage A adapter SHA-256 `604bff63149edde707267a80353e9ffb080c3bcb4f280e6045101a43ebba7990`.
- Hardware: NVIDIA H200 NVL × 1 on Slurm job `402305`.
- Model/adapter load and placement: 1439.3 s; scoring plus preprocessing: 80.1 s for 144 forward calls.
- Peak CUDA allocation: E0 17.94 GiB; E1/E2 18.32 GiB. Peak reserved was 20.75 GiB for crop arms.
- Per arm: E0 106,926 input tokens and 0 auxiliary-image tokens; E1/E2 each 126,006 input tokens, including 15,360 auxiliary-image vision tokens. Each arm had 48 forward calls and 96,000 video vision tokens.
- Slurm allocation: 27:33 (27.55 H200 GPU-minutes), 12 CPUs and 96 GB requested; batch MaxRSS 30,866,748 KB (29.4 GiB), total CPU time 5:25. Slurm records `billing=12` as an internal resource TRES; the account exposes no monetary charge, so no dollar cost is claimed.

## Post-hoc crop-content audit

I reviewed the displayed E1/E2 crops after the run. These are qualitative, non-blinded judgments and do not change the frozen rankings or answer scores:

| QA ID | Retrieved-crop assessment |
|---|---|
| train_014154 | Both retrieved regions show the commercial-looking building and apartment-complex scene at coarse scale; this supports category/context, not exact boundaries. |
| train_005918 | The retrieved green-space crop is narrow but relevant; the roundabout and road crops are broad contextual views. The source review says the island has no stable orientation, so these do not make the answer geometry assessable. |
| train_011679 | Retrieved crops show the mall/theater block and parking context, but signage and building boundaries are not consistently legible at crop scale; the comparison still relies on the original video and broad scene context. |
| train_012957 | The vehicle crop is only 24×21 pixels and cannot independently identify the target set. Intersection and green-roof-building crops provide context, but the source review could not uniquely verify the named building. |
| train_008503 | The retrieved white-roof building crop is relevant and large enough to show the building; its first appearance is nevertheless left-censored at clip start. |
| train_008163 | The construction/cleared parcel is represented by a useful contextual crop. Random and retrieved crops are nearly the same view, so this pair is a weak semantic-selection contrast. |
| train_001478 | The bus crop is 49×35 pixels: bus-like content is present, but the crop cannot establish the question's white/red identity, consistent with the source-review ambiguity. |
| train_010200 | The baseball-field crop is scene-relevant and readable, but the video contains multiple diamonds; the singular field identity remains ambiguous. |
| train_003224 | The crops show a field-side fence/dugout-like structure, not enough detail to certify the structure type or unique left-of relation; the source review marks that instance ambiguous. |
| train_002092 | The retrieved contextual regions show the vegetated central median and adjacent road. They are scene-relevant boxes, not precise land-cover masks. |
| train_012936 | The motorcycle crop is only 12×19 pixels and does not resolve a unique motorcycle; neither crop representation fixes the source-level identity failure. |
| train_014264 | The vehicle crop is only 20×22 pixels while the highway crop is broad context. This supports group-level scene context, not individual vehicle identity or motion tracking. |

## Conclusion and limits

This diagnostic does not justify held-out Qwen validation yet. The baseline is already 11/12 on a small, selected training set, leaving little headroom; E2 ties it, E1 is worse, and crop arms lower average gold probability/margin. The unchanged baseline plus image-plus-video framing is safe enough to retain as a hypothesis, but the current R0 crop selection has no demonstrated downstream utility. Because several questions are visually ambiguous or censored, these counts are not a generalization estimate. A next run should be considered only after a predeclared, independently selected held-out `valT` set and a refined evidence format for small objects; do not infer improvement from this cohort.

## Artifacts

- Result contact sheet: `runs/phase5_crop_ablation/contact_sheets/phase5_answer_outcomes.jpg`
- Raw score rows and resource summary: `runs/phase5_crop_ablation/results_hpc/scores_E0.jsonl`, `runs/phase5_crop_ablation/results_hpc/scores_E1.jsonl`, `runs/phase5_crop_ablation/results_hpc/scores_E2.jsonl`, `runs/phase5_crop_ablation/results_hpc/hpc_run_summary.json`
- Evaluation JSON and Slurm accounting: `runs/phase5_crop_ablation/results_hpc/phase5_evaluation.json`, `runs/phase5_crop_ablation/hpc_accounting.txt`
- Original E1/E2 crop sheet: `runs/phase5_crop_ablation/contact_sheets/phase5_crop_pairs.jpg`
