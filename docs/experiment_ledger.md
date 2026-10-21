# ReVA experiment ledger

**Audit date:** 2026-10-21 (Asia/Kolkata)  
**Scope:** experiment, review, manifest, and result artifacts currently present in this checkout.

The public mirror contains source, experiment designs, aggregate metrics, and
text reports. It excludes dataset files, frames and masks, caches, raw
per-question prediction JSONL/CSV, model checkpoints, and the hidden-test ZIP.
Inline paths to excluded artifacts below are local-only provenance references.

This ledger is an evidence index. It does not infer a “since last commit” window:
the checkout contains an empty `.git` directory, with no readable commit, branch,
or remote history. The dates below come from report contents, named manifests,
and filesystem records where they are available. A result is called complete only
when a report or machine-readable result artifact supports that status. Plans,
preflights, caches, and selections remain marked as incomplete when no scored
result is present.

## Current reference and gates

| Track | Current status | Evidence and exact result |
| --- | --- | --- |
| Stage A frozen reference | Active comparison reference | Qwen3-VL-8B-Instruct revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, language LoRA r16, 16 legacy frames, offsets `0`/`0.5`, two option rotations. `valQ`: 498/595 (83.70%); `valT`: 885/1,059 (83.57%); official `val`: 1,757/2,000 (87.85%). Evidence: [`runs/stageA_q3vl8b_r16/legacy/scores_valQ_summary.json`](../runs/stageA_q3vl8b_r16/legacy/scores_valQ_summary.json), [`scores_valT_summary.json`](../runs/stageA_q3vl8b_r16/legacy/scores_valT_summary.json), [`scores_val_summary.json`](../runs/stageA_q3vl8b_r16/legacy/scores_val_summary.json). |
| Prior private-test submission | Reported result, local provenance incomplete | The README records 3,358/4,000 (83.95%) for the existing submission. No test score manifest or leaderboard capture is present in this checkout. Evidence: [`README.md`](../README.md). |
| New 4,000-pair candidate result | User-reported; provenance not yet attached | The user supplied overall 83.75% (3,350/4,000) and category/task accuracies recorded below. If it is the same test set and protocol as the prior result, it is 8 answers lower (−0.20 percentage points); no paired prediction artifact is available here to attribute the difference. |
| Endpoint frame sampler | Completed diagnostic; not promoted | `valQ`: 495/595 (83.19%); `valT`: 890/1,059 (84.04%); official `val`: 1,755/2,000 (87.75%). Evidence: endpoint summary files under [`runs/stageA_q3vl8b_r16/`](../runs/stageA_q3vl8b_r16/). |
| Stage B full retrain | No local result artifact | The repository documents the Stage B procedure and gate, but no Stage B adapter, score manifest, or submission result is present under `runs/`. Do not report Stage B as run. |
| Adaptive frame selector | Closed; no promotion | On official `val`, adaptive frames scored 1,745/2,000 (87.25%) versus 1,757/2,000 (87.85%) for legacy. Temporal Grounding fell 243/320 to 237/320; Geometric Relation moved 180/200 to 181/200. Evidence: [`results/error_review/adaptive_temporal_geometry_audit.md`](../results/error_review/adaptive_temporal_geometry_audit.md). |
| Visual adaptation | Closed as no-go for A1–A4 under the recorded gates | A2 was +1.61 pp over A1 on `valT`, below the predeclared +2.0 pp gate. A3 scored 858/1,059 (81.02%) and A4 863/1,059 (81.49%) on `valT`, versus A0 885/1,059 (83.57%). Evidence: [`runs/phase6_visual_adaptation_pilot/paired_report.md`](../runs/phase6_visual_adaptation_pilot/paired_report.md), [`A3 report`](../runs/phase6_A3_language_plus_vision_from_stageA/phase6b_final_report.md), [`A4 report`](../runs/phase6_A4_language_plus_vision_and_merger_from_stageA/phase6b_final_report.md). |
| Phase 8 Qwen3.8/32B backbone comparison and test package | No complete paired validation comparison; later H100 test package is prepared but not submitted | The local checkout has Qwen3.8 `valQ` complete and `valT` partial files, a validated test ZIP from job `407542`, and user-reported 4,000-pair metrics. The score manifest/lineage and exact paired predictions for the 4,000-pair result are not local. No Codabench upload is recorded. |

## Chronological record

### Baseline diagnostics — 2026-10-13 onward

The Stage A error review audited all 595 `valQ`, 1,059 `valT`, and 2,000
official-validation predictions, then inspected a focused set of errors and
controls. The review found 97, 174, and 243 errors on those splits, with 72/97,
127/174, and 168/243 errors agreeing across all four views. Temporal Grounding
was the weakest task on all three splits: 73.91% on `valQ`, 74.44% on `valT`,
and 75.94% on official `val`. The review also found possible clip/label
inconsistencies and concluded that the 16-frame cap is a plausible timing
bottleneck but not a proven cause of all errors.

Evidence: [`results/error_review/REVIEW.md`](../results/error_review/REVIEW.md),
[`results/error_review/audit_statistics.json`](../results/error_review/audit_statistics.json).

The Phase 0 visual review sampled 60 `valT` items: 20 temporal errors, 10
temporal controls, 20 geometry errors, and 10 geometry controls. The most common
reviewed temporal pattern was a wrong visibility boundary; the most common
geometry pattern was uncalibrated depth or distance. The review retained
possible annotation concerns, including one seven-frame clip whose answer
options describe longer intervals.

Evidence: [`results/phase0_temporal_geometry_review/REVIEW.md`](../results/phase0_temporal_geometry_review/REVIEW.md).

### Temporal-data feasibility audit — 2026-10-17

The train-only temporal supervision audit reviewed 100 questions from 100
videos, excluding `valQ` and `valT` videos. It found:

| Disposition | Count |
| --- | ---: |
| Directly usable | 10/100 |
| Conditional repair or re-review | 57/100 |
| Exclude | 33/100 |

The predeclared usable-evidence gate required at least 80/100. Even counting
every conditional case as repairable gives 67/100, so targeted temporal training
was not started. Three gold choices also exceeded the clip duration in the
numeric precheck. The prepared 20-case independent review packet remains an
open data-quality gate.

Evidence: [`results/temporal_training_data_audit/phase1_report.md`](../results/temporal_training_data_audit/phase1_report.md),
[`results/temporal_training_data_audit/README.md`](../results/temporal_training_data_audit/README.md).

### Phase 2 temporal sampling — completed, no promotion

The coarse-to-fine temporal run changed only Temporal Grounding predictions and
kept the Stage A model and other-task predictions fixed:

| Arm | `valQ` temporal | `valT` temporal | Decision |
| --- | ---: | ---: | --- |
| T0 legacy | 51/69 | 99/133 | Reference |
| T1 coarse-to-fine | 49/69 | 98/133 | Gate failed: −2 and −1 net answers |
| T2 timestamp context | 47/69 | 91/133 | Gate failed: −4 and −8 net answers |

The run took 1,608.6 seconds on the recorded A100 path. Its predeclared gate
required at least +3 temporal answers on `valT`, nonnegative change on `valQ`,
and no more than a 1 pp overall drop. Neither arm passed.

Evidence: [`runs/phase2_temporal_coarsefine/phase2_report.md`](../runs/phase2_temporal_coarsefine/phase2_report.md).
The paired per-question comparison is retained locally at
`runs/phase2_temporal_coarsefine/phase2_comparison.json` and is not published.

The reviewed-window T1R pilot then scored 24 selected questions (12 per split,
six saved errors and six controls per split). T1R scored 13/24 (54.2%), tied T1,
and was one net answer above T0: three fixes and two regressions. This was a
selected diagnostic, dominated by opening-boundary brackets (20/24), and is not
a split-wide temporal result.

Evidence: [`runs/phase2_temporal_verified_window/review_report.md`](../runs/phase2_temporal_verified_window/review_report.md),
[`modal_full/report/verified_window_report.md`](../runs/phase2_temporal_verified_window/modal_full/report/verified_window_report.md).

### Phase 3 segmentation and geometry — diagnostics only

The fixed 32-question SAM 2 mask pilot scored 23/32 (71.9%) for baseline,
detector-only, and detector-plus-mask. Only 9/32 questions received a matched
mask overlay, and no answer changed. Mean correct-answer probability decreased
by 4.71% on overlaid questions and by 1.32% overall.

Evidence: [`runs/segmentation_pilot_stageA_modal/comparison_report.md`](../runs/segmentation_pilot_stageA_modal/comparison_report.md).

The combined 32-question direct-text SAM 3 comparison scored 23/32 (71.9%) for
baseline and 24/32 (75.0%) for SAM 3, with three fixes and two regressions. The
17-question supplement was 14/17 for both baseline and SAM 3. These selected
pilot changes do not establish a general gain; the reports record broad field
masks, unstable building/barn masks, and non-unique car instances.

Evidence: [`runs/sam3_remaining17_stageA_replicate/comparison_report_32.md`](../runs/sam3_remaining17_stageA_replicate/comparison_report_32.md),
[`runs/sam3_pilot_stageA_replicate/comparison_report.md`](../runs/sam3_pilot_stageA_replicate/comparison_report.md),
[`runs/sam3_remaining17_stageA_replicate/comparison_report.md`](../runs/sam3_remaining17_stageA_replicate/comparison_report.md).

The completed explicit-geometry report compares G0 baseline, G1
GroundingDINO box context, and G2 SAM 3 mask context on 30 selected geometry
questions per split (20 baseline errors plus 10 controls):

| Split | G0 | G1 | G2 | Answer changes |
| --- | ---: | ---: | ---: | --- |
| `valQ` | 10/30 | 10/30 | 10/30 | 0/30 for either arm |
| `valT` | 10/30 | 10/30 | 10/30 | 0/30 for either arm |

Usable strict 2D evidence covered 1/30 `valQ` questions and 0/30 `valT`
questions. The supported-looking example was manually judged unreliable and did
not encode the queried relation. The report concludes that the recipe should
not proceed to larger geometry or fine-tuning phases.

**Record conflict:** `runs/phase3_explicit_geometry/results_summary.json` and
`execution_status.json` retain an earlier `not_run`/blocked preflight snapshot,
while the full score outputs, [`full/phase3_report.md`](../runs/phase3_explicit_geometry/full/phase3_report.md),
and [`full/qualitative_review/phase3_manual_review.md`](../runs/phase3_explicit_geometry/full/qualitative_review/phase3_manual_review.md)
document the later completed comparison. Both are preserved; use the full report
and its score artifacts for the result, and reconcile the stale status files
before using them as an automated gate.

### Phase 3A retrieval smoke and Grounding DINO pilot — completed diagnostics, no promotion

The two-video Phase 3A retrieval smoke completed 20 planned SAM 3 proposal
records for `train_010397` and `train_009869`, then ranked 148 candidates with
SigLIP2 over padded and masked crops. The manual audit is category/frame-level
only: it did not establish target-instance identity, cross-frame identity, or a
general 32-question gate. For the question-conditioned arm, black-car Recall@5
was 0.313 for both crop styles versus 0.154 (padded) and 0.157 (masked) for
the saved random means; parking-lot strict Recall@5 was 0.000 versus 0.157 and
0.154 random; baseball-field Recall@5 was 0.250 for both styles versus 0.335
and 0.334 random. Candidate generation and ambiguous boundaries remained the
dominant limitations. Qwen scoring and 3D/depth scoring were not run.

The saved smoke manifest retains an earlier `ranking_status: pending` field,
while `hpc_report.json` and the manual audit report document the later ranking
completion. Use the latter reports for the result and preserve the manifest as
execution provenance until it is reconciled.

Evidence: [`runs/phase3a_retrieval_smoke/review/manual_audit/diagnostic_report.md`](../runs/phase3a_retrieval_smoke/review/manual_audit/diagnostic_report.md).
The raw report and annotation files (`hpc_report.json` and `manual_audit.json`)
remain local because they contain candidate-level records and are not part of
the public mirror.

The Stage A Grounding DINO pilot is also present as score JSONL and manifests
for the fixed 16-question pilot in each split. `valQ` scored 12/16 (75.0%) and
`valT` scored 11/16 (68.75%), exactly matching the same-ID Stage A baseline on
each split. The detector revision is
`IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71`,
with legacy frames, two offsets, and two option rotations. These selected
scores do not establish a full-split gain, and the manifests contain no added
runtime/cache-size summary or independent detector-quality audit; the arm is
therefore not promoted. The adjacent SAM 2 report records that mask boundaries
were not manually audited.

Evidence: [`runs/segmentation_pilot_stageA_modal/comparison_report.md`](../runs/segmentation_pilot_stageA_modal/comparison_report.md).
The detector score JSONL and manifests are local-only files under
`runs/grounding_pilot_stageA_modal/`; they are not published.

### Phase 3B retrieval gates — exploratory, no Qwen promotion

Gate 1 froze SigLIP2 rankings over 2,319 candidates but left every
candidate-level annotation blank. It therefore reports no Recall@1/5,
correct-instance metric, or random comparison. Gate 2 reviewed two videos and
482 candidate/entity rows. It found a narrow padded-RGB parking-context signal
and a one-frame baseball-field signal, but black-car identity and temporal
continuity remained unassessable. No Qwen scoring was run.

Evidence: [`runs/phase3b_gate1/scientific_evaluation_2026-10-18.md`](../runs/phase3b_gate1/scientific_evaluation_2026-10-18.md),
[`runs/phase3b_gate1/gate2/report.md`](../runs/phase3b_gate1/gate2/report.md).

### Phase 4 QCSRR — completed fixed-frame gate diagnostic

The initial provider run was incomplete after 41 SAM 3 create attempts were
exhausted by throttling; it produced only partial question-conditioned masks
for three videos. The derived Gate report later evaluated frozen artifacts in a
470-alias fixed-frame packet. Its category metrics were:

| Arm | Eligible pools | Category Hit@1 | Category Hit@5 | Exact random Hit@1 / Hit@5 |
| --- | ---: | ---: | ---: | ---: |
| Fixed vocabulary | 19 | 12/19 (63.2%) | 16/19 (84.2%) | 59.6% / 82.3% |
| Question conditioned | 19 | 14/19 (73.7%) | 17/19 (89.5%) | 65.6% / 85.5% |

For the stricter singular-instance packet, fixed vocabulary was 6/11 at
Hit@1 and 9/11 at Hit@5; question-conditioned was 5/11 and 9/11. These are
frame-pool diagnostics with 11 eligible instance slots per arm, not video-global
retrieval or temporal identity accuracy. The report recommends only a small
paired frozen-Qwen crop diagnostic, which became Phase 5.

Evidence: [`runs/phase4_qcsrr/PHASE4_FINAL_REPORT.md`](../runs/phase4_qcsrr/PHASE4_FINAL_REPORT.md),
[`PHASE4_PRE_RECOVERY_REPORT.md`](../runs/phase4_qcsrr/PHASE4_PRE_RECOVERY_REPORT.md).

### Phase 5 frozen crop-evidence ablation — completed, no promotion

On the selected 12-question training diagnostic cohort, the video-only E0 arm
scored 11/12 (91.7%), random-crop E1 scored 10/12 (83.3%), and retrieved-crop
E2 scored 11/12 (91.7%). E2 made zero fixes and zero regressions against E0;
E1 regressed one answer. Mean gold probability and margin were lower in both
crop arms than E0. The cohort is selected training data, not a held-out estimate.

Evidence: [`runs/phase5_crop_ablation/PHASE5_FINAL_REPORT.md`](../runs/phase5_crop_ablation/PHASE5_FINAL_REPORT.md).
The per-question evaluation is local at
`runs/phase5_crop_ablation/results_hpc/phase5_evaluation.json` and is not published.

### Phase 6 visual adaptation — complete, no-go under predeclared gate

The A1/A2 comparison on full `valQ` and `valT` scored:

| Arm | `valQ` | `valT` |
| --- | ---: | ---: |
| A0 frozen Stage A | 498/595 (83.70%) | 885/1,059 (83.57%) |
| A1 language continuation | 490/595 (82.35%) | 854/1,059 (80.64%) |
| A2 language + merger | 499/595 (83.87%) | 871/1,059 (82.25%) |

A2 improved over A1 by 1.51 pp on `valQ` and 1.61 pp on `valT`, but the
predeclared valT screening threshold was +2.0 pp; the overall result was
therefore no-go. A3 (language plus upper-block vision LoRA) scored 495/595 and
858/1,059; A4 (language plus vision and merger LoRA) scored 500/595 and
863/1,059. Both regressed against A0 on `valT`; A4's +0.47 pp over A3 was
inconclusive.

A4 was first canceled at 16/200 steps, then later rerun from step 0 under the
unchanged configuration and completed. The later report is the result record;
the cancellation remains historical provenance.

Evidence: [`runs/phase6_visual_adaptation_pilot/paired_report.md`](../runs/phase6_visual_adaptation_pilot/paired_report.md),
[`runs/phase6_A3_language_plus_vision_from_stageA/phase6b_final_report.md`](../runs/phase6_A3_language_plus_vision_from_stageA/phase6b_final_report.md),
[`runs/phase6_A4_language_plus_vision_and_merger_from_stageA/phase6b_final_report.md`](../runs/phase6_A4_language_plus_vision_and_merger_from_stageA/phase6b_final_report.md).

### Phase 7 evidence and tracking feasibility — gates remain closed

Phase 7A generated a canonical frame manifest for 1,047 videos. Twenty-three
records use exact presentation timestamps from a decoded source manifest; the
remaining 1,024 use an explicit constant-FPS index-divided-by-FPS fallback.

Phase 7B validated 24 reviewed cases across 23 videos. Five videos were
policy-exact, all five staged source clips matched their cache checksums, and
the cache-only manual review found no question with both a correct automatic
proposal and an independently assessable target/event label. The downstream
evidence gate therefore failed; Qwen, SAM 3 native, and DA3 scoring was not run
for this tranche.

Phase 7C later ran a two-video Replicate diagnostic on an archway and a marked
white van. The report records 32 unique prediction IDs (27 successes and 5
failures) across SAM 3, Grounding DINO, and SAM 2 paths. The archway mask began
0.47 seconds after the first recognizable source frame; the marked van target
was missed by SAM 3. DINO-to-SAM2 automatic re-anchoring recovered sparse masks
at several later frames but left gaps and unverified cross-frame identity. No
Qwen or DA3 result was run.

**Record conflict:** `dry_run_report.json` is an earlier no-call credential
preflight. The later report aggregates 32 unique prediction IDs (27 successes,
5 failures), while the compact `execution_summary.json` records an earlier
22-ID subset (19 successes, 3 failures). Treat these as different scopes and
use the report plus raw manifests for the aggregate; neither supports a broad
tracking or Qwen result.

Evidence: [`runs/phase7B_cache_validation/report.md`](../runs/phase7B_cache_validation/report.md),
[`runs/phase7B_cache_validation/manual_source_review.md`](../runs/phase7B_cache_validation/manual_source_review.md),
[`runs/phase7C_replicate_smoke/report.md`](../runs/phase7C_replicate_smoke/report.md).
The canonical frame manifest remains local at
`runs/phase7A_feasibility/canonical_frame_manifest.json`.

### Phase 8 backbone comparison and 4,000-pair test result — incomplete provenance

The repository contains a frozen, stratified 110-question pilot selection:
55 `valQ` and 55 `valT`, with five IDs per task. The selection was frozen before
candidate scoring and records `candidate_predictions_inspected: false`.
Qwen3.8-27B has a complete 595-row `valQ` file and a 658-row prefix of the
1,059-row `valT` split. Its complete `valQ` result is 506/595 (85.04%) versus
498/595 (83.70%) for Stage A, with 23 fixes and 15 regressions. Its target-task
counts are Geometric Relation 92/110 versus 87/110 and Temporal Grounding
54/69 versus 51/69. On the processed `valT` rows it scores 558/658 (84.80%),
matching Stage A on those rows, with 35 fixes and 35 regressions. There is no
complete paired `valT` result or Qwen3-VL-32B result in this checkout. Score
rows identify Qwen/Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` and four views; the directory name
`checkpoint1641` is path provenance only.

Evidence: [`tasks/phase8_pretrained_scorer.md`](../tasks/phase8_pretrained_scorer.md).
The frozen selection and per-question validation score files remain local under
`runs/phase8_backbone_comparison/` and
`runs/phase8_qwen38_27b_r16_h200_checkpoint1641/`.

An older live snapshot recorded H100 validation-scoring job `407497` as running
with a PEFT adapter and partial `valQ` rows. That snapshot is stale and is not a
current queue status. A later H100 test package attributed to job `407542` is
present locally at
`submissions/phase8_qwen38_h100_407542/submission.zip` is local-only and is not
published because it contains hidden-test predictions.
It is 320,757 bytes, has SHA-256
`0784a48650189ddd673aaee4c45e71065248f7989f771e3135a33d102b4cca34`, and was
validated as containing 4,000 unique A–D predictions. It has not been uploaded
to Codabench. No score JSONL or run manifest for the 4,000-pair result is in
this checkout, so the package's exact adapter lineage and the reported metrics
cannot yet be independently paired or audited.

The user reported this 4,000-pair result:

| Metric | Accuracy |
| --- | ---: |
| Overall | 83.75% (3,350/4,000) |
| Factual / Temporal / Spatial / Causal | 82.76% / 78.60% / 86.80% / 90.00% |
| General understanding / Object-land-cover / Change detection | 98.33% / 82.58% / 77.40% |
| Temporal grounding / Trend-pattern | 75.00% / 85.00% |
| Geometric relation / Structural layout / Perspective-viewpoint | 86.50% / 87.65% / 86.46% |
| Causation / Consequence / Hypothetical | 92.78% / 92.00% / 85.63% |

The prior private-test score recorded in the README is 3,358/4,000 (83.95%).
If the user-reported result is from the same test set and protocol, the difference
is 8 answers (−0.20 percentage points). The aggregate alone cannot show whether
the new model fixed or regressed the same questions, and cannot establish that
fine-tuning caused the difference.

The last `browser-harness --doctor` check found no running Chrome, a dead
daemon, and zero active browser connections. The user said they would enable
remote debugging. No upload is recorded. The existing request to upload this
specific package through browser-harness remains pending on the user's browser
setup; do not change their browser setting or use another authenticated-browser
path to bypass the requested method.

## Research-only records

These documents inform possible future work and are not experiment results:

- [`docs/research/geovlm_r1_transfer_assessment.md`](research/geovlm_r1_transfer_assessment.md): no weights, jobs, or ReVA code changes; method-transfer assessment only. It recommends a small structured-reward audit rather than replacing Stage A.
- [`docs/research/reva_rl_recipe_sources.md`](research/reva_rl_recipe_sources.md): no RL training or provider job; source-backed design constraints and a proposed bounded pilot.

## Current next gates

1. Keep the frozen Stage A adapter and legacy score files as the comparison
   reference. No adaptive-frame, crop-evidence, segmentation, QCSRR, or A1–A4
   result is promoted in the current records.
2. Treat Phase 7 evidence work as blocked until a target is both automatically
   proposed and independently assessable; do not run Qwen or geometry scoring on
   missing or identity-ambiguous tracks.
3. For Phase 8, retrieve the exact score JSONL and run manifest for the
   user-reported 4,000-pair metrics and the H100 test package. Confirm the model,
   adapter checksum, split, and scoring protocol; then compare same-ID
   predictions with the prior submission and frozen Stage A. The old `407497`
   queue snapshot is historical, not live status. No backbone promotion decision
   or Codabench upload is recorded.
4. For temporal training, complete the independent 20-case audit and re-audit a
   fresh sample before constructing supervision; the 80/100 evidence gate has
   not passed.
5. Reconcile stale status snapshots before using automation: the Phase 3 and
   Phase 7C conflicts are called out above, and plans under `tasks/` describe
   intended workflows rather than the final state of every run.

## Artifact reading guide

- **Decision reports:** `runs/**/{*REPORT*.md,*report.md,paired_report.md}` and the linked result JSON files.
- **Execution contracts:** `tasks/`, `runs/**/manifest*.json`, and preflight reports. These establish scope and provenance; they do not prove a gain.
- **Raw evidence:** score JSONL, per-question CSV/JSON, contact sheets, source reviews, and cache records. Use these to audit a report's denominators and identity limits.
- **Research:** `docs/research/` contains source-backed analysis and proposals, not executed runs.

Known limitations: this ledger was assembled from the current filesystem only;
the checkout has no Git history or remote, no automated commit boundary can be
reconstructed. The new 4,000-pair metrics are user-reported and lack a local
score manifest. No tests were run for this documentation-only update.
