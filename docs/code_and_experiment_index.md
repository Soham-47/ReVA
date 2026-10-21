# ReVA code and experiment index

**Updated:** 2026-10-21  
**Workspace:** `/home/soham/ReVA`

This index maps the principal implementation files in this checkout to the
experiments and reports they support. The `.git` directory is empty, so this is
an inventory of code currently present, **not** a provable list of files added
since a particular commit. For status, metrics, decision gates, and caveats, use
the [experiment ledger](experiment_ledger.md). For the latest 4,000-pair result
and pending submission state, see the [session handoff](handoff_2026-10-21.md).

## Core training and evaluation pipeline

| Area | Main implementation | Role |
|---|---|---|
| Dataset and split provenance | [`code/download_data.py`](../code/download_data.py), [`code/build_manifest.py`](../code/build_manifest.py), [`code/make_splits.py`](../code/make_splits.py), [`code/common.py`](../code/common.py) | Downloads only referenced clips, validates media, pins dataset revisions, and creates train-only `fit_A`, `valT`, `valQ`, and full-train partitions. |
| Stage A/B recipes | [`code/train.py`](../code/train.py), [`code/score.py`](../code/score.py), [`code/evaluate.py`](../code/evaluate.py), [`code/configs/stageA.yaml`](../code/configs/stageA.yaml), [`code/configs/stageB.yaml`](../code/configs/stageB.yaml) | Resumable LoRA SFT, letter-logit scoring with temporal/option-order views, evaluation, and gated full-train configuration. |
| Cluster orchestration | [`jobs/pipeline.sh`](../jobs/pipeline.sh), [`jobs/train.sbatch`](../jobs/train.sbatch), [`jobs/score.sbatch`](../jobs/score.sbatch), [`jobs/submit_stageB.sh`](../jobs/submit_stageB.sh) | Sharanga Slurm wrappers, dependencies, resumable execution, and Stage B gate. The existence of a Stage B recipe does not mean its result is present; see the ledger. |
| Submission validation | [`code/make_submission.py`](../code/make_submission.py) | Converts test score records into the required ZIP shape and validates IDs, answer choices, probabilities, and archive contents. |

## Inference-time evidence experiments

| Workstream | Main implementation | Experiment records |
|---|---|---|
| Endpoint and adaptive frame sampling | [`code/build_feature_cache.py`](../code/build_feature_cache.py), [`jobs/adaptive_cache.sbatch`](../jobs/adaptive_cache.sbatch), [`jobs/adaptive_score.sbatch`](../jobs/adaptive_score.sbatch), [`jobs/score_endpoints.sbatch`](../jobs/score_endpoints.sbatch) | Endpoint ablation and SigLIP2 query-aware frame selection. Adaptive selection did not pass the recorded promotion gate. |
| Temporal localization | [`code/score_temporal.py`](../code/score_temporal.py), [`code/evaluate_temporal.py`](../code/evaluate_temporal.py), [`code/score_temporal_verified.py`](../code/score_temporal_verified.py), [`code/evaluate_temporal_verified.py`](../code/evaluate_temporal_verified.py), [`jobs/modal_phase2_temporal.py`](../jobs/modal_phase2_temporal.py), [`jobs/modal_phase2_verified_window.py`](../jobs/modal_phase2_verified_window.py) | Coarse-to-fine sampling, explicit timestamp context, reviewed-window diagnostics, and paired evaluation. Full results are in [`runs/phase2_temporal_coarsefine/`](../runs/phase2_temporal_coarsefine/) and [`runs/phase2_temporal_verified_window/`](../runs/phase2_temporal_verified_window/). |
| Detector and segmentation context | [`code/grounding.py`](../code/grounding.py), [`code/segmentation.py`](../code/segmentation.py), [`code/build_grounding_cache.py`](../code/build_grounding_cache.py), [`code/build_segmentation_cache.py`](../code/build_segmentation_cache.py), [`code/replicate_sam3.py`](../code/replicate_sam3.py), [`code/sam3_overlay.py`](../code/sam3_overlay.py) | GroundingDINO boxes/tracks, SAM masks, and bounded inference context. Pilot results are diagnostic and not broadly promoted. |
| Region retrieval and explicit geometry | [`code/phase3a_retrieval.py`](../code/phase3a_retrieval.py), [`runs/phase3_explicit_geometry/`](../runs/phase3_explicit_geometry/), [`runs/phase3b_gate1/`](../runs/phase3b_gate1/), [`runs/phase4_qcsrr/`](../runs/phase4_qcsrr/) | SAM 3 proposal retrieval, SigLIP2 crop ranking, GroundingDINO/SAM 3 relation evidence, and fixed-frame QCSRR gates. Several phase-specific Python tools are kept with their run artifacts. |
| Crop evidence ablation | [`jobs/phase5_prepare_crop_ablation.py`](../jobs/phase5_prepare_crop_ablation.py), [`jobs/phase5_qwen_crop_ablation_hpc.py`](../jobs/phase5_qwen_crop_ablation_hpc.py), [`jobs/phase5_analyze_crop_ablation.py`](../jobs/phase5_analyze_crop_ablation.py) | Video-only, random-crop, and retrieved-crop evidence comparison. The selected cohort showed no retrieved-crop fixes over video-only. |

## Training and tracking experiments

| Workstream | Main implementation | Experiment records |
|---|---|---|
| Visual adapter variants | [`code/phase6b_analysis.py`](../code/phase6b_analysis.py), [`code/phase6b_training_diagnostics.py`](../code/phase6b_training_diagnostics.py), [`jobs/phase6_train.sbatch`](../jobs/phase6_train.sbatch), [`code/configs/phase6_A1.yaml`](../code/configs/phase6_A1.yaml)–[`phase6_A4.yaml`](../code/configs/phase6_A4.yaml) | Language, merger, and vision-adapter variants. Recorded variants did not pass the `valT` promotion gate. |
| Frame manifests and source evidence | [`code/phase7_manifest.py`](../code/phase7_manifest.py), [`code/validate_phase7_cache.py`](../code/validate_phase7_cache.py), [`code/phase7b_source_evidence.py`](../code/phase7b_source_evidence.py) | Canonical frame/timestamp provenance and validation of automatically proposed targets against source review. |
| Replicate tracking diagnostics | [`code/phase7c_replicate_diagnostics.py`](../code/phase7c_replicate_diagnostics.py), [`code/phase7c_replicate_runner.py`](../code/phase7c_replicate_runner.py) | Bounded two-video SAM 3 / GroundingDINO / SAM 2 diagnostic. The evidence gate remained closed; no broad tracking or Qwen result was established. |
| Qwen3.8 / Qwen3-VL-32B comparison | [`code/phase8_adapters.py`](../code/phase8_adapters.py), [`code/phase8_scorer.py`](../code/phase8_scorer.py), [`code/phase8_analysis.py`](../code/phase8_analysis.py), [`code/phase8_pilot_compare.py`](../code/phase8_pilot_compare.py), [`jobs/phase8_download.sbatch`](../jobs/phase8_download.sbatch), [`jobs/phase8_score.sbatch`](../jobs/phase8_score.sbatch) | Pinned alternative-backbone inference, optional PEFT adapter loading, pilot comparison, and paired analysis. Local validation remains incomplete; see the ledger. |

## Verification code

The tests currently present are under [`tests/`](../tests/):

- Core logic and job wrappers: `test_pure_logic.py`, `test_jobs.sh`.
- Phase 3A/7 evidence: `test_phase3a_hpc.py`, `test_phase7_manifest.py`,
  `test_phase7_cache_validation.py`, `test_phase7b_source_evidence.py`,
  `test_phase7c_replicate_runner.py`.
- Phase 8 scoring and comparisons: `test_phase8_scorer.py`,
  `test_phase8_analysis.py`, `test_phase8_pilot_compare.py`,
  `test_qwen38_training.py`.

These tests were not run as part of the documentation update.

## Reconstructing a true commit delta

To report exactly what changed since a particular commit, restore the repository
Git metadata or provide the commit hash/remote checkout. The current `.git`
directory is empty; filesystem timestamps and directory names are not a safe
substitute for a commit diff. Until then, this index and the ledger should be
read as a best-effort record of the files and results available in this
workspace, with missing artifacts and provenance called out explicitly.
