# Phase 6B A3 Evaluation Report

## Result

A3 did **not** improve ReVA VideoQA on video-held-out `valT`. Its four-view accuracy was **81.02% (858/1,059)**, compared with **83.57% (885/1,059)** for Stage A A0, a **−2.55 percentage-point** change. The paired video-cluster 95% interval was **[−4.52, −0.66] pp** across 72 videos (10,000 bootstrap draws). A3 therefore fails the predeclared primary gate and provides evidence of a regression on this split.

| Arm | valQ | valT | valT change vs A0 |
|---|---:|---:|---:|
| A0 | 498/595 (83.70%) | 885/1,059 (83.57%) | — |
| A1 | 490/595 (82.35%) | 854/1,059 (80.64%) | −2.93 pp |
| A2 | 499/595 (83.87%) | 871/1,059 (82.25%) | −1.32 pp |
| A3 | 495/595 (83.19%) | 858/1,059 (81.02%) | −2.55 pp |

A3 was only **+0.38 pp** above A1 on valT (12 fixes, 8 regressions; video-cluster 95% interval **[−0.36, +1.10] pp**), so the apparent gain over language-only continuation is inconclusive. It was **−1.23 pp** below A2 (10 fixes, 23 regressions; interval **[−2.62, 0.00] pp**). Against A0, A3 produced 34 fixes and 61 regressions. The mean gold probability and margin also fell from **0.7924 / 0.6234** for A0 to **0.7699 / 0.5820** for A3 on valT.

## Task results

| valT task | A0 | A3 | Change |
|---|---:|---:|---:|
| Geometric Relation (n=171) | 78.36% | 75.44% | −2.92 pp |
| Temporal Grounding (n=133) | 74.44% | 71.43% | −3.01 pp |
| Object and Land Cover Recognition (n=109) | 78.90% | 77.98% | −0.92 pp |

All four top-level valT categories with at least 50 questions breached the predeclared −1 pp guard versus A0: Causal Reasoning **−2.20 pp**, Factual Perception **−1.94 pp**, Spatial Reasoning **−2.78 pp**, and Temporal Understanding **−2.88 pp**. No separate counting task label was present in the split taxonomy. These slices do not identify the cause of an error; no manual error adjudication was performed for this run.

## A3 training and validation

A3 continued from the frozen Stage A adapter with the same `fit_A`, seed 42, deterministic sample order, 200 optimizer steps, effective batch protocol, bf16, gradient checkpointing, and answer loss as the Phase 6 reference. It trained language LoRA plus rank-16 vision LoRA on `visual.blocks.{23,24,25,26}.attn.{qkv,proj}`; MLP projections and visual merger stayed frozen. The language and vision learning rates were `2e-4` and `2e-5`, respectively. Trainable parameters were **44,089,344** total: 43,646,976 language and 442,368 vision.

The 200-step run completed with train loss **0.3526**. Logged loss averaged 0.3655 in the first half and 0.3397 in the last half; logged gradient norms were finite (0.1758–5.643, mean 2.871). Adapter deltas were nonzero: language LoRA L2 norm 9.760 (max absolute change 0.00904) and vision LoRA L2 norm 6.539 (max absolute change 0.03027). This confirms updates occurred, but does not show that they learned useful transferable visual features. Intermediate checkpoints were not evaluated, and validation loss/per-group full-run gradient trajectories were not recorded; the accuracy regression cannot be attributed specifically to overfitting, learning rate, or catastrophic forgetting.

The A3 save/reload smoke check matched unmerged logits exactly and confirmed finite nonzero vision gradients. Because the scorer uses the historical merged-adapter mode, a follow-up five-question answer-token probe compared merged and unmerged A3 inference: all five answer choices matched, with maximum answer-logit difference 0.125. The larger full-sequence logit differences seen in the smoke diagnostic therefore did not change these five answer decisions; the small probe does not prove parity over all validation examples.

## A4 status

A4 passed its bounded smoke check, including nonzero gradients for the intended vision and main/DeepStack merger targets, but was canceled at **16/200** training steps at the user's direction. It has no complete checkpoint or evaluation result and is excluded from the comparison. Consequently, this run does not answer whether jointly adapting vision and merger helps.

## Compute and reproducibility

The complete A3 train and four-view scoring jobs ran on **H200 NVL `gpunode7`**: Slurm 403043 (36:47) and 403109 (24:46). Slurm reported peak GPU memory of 28,270 MiB for training and 27,154 MiB for scoring; peak host RSS was 4.96 GiB and 15.24 GiB. The five-question merge probe (403166) took 16 seconds and reported 18.1 GiB peak GPU allocation. The A3/A4 smoke jobs took 31/26 seconds and peaked at 25.3/25.9 GiB. A4's canceled allocation used 3:08. The selected H200 scorer (403109) started first; Slurm's `QOSMaxCpuPerUserLimit` allowed only one 12-CPU job, so duplicate pending scorer 403110 was canceled before start. No dollar billing rate was available from HPC accounting.

Full score files and manifests, adapter checkpoint, training/scoring logs, diagnostics, and merge probe are retained in this run directory. The pinned base model revision is `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`; score manifests validate 595 valQ and 1,059 valT questions, each with the expected four views.

## Recommendation

Do not continue with this upper-block attention-only vision-LoRA configuration: it did not beat A0, did not reliably beat A1, underperformed A2, and reduced geometric and temporal accuracy. Do not infer that all vision adaptation or a stronger pretrained VLM is ineffective; neither was isolated by this experiment. A4 remains untested. Any future backbone comparison should begin with a small frozen-backbone evaluation, while another fine-tuning run should wait for a specific data or optimization hypothesis. The validation results are development evidence, not a final test-set generalization claim.

Detailed paired metrics, per-question transitions, task/category/source slices, and video-cluster intervals are in [phase6b_a3_analysis.md](phase6b_a3_analysis.md), phase6b_a3_analysis.json (local-only artifact: `phase6b_a3_analysis.json`), and phase6b_a3_analysis.csv (local-only artifact: `phase6b_a3_analysis.csv`). Slurm resource accounting is in phase6b_resources.json (local-only artifact: `phase6b_resources.json`).
