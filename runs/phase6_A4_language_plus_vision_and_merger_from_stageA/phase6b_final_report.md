# Phase 6B A4 evaluation report

## Result

A4 completed the previously canceled 200-step run and full four-view scoring. It did **not** pass the predeclared valT gate against frozen Stage A A0.

| Arm | valQ | valT |
|---|---:|---:|
| A0 | 498/595 (83.70%) | 885/1,059 (83.57%) |
| A1 | 490/595 (82.35%) | 854/1,059 (80.64%) |
| A2 | 499/595 (83.87%) | 871/1,059 (82.25%) |
| A3 | 495/595 (83.19%) | 858/1,059 (81.02%) |
| **A4** | **500/595 (84.03%)** | **863/1,059 (81.49%)** |

On valQ, A4 was +0.34 percentage points versus A0 (24 fixes, 22 regressions); the paired video-cluster 95% interval was [−1.86, +2.61] pp. On valT, A4 was −2.08 pp versus A0 (32 fixes, 54 regressions), with interval [−3.94, −0.19] pp across 72 videos. This is evidence of a regression on this development split, not held-out test generalization.

Compared directly with A3 on valT, A4 was +0.47 pp (13 fixes, 8 regressions), but the interval [−0.55, +1.53] pp includes zero. The added merger adaptation therefore did not show a reliable improvement over A3. A4's mean gold probability and margin on valT also fell from A0's 0.7924 / 0.6234 to 0.7667 / 0.5774.

## Task slices

| valT slice | A0 | A4 | Change |
|---|---:|---:|---:|
| Geometric Relation (n=171) | 78.36% | 76.02% | −2.34 pp |
| Temporal Grounding (n=133) | 74.44% | 70.68% | −3.76 pp |
| Object and Land Cover Recognition (n=109) | 78.90% | 78.90% | 0.00 pp |

The top-level category guard failed in Causal Reasoning (−1.10 pp), Spatial Reasoning (−2.57 pp), and Temporal Understanding (−2.88 pp); Factual Perception was −0.78 pp. No separate counting task was present. These aggregate slices identify where accuracy changed, not why individual answers changed.

## Training and implementation

The old A4 attempt had stopped at 16/200 and left only smoke-test adapters, so the authorized completion restarted from step 0 using the unchanged A4 configuration and frozen Stage A adapter. Training completed 200 steps on `fit_A`, seed 42, with the same answer loss, data order, four-block vision LoRA, and learning rates used by the approved design. A4 trained language LoRA plus vision attention LoRA on `visual.blocks.{23,24,25,26}.attn.{qkv,proj}` and LoRA on the final and three DeepStack visual mergers. The merged adapter was used for scoring, matching A0–A3's four-view protocol.

Trainable parameters totaled 45,236,224: 43,646,976 language, 442,368 vision, and 1,146,880 merger. Final training loss was 0.3546. The saved deltas were nonzero: language L2 9.777, vision L2 6.540, and merger L2 6.549. This verifies parameter updates, not useful transfer.

## Resources and reproducibility

Training job 403239 ran on H200 NVL `gpunode7` for 37:06; peak GPU memory was 28,628 MiB (85% utilization) and peak host RSS was 5,185,724 KiB. Scoring job 403248 ran on the same GPU for 24:59; peak GPU memory was 27,154 MiB (75% utilization) and peak host RSS was 15,392,476 KiB. Both score files and manifests contain the full 595 valQ and 1,059 valT rows, each with four views. The A4 smoke job 403035 took 26 seconds. The earlier user-canceled allocation 403045 used 3:08 at 16/200. HPC accounting did not expose a dollar billing rate.

The pinned model revision was `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`. The adapter, score files, manifests, logs, analysis outputs, and resource record are preserved in this run directory. Per-question transitions and video-cluster intervals are in [phase6b_a4_analysis.md](phase6b_a4_analysis.md), with machine-readable results in phase6b_a4_analysis.json (local-only artifact: `phase6b_a4_analysis.json`) and per-question CSV in phase6b_a4_analysis.csv (local-only artifact: `phase6b_a4_analysis.csv`).

## Recommendation and limits

Do not proceed with this A4 vision-plus-merger LoRA configuration: it failed the primary valT gate and regressed from A0, while its apparent gain over A3 was inconclusive. A3 and A4 therefore provide no evidence that these upper-block vision/merger adaptations improve VideoQA under this setup. This does not test every possible visual adaptation method. Results are one-seed development evidence; there was no manual adjudication of changed answers, no intermediate-checkpoint evaluation, and no final test-set confirmation.
