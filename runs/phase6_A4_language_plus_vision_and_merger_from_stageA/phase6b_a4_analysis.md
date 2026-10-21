# Phase 6B paired analysis

This report is descriptive. It uses the named score arms and the same four-view protocol; paired changes do not by themselves establish causation or held-out generalization.


## Overall metrics

### valQ

| Arm | TTA accuracy | Single-view accuracy | mean p_gold | mean gold margin |
| --- | --- | --- | --- | --- |
| A0 | 498/595 (83.70%) | 83.03% | 0.7940 | 0.6249 |
| A1 | 490/595 (82.35%) | 82.02% | 0.7832 | 0.6039 |
| A2 | 499/595 (83.87%) | 84.03% | 0.7887 | 0.6151 |
| A3 | 495/595 (83.19%) | 82.86% | 0.7867 | 0.6102 |
| A4 | 500/595 (84.03%) | 83.03% | 0.7854 | 0.6096 |

### valT

| Arm | TTA accuracy | Single-view accuracy | mean p_gold | mean gold margin |
| --- | --- | --- | --- | --- |
| A0 | 885/1059 (83.57%) | 83.29% | 0.7924 | 0.6234 |
| A1 | 854/1059 (80.64%) | 80.55% | 0.7666 | 0.5766 |
| A2 | 871/1059 (82.25%) | 80.26% | 0.7702 | 0.5834 |
| A3 | 858/1059 (81.02%) | 80.36% | 0.7699 | 0.5820 |
| A4 | 863/1059 (81.49%) | 80.17% | 0.7667 | 0.5774 |

## Paired transitions

### valQ

| Candidate vs reference | Fixes | Regressions | Accuracy delta | 95% video-cluster interval |
| --- | --- | --- | --- | --- |
| A1_vs_A0 | 19 | 27 | -1.34 pp | [-3.61 pp, +0.87 pp] |
| A2_vs_A0 | 22 | 21 | +0.17 pp | [-2.03 pp, +2.29 pp] |
| A2_vs_A1 | 12 | 3 | +1.51 pp | [+0.33 pp, +2.83 pp] |
| A3_vs_A0 | 22 | 25 | -0.50 pp | [-2.78 pp, +1.79 pp] |
| A3_vs_A1 | 9 | 4 | +0.84 pp | [-0.33 pp, +2.04 pp] |
| A3_vs_A2 | 4 | 8 | -0.67 pp | [-1.83 pp, +0.50 pp] |
| A4_vs_A0 | 24 | 22 | +0.34 pp | [-1.86 pp, +2.61 pp] |
| A4_vs_A1 | 13 | 3 | +1.68 pp | [+0.49 pp, +3.05 pp] |
| A4_vs_A2 | 4 | 3 | +0.17 pp | [-0.67 pp, +1.03 pp] |
| A4_vs_A3 | 9 | 4 | +0.84 pp | [-0.34 pp, +2.02 pp] |

### valT

| Candidate vs reference | Fixes | Regressions | Accuracy delta | 95% video-cluster interval |
| --- | --- | --- | --- | --- |
| A1_vs_A0 | 33 | 64 | -2.93 pp | [-4.72 pp, -1.18 pp] |
| A2_vs_A0 | 39 | 53 | -1.32 pp | [-3.19 pp, +0.54 pp] |
| A2_vs_A1 | 27 | 10 | +1.61 pp | [+0.45 pp, +2.80 pp] |
| A3_vs_A0 | 34 | 61 | -2.55 pp | [-4.52 pp, -0.66 pp] |
| A3_vs_A1 | 12 | 8 | +0.38 pp | [-0.36 pp, +1.10 pp] |
| A3_vs_A2 | 10 | 23 | -1.23 pp | [-2.62 pp, +0.00 pp] |
| A4_vs_A0 | 32 | 54 | -2.08 pp | [-3.94 pp, -0.19 pp] |
| A4_vs_A1 | 16 | 7 | +0.85 pp | [+0.09 pp, +1.65 pp] |
| A4_vs_A2 | 10 | 18 | -0.76 pp | [-1.82 pp, +0.29 pp] |
| A4_vs_A3 | 13 | 8 | +0.47 pp | [-0.55 pp, +1.53 pp] |

## Predeclared valT gate

A candidate must exceed A0 on the valT point estimate to pass screening. Strong positive evidence requires the video-cluster 95% interval to lie above zero. The historical category guard requires every valT top-level category with n >= 50 to stay within -1 percentage point.

| Arm | valT delta | Point screen | Video 95% interval | Strong positive | Category guard |
| --- | --- | --- | --- | --- | --- |
| A1 | -2.93 pp | FAIL | [-4.72 pp, -1.18 pp] | NO | FAIL |
| A2 | -1.32 pp | FAIL | [-3.19 pp, +0.54 pp] | NO | FAIL |
| A3 | -2.55 pp | FAIL | [-4.52 pp, -0.66 pp] | NO | FAIL |
| A4 | -2.08 pp | FAIL | [-3.94 pp, -0.19 pp] | NO | FAIL |

**A3 and A4 both fail the predeclared primary valT point-estimate gate versus A0. A4's interval is below zero, indicating a regression on this development split.**

## Task/category/source slices

### valQ

#### task

| task | n | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- | --- |
| Causation Reasoning | 15 | 93.33% | 93.33% | 93.33% | 93.33% | 93.33% |
| Change Detection | 38 | 78.95% | 81.58% | 89.47% | 81.58% | 86.84% |
| Consequence Reasoning | 17 | 100.00% | 94.12% | 94.12% | 94.12% | 94.12% |
| General Understanding | 39 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| Geometric Relation | 110 | 79.09% | 78.18% | 78.18% | 77.27% | 79.09% |
| Hypothetical Reasoning | 15 | 86.67% | 86.67% | 86.67% | 86.67% | 86.67% |
| Object and Land Cover Recognition | 66 | 87.88% | 83.33% | 84.85% | 84.85% | 86.36% |
| Perspective and Viewpoint | 78 | 85.90% | 85.90% | 85.90% | 84.62% | 84.62% |
| Structural Layout | 85 | 83.53% | 82.35% | 83.53% | 83.53% | 83.53% |
| Temporal Grounding | 69 | 73.91% | 71.01% | 73.91% | 76.81% | 75.36% |
| Trend and Pattern | 63 | 80.95% | 79.37% | 82.54% | 80.95% | 82.54% |

#### category

| category | n | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- | --- |
| Causal Reasoning | 47 | 93.62% | 91.49% | 91.49% | 91.49% | 91.49% |
| Factual Perception | 143 | 88.81% | 87.41% | 90.21% | 88.11% | 90.21% |
| Spatial Reasoning | 273 | 82.42% | 81.68% | 82.05% | 81.32% | 82.05% |
| Temporal Understanding | 132 | 77.27% | 75.00% | 78.03% | 78.79% | 78.79% |

#### source

| source | n | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- | --- |
| ERA_Tra | 164 | 85.98% | 87.80% | 88.41% | 87.80% | 89.02% |
| Hawk_UAV | 305 | 83.93% | 81.64% | 82.95% | 82.62% | 83.28% |
| UAVDT | 21 | 85.71% | 80.95% | 85.71% | 85.71% | 85.71% |
| VisDrone | 105 | 79.05% | 76.19% | 79.05% | 77.14% | 78.10% |

### valT

#### task

| task | n | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- | --- |
| Causation Reasoning | 31 | 83.87% | 83.87% | 87.10% | 87.10% | 83.87% |
| Change Detection | 74 | 83.78% | 79.73% | 78.38% | 79.73% | 82.43% |
| Consequence Reasoning | 31 | 96.77% | 93.55% | 93.55% | 93.55% | 93.55% |
| General Understanding | 75 | 98.67% | 97.33% | 97.33% | 97.33% | 97.33% |
| Geometric Relation | 171 | 78.36% | 74.27% | 77.78% | 75.44% | 76.02% |
| Hypothetical Reasoning | 29 | 93.10% | 93.10% | 93.10% | 86.21% | 93.10% |
| Object and Land Cover Recognition | 109 | 78.90% | 76.15% | 80.73% | 77.98% | 78.90% |
| Perspective and Viewpoint | 143 | 82.52% | 81.82% | 82.52% | 81.82% | 81.12% |
| Structural Layout | 153 | 88.89% | 83.66% | 84.31% | 84.31% | 84.97% |
| Temporal Grounding | 133 | 74.44% | 69.92% | 73.68% | 71.43% | 70.68% |
| Trend and Pattern | 110 | 84.55% | 83.64% | 82.73% | 81.82% | 82.73% |

#### category

| category | n | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- | --- |
| Causal Reasoning | 91 | 91.21% | 90.11% | 91.21% | 89.01% | 90.11% |
| Factual Perception | 258 | 86.05% | 83.33% | 84.88% | 84.11% | 85.27% |
| Spatial Reasoning | 467 | 83.08% | 79.66% | 81.37% | 80.30% | 80.51% |
| Temporal Understanding | 243 | 79.01% | 76.13% | 77.78% | 76.13% | 76.13% |

#### source

| source | n | A0 | A1 | A2 | A3 | A4 |
| --- | --- | --- | --- | --- | --- | --- |
| ERA_Tra | 289 | 85.47% | 83.04% | 83.04% | 82.35% | 82.35% |
| Hawk_UAV | 493 | 82.15% | 78.09% | 80.73% | 78.70% | 79.92% |
| UAVDT | 84 | 76.19% | 75.00% | 76.19% | 76.19% | 73.81% |
| VisDrone | 193 | 87.56% | 86.01% | 87.56% | 87.05% | 87.56% |

## Provenance

The machine-readable JSON records split checksums, adapter checksums when present, merge mode, view protocol, and score-file SHA256 values. Slurm GPU/memory accounting for A4 is recorded in `phase6b_resources.json`.

