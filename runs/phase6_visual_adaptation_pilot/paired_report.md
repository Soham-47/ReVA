# Phase 6 paired visual adaptation analysis

This report uses the frozen A0 score JSONL under `runs/stageA_q3vl8b_r16/legacy/` and the copied A1/A2 score artifacts under this run directory. `valQ` and `valT` are development splits; these results do not establish held-out or test generalization.

Validation: A0 copied references match the frozen baseline byte-for-byte; A1/A2 manifests agree on pinned settings, offsets `[0.0, 0.5]`, two rotations, legacy sampling, and four ordered view signatures. Bootstrap uses 10,000 deterministic video-cluster resamples (seed `20261019`).

## Compute and provenance

The supplied Slurm accounting is recorded in [`phase6_provenance.md`](phase6_provenance.md): A1 training job 402807 on H200 took 00:35:07 (train loss 0.3526, peak 27,926 MiB, 85% GPU utilization); A2 training job 402810 on H100 took 00:42:31 (train loss 0.3567, peak 28,070 MiB, 81% utilization); A1/A2 scoring jobs 402864/402865 took 00:24:38/00:26:23 and completed 1,654 rows each. The A2 manifest points to the remote H100 adapter run and its adjacent `RUN_PROVENANCE`; no direct HPC currency charge is exposed.

## Overall metrics

### valQ

| Arm | TTA accuracy | Single-view accuracy | mean p_gold | median p_gold | p_gold > 0.5 | mean margin | median margin | positive margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 498/595 (83.70%) | 494/595 (83.03%) | 0.7940 | 0.9720 | 81.01% | 0.6249 | 0.9544 | 83.70% |
| A1 | 490/595 (82.35%) | 488/595 (82.02%) | 0.7832 | 0.9649 | 80.84% | 0.6039 | 0.9441 | 82.35% |
| A2 | 499/595 (83.87%) | 500/595 (84.03%) | 0.7887 | 0.9652 | 82.18% | 0.6151 | 0.9436 | 83.87% |

### valT

| Arm | TTA accuracy | Single-view accuracy | mean p_gold | median p_gold | p_gold > 0.5 | mean margin | median margin | positive margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | 885/1059 (83.57%) | 882/1059 (83.29%) | 0.7924 | 0.9631 | 81.59% | 0.6234 | 0.9381 | 83.57% |
| A1 | 854/1059 (80.64%) | 853/1059 (80.55%) | 0.7666 | 0.9544 | 78.19% | 0.5766 | 0.9217 | 80.64% |
| A2 | 871/1059 (82.25%) | 850/1059 (80.26%) | 0.7702 | 0.9570 | 79.41% | 0.5834 | 0.9309 | 82.25% |

## A2-minus-A1 confidence deltas

| Split | mean p_gold | median p_gold | p_gold > 0.5 | mean margin | median margin | positive margin |
| --- | --- | --- | --- | --- | --- | --- |
| valQ | +0.0056 | +0.0003 | +0.0134 | +0.0111 | -0.0006 | +0.0151 |
| valT | +0.0036 | +0.0026 | +0.0123 | +0.0068 | +0.0091 | +0.0161 |

## Paired transitions and clustered intervals

### valQ

| Candidate vs reference | fixes | regressions | unchanged correct | unchanged wrong | accuracy delta | 95% video-cluster interval |
| --- | --- | --- | --- | --- | --- | --- |
| A1 vs A0 | 19 | 27 | 471 | 78 | -1.34 pp | [-3.61 pp, +0.87 pp] |
| A2 vs A0 | 22 | 21 | 477 | 75 | +0.17 pp | [-2.03 pp, +2.29 pp] |
| A2 vs A1 | 12 | 3 | 487 | 93 | +1.51 pp | [+0.33 pp, +2.83 pp] |

### valT

| Candidate vs reference | fixes | regressions | unchanged correct | unchanged wrong | accuracy delta | 95% video-cluster interval |
| --- | --- | --- | --- | --- | --- | --- |
| A1 vs A0 | 33 | 64 | 821 | 141 | -2.93 pp | [-4.72 pp, -1.14 pp] |
| A2 vs A0 | 39 | 53 | 832 | 135 | -1.32 pp | [-3.26 pp, +0.58 pp] |
| A2 vs A1 | 27 | 10 | 844 | 178 | +1.61 pp | [+0.38 pp, +2.79 pp] |

## Task, category, and source slices

### valQ

#### Task

| Task | n | A0 TTA | A1 TTA | A2 TTA | A2-A1 |
| --- | --- | --- | --- | --- | --- |
| Causation Reasoning | 15 | 93.33% | 93.33% | 93.33% | +0.00 pp |
| Change Detection | 38 | 78.95% | 81.58% | 89.47% | +7.89 pp |
| Consequence Reasoning | 17 | 100.00% | 94.12% | 94.12% | +0.00 pp |
| General Understanding | 39 | 100.00% | 100.00% | 100.00% | +0.00 pp |
| Geometric Relation | 110 | 79.09% | 78.18% | 78.18% | +0.00 pp |
| Hypothetical Reasoning | 15 | 86.67% | 86.67% | 86.67% | +0.00 pp |
| Object and Land Cover Recognition | 66 | 87.88% | 83.33% | 84.85% | +1.52 pp |
| Perspective and Viewpoint | 78 | 85.90% | 85.90% | 85.90% | +0.00 pp |
| Structural Layout | 85 | 83.53% | 82.35% | 83.53% | +1.18 pp |
| Temporal Grounding | 69 | 73.91% | 71.01% | 73.91% | +2.90 pp |
| Trend and Pattern | 63 | 80.95% | 79.37% | 82.54% | +3.17 pp |

#### Top-level category

| Top-level category | n | A0 TTA | A1 TTA | A2 TTA | A2-A1 |
| --- | --- | --- | --- | --- | --- |
| Causal Reasoning | 47 | 93.62% | 91.49% | 91.49% | +0.00 pp |
| Factual Perception | 143 | 88.81% | 87.41% | 90.21% | +2.80 pp |
| Spatial Reasoning | 273 | 82.42% | 81.68% | 82.05% | +0.37 pp |
| Temporal Understanding | 132 | 77.27% | 75.00% | 78.03% | +3.03 pp |

#### Source

| Source | n | A0 TTA | A1 TTA | A2 TTA | A2-A1 |
| --- | --- | --- | --- | --- | --- |
| ERA_Tra | 164 | 85.98% | 87.80% | 88.41% | +0.61 pp |
| Hawk_UAV | 305 | 83.93% | 81.64% | 82.95% | +1.31 pp |
| UAVDT | 21 | 85.71% | 80.95% | 85.71% | +4.76 pp |
| VisDrone | 105 | 79.05% | 76.19% | 79.05% | +2.86 pp |

### valT

#### Task

| Task | n | A0 TTA | A1 TTA | A2 TTA | A2-A1 |
| --- | --- | --- | --- | --- | --- |
| Causation Reasoning | 31 | 83.87% | 83.87% | 87.10% | +3.23 pp |
| Change Detection | 74 | 83.78% | 79.73% | 78.38% | -1.35 pp |
| Consequence Reasoning | 31 | 96.77% | 93.55% | 93.55% | +0.00 pp |
| General Understanding | 75 | 98.67% | 97.33% | 97.33% | +0.00 pp |
| Geometric Relation | 171 | 78.36% | 74.27% | 77.78% | +3.51 pp |
| Hypothetical Reasoning | 29 | 93.10% | 93.10% | 93.10% | +0.00 pp |
| Object and Land Cover Recognition | 109 | 78.90% | 76.15% | 80.73% | +4.59 pp |
| Perspective and Viewpoint | 143 | 82.52% | 81.82% | 82.52% | +0.70 pp |
| Structural Layout | 153 | 88.89% | 83.66% | 84.31% | +0.65 pp |
| Temporal Grounding | 133 | 74.44% | 69.92% | 73.68% | +3.76 pp |
| Trend and Pattern | 110 | 84.55% | 83.64% | 82.73% | -0.91 pp |

#### Top-level category

| Top-level category | n | A0 TTA | A1 TTA | A2 TTA | A2-A1 |
| --- | --- | --- | --- | --- | --- |
| Causal Reasoning | 91 | 91.21% | 90.11% | 91.21% | +1.10 pp |
| Factual Perception | 258 | 86.05% | 83.33% | 84.88% | +1.55 pp |
| Spatial Reasoning | 467 | 83.08% | 79.66% | 81.37% | +1.71 pp |
| Temporal Understanding | 243 | 79.01% | 76.13% | 77.78% | +1.65 pp |

#### Source

| Source | n | A0 TTA | A1 TTA | A2 TTA | A2-A1 |
| --- | --- | --- | --- | --- | --- |
| ERA_Tra | 289 | 85.47% | 83.04% | 83.04% | +0.00 pp |
| Hawk_UAV | 493 | 82.15% | 78.09% | 80.73% | +2.64 pp |
| UAVDT | 84 | 76.19% | 75.00% | 76.19% | +1.19 pp |
| VisDrone | 193 | 87.56% | 86.01% | 87.56% | +1.55 pp |

## Temporal Grounding subset

| Split | n | A0 TTA | A1 TTA | A2 TTA | A2-A1 |
| --- | --- | --- | --- | --- | --- |
| valQ | 69 | 73.91% | 71.01% | 73.91% | +2.90 pp |
| valT | 133 | 74.44% | 69.92% | 73.68% | +3.76 pp |

## Predeclared A2-vs-A1 screening gate

The thresholds below are applied exactly as declared in `tasks/phase6_visual_adaptation_pilot.md`; bootstrap intervals are reported for context and do not replace these point thresholds.

| Gate | Observed | Threshold | Result |
| --- | --- | --- | --- |
| valT TTA accuracy | +1.61 pp | +2.00 pp | FAIL (below threshold) |
| valQ TTA accuracy | +1.51 pp | +1.00 pp | PASS |
| valQ mean p_gold | +0.0056 | >= -0.02 | PASS |
| valQ mean gold margin | +0.0111 | >= -0.02 | PASS |
| valT mean p_gold | +0.0036 | >= -0.02 | PASS |
| valT mean gold margin | +0.0068 | >= -0.02 | PASS |
| valQ category=Factual Perception (n=143) | +2.80 pp | -1.00 pp | PASS |
| valQ category=Spatial Reasoning (n=273) | +0.37 pp | -1.00 pp | PASS |
| valQ category=Temporal Understanding (n=132) | +3.03 pp | -1.00 pp | PASS |
| valQ source=ERA_Tra (n=164) | +0.61 pp | -1.00 pp | PASS |
| valQ source=Hawk_UAV (n=305) | +1.31 pp | -1.00 pp | PASS |
| valQ source=VisDrone (n=105) | +2.86 pp | -1.00 pp | PASS |
| valT category=Causal Reasoning (n=91) | +1.10 pp | -1.00 pp | PASS |
| valT category=Factual Perception (n=258) | +1.55 pp | -1.00 pp | PASS |
| valT category=Spatial Reasoning (n=467) | +1.71 pp | -1.00 pp | PASS |
| valT category=Temporal Understanding (n=243) | +1.65 pp | -1.00 pp | PASS |
| valT source=ERA_Tra (n=289) | +0.00 pp | -1.00 pp | PASS |
| valT source=Hawk_UAV (n=493) | +2.64 pp | -1.00 pp | PASS |
| valT source=UAVDT (n=84) | +1.19 pp | -1.00 pp | PASS |
| valT source=VisDrone (n=193) | +1.55 pp | -1.00 pp | PASS |

**Overall A2-vs-A1 screening result: NO-GO.**

## Highest-confidence paired changes

Taxonomy labels are deterministic task/category proxies for review: factual perception, spatial and structural, temporal and change, reasoning, or domain and ambiguity. Input/readout metadata is validated separately; it cannot be inferred from answer probabilities alone.

### valQ: A1 vs A0

#### Repaired baseline errors

| qa_id | task | source | taxonomy | transition | candidate confidence | question |
| --- | --- | --- | --- | --- | --- | --- |
| train_003309 | Trend and Pattern | ERA_Tra | temporal_and_change | B->D | 0.917 | How do the positions of the outfield players change throughout the video? |
| train_003731 | Geometric Relation | ERA_Tra | spatial_and_structural | B->D | 0.888 | How is the orientation of the outrigger canoe in the foreground relative to the outrigger canoe in t |
| train_010604 | Structural Layout | Hawk_UAV | spatial_and_structural | D->B | 0.882 | How is the road network pattern organized in the video? |

#### Regressions

| qa_id | task | source | taxonomy | transition | candidate confidence | question |
| --- | --- | --- | --- | --- | --- | --- |
| train_001961 | Structural Layout | VisDrone | spatial_and_structural | D->C | 0.904 | How would you describe the spatial arrangement of vehicles in the intersection from the overhead vie |
| train_009538 | Geometric Relation | Hawk_UAV | spatial_and_structural | D->B | 0.878 | How does the distance between the residential street and the train tracks compare to the distance be |
| train_006260 | Perspective and Viewpoint | ERA_Tra | spatial_and_structural | D->B | 0.846 | What is the camera's motion state during the video? |

### valQ: A2 vs A0

#### Repaired baseline errors

| qa_id | task | source | taxonomy | transition | candidate confidence | question |
| --- | --- | --- | --- | --- | --- | --- |
| train_003309 | Trend and Pattern | ERA_Tra | temporal_and_change | B->D | 0.916 | How do the positions of the outfield players change throughout the video? |
| train_003731 | Geometric Relation | ERA_Tra | spatial_and_structural | B->D | 0.887 | How is the orientation of the outrigger canoe in the foreground relative to the outrigger canoe in t |
| train_003251 | Geometric Relation | ERA_Tra | spatial_and_structural | B->A | 0.871 | How does the distance from the American flag to the baseball field compare to the distance from the  |

#### Regressions

| qa_id | task | source | taxonomy | transition | candidate confidence | question |
| --- | --- | --- | --- | --- | --- | --- |
| train_009538 | Geometric Relation | Hawk_UAV | spatial_and_structural | D->B | 0.871 | How does the distance between the residential street and the train tracks compare to the distance be |
| train_001961 | Structural Layout | VisDrone | spatial_and_structural | D->C | 0.856 | How would you describe the spatial arrangement of vehicles in the intersection from the overhead vie |
| train_001402 | Object and Land Cover Recognition | VisDrone | factual_perception | B->C | 0.769 | How many vehicles are visible on the road in the upper-left portion of the frame at the beginning of |

### valT: A1 vs A0

#### Repaired baseline errors

| qa_id | task | source | taxonomy | transition | candidate confidence | question |
| --- | --- | --- | --- | --- | --- | --- |
| train_000678 | Object and Land Cover Recognition | VisDrone | factual_perception | D->B | 0.969 | Where is the wooden bridge located in relation to the lily pad-covered pond? |
| train_001780 | Perspective and Viewpoint | VisDrone | spatial_and_structural | D->C | 0.855 | What is the camera's motion state in the video? |
| train_009907 | Structural Layout | Hawk_UAV | spatial_and_structural | D->C | 0.849 | What is the pattern of the road network in the video? |

#### Regressions

| qa_id | task | source | taxonomy | transition | candidate confidence | question |
| --- | --- | --- | --- | --- | --- | --- |
| train_002916 | Consequence Reasoning | ERA_Tra | reasoning | D->C | 0.950 | Given the high number of spectators and the full parking lot, what is likely to happen after the gam |
| train_014049 | Temporal Grounding | Hawk_UAV | temporal_and_change | A->B | 0.930 | When does the church with the tall steeple become visible in the video? |
| train_012862 | Structural Layout | UAVDT | spatial_and_structural | C->D | 0.927 | How are the vehicles distributed across the intersection in terms of spatial arrangement? |

### valT: A2 vs A0

#### Repaired baseline errors

| qa_id | task | source | taxonomy | transition | candidate confidence | question |
| --- | --- | --- | --- | --- | --- | --- |
| train_000678 | Object and Land Cover Recognition | VisDrone | factual_perception | D->B | 0.981 | Where is the wooden bridge located in relation to the lily pad-covered pond? |
| train_009907 | Structural Layout | Hawk_UAV | spatial_and_structural | D->C | 0.889 | What is the pattern of the road network in the video? |
| train_013535 | Geometric Relation | Hawk_UAV | spatial_and_structural | D->A | 0.830 | Is the circular sign adjacent to the gas station or farther away from it? |

#### Regressions

| qa_id | task | source | taxonomy | transition | candidate confidence | question |
| --- | --- | --- | --- | --- | --- | --- |
| train_012862 | Structural Layout | UAVDT | spatial_and_structural | C->D | 0.945 | How are the vehicles distributed across the intersection in terms of spatial arrangement? |
| train_007337 | Structural Layout | Hawk_UAV | spatial_and_structural | D->A | 0.944 | How are the fairways arranged across the golf course in the video? |
| train_015760 | Geometric Relation | UAVDT | spatial_and_structural | D->C | 0.891 | How does the distance between the blue bus and the white car in the center of the intersection compa |

## Outputs

- Per-question paired transitions: `runs/phase6_visual_adaptation_pilot/paired_transitions.csv`
- Source score manifests: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` model revision and dataset revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`
