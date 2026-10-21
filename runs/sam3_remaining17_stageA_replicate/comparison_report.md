# SAM 3 direct-text overlay comparison (supplemental-17)

| Arm | Correct | Accuracy |
|---|---:|---:|
| baseline | 14/17 | 82.4% |
| detector | 14/17 | 82.4% |
| sam2 | 14/17 | 82.4% |
| sam3 | 14/17 | 82.4% |

| Task | n | Baseline | Detector | SAM 2 | SAM 3 |
|---|---:|---:|---:|---:|---:|
| Change Detection | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| Consequence Reasoning | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| General Understanding | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| Hypothetical Reasoning | 2 | 2/2 | 2/2 | 2/2 | 2/2 |
| Object and Land Cover Recognition | 8 | 6/8 | 6/8 | 6/8 | 5/8 |
| Perspective and Viewpoint | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| Structural Layout | 1 | 1/1 | 1/1 | 1/1 | 1/1 |
| Temporal Grounding | 1 | 0/1 | 0/1 | 0/1 | 1/1 |
| Trend and Pattern | 1 | 1/1 | 1/1 | 1/1 | 1/1 |

Supplemental SAM 3 API outputs: 44; prediction attempts: 45 (approx. $0.810). Comparison questions: 17.

| Pair | Prediction changes |
|---|---:|
| baseline_to_sam3 | 4 (gained 2, lost 2) |
| detector_to_sam3 | 4 (gained 2, lost 2) |
| sam2_to_sam3 | 4 (gained 2, lost 2) |

SAM 3 uses all direct prompt masks for each question and the same detector context, Stage A adapter, legacy sampling, offsets, and rotations. The SAM 3 arm never loads the SAM 2 cache or filters direct masks through DINO entities.
