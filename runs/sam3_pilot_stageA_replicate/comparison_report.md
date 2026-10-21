# SAM 3 direct-text overlay comparison

| Arm | Correct | Accuracy |
|---|---:|---:|
| baseline | 9/15 | 60.0% |
| detector | 9/15 | 60.0% |
| sam2 | 9/15 | 60.0% |
| sam3 | 10/15 | 66.7% |

| Task | n | Baseline | Detector | SAM 2 | SAM 3 |
|---|---:|---:|---:|---:|---:|
| Geometric Relation | 8 | 5/8 | 5/8 | 5/8 | 6/8 |
| Temporal Grounding | 7 | 4/7 | 4/7 | 4/7 | 4/7 |

Completed prompt outputs: 14; prediction attempts: 15 (approx. $0.270).

| Pair | Prediction changes |
|---|---:|
| baseline_to_sam3 | 2 (gained 1, lost 0) |
| detector_to_sam3 | 2 (gained 1, lost 0) |
| sam2_to_sam3 | 2 (gained 1, lost 0) |

SAM 3 uses all direct prompt masks for each question and the same detector context, Stage A adapter, legacy sampling, offsets, and rotations. The SAM 3 arm never loads the SAM 2 cache or filters direct masks through DINO entities.
