# Phase 3: SAM 2 visual-mask pilot

## Result

| Arm | Correct | Accuracy |
|---|---:|---:|
| baseline | 23/32 | 71.9% |
| detector | 23/32 | 71.9% |
| mask | 23/32 | 71.9% |

All 32 predicted answers matched across baseline, detector-only, and detector-plus-mask. The mask overlay ran on 9/32 questions. 2/9 unchanged errors received a mask overlay (train_010397, train_000073).

Correct-answer probability changed by -4.71% on overlaid questions on average (-1.32% across all 32). The no-overlay cases retained detector-only visual input. These probability movements did not change any answer.

## Task groups

| Task | n | Baseline | Detector | Mask |
|---|---:|---:|---:|---:|
| Change Detection | 1 | 1/1 | 1/1 | 1/1 |
| Consequence Reasoning | 1 | 1/1 | 1/1 | 1/1 |
| General Understanding | 1 | 1/1 | 1/1 | 1/1 |
| Geometric Relation | 8 | 5/8 | 5/8 | 5/8 |
| Hypothetical Reasoning | 2 | 2/2 | 2/2 | 2/2 |
| Object and Land Cover Recognition | 8 | 6/8 | 6/8 | 6/8 |
| Perspective and Viewpoint | 1 | 1/1 | 1/1 | 1/1 |
| Structural Layout | 1 | 1/1 | 1/1 | 1/1 |
| Temporal Grounding | 8 | 4/8 | 4/8 | 4/8 |
| Trend and Pattern | 1 | 1/1 | 1/1 | 1/1 |

## Method

- Fixed Phase 3 manifest: 16 valQ and 16 valT questions; split checksums verified on Modal.
- Same Stage A adapter, Grounding DINO context, legacy frame sampling, offsets 0 and 0.5, and two option rotations.
- SAM 2 Hiera Tiny (`facebook/sam2-hiera-tiny`, revision `7c218beaf0bb87874785f32b582f640134fc1c09`) produced masks from the existing Grounding DINO boxes.
- For question-matched masks, each sampled frame was paired side by side with its mask overlay. Questions without matched masks kept the detector-only visual input. No additional SAM text header was added.
- SAM cache, score JSONL, and manifests are saved next to this report.

## Interpretation and limits

This pilot does not show an accuracy improvement from the visual masks. Only nine questions received matched masks, and only two of the nine unchanged errors received an overlay. The upstream detector/entity-matching path therefore limits which questions can benefit. Treat the small confidence shifts as exploratory, not a promotion result.

Transformers emitted a SAM checkpoint/model-type compatibility warning (`sam2_video` checkpoint instantiated as `sam2`). The smoke pass and full cache/scoring runs completed, but the warning should be resolved or explicitly validated before scaling beyond this pilot.

Not manually audited: segmentation boundaries against ground-truth masks. The score run validates execution and answer outcomes, not mask quality.
