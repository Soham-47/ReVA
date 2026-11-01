# Stage A prediction error review

Reviewed 13 October 2026. Run: `stageA_q3vl8b_r16`; training job 392966 and scoring job 392967 completed with exit code 0.

## Main finding

The failures have several causes. There are genuine errors in identifying objects, applying spatial relations, counting fully visible objects, and using motion evidence. There are also strong candidates for inconsistencies between temporal labels and the supplied clips. The 16-frame sampling cap is a plausible limit on timing precision, but this review does not demonstrate that increasing the frame count improves accuracy.

No production code, dataset labels, model weights, split, training configuration, or scoring configuration was changed during this review. Stage B remains gated.

## Evidence and scope

- Audited the saved predictions for all 595 valQ, 1,059 valT, and 2,000 official validation questions.
- Selected 12 valQ errors: four temporal grounding, three geometric relation, two change detection, two trend/pattern, and one structural layout. Selection prioritized high confidence within each task; this is a diagnostic sample, not a representative label-quality survey.
- Inspected offset-0 contact sheets for all 12 questions and offset-0.5 sheets for four important cases. Both sets are saved for every selected question.
- Checked original endpoint frames for five cases and decoded every frame in the 11 distinct clips represented by these questions.
- Contact sheets show the source frames selected by the current sampler, before the processor's resizing. They are not reconstructions of the exact model pixel tensors.
- Audited score aggregation, prediction argmax, four-view completeness, finite probabilities, split membership, and video timestamps. No new GPU inference or retraining was performed.

## Aggregate results

| Split | Correct / questions | Accuracy | Errors | Errors where all four views agree | Wrong predictions with probability >= 0.90 |
|---|---:|---:|---:|---:|---:|
| valQ | 498 / 595 | 83.70% | 97 | 72 | 28 |
| valT | 885 / 1,059 | 83.57% | 174 | 127 | 39 |
| official val | 1,757 / 2,000 | 87.85% | 243 | 168 | 48 |

The probabilities are softmax values over four answer letters, not calibrated guarantees of correctness.

| Split | Single-view correct | TTA rescues | TTA harms | TTA net gain |
|---|---:|---:|---:|---:|
| valQ | 494 | 8 | 4 | +4 |
| valT | 882 | 13 | 10 | +3 |
| official val | 1,742 | 34 | 19 | +15 |

A rescue means single-view wrong and TTA correct; a harm means single-view correct and TTA wrong. Changes from one wrong answer to another are excluded. The four views combine two temporal offsets and two cyclic option orders. Since 72/97 valQ errors survive every view, changing option order or averaging more similar views is unlikely to solve most errors.

### Error concentration

valQ error counts: geometric relation 23, temporal grounding 18, structural layout 14, trend/pattern 12, perspective/viewpoint 11, object recognition 8, change detection 8, hypothetical reasoning 2, causation 1. Counts depend on the number of questions per task; they are not task error rates.

Temporal grounding is the weakest task by accuracy on every split: 73.91% valQ, 74.44% valT, 75.94% official val. It accounts for 77/243 official validation errors.

The valQ target is 88%: at least 524/595 correct. The current checkpoint is 26 correct answers short. The visual sample below does not explain or classify all 97 errors.

## What the clip review shows

These are visual interpretations, not formal relabeling decisions. Gold answers remain unchanged.

| Question ID | Gold / model | Visual observation | Likely cause and confidence |
|---|---|---|---|
| train_005036 | D: unloading begins around the middle / A: at the beginning | Material is already falling from the tipped container into the red trailer in the first frame, and continues throughout both sampled views. Original clip ends at 5.75s. | Strong question/label concern. The supplied clip supports an event already underway at the beginning; its actual onset before the clip cannot be determined. Also the question's combine-harvester description does not clearly match the visible tipped container. |
| train_009393 | C: soccer field first visible at 0.0s / D: 4.0s | The first frame predominantly shows baseball diamonds. The soccer pitch is identifiable later, including in the final frame. | Strong label or object-reference concern. The predicted later appearance is visually plausible. Exact first visibility needs a defined rule for partial objects and dense frame review. |
| train_007912 | B: parking lot visible 2.5–5.5s / A: 3.0–6.0s | The lot enters along the upper/right edge later in the clip and remains visible near the end. | Boundary/partial-visibility ambiguity plus timing precision. Neither option's exact boundary was independently certified. |
| train_008994 | A: beach disappears at 5.0s / D: disappears at 5.5s | The beach remains plainly visible at 5.81s and in the original final frame at 5.973s. | Strong temporal label concern. Even the model's 5.5s answer is not an exact disappearance boundary; the listed options appear inconsistent with the supplied clip. |
| train_009453 | C: soccer field below baseball fields / D: to the left | Later frames show soccer lower/left and baseball upper/right; the camera changes orientation and framing. | Spatial-reference ambiguity and incomplete relation grounding. Both axes have visual support; which relation the benchmark intends is not stated explicitly. |
| train_005711 | A: car approximately equally distant from two tree groups / C: closer to yellow trees | The car is small and trees form several overlapping clusters. | Target identification/distance estimation is plausible, but this review cannot establish a unique reference tree or a quantitative gold distance. Unresolved. |
| train_013105 | B: bridge parallel to road / A: perpendicular | The bridge deck follows the road. The bridge crosses the river, a different reference object. | Strong evidence for a genuine model relation error. A bridge-crosses-river association may explain the wrong answer, but the model's internal reasoning is unobserved. |
| train_001154 | A: buildings shrink / D: buildings grow | A prominent foreground facade occupies progressively less vertical image area as the camera advances past it. | Genuine camera-motion/scale reasoning weakness is plausible. The prediction resembles the usual forward-motion assumption rather than this facade's observed trajectory; other buildings create some ambiguity. |
| train_006745 | D: four fully visible people / A: five | Four full people and part of a fifth person at the right border are visible in the first frame. | Strong evidence for a genuine qualifier/boundary counting error: counting the partial person despite 'fully visible'. |
| train_011338 | C: lots consistently full / B: occupancy increases during peak hours | The approximately eight-second clip does not establish a repeated occupancy cycle or business peak hours. There are visibly unoccupied spaces. | Unsupported temporal/business inference by the model, plus an overstrong gold statement. The question/options demand claims the short clip cannot establish. |
| train_000590 | D: random movement / A: offense toward basket and retreat on defense | Small players move on the court; the clip is short and camera motion is substantial. | A generic basketball explanation is a plausible language prior, but sampled stills do not establish a definitive motion classification. Unresolved without trajectories or continuous viewing. |
| train_009455 | A: one linear road parallel to park / B: interconnected converging/diverging paths | The video contains a main road, an intersection, and curved park paths. | Reference/extent ambiguity and possible road-versus-path confusion. Gold and prediction summarize different parts of the visible scene. |

### Frame evidence for the strongest concerns

Unloading at the start:

*[Image omitted from public mirror: `train_005036_offset0.0.jpg`]*

Soccer question: original first and last frames:

*[Image omitted from public mirror: `train_009393_first.jpg`]*
*[Image omitted from public mirror: `train_009393_last.jpg`]*

Beach is still visible in the original final frame, despite a 5.0s disappearance label:

*[Image omitted from public mirror: `train_008994_last.jpg`]*

Four fully visible people plus a partial fifth person at the right border:

*[Image omitted from public mirror: `train_006745_first.jpg`]*

## Sampling and pipeline audit

### Confirmed sampling limit

`code/common.py` sets both `MIN_FRAMES` and `MAX_FRAMES` to 16. The nominal `FPS_TARGET=4.0` therefore does not deliver four frames per second for these 5.7–8.0 second clips. The sampler yields roughly 2–2.8 selected frames per second, with about 0.36–0.50s gaps. Some temporal answer choices differ by 0.5s.

The offset-0 view includes the first frame but stops one grid interval before the end. The offset-0.5 view shifts the grid and gets closer to the end, but misses the first frame. For reviewed clips its last sample is about 0.17–0.23s before the final decoded frame. No view contains both exact endpoints.

This is a plausible precision bottleneck, not a demonstrated explanation of all temporal misses. Several visible contradictions remain even with the original final frame, so frame density alone cannot explain those cases.

### Checks that passed

- All 11 distinct reviewed clips have the same decoded frame count as their manifest entry.
- Ten clips have negligible timestamp drift between decoded presentation timestamps and frame-index/FPS estimates. One variable-rate clip has maximum drift 0.079s; this is a spatial question and does not explain the large temporal-label disagreements above.
- Every saved score in the three splits has four views, finite normalized probabilities, the correct argmax letter, and probabilities equal to the mean of its views.
- fit_A shares no QA IDs with valQ or valT, no exact video/question groups with valQ, and no video paths with valT. valQ intentionally holds out questions on otherwise potentially seen videos.
- The scorer maps shown-option probabilities back to original letters before averaging. Its saved letter histograms do not show a gross collapse to one letter.

These checks do not prove all model preprocessing is correct. Exact processor pixel tensors, all semantic near-duplicate questions, and the full dataset's timing annotations were not independently audited.

## Recommended follow-up

1. Preserve the present checkpoint and all gold-based metrics as the baseline. Keep a separate annotation-concern ledger; do not remove questionable questions or alter the validation labels to improve the gate result.
2. Establish an object-visibility rule and verify the flagged temporal examples at dense timestamps. If source annotation provenance is available, check whether labels refer to a different interval or the longer parent video. No parent-video misalignment was proven here.
3. Run a bounded inference ablation with the same checkpoint: current 16 frames versus a denser, endpoint-inclusive sequence. Keep question IDs, prompt, option-order handling, and score aggregation fixed. Compare overall accuracy and temporal-grounding accuracy on the fixed splits. This is a proposed experiment; no configuration change or GPU run was made.
4. If extra temporal evidence helps, consider a lightweight temporal verifier that tracks first/last object visibility with timestamps. For counting and geometry, inspect target object crops, border inclusion, and explicit reference objects. Measure each component separately before building a larger framework.
5. A zero-shot comparison or checkpoint comparison is needed before claiming that fine-tuning helped or that the model overfit. Falling training loss alone cannot establish either claim.

The current evidence supports targeted temporal/spatial verification and annotation review. It does not justify automatically retraining Stage B or promising that a particular architecture will close the 26-answer valQ gap.

## Reproducing the audit

Analysis helpers and machine-readable evidence are in this directory:

- `collect_review.py`: selects the 12 cases, computes error/view statistics, and creates sampled frame sheets.
- `validate_review.py`: checks score aggregation and split integrity, decodes the full selected clips, and saves original endpoints.
- `audit_statistics.json`: final corrected counts and decode checks.
- `selected_examples.json`: original questions, options, predictions, clip metadata, and frame indices.
- `valQ_errors.json`: all 97 gold-based valQ errors.

On HPC, use `WORK=/scratch/ambpdc/soham/ReVA` with the existing `envs/reva/bin/python` interpreter. The review is mirrored under `results/error_review` in the HPC project. The completed experiment's official evaluation files remain under `runs/stageA_q3vl8b_r16`.
