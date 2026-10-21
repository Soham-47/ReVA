# Adaptive-frame audit: temporal grounding and geometric relation

## Scope and comparison

Question-level paired comparison of the adaptive-frame evaluation against the canonical saved baseline (`scores_<split>_eval.csv`) on identical `qa_id` sets. The same Stage A adapter and scorer were used. `val` is the primary held-out comparison; `valQ` and `valT` are reported separately. The case CSV includes every target-task question where either recipe was wrong, with question, options, gold answer, both predictions, confidence, source, and transition.

**Important:** this is an answer-level audit. It identifies exact fixes, regressions, and persistent errors; it does not prove why a model missed an item without reviewing the video frames/evidence.

## Overall accuracy on val

| Recipe | Correct | Accuracy | Difference vs canonical baseline |
|---|---:|---:|---:|
| Canonical baseline | 1,757 / 2,000 | 87.85% | — |
| Adaptive frames | 1,745 / 2,000 | 87.25% | −0.60 percentage points |
| Endpoint sampler reference | 1,755 / 2,000 | 87.75% | −0.50 pp vs endpoints |

The adaptive recipe is not an overall improvement. Its target-task net change is −5 correct answers out of 520 val questions across the two tasks.

## Temporal grounding and geometry results

| Split | Task | Baseline | Adaptive | Change | Fixed | Regressed | Still wrong in both |
|---|---|---:|---:|---:|---:|---:|---:|
| val | Temporal Grounding | 243/320 (75.94%) | 237/320 (74.06%) | -1.88 pp | 12 | 18 | 65 |
| val | Geometric Relation | 180/200 (90.00%) | 181/200 (90.50%) | +0.50 pp | 1 | 0 | 19 |
| valQ | Temporal Grounding | 51/69 (73.91%) | 48/69 (69.57%) | -4.35 pp | 2 | 5 | 16 |
| valQ | Geometric Relation | 87/110 (79.09%) | 86/110 (78.18%) | -0.91 pp | 0 | 1 | 23 |
| valT | Temporal Grounding | 99/133 (74.44%) | 95/133 (71.43%) | -3.01 pp | 1 | 5 | 33 |
| valT | Geometric Relation | 134/171 (78.36%) | 137/171 (80.12%) | +1.75 pp | 3 | 0 | 34 |

### Primary val findings

- **Temporal Grounding:** baseline 243/320 (75.94%); adaptive 237/320 (74.06%), **−1.88 pp**. It fixed 12 baseline mistakes but introduced 18 new mistakes, leaving 65 wrong under both. Net: six fewer correct answers.
- **Geometric Relation:** baseline 180/200 (90.00%); adaptive 181/200 (90.50%), **+0.50 pp**. It fixed one baseline mistake, introduced no regressions, and left 19 wrong under both.
- The two tasks combined: baseline 423/520 (81.35%); adaptive 418/520 (80.38%), **−0.96 pp**. Thirteen fixes were outweighed by 18 regressions.
- Paired exact McNemar checks on `val` are not significant for either task (temporal p=0.362; geometric p=1.000); treat the +1 geometric answer as a tiny, unconfirmed gain.
- Against the separate 16-frame endpoint run, adaptive geometry is tied at 90.50%; adaptive temporal grounding is lower by 1.88 pp (74.06% vs 75.94%).

## Split stability

- `valQ`: both target tasks declined: temporal −4.35 pp (2 fixed / 5 regressed); geometry −0.91 pp (0 / 1).
- `valT`: geometry improved +1.75 pp (3 / 0); temporal declined −3.01 pp (1 / 5).
- The geometric change varies by split (+1.75 pp on valT, −0.91 pp on valQ, +0.50 pp on val). Temporal grounding declined in all three (−3.01 to −4.35 pp). This does not support a stable temporal benefit.

## What changed in the predictions on val

### Temporal Grounding repairs (12)

Adaptive frames corrected 12 cases. Corrected question IDs and answer transitions:
- `val_000002` gold A: C → A — How long does the large gray-roofed barn with two silos remain visible in the video?
- `val_000180` gold C: A → C — How long does the group of pedestrians remain visible at the crosswalk in the center of the intersection?
- `val_000196` gold B: D → B — At what point in the video does the large circular driveway first appear in the frame?
- `val_000329` gold A: C → A — How long does the red truck on the overpass remain visible in the video?
- `val_000395` gold C: A → C — When does the train disappear from the frame?
- `val_000527` gold B: C → B — At what point in the video does the first visible swimming pool appear?
- `val_000794` gold A: C → A — At what point in the video does the small white truck first appear in the frame?
- `val_000842` gold C: D → C — At what point in the video does the swimming pool in the upper right corner first appear?
- `val_000973` gold B: C → B — How long does the white car visible in the center of the road at 0:00 remain in the frame before it moves out of view or the video ends?
- `val_001036` gold C: D → C — At what time does the blue car appear in the video and when does it disappear?
- `val_001089` gold C: B → C — At what point in the video does the first house appear in the frame?
- `val_001223` gold B: A → B — At what point in the video does the white van first appear in the frame?

### Temporal Grounding regressions (18)

Adaptive frames changed 18 previously correct answers to wrong answers:
- `val_000080` gold D: D → B — At what point does the highway first appear in the video?
- `val_000084` gold A: A → D — How long does the white sand trap remain visible in the video?
- `val_000173` gold A: A → D — When does the tennis court and playground area disappear from the frame?
- `val_000223` gold B: B → D — At what point in the video does the cluster of buildings, including the white-roofed structure, first appear in the frame?
- `val_000255` gold A: A → B — When does the baseball field first become visible in the video?
- `val_000479` gold D: D → B — How long do the circular water treatment tanks appear in the video?
- `val_000525` gold C: C → A — How long does the blue car remain visible in the video?
- `val_000604` gold A: A → C — At what time does the blue car first appear in the video?
- `val_000611` gold B: B → D — How long does the aerial view of the cornfield remain visible in the video?
- `val_000655` gold C: C → D — At what point in the video does the red car first appear on the road?
- `val_000656` gold A: A → D — How long does the red car remain visible in the frame?
- `val_000857` gold D: D → B — At what point in the video does the red truck with the green container completely disappear from view?
- `val_000869` gold A: A → C — How long does the yellow car remain visible in the video?
- `val_000895` gold D: D → C — At what point in the video does the residential neighborhood first appear?
- `val_000959` gold C: C → B — How long does the building structure visible in the upper left portion of the frame remain in view during the video?
- `val_001060` gold A: A → B — At what time does the green bus first appear in the video?
- `val_001191` gold D: D → C — When do the white tents first become visible in the video?
- `val_001260` gold C: C → B — When does the small lake become visible in the video?

### Geometric Relation change

- One repair: `val_000344` (Hawk_UAV), gold B; baseline predicted C, adaptive predicted B. The question compares the distance from a parking lot to a community-center building versus to a basketball court.
- No geometric regressions on `val`. The other 19 geometric errors persisted.

## Remaining error inventory on val

- **Temporal:** 83 adaptive errors: 65 were already wrong and remained wrong; 18 were regressions. 12 baseline errors were repaired. The questions center on event onset/first appearance, disappearance, and duration/visibility; these are directly listed in the CSV.
- **Geometry:** 19 adaptive errors, all also wrong in the baseline; one baseline geometry error was repaired. No geometry regression appeared in this split.
- By source, temporal gains/losses were concentrated in Hawk_UAV (7 fixed, 13 regressed; net −6/206, −2.91 pp). VisDrone had 5 fixed and 5 regressed (net zero across 109); ERA_Tra had no changed cases (5 total). Geometry improved by one Hawk_UAV case (1/91, +1.10 pp); VisDrone had no changes (98/109 correct in both).

## Audit files

- `adaptive_target_task_error_cases.csv` — every temporal/geometric case that was wrong in at least one recipe across `val`, `valQ`, and `valT`. It includes persistent errors, fixes, and regressions with all answer choices and both predictions. The three split rows can overlap; do not sum them as independent samples.
- The remaining limitation is root-cause attribution: each listed mistake needs video/frame review to label it as missed event timing, object visibility, viewpoint, relation direction, distance comparison, or an annotation ambiguity.
