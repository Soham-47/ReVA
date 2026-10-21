# Phase 0 visual failure review

**Review set:** 60 `valT` QA items: 20 Temporal Grounding baseline errors + 10 correct controls, and 20 Geometric Relation baseline errors + 10 correct controls. Selection was uniform pseudo-random within each pool without replacement (seeds 20261017/20261018). See `selection.json` and per-item `review_annotations.json`.

**Evidence reviewed:** 36 video contact sheets, each with 24 timestamped samples; metadata includes decoded fps, frame count, duration, and sample times. Labels and predictions were not changed. Per-item notes are diagnostic hypotheses, not definitive causal attribution.

## Findings

| Task | Baseline errors reviewed | Observed primary categories |
|---|---:|---|
| Temporal Grounding | 20 | T3 visibility/event boundary: 6; T1 event already visible at opening / onset: 4; T4 small-object identity: 3; T2 precise onset timing: 3; T7 ambiguous wording or possible data/label issue: 2; T5 camera motion: 1; T6 occlusion/reappearance: 1 |
| Geometric Relation | 20 | G7 uncalibrated depth/distance: 7; G1 object/region localization: 5; G3 image/world/reference-frame ambiguity: 3; G5 camera motion or perspective: 3; G2 image-plane direction: 1; G6 semantic/relation wording: 1 |

The leading temporal pattern is choosing the wrong visibility boundary, especially for duration/exit questions and small targets in wide aerial footage. Several first-appearance errors have the target visible in the opening sheet frame, indicating missed evidence or inconsistent onset interpretation rather than a need to add segmentation. Small red cars, white vehicles, and a freight train also illustrate how target identity can confound event timing.

The leading geometry pattern is being asked to infer depth or relative distance from uncalibrated, often oblique aerial footage. Such questions may lack a clear reference point for extended regions. Other errors combine small-object localization with compound left/right/front/behind wording, or use a direction that shifts with the camera viewpoint. Masks can sharpen a region boundary but cannot provide metric depth or define an ambiguous reference frame.

## Data/label review flag

`train_001774` (`VisDrone/uav0000279_00001_v_04.mp4`) decodes to 7 frames at 30 fps (0.233 s), while every answer option describes 1.5–4.0 seconds of visibility. This strongly suggests a clip/metadata or annotation mismatch. The item remains in evaluation; no answer was changed or excluded.

Other possible ambiguity flags include subjective phrases such as “begin to reveal the full extent,” apparent visibility under camera motion, and timing options close to the temporal sampling resolution. These are review flags, not confirmed label errors.

## Controls and limits

The 20 correct controls show the same videos/questions are not uniformly impossible: some opening-time, duration, orientation, and clear image-relative relations are answerable from the sampled views. A control match does not establish that all evidence is unambiguous.

This is a sampled qualitative diagnosis of 40 of the `valT` failures (not every error in either split), plus 20 controls. It uses contact-sheet thumbnails rather than frame-by-frame, full-resolution annotation. It does not quantify how many errors a proposed method will fix, and it does not establish causality.

## Phase 2 implications

1. Test question-aware local sampling and timestamp brackets on all 202 Temporal Grounding questions, keeping the final per-offset frame budget at 16 and the Stage A model/adapter fixed.
2. Keep event identity and boundary confidence explicit; low-confidence localization falls back to legacy sampling.
3. Preserve suspicious-label items in paired analysis and flag them separately; do not silently relabel or drop them.
4. Keep Phase 3 geometry conditional. Its first box-based comparison should report image-plane relations separately from depth/distance claims; masks cannot resolve missing calibration.

## Baseline reference metrics

These are frozen metrics recorded in the project plan (not recomputed by this sampled review):

| Split | Temporal Grounding | Geometric Relation |
|---|---:|---:|
| `valQ` | 51/69 (73.91%) | 87/110 (79.09%) |
| `valT` | 99/133 (74.44%) | 134/171 (78.36%) |
| Official `val` | 243/320 (75.94%) | 180/200 (90.00%) |
