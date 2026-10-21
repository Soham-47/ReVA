# SAM 3 pilot qualitative review

## Scope and scoring setup

This review covers the selected 15-video pilot: 8 `valQ` and 7 `valT` questions, with 8 Geometric Relation and 7 Temporal Grounding items. The run made 28 direct text-prompt calls and produced 5,224 masks in the overlay cache.

The comparison used the same Stage A adapter, Grounding DINO detector context, legacy frame sampling, offsets `0` and `0.5`, and two rotations for baseline, detector, SAM 2, and SAM 3. SAM 3 direct masks were added as visual overlays. This is a selected, biased pilot, so it is not evidence of general improvement.

## Results

| Arm | Correct | Accuracy |
|---|---:|---:|
| Baseline | 9/15 | 60.0% |
| Detector | 9/15 | 60.0% |
| SAM 2 | 9/15 | 60.0% |
| SAM 3 | 10/15 | 66.7% |

By task, Geometric Relation increased from 5/8 to 6/8 for SAM 3. Temporal Grounding stayed at 4/7.

The baseline-to-SAM 3 changes were:

- `train_012156`: A → C, wrong → right; correct answer C.
- `train_012502`: B → C, wrong → wrong; correct answer D.

No correct answer was lost. The pilot therefore produced one net answer correction. The unchanged geometry miss `train_013068` predicted the barn was closer to the road (A), while the correct answer was closer to the river (C).

All three temporal misses remained:

- `train_009393`: predicted 4.0 seconds; correct answer 0.0 seconds.
- `train_009869`: predicted 3.0 seconds; correct answer 3.5 seconds.
- `train_010397`: predicted 0–7 seconds; correct answer 0–3 seconds.

## Qualitative observations

The contact sheets below cover every baseline-wrong question in this subset.

### `train_012156`

Open contact sheet (local-only artifact: `qualitative_review/train_012156.png`)

The court and road masks reasonably align with the scene. The parking-lot prompt mostly selects parked cars, so the overlay gives weak evidence for the full parking-lot region. The answer nevertheless changed to the correct option.

### `train_012502`

Open contact sheet (local-only artifact: `qualitative_review/train_012502.png`)

The barn and house masks are sparse and unstable across frames. They did not fix the relation or elevation judgment, consistent with the wrong answer remaining wrong.

### `train_013068`

Open contact sheet (local-only artifact: `qualitative_review/train_013068.png`)

The masks broadly locate the barn, road, and river, but the relative-distance judgment remains wrong.

### `train_009393`

Open contact sheet (local-only artifact: `qualitative_review/train_009393.png`)

The soccer-field prompt marks broad grass from frame 0 rather than precise soccer-field evidence. This does not resolve the visibility timing question.

### `train_009869`

Open contact sheet (local-only artifact: `qualitative_review/train_009869.png`)

The baseball-field mask spills onto neighboring grass and fields. The predicted visibility end remains off by 0.5 seconds.

### `train_010397`

Open contact sheet (local-only artifact: `qualitative_review/train_010397.png`)

The black-car prompt marks many parked cars rather than one unique vehicle. The masks therefore do not establish the target car's appearance and disappearance interval.

These observations suggest that broad or unstable direct masks limited the temporal benefit and some spatial judgments. That is an interpretation of the contact sheets, not a measurement beyond the paired scores.

## API and execution history

The final full-run manifest reports 14 new outputs, 14 reused outputs, 15 prediction attempts, and an approximate cost of `$0.270` for that invocation at the displayed `$0.018` per run rate. Earlier smoke execution and full-run OOM retries bring the observed total to about 31 requests, or approximately `$0.56` if every request was billed at that displayed rate; actual billing was not audited.

Three full-clip OOM attempts were observed across two videos. Two fallback outputs succeeded using the exact 32-frame union of the Stage A legacy offset `0` and `0.5` samples encoded at 4 fps. No credentials were stored. The prior SAM 2 artifacts remain unchanged.
