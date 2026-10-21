# Phase 2 verified temporal-window pilot

This is a bounded diagnostic arm for 24 first-appearance Temporal Grounding
questions: 12 from `valQ` and 12 from `valT`, with six saved-baseline errors
and six correct controls per split. The checked-in manifest selects four
records for the first tranche (one error and one control per split).

The arms are:

- `T0`: the frozen legacy score already saved under
  `runs/stageA_q3vl8b_r16/legacy/`.
- `T1`: the frozen question-only guessed-window score already saved under
  `runs/phase2_temporal_coarsefine/full/`.
- `T1R`: the same Stage A model, offsets, option rotations, 16-frame budget,
  prompt, and aggregation, with the window source changed to an independently
  reviewed numeric event window.

## Review contract

Review each selected clip with the question and contact sheet. Fill only the
manifest fields for the reviewed record:

1. Set `review_status` and `boundary_status` to `reviewed` only when the target
   identity and first-visibility boundary are supported by the clip.
2. Set `identity_status` to `confirmed` and fill `window_start_s` and
   `window_end_s` with the supported visibility bracket in source presentation
   seconds. Leave `onset_censoring` as `none` only when an onset inside the
   clip is supported.
3. Record concise `boundary_evidence`, `identity_evidence`, `reviewer`, and
   `reviewed_at` values for audit.

Leave both boundaries `null` while a record is `unreviewed`. `unknown` and
`ambiguous` records also keep null boundaries and use the T0 legacy selection.
If the target is already visible in the opening frame and no pre-clip onset can
be established, set `onset_censoring` to `opening_censored`, mark the boundary
unknown, and keep the T0 fallback. Do not invent a pre-clip onset.
The scorer ignores all review prose; it sends no onset text or review metadata
to the answer model. For a qualified review, T1R recenters the saved T1
candidate window on the reviewed bracket midpoint while preserving that T1
window's duration (clamped to the clip). Missing, invalid, or narrower T1
candidate windows produce an explicit T0 fallback reason. A saved window
narrower than the reviewed bracket cannot cover the reviewed evidence; the
scorer preflights every qualified record before model loading. A tranche run
requires all 4 records to be qualified and recenterable; a full run requires
at least 20/24 such records. Qualified records that fail preflight use the
explicit T0 fallback when the full-run threshold still holds; otherwise the
run is refused before model loading. Short clips also fall back to T0.

## Read-only validation and scoring

From the repository/runtime tree, validate the four-record tranche without
loading a model or decoding a video:

```bash
python3 code/score_temporal_verified.py \
  --manifest tasks/phase2_temporal_verified_window_manifest.json \
  --subset tranche
```

After the tranche has independently reviewed records, an explicit future GPU
run is:

```bash
python3 code/score_temporal_verified.py \
  --manifest tasks/phase2_temporal_verified_window_manifest.json \
  --subset tranche --run \
  --video-manifest runs/phase2_temporal_coarsefine/full/source_manifest.json \
  --output-dir runs/phase2_temporal_verified_window/tranche
```

Run the same commands with `--subset full` after the tranche review is
accepted. The scorer is validation-only unless `--run` is supplied, so merely
preparing or checking the manifest cannot launch inference.

Once `scores_{split}_T1R.jsonl` exists, generate the paired report:

```bash
python3 code/evaluate_temporal_verified.py \
  --manifest tasks/phase2_temporal_verified_window_manifest.json \
  --subset tranche \
  --manual-dir runs/phase2_temporal_verified_window/tranche \
  --output-dir runs/phase2_temporal_verified_window/tranche/report
```

The report keeps fixes, regressions, wrong-to-wrong changes, unchanged
predictions, controls, split-level accuracy, T1/T1R fallback offsets, and
boundary/identity evidence, plus whether the eight T1R local-search timestamps
cover the reviewed bracket. Each offset reports local samples before, inside,
and after the bracket. It does not promote a variant or treat the blank
template as completed annotation.
