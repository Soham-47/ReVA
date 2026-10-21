# SAM 3 Replicate pilot preflight

Status: completed. The authorized 15-video Replicate SAM 3 pilot produced all 28 prompt outputs and 5,224 cached masks. Both splits were scored on Modal A100 with the fixed Stage A adapter and detector control. The qualitative audit is in [`qualitative_review.md`](qualitative_review.md). No credentials were stored.

## Pinned endpoint and inputs

- Model: `lucataco/sam3-video`
- Pinned Replicate version: `408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269`
- Runner: [`jobs/replicate_sam3_pilot.py`](../../jobs/replicate_sam3_pilot.py)
- Each direct text call submits one short noun phrase and requests `mask_only=true` and `return_zip=true`.
- The first selected video is a smoke pass: every prompt for that video is called, each ZIP is validated, and the direct overlay cache is loaded before the remaining videos are submitted.
- The cache contains all returned masks for each direct prompt. It is intentionally independent of Grounding DINO entity filtering and is consumed by the distinct `sam3-direct-overlay-v1` visual scorer.
- A valid no-detection response may contain only `overlay.mp4`; this is recorded as a completed zero-mask call, reused on resume, and leaves the corresponding VLM frames unchanged.

Official Replicate references:

- [SAM 3 video model](https://replicate.com/lucataco/sam3-video)
- [Pinned version](https://replicate.com/lucataco/sam3-video/versions/408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269)
- [Pinned API schema](https://replicate.com/lucataco/sam3-video/versions/408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269/api)
- [Creating predictions](https://replicate.com/docs/topics/predictions/create-a-prediction)
- [Input files](https://replicate.com/docs/topics/predictions/input-files)

The public model page lists an approximate **$0.018 per run**. The fixed prompts below require **28 completed prompt outputs** (21 geometric operands plus 7 temporal objects), for an estimate of **about $0.504** before retries or pricing changes. The runner records completed outputs, prediction attempts (including OOM retries), reused validated calls, and the estimate in `manifest.json`.

If a full-clip prediction fails with an explicit CUDA out-of-memory error, and only then, the runner retries that prompt on a temporary MP4 containing the sorted union of the two Stage A legacy sample grids (offsets `0` and `0.5`). The temporary clip is encoded at the Stage A target rate of 4 fps, so its 32 input frames preserve the original time span. A durable per-call mapping sidecar maps returned fallback frame ordinals back to source frame indices before cache creation. Other failures do not trigger this retry, and successful full-clip outputs remain unchanged.

## Fixed subset and direct prompts

This is the existing approved Phase 3 subset narrowed to 15 unique videos: all eight `Geometric Relation` questions and seven `Temporal Grounding` questions. Geometric operands are split into one noun phrase per call so both/all named objects are available to the relation question. Temporal prompts target the referenced object or field. The split IDs and source revision remain unchanged.

| split | question | task | manifest frames | direct SAM 3 prompt calls | source question |
|---|---|---|---:|---|---|
| valQ | `train_014717` | Geometric Relation | 240 | `parking lot`, `baseball field`, `main grassy field` | Is the parking lot closer to the baseball field or to the main grassy field? |
| valQ | `train_002581` | Geometric Relation | 180 | `volleyball court`, `basketball court` | In the video, how is the volleyball court positioned relative to the basketball court? |
| valQ | `train_013068` | Geometric Relation | 240 | `red barn`, `road`, `river` | How does the distance between the red barn and the road compare to the distance between the red barn and the river? |
| valQ | `train_003645` | Geometric Relation | 140 | `rowing boats`, `yellow boat`, `motorboat` | In the video, how does the distance between the two rowing boats compare to the distance between the yellow boat and the motorboat in the background? |
| valQ | `train_003181` | Temporal Grounding | 140 | `sports court` | How long does the game on the sports court appear to be in progress throughout the video? |
| valQ | `train_009393` | Temporal Grounding | 180 | `soccer field` | At what point in the video does the soccer field first become visible in the frame? |
| valQ | `train_009869` | Temporal Grounding | 180 | `baseball field` | From which timestamp to which timestamp is the baseball field continuously visible in the video? |
| valQ | `train_015573` | Temporal Grounding | 240 | `car` | How long is the car visible in the parking lot before it moves out of the frame? |
| valT | `train_011808` | Geometric Relation | 240 | `large parking lot`, `road`, `industrial buildings` | What is the relative position of the large parking lot on the left side of the road compared to the industrial-looking buildings on the right side? |
| valT | `train_012156` | Geometric Relation | 240 | `basketball court`, `parking lot`, `road` | How does the distance between the basketball court and the parking lot compare to the distance between the basketball court and the road? |
| valT | `train_012502` | Geometric Relation | 240 | `red barn`, `house` | How is the red barn positioned relative to the house in the scene? |
| valT | `train_014232` | Geometric Relation | 265 | `bridge`, `line of boats` | How is the orientation of the bridge relative to the long line of boats on the left side of the frame? |
| valT | `train_002906` | Temporal Grounding | 140 | `baseball fields` | How long does the activity on the baseball fields last throughout the video? |
| valT | `train_009901` | Temporal Grounding | 180 | `soccer fields` | How long are the soccer fields visible in the video? |
| valT | `train_010397` | Temporal Grounding | 240 | `black car` | At what time does the black car appear in the lower right section of the parking lot and when does it disappear? |

Totals: 15 unique videos, 8 valQ/7 valT, 8 geometric/7 temporal, 3,085 manifest frames, and 28 direct prompt calls.

## Data gate and output layout

The local checkout has no MP4 assets (`find data -iname '*.mp4'` returned zero files). After the token gate passes, the runner stages only the 15 selected source paths from dataset `ReVA-Benchmark/ReVA` at revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee` into a temporary directory under `/tmp`. The Replicate client receives only those selected clips; no HF model or SAM 3 checkpoint is installed.

Successful API artifacts are isolated under `runs/sam3_pilot_stageA_replicate/`:

- one `*.call-XX.masks.zip` per direct prompt call;
- `overlay_cache/` with validated per-frame PNG masks and cache JSON;
- `subset_ids.json` for the exact 8+7 score selection;
- `manifest.json` with pinned model/version, all prompts, mask counts, newly
  billed calls, reused validated calls, and estimated cost.

The runner never records the API token or output authorization URLs.

## Exact launch sequence

Run the one-video smoke first. It stages only the first selected source clip,
executes its three direct prompts, validates all returned ZIPs, builds and
reloads the overlay cache, and writes only below `runs/sam3_pilot_stageA_replicate/smoke/`:

```bash
export REPLICATE_API_TOKEN  # configure securely in the process environment
python3 jobs/replicate_sam3_pilot.py --smoke-only
```

After inspecting the smoke ZIPs and `smoke/overlay_cache/`, run the full 15
video set with the same process environment:

```bash
python3 jobs/replicate_sam3_pilot.py
```

The full run writes the complete `subset_ids.json`, `manifest.json`, 28 prompt
ZIPs, and `overlay_cache/` below the main result directory. The smoke run does
not remove or overwrite full-run artifacts.

The full runner is resumable: before each call it validates the deterministic
per-question/per-prompt ZIP already on disk. A valid ZIP is reused without a
new Replicate request; a missing or malformed ZIP triggers only that one call.
After either a resumed or fresh run, every prompt ZIP is re-read to rebuild the
per-video overlay cache. The manifest separates completed prompt outputs,
prediction attempts (which drive the approximate cost), and reused validated
calls. A valid fallback ZIP plus its mapping sidecar is reused on resume without
another request; malformed or out-of-range cached outputs are regenerated only
for that prompt.

## Scoring command after the API cache is complete

The existing Stage A adapter and detector cache are reused for the detector context. The scorer adds only the direct SAM 3 overlay cache, without SAM 2 cache loading or DINO-based mask selection:

```bash
python3 code/score.py \
  --adapter runs/stageA_q3vl8b_r16/final_adapter \
  --split valQ \
  --grounding-context \
  --grounding-cache-dir runs/grounding_pilot_stageA_modal/grounding_cache \
  --sam3-overlay-dir runs/sam3_pilot_stageA_replicate/overlay_cache \
  --qa-ids-file runs/sam3_pilot_stageA_replicate/subset_ids.json \
  --offsets 0,0.5 --rots 2 --frame-sampling legacy \
  --out runs/sam3_pilot_stageA_replicate/scores_valQ_sam3.jsonl
```

Repeat with `--split valT`. The output is versioned as
`scores_<split>_sam3.segmentation.sam3-direct-overlay-v1.jsonl`, preserving the baseline, detector-only, and SAM 2 output files. The same adapter, question text/options, detector context, frame sampling, offsets, and rotations are used.

For the authorized A100 scoring environment, [`jobs/modal_sam3_score_pilot.py`](../../jobs/modal_sam3_score_pilot.py) stages only the 15 selected clips, the generated SAM 3 overlay cache, the Stage A adapter, and the existing reference score files. It builds the required DINO cache on the selected subset, runs both split scores, and writes paired prediction changes and per-task metrics to `comparison_report.md` and `comparison_summary.json`. It does not install or run SAM 3 weights.

Launch that separate scoring step only after the full API cache exists:

```bash
modal run jobs/modal_sam3_score_pilot.py
```

Current API execution has not started because `REPLICATE_API_TOKEN` is absent. Existing subset references in [`runs/segmentation_pilot_stageA_modal/comparison_report.md`](../segmentation_pilot_stageA_modal/comparison_report.md) remain unchanged.
