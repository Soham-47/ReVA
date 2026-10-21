# Phase 7 execution plan and feasibility record

**Status:** Phase 7A inspection and offline preflight completed. No new model
inference, training, paid provider call, Codabench action, or broad video
scoring was launched.

**Reference proposal:** the complete proposal attached to this task, `ReVA_Geometric_Temporal_Improvement_Plan.md`.

## Current repository facts

The frozen Stage A reference remains:

- Base model: `Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
- Adapter: `runs/stageA_q3vl8b_r16/final_adapter`.
- Score protocol: 16 `legacy` frames, offsets `0` and `0.5`, two cyclic option
  rotations, four aggregated views. The original `legacy/` score files remain
  untouched.
- Development references: `valQ` 498/595 and `valT` 885/1,059 for the frozen
  Stage A arm. The Phase 6 A3 and A4 adapters both regress on `valT`; they are
  retained as historical artifacts and are not Phase 7 candidates.

Prior experiments already answer several feasibility questions:

- Coarse-to-fine temporal sampling did not pass its gate: T1 was −1 answer on
  `valT` and −2 on `valQ`; timestamp-context T2 was −8 and −4.
- The reviewed 24-case temporal window pilot produced a net +1 answer versus
  T0, with 3 fixes and 2 regressions; it is a diagnostic result, not a broad
  promotion signal.
- The completed Replicate SAM 3 pilot has 28 prompt outputs and 5,224 cached
  masks across 15 videos. The combined 32-question comparison is 24/32 for
  SAM 3 versus 23/32 baseline, with 3 fixes and 2 regressions. The review
  identifies broad field masks, unstable barn/house masks, and non-unique car
  masks. This is feasibility evidence only.
- The Phase 3 explicit-geometry run has a frozen 60-question G0 selection and
  relation-context schema, but the preflight remains blocked for new paid calls
  because the Replicate token and matching cache/score inputs are absent.
- `code/grounding.py` already implements deterministic adjacent-frame greedy IoU
  association. Existing Grounding DINO caches contain 50 files, 1,600 sampled
  frames and 1,726 tracks (about 1.94 MiB), but they use 15 frame policies, so
  they require an exact policy/checksum gate before reuse.

## Phase 7A artifact

`code/phase7_manifest.py` adds a versioned, offline-only canonical frame
manifest contract. It accepts the aggregate `data/manifest.json` and optional
run-local source manifests. It records:

- source frame count, dimensions, FPS and duration;
- one timestamp for every source frame;
- timestamp provenance (`presentation_timestamp` when supplied by a prior
  decoded source manifest, otherwise the explicit
  `index_divided_by_fps` fallback);
- frozen Stage A legacy samples at offsets `0.0` and `0.5` with their source
  frame indices and bound timestamps;
- dataset revision and a content digest.

The generated artifact is
`runs/phase7A_feasibility/canonical_frame_manifest.json` with 1,047 video
records and digest
`8908636e3c76b3dc8f6058bbf817eb5c44d231a19e35c7cb7a1fdd6a157164a2`.
Twenty-three records use exact presentation timestamps from the reviewed Phase
2 source manifest; the remaining 1,024 records are explicitly marked as the
constant-FPS fallback. This artifact is new and does not replace
`data/manifest.json` or any prior run manifest.

The validator rejects non-monotone timelines, out-of-range frame indices,
sample timestamps that do not bind to the source timeline, malformed records,
and digest tampering. A source manifest with exact timestamps is preferred for
any future temporal or tracking run; the FPS fallback is suitable for
preflight and must not be described as a variable-frame-rate presentation
timeline.

## Phase 7B cache-validation artifact

The bounded offline validator is `code/validate_phase7_cache.py`; its focused
contract checks are in `tests/test_phase7_cache_validation.py`. The generated
report is `runs/phase7B_cache_validation/report.json` with a short summary in
`runs/phase7B_cache_validation/report.md`. It reads the reviewed Phase 2 packet
and Phase 3 G1 cache sidecars only; it does not run inference or read answer
fields.

The packet contains 24 cases across 23 videos. Five videos have valid mapped
Grounding DINO caches with exact legacy frame policy and canonical timestamps;
the caches contain 225 tracks and 446 observations, including 167
single-observation tracks, with zero gap or label-switch proxies in this
subset. Prompt coverage makes 3 cases exact cache reuses across 2 videos; one
same-video cache has a prompt gap and two cases remain prompt-unverified. The
other 18 videos have no trusted cache mapping.

The lead agent independently compared the five mapped HPC source SHA-256
values against every cache `video_checksum`; all five matched at dataset
revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`. All five clips are now
staged locally and all five cache overlays are rendered. The source-checksum
and overlay manifests are in `runs/phase7B_cache_validation/`; the prior
offline status remains preserved in `report_offline_snapshot.json` and
`report_offline_snapshot.md`.

The cache-only manual review is recorded in
`runs/phase7B_cache_validation/manual_source_review.md`. It found that the
three exact question/cache matches do not provide a valid denominator for
instance-level first-visibility evaluation: one target is already present at
the clip start, and two references are not uniquely identifiable. Two
additional source targets (the marked white van and the Chinese-character
archway) are visually identifiable, but the fixed-vocabulary cache does not
propose the target regions at the relevant frames. The field control is also
visible at the clip start and therefore onset-censored. Cached track IDs come
from adjacent greedy IoU association, not SAM video tracking or verified
identity labels.

Phase 7B cache validation and source review are complete. The downstream
evidence gate does not pass: no question has both a usable target/event label
and a correct automatic proposal from the existing candidate cache. No new
SAM, DA3, or Qwen inference was run. The HPC account has no configured
Hugging Face token and no cached SAM 3 or DA3 weights, so a native SAM 3 run
requires authenticating an account with access to the gated checkpoint before
we can test the proposed identity-preserving pipeline. A frozen-Qwen arm is
deferred until a target is automatically proposed and independently
assessable.

## Phase 7C Replicate smoke (current status)

Phase 7C is a new, isolated Replicate preflight in
`runs/phase7C_replicate_smoke/`. It is limited to two source-reviewed `valT`
clips: `train_000047` (the Chinese-character archway under camera motion) and
`train_001138` (the marked white van). Prompts are literal spans from the
questions plus the fixed controls `vehicle`, `building`, and `road`; answer
labels, baseline predictions, manual target boxes, and model-ranked identity
are not inference inputs.

The exact pinned endpoints are:

- `lucataco/sam3-video@408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269`, a merged-mask reference;
- `8digit/gener8-sam3@a6f50615da4cfa8b03fe95ec56681bc252a5e60c2ed368938f72bfa4f55dfcd1`, a third-party ID-preserving candidate.

The ID-preserving wrapper has limited public run history and an advertised
estimate of about $0.078 per run; those claims and its output behavior remain
unverified until a credentialed run passes the validator and manual review.

The plan contains 16 primary prediction creates (two videos, two endpoints,
four prompt arms) and a shared allowance of at most four retry creates. The
runner uses Replicate's asynchronous `predictions.create` and
`predictions.get` flow, submits sequentially, persists the prediction ID
before polling, and resumes an existing ID without creating a duplicate. A
create response without an ID is unresolved and requires provider
reconciliation. The request and output contracts are implemented in
`code/phase7c_replicate_runner.py`; they follow Replicate's prediction API and
the pinned model pages:
`https://replicate.com/docs/topics/predictions/create-a-prediction`,
`https://replicate.com/lucataco/sam3-video`, and
`https://replicate.com/8digit/gener8-sam3`.

The isolated runtime pin is
`runs/phase7C_replicate_smoke/requirements.txt` (`replicate==1.0.7`). The
official 1.0.7 client exports `replicate.Client`, supports
`client.predictions.create(version=..., input=...)` and
`client.predictions.get(prediction_id)`, uploads local file handles through
its file API, and exposes file outputs with `FileOutput.read()`; the runner
uses those APIs. The repository Python environment does not currently have
this package installed. The setup command for a separate Phase 7C virtual
environment is:

```bash
python3 -m venv .venvs/phase7c-replicate \
  && . .venvs/phase7c-replicate/bin/activate \
  && python -m pip install --requirement runs/phase7C_replicate_smoke/requirements.txt
```

This command has not been run as part of the no-call preflight and does not
modify Stage A or project-wide requirements. The SDK API references are the
official 1.0.7 source tree and client/helper files linked from the runner's
manifest.

The dry run is complete and records a planning estimate of about $0.6448 for
the 16 primary requests using the pinned model-page estimates; retries are
excluded. This is a planning estimate, not provider billing. The execution
manifest separates create attempts, observed prediction IDs, validated
successful predictions, and any unresolved billing uncertainty.
`REPLICATE_API_TOKEN` is absent in this environment, so zero paid calls have
been made. No native SAM 3 weights, DA3, or Qwen inference is
authorized by this preflight. Before any call, the credential must be
provided through the execution environment and the dry-run manifest must
remain unchanged.

Output validation rejects unnamed or sparse merged-mask archives. Because the
merged-mask endpoint does not expose verified frame metadata in its contract,
its explicit full-frame output is accepted with timestamp mapping marked
unverified. The ID-preserving candidate must provide `objects.json`, 16-bit
labels for every source frame, and matching source frame count, dimensions,
and FPS before timestamps are mapped. Source checksums, canonical frame
timestamps, output frame indices, and validation uncertainty are recorded.
Manual output review and this evidence gate must pass before considering any
future DA3 or Qwen tranche.

## Phase 7B first experiment recommendation

Use a cache-only tracking/evidence feasibility tranche before any new DA3 or
SAM3 inference:

1. Freeze the reviewed 24-case cohort in
   `runs/phase2_temporal_verified_window/review_material/review_packet.json` as
   the diagnostic set: 12 `valQ` and 12 `valT`, each split containing six
   baseline-error cases and six correct controls. Keep those labels for audit
   only; do not select new inference targets from answer errors.
2. Add the deterministic independent controls from
   `tasks/phase3_grounding_pilot_ids.json` only after deduplicating video paths
   and documenting its fixed task strata. Use the manifest's IDs and prompts as
   inputs, never answer labels.
3. For videos with a matching canonical timeline and existing Grounding DINO
   cache, validate frame-policy identity, source checksum, timestamp binding,
   adjacent-frame track continuity, and bounded context serialization. Record
   coverage, single-observation uncertainty, track switches, and fallback
   counts. Do not score Qwen in this tranche.
4. Use the existing baseline and SAM 3 overlay caches only as evidence to
   inspect. A cache is reusable only when the exact source frame policy,
   source checksum, prompt and model revision match; cached tracks are not
   ground truth.

The first Qwen experiment, if this tranche passes its evidence gate, is a
matched frozen-Stage-A comparison on the fixed tranche: original frames versus
the same frames plus compact, explicitly uncertain 2D track/event context.
Keep the four-view scorer, options, frame count and adapter identical. Report
fixed, broken, wrong-to-wrong and unchanged predictions, evidence coverage and
qualitative track correctness separately. DA3/VGGT geometry belongs after this
gate and requires a separate checkpoint/API compatibility preflight.

## Model and environment feasibility

Pinned identities found in repository code and manifests:

| Component | Identity | Local status |
|---|---|---|
| Qwen Stage A | `Qwen/Qwen3-VL-8B-Instruct@0c351dd...` | Adapter exists; base weights are not staged in this checkout |
| SAM 3 video | `lucataco/sam3-video@408f82fc...` | Masks/overlays exist; no native checkpoint |
| Grounding DINO | `IDEA-Research/grounding-dino-tiny@a2bb814...` | Cache outputs exist; staged model snapshot is absent locally |
| SigLIP2 | `google/siglip2-base-patch16-224@022b6f...` | Existing cache code is pinned; model snapshot is absent locally |
| DA3 | proposal candidate only | No code, checkpoint, or compatible environment found |
| VGGT | proposal candidate only | No code, checkpoint, or compatible environment found |

The current host has Python 3.12.3 but no working NVIDIA driver and no
`sbatch`/`sinfo` command. The prior H200 records show that the frozen Qwen
scorer used about 27.2 GiB GPU memory and A4 training used 28.6 GiB, with 12
CPUs under the cluster QOS. Those are historical measurements, not a current
allocation. No Phase 7 H200 job should be submitted from this host.

## Resource gates and estimates

- Phase 7A manifest generation: CPU-only, 1,047 records, 7.2 MiB JSON in this
  checkout, completed in about 1.6 seconds; no model or video decode required.
- Phase 7B cache validation: CPU-only and bounded by existing JSON caches; the
  50-cache inventory is about 1.94 MiB. A matching policy subset must be
  measured before choosing a question count.
- First frozen-Qwen tranche: one H200 allocation with the existing 12-CPU
  wrapper is the upper bound. Reuse the prior scorer measurements and request a
  short smoke plus a bounded tranche; do not estimate a full split until the
  evidence context size and model throughput are measured.
- Native DA3/VGGT and native or HF SAM3 remain blocked pending exact
  checkpoint/code/API identity, staged local assets, output schema checks and
  a separate cost/time gate. The selected pinned Replicate endpoints have a
  completed local schema and dry-run preflight, but remain credential-blocked;
  no paid calls or multi-hour job is part of this record.

## Promotion gates

Advance from Phase 7B evidence validation only when source timestamps and frame
mapping pass, evidence coverage is reported, track identity is hand-checked on
representative camera-motion/occlusion cases, and fallback behavior is
explicit. Advance to frozen-Qwen scoring only with a new unique run directory,
the unchanged Stage A adapter and a predeclared fixed cohort. Advance to DA3 or
training only after a paired Qwen result shows that simpler evidence addresses
the reviewed failure subtype.

