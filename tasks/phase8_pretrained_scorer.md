# Phase 8 backbone scorer

`code/phase8_scorer.py` scores the two Phase 8 backbone candidates with either
the pinned base weights alone or an optional PEFT adapter. Only the no-adapter
mode is zero-shot. H100 job `407497` uses an adapter, so its results are from an
adapted Qwen3.8-27B model, not the zero-shot base model. Set
`PHASE8_ADAPTER` to the adapter directory; the wrapper passes it to the scorer,
which records its path and checksum in the score manifest. It reuses the Stage A
contract from `code/common.py`:
legacy sampling, 16 frames per view, offsets `0` and `0.5`, cyclic option
orders, one-token A-D logits, and original-letter aggregation. It writes a
Phase 8 manifest beside the JSONL output and resumes only when that manifest is
an exact match.

The model-family adapters are in `code/phase8_adapters.py`:

* `qwen38` loads `Qwen/Qwen3.8-27B` with `AutoModelForMultimodalLM` and passes
  the direct Transformers argument `enable_thinking=False` to the chat
  template.
* `qwen3vl32b` loads `Qwen/Qwen3-VL-32B-Instruct` with
  `Qwen3VLForConditionalGeneration` and the standard chat template.

The family defaults are pinned to these Hugging Face revisions:

* Qwen3.8-27B: `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`
* Qwen3-VL-32B-Instruct: `0cfaf48183f594c314753d30a4c4974bc75f3ccb`

Qwen3.8's Transformers 5.19.0 implementation keeps `video_grid_thw` on the
model device while the visual encoder builds its positional and interpolation
metadata from that tensor.  Its `get_image_features` method then performs a
small `grid_thw.prod(...).tolist()` split-size calculation, and non-Flash
vision attention repeatedly converts `cu_seqlens` to Python split sizes.  The
Phase 8 Qwen3.8 adapter leaves the grid on CUDA for the encoder, while
computing both host-only split metadata values from detached CPU copies.  This
avoids the reported H200 CUDA invalid-argument failure at the GPU
reduction/device-to-host boundary without changing the video pixels,
precision, attention backend, or scoring rule.  Qwen3-VL uses the unpatched
native adapter.  If a rerun with
`CUDA_LAUNCH_BLOCKING=1` reports a failure inside the Qwen3.8 recurrent/GDN
kernel instead, that is a separate runtime issue and needs a Transformers or
optional-kernel investigation.

An explicit `--revision` overrides the family pin and is recorded in the
manifest. Outputs must use a new Phase 8 path under `$WORK`; the manifest
rejects reuse of a Stage A score path or a run with different model, split,
sampling, or TTA settings.

```bash
python code/phase8_scorer.py \
  --family qwen38 \
  --split valQ \
  --out runs/phase8_qwen38_27b/scores_valQ.jsonl

python code/phase8_scorer.py \
  --family qwen3-vl-32b \
  --split valQ \
  --out runs/phase8_qwen3vl32b/scores_valQ.jsonl
```

Before scoring, pre-download both pinned snapshots with the CPU-only job. It
writes only the Hugging Face cache and does not invoke or modify the Stage A
scorer or weights:

```bash
sbatch jobs/phase8_download.sbatch
```

The download job writes under `$HF_HUB_CACHE` (default
`$WORK/hf_cache/hub`), then resolves both pinned processor/config records with
`local_files_only=True` through that same cache before it exits.

After that job succeeds, use the Phase 8-only H200 wrapper. It runs offline,
requests one H200 and 12 CPUs with an 11:50 limit, stops after the current
complete question on `USR1`, and requeues so JSONL outputs resume by `qa_id`.
Full runs use `phase8_*` directories:

```bash
sbatch jobs/phase8_score.sbatch qwen38 valQ valT val
sbatch jobs/phase8_score.sbatch qwen3vl32b valQ valT val
```

The scorer receives all requested splits in one process and loads each
candidate model once. Its standalone equivalent uses an output template:

```bash
python code/phase8_scorer.py --family qwen38 \
  --split valQ valT --out runs/phase8_qwen38_27b/scores_{split}.jsonl
```

Run the frozen pilot first by setting `PHASE8_MODE=pilot`. The wrapper passes
`runs/phase8_backbone_comparison/pilot_selection.json` to every split through
`--qa-ids-file` and writes under a separate `phase8_pilot_*` directory. Set
`PHASE8_PILOT_SELECTION` only when using an explicitly copied, unchanged
selection file under the runtime tree:

```bash
PHASE8_MODE=pilot sbatch jobs/phase8_score.sbatch qwen38 valQ valT
PHASE8_MODE=pilot sbatch jobs/phase8_score.sbatch qwen3vl32b valQ valT
```

The pilot selection is restricted to `valQ` and `valT`; full mode can also
score `val` and `test`. `PHASE8_RUN` is checked so pilot outputs must be named
`phase8_pilot_*`, while full outputs use a separate `phase8_*` directory.

Pre-stage the pinned model snapshots in `$WORK/hf_cache` before setting the
offline job environment. Set `PHASE8_REVISION` only when intentionally using
an explicit revision override. The wrapper writes to `runs/phase8_*` and
rejects paths containing Stage A or Stage B names.

When a canonical Phase 7 frame manifest is available, pass it with
`--frame-manifest`. Each view records the actual source frame indices, bound
timestamps, and timestamp provenance. Without that file, timestamps are
explicitly recorded as the constant-FPS `index_divided_by_fps` fallback.

Each view also records the actual frame count, preprocessing and forward
latency, and processor shape indicators when those tensors expose shapes
(`input_ids`, `video_grid_thw`, and pixel tensors). The manifest records the
selection file checksum, selected question count, ID policy, and `bfloat16`
precision, so a changed pilot file rejects resume. A separate `.summary.json`
accumulates across resumptions and records model-load time/count, mean/median
latency and throughput, dtype, GPU name, peak GPU memory, host max RSS, and
available vision-grid values/token counts. It also records the actual tokenizer
IDs for plain `A`-`D` and leading-space ` A`-` D` forms in the manifest and
summary. Only the plain one-token IDs drive scoring; leading-space forms are
diagnostic and may be explicitly unavailable. Vision-grid values and token
counts are explicitly `null` when the processor does not expose them; no
unavailable metric is inferred.

## Paired Phase 8 analysis

After both a complete A0 run and one or more complete candidate runs exist,
`code/phase8_analysis.py` validates each score file against its own manifest.
It requires all four views for every question in both `valQ` and `valT`, so an
interrupted or partial split cannot be analyzed. Model IDs and revisions may
differ between arms; they are retained separately in the provenance output.

Pass a run directory when it contains exactly one matching score file per
split, or pass an explicit `{split}` file pattern:

```bash
python code/phase8_analysis.py \
  --work "$WORK" \
  --a0 runs/phase8_a0 \
  --candidate qwen38=runs/phase8_qwen38_27b \
  --candidate qwen3vl32b=runs/phase8_qwen3vl32b \
  --output-prefix runs/phase8_analysis/candidates \
  --bootstrap-reps 10000 \
  --bootstrap-seed 20261019
```

The JSON/Markdown/CSV outputs include overall accuracy and counts, task and
category slices, fixes/regressions/both-correct/both-wrong transitions,
candidate disagreements, confidence and margin deltas, per-question paired
records, and deterministic video-cluster 95% bootstrap intervals. The same
prefix also emits a dependency-free `.svg` chart of candidate-minus-A0 task
accuracy deltas for both splits; its bar data attributes contain the A0 and
candidate denominators.

