# Phase 6: controlled visual adaptation pilot

Status: preparation only. No Slurm job was submitted and the Stage A adapter or
its score artifacts were not modified.

## Question and arms

The first comparison is whether a small visual adaptation helps beyond
continuing the existing language-only adapter.

| Arm | Initialization | Trainable modules | Output |
| --- | --- | --- | --- |
| A0 | Existing Stage A adapter, unchanged | none | `runs/stageA_q3vl8b_r16/final_adapter` |
| A1 | Load A0 with `is_trainable=True` | Stage A language LoRA only | `runs/phase6_A1_language_from_stageA` |
| A2 | Load A0 with `is_trainable=True` | A1 modules plus the main visual merger LoRA | `runs/phase6_A2_language_plus_merger_from_stageA` |
| A3 (future) | Load A0 with `is_trainable=True` | A1 modules plus selected vision attention/projection LoRA | to be prepared after A1/A2 review |
| A4 (conditional future) | Load A0 with `is_trainable=True` | A1 plus visual modules selected by the predeclared A2/A3 gate | only if the conditional gate passes |

A0 stays the immutable evaluation reference. A1 and A2 use the same
`data/splits/fit_A.json`, row sampler, option permutation, frame sampling,
loss, seed, micro-batch, gradient accumulation, optimizer, scheduler, and 200
optimizer-step pilot budget. The Phase 6 configs set `deterministic_sampling`
and `data_seed` to 42; existing Stage A configs leave the old per-sample random
sampling behavior unchanged.

The pilot uses one-token answer loss, micro-batch 1, gradient accumulation 16,
bf16, SDPA, gradient checkpointing, 12,000-token protection, and cosine decay.
The A2 visual adapter uses `5e-5` versus `2e-4` for language LoRA. This is a
predeclared 4x lower rate because the visual merger is a small input bridge
being added on top of a trained adapter; it limits disruption while retaining
an independent visual update path. The rate should be changed only as a
follow-up experiment, not after seeing pilot outcomes.

## A3/A4 follow-on designs

A3 is a separate visual-location test. If A1/A2 pass their smoke and paired
evaluation gates, A3 would initialize from the same immutable A0 and retain
the A1 language targets while adding rank-16 LoRA to the last four vision
blocks' attention projections:
`model.visual.blocks.{23,24,25,26}.attn.qkv` and
`model.visual.blocks.{23,24,25,26}.attn.proj`. The v5.19.0 implementation
defines the vision block as `attn.qkv: Linear(1152, 3456)` and
`attn.proj: Linear(1152, 1152)`. Four blocks add 442,368 rank-16 LoRA
parameters, giving an inferred 44,089,344 trainable parameters with A1's
language adapter. These paths and counts require the first live smoke to
confirm the installed model tree; they are not enabled by the current A2 code.
The A3 visual group would use a separately predeclared lower learning rate and
the same rows, order, loss, seed, optimizer-step budget, and evaluation views.

A4 is conditional and should not be started by default. Before scoring A2 and
A3, predeclare that A4 is eligible only when at least one candidate has a
positive paired net change on both valQ and valT without a material regression
in a top-level category, or when A2 and A3 repair complementary baseline
errors. If eligible, A4 is the union of the winning visual modules (merger,
selected vision attention/projection, or both) plus the A1 language adapter,
with the same 200-step screening budget and a lower visual learning rate. If
neither candidate meets the gate, stop after A2/A3 and do not use A4 to search
validation noise. A4 results remain a development finding until a frozen
follow-up protocol is approved.

## Architecture inspection

The pinned config identifies `Qwen3VLForConditionalGeneration`, 36 text layers,
text hidden size 4096, vision hidden size 1152, `spatial_merge_size=2`,
`out_hidden_size=4096`, and deepstack indexes `[8, 16, 24]`:

The repository pins `transformers==5.19.0`; the implementation link below is
the matching v5.19.0 tag.

* [pinned Qwen3-VL-8B-Instruct config](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b/config.json)
* [Transformers v5.19.0 Qwen3-VL visual merger implementation](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L155-L167)

The visual tree is `model.visual.merger`, whose main bridge is:

```text
model.visual.merger.norm
model.visual.merger.linear_fc1   Linear(4608, 4608)
model.visual.merger.act_fn       GELU
model.visual.merger.linear_fc2   Linear(4608, 4096)
```

The 4,608 input is `1152 * 2 * 2`. The separate
`model.visual.deepstack_merger_list.{0,1,2}` modules are deliberately excluded
from A2 so this first comparison changes one visual bridge only. This
inspection uses the pinned model config and official Transformers source;
it was not a live PyTorch `named_modules()` dump because the local checkout has
no compatible Transformers/PEFT CUDA runtime or base-model snapshot. The
training path will discover these suffixes at runtime and fail closed unless
both main merger linears are present as `nn.Linear` modules.

| Model path or target pattern | Base shape | A1 | A2 |
| --- | --- | --- | --- |
| `model.language_model.layers.*.self_attn.{q,o}_proj` | `Linear(4096, 4096)` | LoRA trainable | LoRA trainable |
| `model.language_model.layers.*.self_attn.{k,v}_proj` | `Linear(4096, 1024)` | LoRA trainable | LoRA trainable |
| `model.language_model.layers.*.mlp.{gate,up}_proj` | `Linear(4096, 12288)` | LoRA trainable | LoRA trainable |
| `model.language_model.layers.*.mlp.down_proj` | `Linear(12288, 4096)` | LoRA trainable | LoRA trainable |
| `model.visual.merger.norm` | 4608-wide normalization | frozen | frozen |
| `model.visual.merger.linear_fc1` | `Linear(4608, 4608)`; 21,238,272 base params including bias (inferred) | frozen | rank-16 LoRA |
| `model.visual.merger.linear_fc2` | `Linear(4608, 4096)`; 18,878,464 base params including bias (inferred) | frozen | rank-16 LoRA |
| `model.visual.deepstack_merger_list.{0,1,2}` | separate deepstack bridges | frozen | frozen |
| all remaining base-model weights | model-defined | frozen | frozen |

## Trainability and parameter accounting

The existing Stage A adapter metadata and safetensor header were inspected
locally. Its target pattern covers 36 layers x 7 language projections = 252
modules, with A/B tensors for 504 trainable tensors and 43,646,976 trainable
parameters. The local Stage A adapter file is 174,663,096 bytes.

The two selected merger linears therefore contain 40,116,736 base parameters
in total, inferred from the v5.19.0 source shapes and the pinned config. These
base counts, like the module paths above, have not been checked against a live
PyTorch module tree in this checkout.

At rank 16, A2 adds two visual modules and four LoRA tensors:

| Scope | LoRA modules | LoRA tensors | Trainable parameters |
| --- | ---: | ---: | ---: |
| A1 language | 252 | 504 | 43,646,976 |
| A2 visual addition | 2 | 4 | 286,720 |
| A2 total | 254 | 508 | 43,933,696 |

The base model, vision encoder blocks, patch embedding, position embedding,
layer norms, and all non-LoRA language weights remain frozen. `train.py` prints
the loaded adapter name, discovered visual targets, active adapter list, and
parameter counts before training. The A2 path activates `default` and
`visual_merger` on the underlying PEFT tuner, which is required by the pinned
PEFT 0.21.2 API.

The resulting A2 adapter saves the existing language adapter at the output root
and the visual adapter below `final_adapter/visual_merger/`. The shared scoring
helper detects that subdirectory, loads both adapters, activates both, and then
merges them. A normal Stage A directory still loads exactly as before.

## Dataset strata and error taxonomy

The repository defines `fit_A` as 14,119 training questions, `valT` as a video
holdout, and `valQ` as a held-out `(video, question)` grouping. The local
`valT` and `valQ` files contain 1,059 and 595 questions respectively; the
complete `fit_A` split is staged only in the runtime tree. Their top-level
category counts are:

| Split | Factual Perception | Spatial Reasoning | Temporal Understanding | Causal Reasoning |
| --- | ---: | ---: | ---: | ---: |
| valT (n=1,059) | 258 | 467 | 243 | 91 |
| valQ (n=595) | 143 | 273 | 132 | 47 |

Both splits cover the same eleven task labels: Object and Land Cover
Recognition, Temporal Grounding, Change Detection, Perspective and Viewpoint,
Geometric Relation, Trend and Pattern, Structural Layout, Causation Reasoning,
General Understanding, Hypothetical Reasoning, and Consequence Reasoning. The
source strata are Hawk_UAV, ERA_Tra, VisDrone, and UAVDT; their local counts
are:

| Split | Hawk_UAV | ERA_Tra | VisDrone | UAVDT |
| --- | ---: | ---: | ---: | ---: |
| valT | 493 | 289 | 193 | 84 |
| valQ | 305 | 164 | 105 | 21 |

Source counts must be reported because camera/viewpoint and compression vary by
source.

Classify every changed or incorrect answer along these dimensions before
aggregating a result:

* **Input and readout:** timestamp or frame-sampling miss, prompt formatting,
  option permutation or letter mapping, tokenizer/readout failure, or a tie
  resolved by the wrong view.
* **Factual perception:** small, blurred, occluded, or low-contrast object;
  land-cover or attribute/color confusion; count or presence miss.
* **Spatial and structural:** left/right, above/below, distance, containment,
  layout, viewpoint, or multi-entity comparison error.
* **Temporal and change:** first/last visibility, onset/exit, duration,
  ordering, camera-versus-object motion, or change detection across frames.
* **Reasoning:** trend/pattern, causation, consequence, hypothetical, or
  general-understanding error after the visual evidence was available.
* **Domain and ambiguity:** source-specific appearance, domain shift, unclear
  target identity, multiple valid instances, or insufficient temporal evidence.

Use the existing task/category/source summaries for quantitative slices and
retain item-level baseline-to-candidate corrections and regressions for the
qualitative taxonomy. Do not relabel a wrong-instance or ambiguous answer as
correct because a category-level cue was present.

## Paired valQ/valT evaluation

Evaluate A0, A1, and A2 on identical IDs, prompts, frame views, offsets, and
option rotations. Report, separately for each split and pooled only as a
secondary view:

| Metric | Required reporting |
| --- | --- |
| Overall accuracy | TTA accuracy and single-view accuracy, with numerator/denominator |
| Paired change | Candidate minus A0 accuracy, net corrections, regressions, unchanged items, and per-item delta |
| Uncertainty | 95% paired bootstrap interval, clustered by video; do not treat repeated questions from one video as independent |
| Slices | Every task, top-level category, and source stratum with counts |
| Temporal subset | Temporal Grounding accuracy separately: 69 valQ questions and 133 valT questions |
| Gold-answer probability | Mean and median TTA probability assigned to the gold letter (`p_gold`), plus the fraction with `p_gold > 0.5` |
| Gold-answer margin | TTA `p_gold - max(p_wrong)`; report mean/median and the fraction with positive margin, plus paired A2-minus-A1 deltas |
| Error review | Highest-confidence regressions and repaired baseline errors, with the taxonomy above |

`valQ` and `valT` are development splits, not blind test evidence. A candidate
should advance only when its paired changes are useful on both splits, do not
hide a material category or source regression, and remain within the approved
runtime budget. The existing Stage B full-train adapter must not be used as a
held-out comparator.

The predeclared A2-versus-A1 screening gate is: TTA accuracy must improve by
at least **+2.0 percentage points on valT** and **+1.0 percentage point on
valQ**; neither split may have a mean gold-answer probability or gold-answer
margin decrease larger than **0.02**; and no top-level category or source
slice with at least 50 questions may regress by more than **1.0 percentage
point** on either split. Smaller slices remain descriptive and must be shown
with counts. Report paired video-cluster bootstrap intervals alongside these
point thresholds, but do not substitute an after-the-fact significance rule.
Missing any primary threshold is a no-go for A2 as the next arm; it does not
justify changing the sampler, loss, or visual learning rate after seeing the
results.

## Pilot procedure and gates

1. Stage the runtime `WORK` tree with the pinned model revision, Stage A
   adapter, `fit_A`, manifest, and referenced clips. The source checkout does
   not contain the base model weights or the complete `fit_A` split.
2. Submit the wrapper in its default smoke-only mode for each config:
   `sbatch jobs/phase6_train.sbatch code/configs/phase6_A1.yaml smoke` (and
   likewise for A2). It writes `SMOKE_OK` only after the Stage A load, visual
   target discovery (A2), one forward/backward probe, and (for the Phase 6
   configs) a finite-gradient, parameter-update check succeed, then exits
   without starting training.
3. Review the smoke log, `SMOKE_OK`, and architecture/parameter printout.
   A missing adapter, unexpected module path, inactive adapter, non-finite loss,
   or zero visual gradient should stop the run. Training mode refuses to start
   unless the reviewed `SMOKE_OK` marker already exists.
4. After explicit review, submit the same wrapper with the distinct `train`
   mode. This launches the bounded 200-step arm and can resume only from its
   own checkpoints.
5. Score A0, A1, and A2 on the same held-out `valT` and `valQ` IDs with the
   existing scorer and TTA settings. Keep A0 scores and all Stage A artifacts
   intact. Do not use the Stage B full-train adapter as held-out evidence.

The 200-step result is a screening pilot. It can reject an implementation or a
clearly unhelpful direction; it cannot establish a final generalization claim.
Only after reviewing paired A1/A2 results should the step budget or deepstack
scope be reconsidered.

## Failure modes and recommendation order

The implementation gates the main risks before training: missing or unexpected
module paths, PEFT adapter activation errors, absent trainable parameters,
non-finite loss or gradients, zero visual gradients, and an optimizer step that
does not update a trainable tensor. The output adapter must preserve the
language adapter at its root and the visual adapter below its named
subdirectory; scoring must load and merge both. A stale `SMOKE_OK`, incomplete
runtime `fit_A`, missing base snapshot, or changed preprocessing manifest is a
stop condition.

The current ordering is:

1. Preserve A0 and run separate smoke-only allocations for A1 and A2.
2. After review, run the bounded A1/A2 arms with identical data/order and
   optimizer-step budgets, then perform the paired valQ/valT evaluation.
3. Prepare A3 only if A2 is promising or the error taxonomy shows failures that
   a merger alone cannot address. Run a fresh architecture smoke before any A3
   training.
4. Prepare conditional A4 only under the predeclared A2/A3 gate. If neither
   candidate is useful on both development splits, stop rather than searching
   validation noise.
5. Defer deepstack or full vision-encoder tuning until a later approved design;
   it is outside this controlled comparison.

Likely interpretation failures include mistaking a source/domain gain for a
general visual gain, improving factual perception while regressing temporal or
spatial reasoning, treating a changed answer as a correction without checking
the gold label, and overinterpreting a 200-step result. Compute matching here
means the same rows, seed, micro-batch, loss, optimizer-step count, and
evaluation protocol; it does not mean equal trainable parameter counts or equal
wall time. Vision-block arms also require extra backward activation storage, so
their H200 time and memory cannot be inferred from the A1/A2 estimate.

## Resource estimate

The prepared wrapper requests one H200, 12 CPUs, 128 GiB host memory, and a
three-hour wall limit per arm. The 200-step budget is intentionally bounded;
the existing full Stage A recipe is much longer. A reasonable planning range
is about 1–3 GPU-hours per arm (2–6 GPU-hours for both), but no local timing
sample exists because the model weights and CUDA runtime are not cached here.
The actual `StepBenchmark` output from the training allocation should replace
this range before any larger run. A2's extra 286,720 LoRA parameters
are about 1.1 MB in fp32 and should have negligible weight/optimizer-memory
impact relative to the frozen 8B bf16 base; activation memory remains the
dominant uncertainty.

A3's inferred 442,368 additional LoRA parameters are small in weight size, but
training selected vision blocks requires gradients through the vision stack and
can materially increase activation memory and step time. A4 can add both the
merger and selected vision-block paths. Neither A3 nor A4 has a submitted
config or approved resource estimate; each requires its own smoke-only H200
allocation and measured `StepBenchmark` before a training budget is approved.

## Read-only HPC readiness check

The actual read-only check on `ambpdc` found the H200 partition up and no
current jobs. `WORK` was unset; the default Python was 3.8.8 without Torch; the
Qwen cache was only about 100 KB; and no ReVA runtime tree or `fit_A` staging
was present under the home directory or `/scratch/ambpdc/soham`. The isolated
environment, pinned model snapshot, Stage A adapter, complete split, manifest,
and referenced clips must be staged before smoke can run. No staging, job
submission, or training was performed during this check.

## Prepared files

* `code/train.py`: optional Stage A adapter initialization, fail-closed visual
  merger discovery, multi-adapter activation, deterministic Phase 6 sampling,
  explicit parameter-group learning rates, and bounded `max_steps` support.
* `code/smoke.py`: accepts `--config` and exercises the exact A1/A2 model
  construction path before training.
* `code/common.py`, `code/score.py`, `code/score_temporal.py`,
  `code/score_temporal_verified.py`: load the optional saved visual adapter
  alongside the default language adapter for inference.
* `code/configs/phase6_A1.yaml` and `code/configs/phase6_A2.yaml`: paired 200
  optimizer-step recipes with separate output roots.
* `jobs/phase6_train.sbatch`: H200 Slurm wrapper with smoke-only default mode;
  explicit `train` mode requires a reviewed `SMOKE_OK` marker. It has not been
  submitted.

## Checks and current blocker

Static YAML, Python compilation, and shell syntax checks are safe locally. The
The GPU forward/backward smoke was not run because the read-only `ambpdc`
check found no isolated runtime, CUDA Torch, Qwen base snapshot, complete
`fit_A`, or staged runtime videos. The first approved H200 allocation must run
the wrapper in smoke-only mode; a separate explicit `train` invocation is
required before any 200-step budget can be consumed.
