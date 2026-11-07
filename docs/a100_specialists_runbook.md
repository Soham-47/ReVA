# A100 specialist preparation

This checkout prepares an A100-80GB kernel smoke and two one-epoch specialists. It does not launch Slurm, and it is not a drop-in replacement for the HPC tree at `/scratch/ambpdc/soham/ReVA`. That tree has local files this GitHub checkout does not contain. Copy the new files, then hand-merge the three core diffs below. Do not `git reset`, `git clean`, or `rsync --delete` the HPC tree.

Nothing here has been trained or scored. Do not treat a passing smoke marker as an accuracy result. EXP-010 already recorded a failed temporal-label quality gate; the temporal job is prepared because this run was approved, and that earlier audit still stands until new labels are reviewed.

## HPC files to keep

Do not overwrite these local HPC files if they exist:

- `tools/cuda_probe_forward.py` and its `.bak434916` / `.bak435040` backups
- `jobs/cuda_probe.sbatch`
- `tools/ensemble_scores.py`
- `tools/paired_gate.py`
- `jobs/activate_reva.sh`
- any `code/configs/b1_hint16.yaml` or `code/configs/b2_frames32.yaml` already edited on the cluster
- the five reported `tmp_sync` directories (not seen from this checkout; do not delete them)

`runs/stageB_q3vl8b_r16_h100/` already has its October scores. These jobs write only under `runs/specialists/<run_name>/<UTC>_<slurm_job_id>/`.

## Hand-merge these existing files

| File | What changed | What may already differ on HPC |
| --- | --- | --- |
| `code/common.py` | `REVA_SDPA_MATH_ONLY=1` also disables flash and memory-efficient SDPA. Unset, behavior stays the previous cuDNN-off path. | Reported HPC helpers `video_size_for`, `motion_hint_text`, `run_meta`, and `assert_score_video_contract`, plus math-only SDPA. Keep those. This patch does not call `video_size_for`. |
| `code/train.py` | Opt-in `train_tasks`, `frame_budget`, required hint checks, safe mirror, `kernel_smoke`, and `isolate_output`. Stage A/B configs leave the new keys unset. | The dataset loop may already differ. Merge the hooks; do not replace the HPC file with this copy. |
| `code/score.py` | Move the model to CUDA before `load_peft_for_inference`. Optional `--tasks` and `--motion-hints` are recorded in the score manifest. Old manifests still match when those flags are absent. | Keep any HPC `assert_score_video_contract` / `run_meta` call. The GitHub file now has the CUDA-before-merge order the A100 forward probe used. |

Before the 32-frame job, inspect HPC `video_size_for` if it exists. `b2_frames32.yaml` sets `preserve_frame_budget: true`, so a sample over `max_tokens: 12000` aborts instead of silently dropping frames. That abort is expected until the HPC pixel budget is wired in. It is not a measured memory result.

`code/smoke.py` is not this kernel gate. Without `explicit_optimizer`, that script backpropagates and then zeroes gradients; it does not run AdamW. The A100 smoke calls `code/train.py` for 8 optimizer steps (5–20 allowed) at the Stage A `grad_accum: 16`.

## Environment

The jobs source `$WORK/jobs/activate_reva.sh`, or else `$WORK/venv312/bin/activate`. They do not activate `envs/reva`.

Reported stack that completed the A100 forward probe: Python 3.12.12, torch 2.14.1+cu130, transformers 5.19, peft 0.21.2. This repo pins `transformers==5.19.0`, `peft==0.21.2`, and `torch==2.14.1` in `requirements.txt` / `requirements-torch.txt`. The CUDA wheel suffix is not pinned. This cloud checkout did not import torch and did not run a GPU kernel.

Offline cache, same as the existing train job:

```bash
export WORK=/scratch/ambpdc/soham/ReVA
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$WORK/hf_cache"
```

The wrappers also set `REVA_SDPA_MATH_ONLY=1`. They do not set `CUDA_LAUNCH_BLOCKING` or `CUDA_MODULE_LOADING`; those were probe diagnostics.

Resources on every new GPU wrapper: partition `gpu_a100_8`, `--gres=gpu:nvidia_a100-sxm4-80gb:1`, 12 CPUs, 128G RAM. Smoke asks for 6 hours. Train and score ask for `11:50:00` with a 15-minute USR1 checkpoint, under the 12-hour job limit. Requeue resumes the same `SLURM_JOB_ID` directory. A new submission creates a new timestamped directory.

## Data rules

Training reads only `data/splits/fit_A.json`, then keeps the task names in the config. `require_split: fit_A` and the existing `train_*` id check both have to pass. Validation and test files are not training inputs.

- Motion specialist (`b1_hint16`): `Perspective and Viewpoint`, `Change Detection`, `frame_budget: 16`, `data/camera_motion_hints.json`.
- Temporal specialist (`b2_frames32`): `Temporal Grounding`, `frame_budget: 32`, no motion hints.
- Mirror ablation (`b1_hint16_mirror`): secondary. Geometric Relation, Perspective and Viewpoint, and Change Detection. A row is flipped only when left/right can be swapped and the gold text lands on exactly one rewritten option. Clockwise, forward/back, up/down, and similar wording skips that row. Score the ablation without flipping.

Hints must be a flat video-path map. Strings must use the audited `Camera motion:` vocabulary from the ORB builder, or `Camera motion: unavailable`. Object values are accepted only through HPC `common.motion_hint_text`. Wrapped documents, QA-id keys, and answer language fail the audit. Row counts are whatever the audit prints on the cluster; this checkout does not contain `data/`.

`valQ` / `valT` are usable for a specialist only when the eligibility audit says `fit_A` shares no QA ids with them and no video with `valT`. `valQ` can still share videos with `fit_A`; that is question holdout, not video holdout. Stage B trained on `full_train`, so A+B comparisons on `valQ` or `valT` are refused. Compare A+B on official `data/ReVA/val.json`, and only if that file is also disjoint from `full_train`.

The routing rule, fixed before any specialist score exists: on the target-task slice, net fixes (fixes minus regressions) at least 2 and regressions at most 1, with every target row present in both files. A pass writes `routed_<split>_<family>.jsonl`. A failure writes only the gate JSON. This is a routing policy, not an accuracy claim. Task-filtered score files are incomplete for `code/evaluate.py`; use that evaluator only on a full split.

## Commands

Run from `$WORK` after the hand-merge. Do not cancel existing jobs.

```bash
export WORK=/scratch/ambpdc/soham/ReVA
cd "$WORK"

# CPU audits. The same audits run again inside the GPU jobs.
python code/specialist_data.py validate-config code/configs/smoke_b1_hint16.yaml --allow-unresolved
python code/specialist_data.py validate-config code/configs/smoke_b2_frames32.yaml --allow-unresolved
python code/specialist_data.py validate-config code/configs/b1_hint16.yaml --allow-unresolved
python code/specialist_data.py validate-config code/configs/b2_frames32.yaml --allow-unresolved

# 1. 16-frame motion kernel smoke, then the 1-epoch motion specialist.
sbatch jobs/a100_train_smoke.sbatch code/configs/smoke_b1_hint16.yaml
# after that job's SMOKE_OK:
sbatch jobs/a100_train_specialist.sbatch code/configs/b1_hint16.yaml

# 2. 32-frame temporal kernel smoke, then the 1-epoch temporal specialist.
sbatch jobs/a100_train_smoke.sbatch code/configs/smoke_b2_frames32.yaml
sbatch jobs/a100_train_specialist.sbatch code/configs/b2_frames32.yaml
```

Score only after `DONE` in that run's timestamped directory. `RUN_DIR` is the printed `runs/specialists/.../<stamp>_<jobid>` path.

```bash
# Motion specialist: official val is the A+B gate; valT is the video holdout vs Stage A.
sbatch jobs/a100_score_specialist.sbatch "$RUN_DIR" val valT

# Temporal specialist uses the same wrapper and its own 32-frame budget.
sbatch jobs/a100_score_specialist.sbatch "$RUN_DIR" val valT

# CPU gate. Point BASELINE at the Nov 5 A+B letter file produced by tools/ensemble_scores.py.
bash jobs/a100_route_specialist.sh "$RUN_DIR" val "$BASELINE" ensemble_ab

# Optional Stage A comparison on valT, only if eligibility_audit.json says the holdout is eligible.
bash jobs/a100_route_specialist.sh "$RUN_DIR" valT "$STAGE_A_VALT_JSONL" stageA
```

Mirror stays off that sequence:

```bash
python code/specialist_data.py validate-config code/configs/b1_hint16_mirror.yaml --allow-unresolved
sbatch jobs/a100_train_specialist.sbatch code/configs/b1_hint16_mirror.yaml
```

## Still required from the HPC tree

1. A diff of `code/common.py`, `code/train.py`, and `code/score.py` against this patch.
2. The signature and call site of `video_size_for` before trusting 32-frame tokens.
3. A sample of `data/camera_motion_hints.json` if the audit rejects its schema.
4. The output schema of `tools/ensemble_scores.py` if it is not score JSONL or a qa-id-to-letter map.
5. Confirmation that `data/ReVA/val.json` and the four split files are the ones the audit should read.

Until those are checked, the GPU commands above are prepared, not validated on the cluster.
