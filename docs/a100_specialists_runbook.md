# A100 specialist preparation

This checkout prepares an A100-80GB kernel smoke and two one-epoch specialists. It does not launch Slurm, and it is not a drop-in replacement for the HPC tree at `/scratch/ambpdc/soham/ReVA`. That tree has local files this GitHub checkout does not contain. Copy the new files, then hand-merge the three core diffs below. Do not `git reset`, `git clean`, or `rsync --delete` the HPC tree.

This checkout is not deployment-ready. Nothing here has been trained or scored. A smoke marker is not an accuracy result.

Temporal Grounding training is blocked in this tree. `assert_training_allowed` rejects `smoke_b2_frames32.yaml` and `b2_frames32.yaml`. The one-epoch motion recipe (`b1_hint16.yaml`) is unchanged: Stage A init, `grad_accum: 16`, one epoch, 16 frames, camera-motion hints.

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
| `code/common.py` | Live file, kept: `video_size_for`, `motion_hint_text`, `run_meta`, `assert_score_video_contract`, and always-on math SDPA with cuDNN off. | Do not replace this with the older GitHub copy. |
| `code/train.py` | Opt-in `train_tasks`, `frame_budget`, required hint checks, safe mirror, `kernel_smoke`, and `isolate_output`. Stage A/B configs leave the new keys unset. | The dataset loop may already differ. Merge the hooks; do not replace the HPC file with this copy. |
| `code/score.py` | Move the model to CUDA before `load_peft_for_inference`. Optional `--tasks` and `--motion-hints` are recorded in the score manifest. Old manifests still match when those flags are absent. | Keep any HPC `assert_score_video_contract` / `run_meta` call. The GitHub file now has the CUDA-before-merge order the A100 forward probe used. |

## EXP-010 and the temporal specialist

`experiments/EXP-010/experiment.md` is `gate_failed_no_training`. The audit is `results/temporal_training_data_audit/phase1_report.md`.

What failed: a plan to build targeted temporal supervision from train labels. The reviewed sample is 100 Temporal Grounding rows from 100 videos, drawn from 913 train rows after valQ/valT videos were removed. Visual dispositions are 10/100 usable, 57/100 conditional, and 33/100 excluded. Counting every conditional case still gives 67/100, under the predeclared 80/100 gate. Gold review of those 100 is 10 supported, 51 partial or censored, 11 not supported, and 28 not assessable. The 10 supported ids are the same 10 visually usable ids. The independent 20-case review was not run. The report says not to train from this sample.

That failed method is uncurated targeted Temporal Grounding letter-SFT, including `b2_frames32.yaml`. Its 32-frame budget and Stage A initialization do not repair those labels. `fit_A` Temporal Grounding rows are also not the same set as the 913-row audit population, and this checkout cannot count the difference because `data/` is absent.

What the failure does not decide: Stage A already trained on mixed tasks and is not retracted. Perspective and Change Detection labels were not in this audit. The official A–D letters remain the evaluation target. The 10 supported ids are not a training allowlist.

## Pixel budget

The live `code/common.py` defines `video_size_for`. For 16 frames and no per-frame override it returns `longest_edge = 16 * 640 * 640`, the same cap as `VIDEO_SIZE`. The motion configs use that 16-frame size. A 32-frame call would scale the cap to `32 * 640 * 640`. Temporal configs stay blocked, so that path is not submitted. `preserve_frame_budget` still aborts a sample above 12000 tokens instead of dropping frames. No GPU run has measured that token count.

## Smoke work and wall cap

`code/smoke.py` is not the gate. Without `explicit_optimizer` it backpropagates and then zeroes gradients.

The motion diagnostic is `code/configs/smoke_b1_hint16.yaml`: `max_steps: 5` and `grad_accum: 1`, so five training microbatches. The previous request was 8 steps at `grad_accum: 16`, which is 128 microbatches. The one-epoch motion specialist still uses `grad_accum: 16`. The diagnostic does not reproduce that accumulation; one backward is enough to see whether the backward kernel and one AdamW write run.

`jobs/a100_train_smoke.sbatch` asks for `00:45:00` and does not resubmit. Probe 435943 put the model on CUDA at 368 seconds and finished one forward at 540 seconds. The forward itself, after encode, was 146.7 seconds. Load plus five forwards of that measured forward is 1102 seconds (18.4 minutes) and still includes no backward. A 20-minute cap sits only about 98 seconds above that lower bound, so it is too tight. Forty-five minutes is the ceiling already used by `jobs/cuda_probe.sbatch`. It is a kill cap, not an ETA. A stop before `KERNEL_SMOKE_STEP_OK` is inconclusive because backward time is unmeasured.

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

Resources on every new GPU wrapper: partition `gpu_a100_8`, `--gres=gpu:nvidia_a100-sxm4-80gb:1`, 12 CPUs, 128G RAM. The motion diagnostic asks for 45 minutes and does not resubmit. The one-epoch motion train and score jobs still ask for `11:50:00` with a 15-minute USR1 checkpoint. A resubmit of those jobs resumes the same `SLURM_JOB_ID` directory. A new submission creates a new timestamped directory. The existing `runs/b1_q3vl8b_hint16` directory is not an output of these jobs.

## Data rules

Training reads only `data/splits/fit_A.json`, then keeps the task names in the config. `require_split: fit_A` and the existing `train_*` id check both have to pass. Validation and test files are not training inputs.

- Motion specialist (`b1_hint16`): `Perspective and Viewpoint`, `Change Detection`, `frame_budget: 16`, `data/camera_motion_hints.json`.
- Temporal recipe (`b2_frames32`): recorded, not launchable. `Temporal Grounding`, `frame_budget: 32`, no motion hints, `training_blocked: EXP-010`.
- Mirror ablation (`b1_hint16_mirror`): secondary. Geometric Relation, Perspective and Viewpoint, and Change Detection. A row is flipped only when left/right can be swapped and the gold text lands on exactly one rewritten option. Clockwise, forward/back, up/down, and similar wording skips that row. Score the ablation without flipping.

Hints must be a flat video-path map. Strings must use the audited `Camera motion:` vocabulary from the ORB builder, or `Camera motion: unavailable`. Object values are accepted only through HPC `common.motion_hint_text`. Wrapped documents, QA-id keys, and answer language fail the audit. Row counts are whatever the audit prints on the cluster; this checkout does not contain `data/`.

`valQ` / `valT` are usable for a specialist only when the eligibility audit says `fit_A` shares no QA ids with them and no video with `valT`. `valQ` can still share videos with `fit_A`; that is question holdout, not video holdout. Stage B trained on `full_train`, so A+B comparisons on `valQ` or `valT` are refused. Compare A+B on official `data/ReVA/val.json`, and only if that file is also disjoint from `full_train`.

The routing rule, fixed before any specialist score exists: on the target-task slice, net fixes (fixes minus regressions) at least 2 and regressions at most 1, with every target row present in both files. A pass writes `routed_<split>_<family>.jsonl`. A failure writes only the gate JSON. This is a routing policy, not an accuracy claim. Task-filtered score files are incomplete for `code/evaluate.py`; use that evaluator only on a full split.

## Baseline paths

`results/ensemble_ab_20261105/` is the reported Nov 5 A+B artifact directory. It is not in this checkout, so no filename inside it is known here.

`tools/ensemble_scores.py` and `tools/paired_gate.py` are HPC-local scripts. They are not the score file. This repo does not contain either script.

`code/score.py` writes JSONL records with `qa_id`, `p` (four floats), `pred` (A–D), and `views`. Stage A legacy copies cited by the handover are `runs/stageA_q3vl8b_r16/legacy/scores_{valQ,valT,val}.jsonl`. `code/route_specialist.py` can also read a list of `{qa_id, pred}` records, a qa-id-to-letter object, or `{"predictions": {qa_id: letter}}`. Point `--baseline` at a file under the Nov 5 directory after that directory is listed. Do not point it at the generator script.

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

# These two must print the block and exit non-zero. Do not submit them.
python code/specialist_data.py assert-training-allowed code/configs/smoke_b2_frames32.yaml
python code/specialist_data.py assert-training-allowed code/configs/b2_frames32.yaml

# Motion diagnostic (5 microbatches, 45-minute kill cap), then the 1-epoch motion specialist.
python code/specialist_data.py assert-training-allowed code/configs/smoke_b1_hint16.yaml
sbatch jobs/a100_train_smoke.sbatch code/configs/smoke_b1_hint16.yaml
# after that job's SMOKE_OK:
sbatch jobs/a100_train_specialist.sbatch code/configs/b1_hint16.yaml
```

Score only after `DONE` in that run's timestamped directory. `RUN_DIR` is the printed `runs/specialists/.../<stamp>_<jobid>` path.

```bash
# Motion specialist: official val is the A+B gate; valT is the video holdout vs Stage A.
sbatch jobs/a100_score_specialist.sbatch "$RUN_DIR" val valT

# BASELINE is a prediction file inside results/ensemble_ab_20261105/, not tools/ensemble_scores.py.
bash jobs/a100_route_specialist.sh "$RUN_DIR" val "$BASELINE" ensemble_ab

# Optional Stage A comparison on valT, only if eligibility_audit.json says the holdout is eligible.
bash jobs/a100_route_specialist.sh "$RUN_DIR" valT "$STAGE_A_VALT_JSONL" stageA
```

Mirror stays off that sequence:

```bash
python code/specialist_data.py validate-config code/configs/b1_hint16_mirror.yaml --allow-unresolved
sbatch jobs/a100_train_specialist.sbatch code/configs/b1_hint16_mirror.yaml
```

## Deploy

Transfer `deploy/a100_motion_overlay/` to the executor. From that directory, with the live tree still matching `MANIFEST_BEFORE.sha256`:

```bash
bash deploy.sh --work /scratch/ambpdc/soham/ReVA
bash preflight_motion.sh --work /scratch/ambpdc/soham/ReVA
cd /scratch/ambpdc/soham/ReVA
sbatch jobs/a100_train_smoke.sbatch code/configs/smoke_b1_hint16.yaml
```

Submit the one-epoch job only after that run's directory contains `SMOKE_OK`:

```bash
cd /scratch/ambpdc/soham/ReVA
sbatch jobs/a100_train_specialist.sbatch code/configs/b1_hint16.yaml
```

Rollback, only if the installed files still match `MANIFEST_AFTER.sha256`:

```bash
bash rollback.sh --work /scratch/ambpdc/soham/ReVA
```

## What still waits for the executor

1. The deploy script must see the before-hashes. If the live files changed after this archive, it refuses.
2. `preflight_motion.sh` rechecks the Stage A weight hash, the 14119/595/1059/15773 split counts, and hints on all 2726 motion rows.
3. Backward time is still unmeasured. The 45-minute diagnostic either writes `KERNEL_SMOKE_STEP_OK` or stops inconclusive.
4. `results/ensemble_ab_20261105/` filenames are still not in this archive. Scoring and the routing gate wait until the diagnostic and the one-epoch run exist.

Do not submit `smoke_b2_frames32.yaml` or `b2_frames32.yaml`. Do not delete `runs/b1_q3vl8b_hint16`. This branch is the overlay source, not a tree to pull.
