# Next plan state

Last updated: 2026-11-02 (UTC+5:30 evening)

## Baseline (EXP-001)

| Split | Correct/total |
| --- | --- |
| valQ | 498/595 |
| valT | 885/1059 |
| val | 1757/2000 |
| private test | 83.95% (fallback) |

Gate subset **S** (TG + Change Detection + Trend and Pattern on valQ+valT): **386/487**.

## Prompt 0 / video path (Brev, transformers 5.19.0)

- Per-frame tags: `<N seconds>` present at 16f and 32f.
- `temporal_patch_size=2`.
- Visual tokens (200× fit_A): 16f mean/p95/max **8480 / 12800 / 12800**; 32f cap **16920 / 25600 / 25600**.

## Cost benchmark (reference)

| Source | 16f mean step | Notes |
| --- | --- | --- |
| HPC EXP-001 job 392966 (H200) | **8.860 s** @ 50 steps | `estimated_train_hours=4.35` for 2-epoch recipe |
| Brev in-process 16f (50 opt steps) | ~14.8 s | Trainer path stalled on Brev after ~1 step |
| Brev B2 smoke Trainer | ~37–44 s/step | 32f |

## Round-1 smoke (Brev `jobs/smoke_round1_body.sh`)

| Arm | Status | Notes |
| --- | --- | --- |
| **B1** `smoke_b1_hint16.yaml` | **PASS** | 5 prompts: `motion_hint_in_prompt=True`, 16f + timestamps; ckpt-10 STOP + resume → ckpt-20; loss finite; downsample **0**; **20/20** valQ scored with `--motion-hints` |
| **B2** `smoke_b2_frames32.yaml` | **INCOMPLETE** | Prompts OK; training **stuck ~8/20** steps on Brev (same Trainer stall as earlier); no final score yet |

Production configs (not started before this launch): `code/configs/b1_hint16.yaml`, `code/configs/b2_frames32.yaml`.  
`b3_both` / `r3_merger_r64` configs **not created yet**.

## Active training launch (approved)

| Field | Value |
| --- | --- |
| **RUN** | `b1_hint16` |
| **Config** | `code/configs/b1_hint16.yaml` |
| **Config SHA256** | `90140b84b0937036682d22d770863d1b58011d7a225f531d516c1b7416b76544` |
| **output_dir** | `runs/b1_q3vl8b_hint16` |
| **Slurm job** | **430018** |
| **Partition / GPU** | `gpu_h100_4`, 1× H100 80GB |
| **Command** | `sbatch -p gpu_h100_4 --gres=gpu:nvidia_h100_80gb_hbm3:1 -t 11:50:00 jobs/train.sbatch code/configs/b1_hint16.yaml` |
| **Submitted** | 2026-11-02 (after B1 smoke PASS; B2 smoke still open on Brev) |
| **Wall estimate** | ~**4.3–5 h** one chunk (1766 optimizer steps × ~8.9–10 s/step from H200 EXP-001 16f; ≪ 11 h50m) |
| **Requeue** | `jobs/train.sbatch` USR1 @ 11h50m; expect **1 chunk** unless step time ≫ benchmark |

### 10-minute health check (job 430018)

**2026-11-02 ~10 min after submit:** still **PENDING (Priority)**; no `reva_train_430018.out` yet — loss/step time **not observable** until the job runs.

### Completion (when `$WORK/runs/b1_q3vl8b_hint16/DONE` exists)

_Pending: final_adapter path, total hours, step-time from log._

## HPC ops

- Venv: `$WORK/venv312` (bootstrap job 429737).
- Motion cache: `$WORK/data/camera_motion_hints.json` (1047 videos, 20 unclear).
- Cancelled pending cost bench **429863** so only **430018** is the approved GPU train job.

## Open issues

- B2 smoke on Brev: finish or re-run on HPC before launching `b2_frames32` full train.
- Brev Trainer 16f/32f micro-runs can hang; full train uses HPC + `dataloader_workers: 8` (watch first steps in `reva_train_430018.out`).
