# ReVA live status

Last verified: 2026-11-08. This is the only status file; update it at the end of each session. Reference material: `docs/reference/`.

## Results to date

- **Baseline EXP-001:** valQ 498/595, valT 885/1,059, val 1,757/2,000, private test 83.95%. Gate subset S (TG + Change Detection + Trend and Pattern, valQ+valT): 386/487.
- **EXP-028:** Qwen3.8 valQ 506/595, valT incomplete (658/1,059). **EXP-029-030:** test 83.75% reported, provenance open.
- **Closed (no promotion):** 004, 007-010, 011-027, 031-036. EXP-005 Stage B not run locally.
- **Ensemble A/B (2026-11-05):** `results/ensemble_ab_20261105/`.

## Round 1 (B1 hints @16f, B2 frames @32f): BLOCKED, nothing has trained

- **B1** smoke PASS on Brev (hints in prompt, 16f, checkpoint resume, 20/20 valQ). **B2** smoke incomplete on Brev (Trainer stall ~8/20 steps).
- **HPC jobs 431617, 432115, 432116, 432729, 432730 all failed** in the `code/smoke.py` preflight after ~5-7 min, on H100:
  `RuntimeError: CUDA driver error: invalid argument ... image cannot be NULL (cuModuleLoadData)` inside Qwen3-VL `get_image_features`.
- Env: `torch 2.14.1+cu130`, `transformers 5.19.0`, driver 580.126.20, archs sm_75/80/86/90/100/120.
- **Only passing run:** CUDA probe 435943 (A100, Nov 7) with `CUDA_MODULE_LOADING=EAGER`. Every lazy-loading run on H100/H200 failed. A100 probes 434916/435040 failed for unrelated reasons.
- **Fix applied (unverified until a job starts):** `CUDA_MODULE_LOADING=EAGER` in `jobs/train.sbatch` and `jobs/score.sbatch` (not `CUDA_LAUNCH_BLOCKING`).
- **Second blocker, fixed 2026-11-08:** `hf_cache/.../snapshots/0c351dd.../` was empty since Nov 7 11:59 (blobs intact), so offline loading failed (this killed A100 job 437051). Re-linked 12 files by content; processor/config/snapshot load offline and 750/750 tensors present.
- **Submitted 2026-11-08:** B1 job **438659** (H200, `-c 4`), B2 job **438660** (H100, `-c 12`). Output dirs `runs/b1_q3vl8b_hint16`, `runs/b2_q3vl8b_f32`. Scheduler start estimates (Nov 21 / Nov 25) are pessimistic backfill numbers.
- **Also submitted (A100, separate output dirs `*_a100`):** B1 **438670**, B2 **438671** (`code/configs/b{1,2}_*_a100.yaml`). Estimated start ~Feb 2027; same recipe, so duplicates of 438659/438660 unless those fail.
- Cluster rules: H200 allows max 4 CPUs/GPU (job 430081 used 12 in early Nov); multi-partition requests are rejected; A100 estimate is Feb 2027.
- **If smoke fails again with the same CUDA error:** try `CUDA_LAUNCH_BLOCKING=1` via `cuda_probe.sbatch` A/B (jobs were cancelled before running).
- Risk: 4 CPUs on H200 may slow video decoding vs the 8.86 s/step benchmark.

## Cost and token reference

- 16f mean step 8.86 s (H200, EXP-001 job 392966; ~4.35 h for the 2-epoch recipe). 32f is ~2x tokens (cap 16,920 mean, 25,600 max).
- Visual tokens at 16f: mean/p95/max 8480 / 12800 / 12800. `temporal_patch_size=2`; per-frame `<N seconds>` tags present at 16f and 32f.

## Where things live

- **Local** (`/home/soham/ReVA`, git, origin `Soham-47/ReVA`): source of truth for `code/ jobs/ tests/ tools/ docs/ experiments/`.
- **HPC** (`/scratch/ambpdc/soham/ReVA`, not git): runs experiments. Holds `data/ runs/ venv312/ envs/ hf_cache/`. Do not clean it. No queued or running ReVA jobs as of 2026-11-08.
- Sync with `tools/sync_to_hpc.sh` (push code, dry-run default, never deletes) and `tools/pull_from_hpc.sh` (pull docs/results).
