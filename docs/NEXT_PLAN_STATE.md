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
- **Only passing run:** CUDA probe job 435943 (A100, Nov 7) with `CUDA_MODULE_LOADING=EAGER` and `CUDA_LAUNCH_BLOCKING=1`. Earlier A100 probes (434916, 435040) failed for unrelated reasons (script bug, timeout). No train/smoke script sets either variable, and the probe was never run on H100.
- **Two open hypotheses:** (1) lazy CUDA module loading (fix: `CUDA_MODULE_LOADING=EAGER`), (2) H100-specific. They are not yet separated.
- `expandable_segments` and cuDNN v8 were already ruled out in `jobs/train.sbatch`.
- Job 437051 (A100 smoke, Nov 8): failed trying to reach huggingface.co offline. `hf_cache` on HPC is ~51 KB; confirm where the model weights are loaded from.
- **Next step (needs user approval to submit):** run `jobs/cuda_probe.sbatch` (EAGER already set) on `gpu_h100_4`. Pass means set EAGER in `train.sbatch`/`smoke_round1_body.sh`; fail means H100 is the issue, try H200.

## Cost and token reference

- 16f mean step 8.86 s (H200, EXP-001 job 392966; ~4.35 h for the 2-epoch recipe). 32f is ~2x tokens (cap 16,920 mean, 25,600 max).
- Visual tokens at 16f: mean/p95/max 8480 / 12800 / 12800. `temporal_patch_size=2`; per-frame `<N seconds>` tags present at 16f and 32f.

## Where things live

- **Local** (`/home/soham/ReVA`, git, origin `Soham-47/ReVA`): source of truth for `code/ jobs/ tests/ tools/ docs/ experiments/`.
- **HPC** (`/scratch/ambpdc/soham/ReVA`, not git): runs experiments. Holds `data/ runs/ venv312/ envs/ hf_cache/`. Do not clean it. No queued or running ReVA jobs as of 2026-11-08.
- Sync with `tools/sync_to_hpc.sh` (push code, dry-run default, never deletes) and `tools/pull_from_hpc.sh` (pull docs/results).
