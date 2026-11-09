# ReVA live status

Last verified: 2026-11-08. This is the only status file; update it at the end of each session. Reference material: `docs/reference/`.

## Results to date

- **Baseline EXP-001:** valQ 498/595, valT 885/1,059, val 1,757/2,000, private test 83.95%. Gate subset S (TG + Change Detection + Trend and Pattern, valQ+valT): 386/487.
- **EXP-028:** Qwen3.8 valQ 506/595, valT incomplete (658/1,059). **EXP-029-030:** test 83.75% reported, provenance open.
- **Closed (no promotion):** 004, 007-010, 011-027, 031-036. EXP-005 Stage B not run locally.
- **Ensemble A/B (2026-11-05):** `results/ensemble_ab_20261105/`.

## Round 1 (B1 hints @16f, B2 frames @32f)

### Root cause of the H100/H200 crash (found 2026-11-09)

`RuntimeError: CUDA driver error: invalid argument ... cuModuleLoadData: image cannot be NULL`, raised at
`(image_grid_thw.prod(-1) // merge**2).tolist()` in Qwen3-VL `get_image_features`.

- `/home/ambpdc` is at its quota (42.02G used; 40G soft / 42G hard). It is a shared account; nothing there was deleted.
- On Nov 3 PyTorch NVRTC-compiled the int64 `prod` reduction for sm_90 and failed to write it under the full home:
  `~/.cache/torch/kernels/reduction_prod_arch9.0_nvrtc13.0_...` was **0 bytes** (same happened Oct 11 with nvrtc 12.6).
- Every later Hopper run loaded that empty file as a NULL module image. The sm_80 (A100) entry is valid, which is why
  only A100 passed. `CUDA_MODULE_LOADING=EAGER` was tested in job 438659 (H200) and did not help: removed.

### Fix

- `jobs/activate_reva.sh` (sourced by every ReVA job): `XDG_CACHE_HOME`, `PYTORCH_KERNEL_CACHE_PATH`, `TRITON_CACHE_DIR`
  under `$WORK/.cache`; `USE_PYTORCH_KERNEL_CACHE=0` (compile in memory per process); quarantine 0-byte cache files.
- The two empty home entries were renamed to `*.empty`. Regression check in `tests/test_jobs.sh`.
- Reverted the other Nov 8 misdiagnosis changes (commit 2c0de69): B1/B2 back to `sdpa` (were `eager`), flash and
  mem-efficient SDPA plus cuDNN back on (only cuDNN-SDPA off, as in EXP-001), `expandable_segments:True` restored.
  Evidence: A100 B1 job 438670 ran at ~1070 s/step (est. 520 h, OOM retries) vs EXP-001's ~8.9 s/step; cancelled.
- Main slowdown (found 2026-11-09): `TORCH_CUDNN_V8_API_DISABLED=1` / `CUDNN_FRONTEND_DISABLE=1` from the Nov 8
  `train.sbatch`. Slurm keeps the submit-time script, so job 438671 still had them and ran ~2,500 s/step. py-spy showed
  it pinned in the Qwen3-VL `Conv3d` patch embed. `tools/bench_patch_embed.py` on A100: 15 ms normal, 4.2 s with those
  vars, 201 s with cuDNN off. `common.check_fast_conv_backend` now makes train/smoke refuse to start in that state.
- Separate fix (Nov 8): `hf_cache` snapshot symlinks were re-linked (blobs intact); confirmed by job 438659 loading weights.

### Jobs (submitted 2026-11-09)

| Job | What | GPU | Gate |
| --- | --- | --- | --- |
| 441171 | `jobs/verify_kernel_cache.sbatch` | H200 | |
| 441172 | `jobs/verify_kernel_cache.sbatch` | H100 | |
| 441668 | B1 `b1_hint16.yaml` | H200 `-c 4` | afterok 441171 |
| 441667 | B2 `b2_frames32.yaml` | H100 `-c 12` | afterok 441172 |
| 441565 / 441666 | B1 / B2 `*_a100.yaml` (separate output dirs) | A100 | |

All four training jobs were (re)submitted with the clean `train.sbatch` (checked with `scontrol write batch_script`). Check step time in the first `benchmark steps=` line (expect ~9 s at 16f).

Verify pass = `VERDICT A=REPRODUCED B=PASS C=PASS`. If a verify job fails, its dependent training job stays
pending as `DependencyNeverSatisfied`: read `reva_kcache_verify_<id>.out` before cancelling.

- Cancelled as too slow: 438670, 438671; replaced with stale script: 438660, 441173.
- Earlier failures: 431617, 432115, 432116, 432729, 432730, 438659 (all this crash); 437051 (empty HF snapshot).
- Cluster rules: H200 max 4 CPUs/GPU (`tests/test_jobs.sh` still asserts `-c 12` in the script header, so pass `-c 4`
  on the command line); multi-partition requests rejected.
- B1 smoke PASS on Brev; B2 smoke incomplete on Brev (Trainer stall ~8/20 steps). Risk: 4 CPUs on H200 may slow decode.

## Cost and token reference

- 16f mean step 8.86 s (H200, EXP-001 job 392966; ~4.35 h for the 2-epoch recipe). 32f is ~2x tokens (cap 16,920 mean, 25,600 max).
- Visual tokens at 16f: mean/p95/max 8480 / 12800 / 12800. `temporal_patch_size=2`; per-frame `<N seconds>` tags present at 16f and 32f.

## Where things live

- **Local** (`/home/soham/ReVA`, git, origin `Soham-47/ReVA`): source of truth for `code/ jobs/ tests/ tools/ docs/ experiments/`.
- **HPC** (`/scratch/ambpdc/soham/ReVA`, not git): runs experiments. Holds `data/ runs/ venv312/ envs/ hf_cache/`. Do not clean it. Round-1 jobs queued as of 2026-11-09 (table above).
- Sync with `tools/sync_to_hpc.sh` (push code, dry-run default, never deletes) and `tools/pull_from_hpc.sh` (pull docs/results).
