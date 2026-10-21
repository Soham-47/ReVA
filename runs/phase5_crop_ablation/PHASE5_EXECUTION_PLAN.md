# Phase 5 — Frozen Qwen3-VL crop-evidence ablation

**Status: completed on the selected BITS HPC; Slurm job 402305 finished successfully with 12/12 questions in each arm.**

## Frozen experiment

Run E0 (video only), E1 (fixed-seed random regions) and E2 (frozen Phase 4 R0 retrieved regions) on the same 12 questions from 12 videos. Keep Stage A frame sampling (16 frames), offsets 0 and 0.5, two answer rotations, original frames, prompt and Stage A adapter. This is 144 answer-logit forward passes. Crop choice is frozen in `crop_selection_manifest.json`; the answer key stays local and is not part of the HPC input packet.

## Selected questions

`train_014154`, `train_005918`, `train_011679`, `train_012957`, `train_008503`, `train_008163`, `train_001478`, `train_010200`, `train_003224`, `train_002092`, `train_012936`, `train_014264`.

E2 uses one question-conditioned R0 rank-1 candidate per entity phrase, target first and capped at four. E1 uses seed `20261018`, draws from the same question-conditioned pool, and matches E2 crop dimensions exactly. There are 23 paired crop slots: 18 exact source-timestamp matches and five nearest-frame matches. Candidate source indices and timestamps were validated against the Phase 4 mapping. The target-prompt proposal pool is empty for `train_002092`; the miss is retained.

## HPC execution

- Host selected by the user: `hpc.bits-hyderabad.ac.in`; run on the fastest listed GPU type, one H200 (`gpu_h200_8`).
- Existing environment: `/scratch/ambpdc/soham/ReVA/envs/reva`; Python 3.12.12, Torch 2.14.1+cu126, Transformers 5.19.0, PEFT 0.21.2, PyAV 19.0.1.
- The pinned public checkpoint revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` is already cached on HPC. The existing Stage A adapter SHA-256 matches `604bff63149edde707267a80353e9ffb080c3bcb4f280e6045101a43ebba7990`.
- Slurm request: one H200, 12 CPUs, 96 GB host memory, one-hour limit. GPU queues were occupied when checked; the job may wait.
- Checkpointed JSONL outputs resume by question. Each completed QA row is flushed and synced. A mixed video/image processor preflight runs before model loading. Runtime, per-arm latency, vision token counts, CUDA peak memory, GPU model and Slurm usage will be recorded.
- HPC allocation charges are not exposed to this account through Slurm. Report measured GPU time and any provider charge only if visible; do not infer an institutional billing amount.

## Artifacts and data separation

- Frozen input packet: `input/` (selected videos, crops, adapter, crop/question/source manifests, and `code/common.py`; no gold answers or Phase 4 annotation labels).
- Local-only answer key: `evaluation_key.json`.
- HPC score implementation: `jobs/phase5_qwen_crop_ablation_hpc.py` and `jobs/phase5_hpc_qwen_crop_ablation.sbatch`.
- Existing Modal implementation retained as an unused reference: `jobs/modal_phase5_qwen_crop_ablation.py`.
- Local contact sheet: `contact_sheets/phase5_crop_pairs.jpg`.
- HPC outputs are isolated under `results_hpc/`, with Slurm logs under `logs/`.
