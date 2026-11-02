# DSE-VTG on HPC (H200 / H100 / A100) — requirements

DSE-VTG is **not implemented** in this repo yet (paper: [arXiv:2609.08850](https://arxiv.org/abs/2609.08850); no public code). Charades grounding already has multi-GPU Slurm wrappers; DSE should mirror that pattern once code exists.

## What you need before any DSE-VTG batch

| Item | Location / action |
| --- | --- |
| **Frozen Stage A adapter** | `$WORK/runs/stageA_q3vl8b_r16/final_adapter` (already on HPC) |
| **ReVA data + splits** | `$WORK/data/ReVA`, `$WORK/data/splits/` |
| **Qwen3-VL-Embedding-8B** | Download into `$WORK/hf_cache/hub` (HF token; accept model license). Stage A scorer stays Qwen3-VL-8B-Instruct; DSE adds a **second** embedder for frame/clip similarity. |
| **Implementation** | New module e.g. `archive/implementation/code/dse_vtg_decoder.py` + scorer or postprocessor: 3 fps (or configurable) frame+clip embeddings, MSF fusion, optional query TTA, map interval → A–D option scores, **gated** fuse with letter logits (do not repeat EXP-036 unconditional α). |
| **Cohort** | TG-only IDs: `results/visual_prompt_pilot/reze_tg_ids.json` (or full valQ/valT). |
| **Isolated run dirs** | Per GPU type, same as Charades: `runs/dse_vtg_h200/`, `runs/dse_vtg_h100/`, `runs/dse_vtg_a100/` |
| **Slurm wrappers** | Copy Charades pattern: `dse_vtg_score_{h200,h100,a100}.sbatch` → partition/GRES from `qwen38_*` jobs |

## Slurm GRES (BITS HPC, verified)

| GPU | Partition | GRES |
| --- | --- | --- |
| H200 | `gpu_h200_8` | `gpu:nvidia_h200_nvl:1` |
| H100 | `gpu_h100_4` | `gpu:nvidia_h100_80gb_hbm3:1` |
| A100 | `gpu_a100_8` | `gpu:nvidia_a100-sxm4-80gb:1` |

## Charades (ready now)

Shared body: `jobs/charades_grounding_train_body.sh`  
Submit all three:

```bash
export WORK=/scratch/ambpdc/soham/ReVA
bash jobs/submit_charades_grounding_multigpu.sh
```

Prerequisites on `$WORK`: prepared Charades pilot data (`runs/charades_grounding/data/manifest.json`), `code/charades_grounding.py`, offline HF cache with Qwen3-VL-8B-Instruct.

## DSE-VTG (after implementation)

1. CPU/GPU smoke: embed one ReVA clip at 3 fps, print similarity curve shape.  
2. Score TG cohort with gated fusion → new JSONL under `runs/dse_vtg_<gpu>/`.  
3. Compare TG numerators vs Stage A on same IDs; register as EXP-039+.

Optional: run **embedding extraction on CPU** and fusion on GPU to save queue time — not required for v1.
