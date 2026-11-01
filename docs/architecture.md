# Pipeline architecture

Agent context: [`CONTEXT.md`](CONTEXT.md) · Experiments: [`experiments/INDEX.md`](../experiments/INDEX.md)

## Task

Four-option video multiple-choice QA (ReVA drone benchmark). Each example: one clip, question, options A–D → predict one letter.

## Shared inference path (every future Stage A/B run)

```text
UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + language LoRA r16
  → four-view TTA (offsets 0/0.5 × two option rotations) → letter-logit argmax
```

```text
download_data → build_manifest → make_splits → train (LoRA) → score (JSONL) → evaluate / make_submission
```

| Stage | Scripts | Config |
| --- | --- | --- |
| Data | `code/download_data.py`, `code/build_manifest.py`, `code/make_splits.py` | — |
| Train | `code/train.py` | `code/configs/stageA.yaml`, `stageB.yaml` |
| Score | `code/score.py` | adapter path, split name |
| Metrics | `code/evaluate.py` | split name |
| Submit | `code/make_submission.py` | test score JSONL |

Shared constants, sampling, prompts: `code/common.py`.

Cluster: `jobs/pipeline.sh`, `jobs/train.sbatch`, `jobs/score.sbatch`.

Historical branches (overlays, grounding, Phase 8, dual-stream) are documented per EXP under `experiments/` and implemented in `archive/implementation/`.

## Runtime layout (cluster)

Heavy assets live under `$WORK` (see root `README.md`): data JSONs, clips, `hf_cache`, adapters, score JSONL, logs.

## Tests

`tests/test_core.py` — registry + Stage A recipe. `tests/test_jobs.sh` — Slurm GRES and smoke-marker guards.
