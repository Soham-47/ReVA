# ReVA

Four-option **drone video QA**. This repository is a small Stage A/B training and scoring pipeline plus a **one-folder-per-experiment** ledger.

Each question is one UAV clip, a question, and options A–D. The model predicts a single letter.

| Split | Size | Role |
| --- | ---: | --- |
| `fit_A` | train subset | LoRA training only |
| `valQ` | 595 | Question holdout |
| `valT` | 1,059 | Video holdout |
| `val` | 2,000 | Official validation |
| `test` | 4,000 | Hidden labels; submission only |

## Status

**Active reference (EXP-001)** — frozen Qwen3-VL-8B language LoRA r16, 16 frames, four-view TTA:

| Split | Accuracy |
| --- | --- |
| valQ | **498/595** |
| valT | **885/1,059** |
| val | **1,757/2,000** |

Stage B (`code/configs/stageB.yaml`) is documented and not run locally (EXP-005). Training-free overlays and prompt variants (EXP-032–036) are closed and not in the production scorer.

Full table: [`experiments/INDEX.md`](experiments/INDEX.md) · narrative status: [`docs/research_status.md`](docs/research_status.md)

## Layout

```text
code/           Train, score, evaluate (used on every Stage A/B run)
jobs/           Slurm: pipeline, train, score, Stage B submit
experiments/    EXP-001 … EXP-036 — one experiment.md each
results/        registry.csv, evaluations.csv, experiments.json
archive/implementation/  One-off experiment code (not the live scorer)
docs/           Architecture, evaluation rules, research status
tests/          Core recipe + job syntax
runs/           Local scores and reports (gitignored; do not overwrite)
data/           Clips and JSON (gitignored)
```

Comparable writeup schema: [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md). Every experiment repeats the **shared architecture** and states **what it changed**.

## Shared architecture

```text
UAV clip → 16-frame sampler → Qwen3-VL-8B-Instruct + LoRA r16
        → four-view TTA (offsets 0 / 0.5 × two option rotations)
        → letter-logit argmax
```

```text
download_data → build_manifest → make_splits → train.py → score.py → evaluate.py
```

Configs: `code/configs/stageA.yaml`, `code/configs/stageB.yaml`. Details: [`docs/architecture.md`](docs/architecture.md).

## Experiments

| Band | EXP | What |
| --- | --- | --- |
| Baseline | [001](experiments/EXP-001/experiment.md)–[006](experiments/EXP-006/experiment.md) | Stage A, audits, endpoints, Stage B recipe, prior test |
| Temporal / AIR | [007](experiments/EXP-007/experiment.md)–[010](experiments/EXP-010/experiment.md), [031](experiments/EXP-031/experiment.md) | Windows, timestamps, A.I.R. selector |
| Grounding / retrieval | [011](experiments/EXP-011/experiment.md)–[019](experiments/EXP-019/experiment.md) | Boxes, masks, QCSRR, crops |
| Adaptation | [020](experiments/EXP-020/experiment.md)–[024](experiments/EXP-024/experiment.md) | Phase 6 LoRA variants |
| Tracking | [025](experiments/EXP-025/experiment.md)–[027](experiments/EXP-027/experiment.md) | Manifest and tracking gates |
| Other backbone | [028](experiments/EXP-028/experiment.md)–[030](experiments/EXP-030/experiment.md) | Qwen3.8-27B (valT incomplete; test provenance open) |
| Training-free | [032](experiments/EXP-032/experiment.md)–[036](experiments/EXP-036/experiment.md) | Progress bar, motion overlay, spatial contract, REZE |

Do not mix valQ, valT, val, test, or selected-pilot denominators. Metrics: [`results/evaluations.csv`](results/evaluations.csv).

## Cluster run

Set `WORK` to the Lustre tree (repo, data, `hf_cache`, adapters, scores).

```bash
export WORK=/path/to/reva_work
sbatch jobs/train.sbatch code/configs/stageA.yaml
sbatch jobs/score.sbatch valQ
sbatch jobs/score.sbatch valT
```

H200 smoke (GRES must match the jobs):

```bash
srun -p gpu_h200_8 --gres=gpu:nvidia_h200_nvl:1 -c 12 -t 0:15:00 --pty bash
python code/smoke.py
```

Full pipeline helper: `jobs/pipeline.sh`. Stage B submit: `jobs/submit_stageB.sh`.

Do not start GPU training or full scoring without an explicit go-ahead.

## Local checks

```bash
python3 code/build_evaluation_registry.py
python3 -m py_compile code/*.py
bash tests/test_jobs.sh
python3 -m unittest discover -s tests -p 'test_*.py' -q
```

The registry builder refreshes `results/registry.csv`, `results/evaluations.csv`, and `experiments/INDEX.md`. It does not overwrite `experiments/EXP-*/experiment.md` or `runs/` score JSONL.

## Evaluation

Letter logits with **four-view TTA**. Accuracy uses the aggregated `pred` field in each score JSONL row (`code/evaluate.py`). Conventions: [`docs/evaluation.md`](docs/evaluation.md).

## Archive

Anything not on the Stage A/B path lives in [`archive/implementation/`](archive/implementation/). Dual-stream training still imports `archive/implementation/code/temporal_grounding.py` when `temporal_stream.enabled` is set in a config.

## Python

See `requirements.txt` and `requirements-torch.txt`. Agents: [`AGENTS.md`](AGENTS.md).
