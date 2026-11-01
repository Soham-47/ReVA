# Project context

## Experiments

Thirty-six numbered experiments (**EXP-001–036**). Each has one comparable writeup:

- [`experiments/INDEX.md`](../experiments/INDEX.md)
- [`experiments/TEMPLATE.md`](../experiments/TEMPLATE.md)

Every writeup repeats the **shared architecture** and states **what this experiment changed**.

Machine-readable copies: [`results/experiments.json`](../results/experiments.json), [`results/registry.csv`](../results/registry.csv).

One-off code is under **`archive/implementation/`**. Active `code/` is the Stage A/B train/score/evaluate path only.

## Active code

`download_data` → `build_manifest` → `make_splits` → `train.py` → `score.py` → `evaluate.py` (`code/common.py`, `code/configs/stageA.yaml`, `stageB.yaml`).

## Splits & scoring

valQ 595 · valT 1,059 · val 2,000 · test 4,000 hidden. Four-view TTA (`pred` in score JSONL). Details: [`evaluation.md`](evaluation.md).

## Pipeline

[`architecture.md`](architecture.md).
