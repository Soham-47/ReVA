# Agent guide (ReVA)

## Experiment knowledge

Canonical writeup per experiment: [`experiments/EXP-NNN/experiment.md`](experiments/INDEX.md) (shared architecture + what changed). Schema: [`experiments/TEMPLATE.md`](experiments/TEMPLATE.md).

| Source | Purpose |
| --- | --- |
| [`experiments/INDEX.md`](experiments/INDEX.md) | EXP-001–039 index |
| [`results/experiments.json`](results/experiments.json) | Machine-readable metadata |
| [`results/registry.csv`](results/registry.csv) | Tabular index with `exp_number` |
| [`results/evaluations.csv`](results/evaluations.csv) | Metrics with denominators |
| [`results/comparisons/overall.md`](results/comparisons/overall.md) | Generated tables |

Refresh: `python code/build_evaluation_registry.py` (does not overwrite `experiments/EXP-*/experiment.md` or `runs/` scores).

**Baseline:** EXP-001 — valQ 498/595, valT 885/1,059, val 1,757/2,000.

## Minimal repo layout

```text
code/              Core pipeline only (every Stage A/B run)
jobs/              pipeline.sh, train.sbatch, score.sbatch, submit_stageB.sh, wait_child.sh
experiments/       One folder per EXP with experiment.md
archive/implementation/  One-off experiment code, jobs, tests
runs/              Artifacts (scores, reports) — do not overwrite
results/           registry, evaluations, experiments.json, comparisons
docs/              CONTEXT, NEXT_PLAN_STATE (live status), architecture, evaluation, reference/
tests/             Core pipeline + job syntax
```

See [`docs/CONTEXT.md`](docs/CONTEXT.md).

For multi-agent / harness continuity (paths, Brev/HPC, TG remediation queue): [`docs/reference/HARNESS_HANDOVER.md`](docs/reference/HARNESS_HANDOVER.md) (read only on request).

## Rules

1. No GPU training/scoring/downloads without user approval.
2. Do not change evaluation math or `runs/` score files during cleanup.
3. Do not claim test/Phase 8 wins without EXP-028–030 provenance.
4. One-off experiment code lives under `archive/implementation/`.

## Verify

```bash
python3 code/build_evaluation_registry.py
python3 -m py_compile code/*.py
bash tests/test_jobs.sh
python3 -m unittest discover -s tests -p 'test_*.py' -q
```
