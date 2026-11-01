# Archived implementation

Historical experiment **code**, **jobs**, and **tests** moved out of the active tree.

**Experiment knowledge is not here** — it lives in [`experiments/INDEX.md`](../../experiments/INDEX.md).

## Layout

| Directory | Contents |
| --- | --- |
| `code/` | Overlays, grounding, temporal, Phase 6–8, dual-stream, REZE, visual prompts |
| `jobs/` | Brev/Modal/Replicate scripts and old Slurm wrappers |
| `tests/` | Unit tests that imported archived modules |
| `tasks/` | Old task manifests |
| `docs/` | One-off research notes |

## Active root code

`code/common.py`, `download_data.py`, `build_manifest.py`, `make_splits.py`, `train.py`, `score.py`, `evaluate.py`, `make_submission.py`, `smoke.py`, `diagnose_gpu.py`, `build_evaluation_registry.py`. Dual-stream training still imports `archive/implementation/code/temporal_grounding.py` when `temporal_stream.enabled` is set.

To rerun an archived experiment, copy the needed files back to `code/` and `jobs/` or run with `PYTHONPATH=archive/implementation/code:code`.
