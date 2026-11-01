# Evaluation conventions

Agent context: [`CONTEXT.md`](CONTEXT.md) · Index: [`experiments/INDEX.md`](experiments/INDEX.md)

## Splits

| Split | Questions | Role |
| --- | ---: | --- |
| `fit_A` | train subset | LoRA training only |
| `valQ` | 595 | Question holdout |
| `valT` | 1,059 | Video holdout |
| `val` | 2,000 | Official validation JSON |
| `test` | 4,000 | Hidden labels; submission only |

Never treat `valQ`, `valT`, and `val` as interchangeable without noting the split.

## Scoring protocol (Stage A reference)

- Letter-logit prediction with **test-time augmentation**: offsets `{0, 0.5}` × two cyclic option permutations → **four views**
- **TTA accuracy** uses the aggregated `pred` field in each score JSONL row (see `code/evaluate.py`)
- **Single-view accuracy** uses the first view only

Phase 8 uses the same four-view protocol when manifests mark `frame_sampling: legacy`.

## Metrics in the registry

| `metric_name` | Meaning |
| --- | --- |
| `overall_accuracy_tta` | Full-split TTA accuracy |
| `temporal_grounding_task_accuracy` | Subset of temporal-task questions only (Phase 2) |
| `overall_accuracy` | Test or other splits without TTA detail in artifact |

`evaluation_kind` values:

- `full_split_tta` — comparable across arms on that split
- `partial_split` — incomplete rows (e.g. Phase 8 valT prefix)
- `selected_pilot` — fixed small cohort (12–32 questions)
- `user_reported` / `leaderboard_reported` — no local JSONL

## Central records

1. [`experiments/INDEX.md`](../experiments/INDEX.md) — comparable writeups
2. [`results/evaluations.csv`](../results/evaluations.csv) — one row per metric instance
3. [`results/registry.csv`](../results/registry.csv) — experiment metadata
4. [`results/comparisons/overall.md`](../results/comparisons/overall.md) — generated tables

Regenerate after adding summaries or JSONL:

```bash
python code/build_evaluation_registry.py
```

Do **not** hand-edit generated comparison Markdown; edit the builder or source artifacts instead.

## Paired comparisons

When both arms have complete score JSONL on the same split:

- Phase 6: `runs/phase6_visual_adaptation_pilot/paired_report.md` (video-cluster bootstrap)
- Phase 8: `archive/implementation/code/phase8_analysis.py` (requires manifests + full coverage)

Report fixes, regressions, and net accuracy delta — not task-only slices as overall accuracy.
