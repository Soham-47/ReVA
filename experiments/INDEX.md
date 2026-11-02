# Experiments

One folder per experiment. Canonical writeup: `experiments/EXP-NNN/experiment.md`.
Schema: [`TEMPLATE.md`](TEMPLATE.md). Metrics: [`../results/evaluations.csv`](../results/evaluations.csv).

**Shared architecture:** 16-frame Qwen3-VL-8B LoRA r16, four-view TTA, letter-logit scoring.

| EXP | ID | Family | Decision | Doc |
| ---: | --- | --- | --- | --- |
| [001](EXP-001/experiment.md) | `reva-stage-a-legacy` | `baseline_sampling` | `active_reference` | [EXP-001](EXP-001/experiment.md) |
| [002](EXP-002/experiment.md) | `reva-error-review-baseline` | `baseline_sampling` | `diagnostic` | [EXP-002](EXP-002/experiment.md) |
| [003](EXP-003/experiment.md) | `reva-endpoint-frames` | `baseline_sampling` | `not_promoted` | [EXP-003](EXP-003/experiment.md) |
| [004](EXP-004/experiment.md) | `reva-adaptive-siglip2-frames` | `baseline_sampling` | `closed_no_promotion` | [EXP-004](EXP-004/experiment.md) |
| [005](EXP-005/experiment.md) | `reva-stage-b-planned` | `baseline_sampling` | `planned` | [EXP-005](EXP-005/experiment.md) |
| [006](EXP-006/experiment.md) | `reva-prior-private-test` | `baseline_sampling` | `historical_reference` | [EXP-006](EXP-006/experiment.md) |
| [007](EXP-007/experiment.md) | `reva-phase2-temporal-t0` | `temporal` | `reference_arm` | [EXP-007](EXP-007/experiment.md) |
| [008](EXP-008/experiment.md) | `reva-phase2-temporal-coarsefine` | `temporal` | `gate_failed` | [EXP-008](EXP-008/experiment.md) |
| [009](EXP-009/experiment.md) | `reva-phase2-verified-window` | `temporal` | `inconclusive_pilot` | [EXP-009](EXP-009/experiment.md) |
| [010](EXP-010/experiment.md) | `reva-temporal-train-audit` | `temporal` | `gate_failed_no_training` | [EXP-010](EXP-010/experiment.md) |
| [011](EXP-011/experiment.md) | `reva-grounding-dino-pilot` | `grounding_geometry` | `not_promoted` | [EXP-011](EXP-011/experiment.md) |
| [012](EXP-012/experiment.md) | `reva-sam2-segmentation-pilot` | `grounding_geometry` | `not_promoted` | [EXP-012](EXP-012/experiment.md) |
| [013](EXP-013/experiment.md) | `reva-sam3-pilot` | `grounding_geometry` | `not_promoted` | [EXP-013](EXP-013/experiment.md) |
| [014](EXP-014/experiment.md) | `reva-phase3-explicit-geometry` | `grounding_geometry` | `closed_no_promotion` | [EXP-014](EXP-014/experiment.md) |
| [015](EXP-015/experiment.md) | `reva-phase3a-retrieval-smoke` | `grounding_geometry` | `exploratory` | [EXP-015](EXP-015/experiment.md) |
| [016](EXP-016/experiment.md) | `reva-phase3b-gates` | `grounding_geometry` | `no_qwen_promotion` | [EXP-016](EXP-016/experiment.md) |
| [017](EXP-017/experiment.md) | `reva-phase4-qcsrr` | `grounding_geometry` | `led_to_phase5_only` | [EXP-017](EXP-017/experiment.md) |
| [018](EXP-018/experiment.md) | `reva-phase5-crop-e0` | `crop_ablation` | `reference_arm` | [EXP-018](EXP-018/experiment.md) |
| [019](EXP-019/experiment.md) | `reva-phase5-crop-ablation` | `crop_ablation` | `not_promoted` | [EXP-019](EXP-019/experiment.md) |
| [020](EXP-020/experiment.md) | `reva-phase6-a0` | `visual_adaptation` | `reference_arm` | [EXP-020](EXP-020/experiment.md) |
| [021](EXP-021/experiment.md) | `reva-phase6-a1` | `visual_adaptation` | `regressed_vs_a0` | [EXP-021](EXP-021/experiment.md) |
| [022](EXP-022/experiment.md) | `reva-phase6-a2` | `visual_adaptation` | `below_screening_gate` | [EXP-022](EXP-022/experiment.md) |
| [023](EXP-023/experiment.md) | `reva-phase6-a3` | `visual_adaptation` | `regressed_vs_a0` | [EXP-023](EXP-023/experiment.md) |
| [024](EXP-024/experiment.md) | `reva-phase6-a4` | `visual_adaptation` | `regressed_vs_a0` | [EXP-024](EXP-024/experiment.md) |
| [025](EXP-025/experiment.md) | `reva-phase7a-manifest` | `tracking` | `infrastructure` | [EXP-025](EXP-025/experiment.md) |
| [026](EXP-026/experiment.md) | `reva-phase7b-source-evidence` | `tracking` | `gate_failed` | [EXP-026](EXP-026/experiment.md) |
| [027](EXP-027/experiment.md) | `reva-phase7c-replicate-tracking` | `tracking` | `gate_failed` | [EXP-027](EXP-027/experiment.md) |
| [028](EXP-028/experiment.md) | `reva-phase8-qwen38-h200` | `alternative_backbones` | `pending_full_comparison` | [EXP-028](EXP-028/experiment.md) |
| [029](EXP-029/experiment.md) | `reva-phase8-test-package-407542` | `alternative_backbones` | `pending_provenance` | [EXP-029](EXP-029/experiment.md) |
| [030](EXP-030/experiment.md) | `reva-phase8-test-user-reported` | `alternative_backbones` | `pending_provenance` | [EXP-030](EXP-030/experiment.md) |
| [031](EXP-031/experiment.md) | `reva-reference-air-temporal-audit` | `temporal` | `not_promoted` | [EXP-031](EXP-031/experiment.md) |
| [032](EXP-032/experiment.md) | `reva-visual-prompt-2x2` | `training_free_overlay` | `not_promoted` | [EXP-032](EXP-032/experiment.md) |
| [033](EXP-033/experiment.md) | `reva-visual-prompt-tg-geometry-bar` | `training_free_overlay` | `not_promoted` | [EXP-033](EXP-033/experiment.md) |
| [034](EXP-034/experiment.md) | `reva-visual-prompt-trend-bar` | `training_free_overlay` | `not_promoted` | [EXP-034](EXP-034/experiment.md) |
| [035](EXP-035/experiment.md) | `reva-spatial-contract` | `training_free_prompt` | `not_promoted` | [EXP-035](EXP-035/experiment.md) |
| [036](EXP-036/experiment.md) | `reva-reze-cross-tg` | `training_free_prompt` | `not_promoted` | [EXP-036](EXP-036/experiment.md) |
| [037](EXP-037/experiment.md) | `reva-cotr-tg-burnin` | `training_free_overlay` | `not_promoted` | [EXP-037](EXP-037/experiment.md) |
| [038](EXP-038/experiment.md) | `reva-patk-tta-reweight` | `training_free_postprocess` | `not_promoted` | [EXP-038](EXP-038/experiment.md) |
| [039](EXP-039/experiment.md) | `reva-dse-vtg-brev` | `training_free_decoder` | `not_promoted` | [EXP-039](EXP-039/experiment.md) |
