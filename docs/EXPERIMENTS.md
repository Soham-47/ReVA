# Experiments

Canonical comparable writeups: [`experiments/INDEX.md`](../experiments/INDEX.md).
Each experiment is `experiments/EXP-NNN/experiment.md` (shared architecture + what changed).

Machine-readable: [`results/experiments.json`](../results/experiments.json), [`results/registry.csv`](../results/registry.csv), [`results/evaluations.csv`](../results/evaluations.csv).

| EXP | ID | Title | Decision |
| ---: | --- | --- | --- |
| [001](../experiments/EXP-001/experiment.md) | `reva-stage-a-legacy` | Frozen Stage A Qwen3-VL-8B legacy 16-frame baseline | `active_reference` |
| [002](../experiments/EXP-002/experiment.md) | `reva-error-review-baseline` | Stage A error review and task slices | `diagnostic` |
| [003](../experiments/EXP-003/experiment.md) | `reva-endpoint-frames` | Endpoint-inclusive 16-frame sampling | `not_promoted` |
| [004](../experiments/EXP-004/experiment.md) | `reva-adaptive-siglip2-frames` | SigLIP2 query-adaptive frame selection | `closed_no_promotion` |
| [005](../experiments/EXP-005/experiment.md) | `reva-stage-b-planned` | Stage B full-train recipe (planned) | `planned` |
| [006](../experiments/EXP-006/experiment.md) | `reva-prior-private-test` | Prior private-test submission (Stage A lineage) | `historical_reference` |
| [007](../experiments/EXP-007/experiment.md) | `reva-phase2-temporal-t0` | Phase 2 temporal legacy reference (T0) | `reference_arm` |
| [008](../experiments/EXP-008/experiment.md) | `reva-phase2-temporal-coarsefine` | Coarse-to-fine temporal sampling (T1) and timestamp context (T2) | `gate_failed` |
| [009](../experiments/EXP-009/experiment.md) | `reva-phase2-verified-window` | Reviewed-window temporal pilot (T1R) | `inconclusive_pilot` |
| [010](../experiments/EXP-010/experiment.md) | `reva-temporal-train-audit` | Train-only temporal supervision feasibility audit | `gate_failed_no_training` |
| [011](../experiments/EXP-011/experiment.md) | `reva-grounding-dino-pilot` | GroundingDINO box-context pilot (16 questions per split) | `not_promoted` |
| [012](../experiments/EXP-012/experiment.md) | `reva-sam2-segmentation-pilot` | SAM 2 mask overlay pilot (32 questions) | `not_promoted` |
| [013](../experiments/EXP-013/experiment.md) | `reva-sam3-pilot` | SAM 3 text-prompt segmentation pilots | `not_promoted` |
| [014](../experiments/EXP-014/experiment.md) | `reva-phase3-explicit-geometry` | Explicit geometry G0/G1/G2 (30 questions per split) | `closed_no_promotion` |
| [015](../experiments/EXP-015/experiment.md) | `reva-phase3a-retrieval-smoke` | SAM3 proposal + SigLIP2 retrieval smoke (2 videos) | `exploratory` |
| [016](../experiments/EXP-016/experiment.md) | `reva-phase3b-gates` | Retrieval annotation gates 1–2 | `no_qwen_promotion` |
| [017](../experiments/EXP-017/experiment.md) | `reva-phase4-qcsrr` | QCSRR fixed-frame ranking gate | `led_to_phase5_only` |
| [018](../experiments/EXP-018/experiment.md) | `reva-phase5-crop-e0` | Phase 5 video-only arm | `reference_arm` |
| [019](../experiments/EXP-019/experiment.md) | `reva-phase5-crop-ablation` | Frozen crop-evidence ablation (12-question cohort) | `not_promoted` |
| [020](../experiments/EXP-020/experiment.md) | `reva-phase6-a0` | Visual adaptation reference (frozen Stage A) | `reference_arm` |
| [021](../experiments/EXP-021/experiment.md) | `reva-phase6-a1` | Language-only continuation LoRA | `regressed_vs_a0` |
| [022](../experiments/EXP-022/experiment.md) | `reva-phase6-a2` | Language + merger LoRA | `below_screening_gate` |
| [023](../experiments/EXP-023/experiment.md) | `reva-phase6-a3` | Language + upper vision-block LoRA | `regressed_vs_a0` |
| [024](../experiments/EXP-024/experiment.md) | `reva-phase6-a4` | Language + vision + merger LoRA | `regressed_vs_a0` |
| [025](../experiments/EXP-025/experiment.md) | `reva-phase7a-manifest` | Canonical frame manifest | `infrastructure` |
| [026](../experiments/EXP-026/experiment.md) | `reva-phase7b-source-evidence` | Source-evidence cache validation | `gate_failed` |
| [027](../experiments/EXP-027/experiment.md) | `reva-phase7c-replicate-tracking` | SAM3 / GroundingDINO / SAM2 tracking feasibility | `gate_failed` |
| [028](../experiments/EXP-028/experiment.md) | `reva-phase8-qwen38-h200` | Qwen3.8-27B Stage-A-style scoring (partial valT) | `pending_full_comparison` |
| [029](../experiments/EXP-029/experiment.md) | `reva-phase8-test-package-407542` | H100 test submission package (job 407542) | `pending_provenance` |
| [030](../experiments/EXP-030/experiment.md) | `reva-phase8-test-user-reported` | User-reported 4,000-pair test metrics (83.75%) | `pending_provenance` |
| [031](../experiments/EXP-031/experiment.md) | `reva-reference-air-temporal-audit` | Reference A.I.R. selector on Temporal Grounding | `not_promoted` |
| [032](../experiments/EXP-032/experiment.md) | `reva-visual-prompt-2x2` | Progress-bar and ORB-motion overlays (60-item valT cohort) | `not_promoted` |
| [033](../experiments/EXP-033/experiment.md) | `reva-visual-prompt-tg-geometry-bar` | Progress bar on all Temporal Grounding and Geometric Relation items | `not_promoted` |
| [034](../experiments/EXP-034/experiment.md) | `reva-visual-prompt-trend-bar` | Progress bar on Trend and Pattern | `not_promoted` |
| [035](../experiments/EXP-035/experiment.md) | `reva-spatial-contract` | Spatial-contract prompt on TG and spatial tasks | `not_promoted` |
| [036](../experiments/EXP-036/experiment.md) | `reva-reze-cross-tg` | CROSS prompt and REZE evidence fusion on Temporal Grounding | `not_promoted` |
| [037](../experiments/EXP-037/experiment.md) | `reva-cotr-tg-burnin` | CoTR visual time anchors on Temporal Grounding | `not_promoted` |
| [038](../experiments/EXP-038/experiment.md) | `reva-patk-tta-reweight` | Phase-aligned TTA view reweighting on Temporal Grounding | `not_promoted` |
| [039](../experiments/EXP-039/experiment.md) | `reva-dse-vtg-brev` | Qwen3-VL-Embedding MSF + gated letter fusion on TG | `not_promoted` |

