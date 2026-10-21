# Phase 6 execution provenance

This note records the read-only Slurm accounting supplied with the copied Phase 6 score artifacts. The configs used for the training arms remain in [`code/configs/phase6_A1.yaml`](../../code/configs/phase6_A1.yaml) and [`code/configs/phase6_A2.yaml`](../../code/configs/phase6_A2.yaml).

| Stage | Slurm job | GPU | Elapsed | Train loss | Peak GPU memory | GPU utilization | Completed score rows |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| A1 training | 402807 | H200 | 00:35:07 (`train_runtime=2086 s`) | 0.3526 | 27,926 MiB | 85% | — |
| A2 training | 402810 | H100 | 00:42:31 (`train_runtime=2329 s`) | 0.3567 | 28,070 MiB | 81% | — |
| A1 scoring | 402864 | H200 | 00:24:38 | — | 27,154 MiB | 75% | valQ 595 + valT 1,059 = 1,654 |
| A2 scoring | 402865 | H100 | 00:26:23 | — | 27,154 MiB | 71% | valQ 595 + valT 1,059 = 1,654 |

All four job steps exited 0. The copied A1/A2 score manifests retain the remote adapter pointers; for A2 they point to `/scratch/ambpdc/soham/reva-phase6-smoke-20261018/runs/phase6_A2_h100_train/final_adapter`. The corresponding remote `RUN_PROVENANCE` record is alongside that runtime run; it was not available in this local checkout. No direct HPC currency charge is exposed by the supplied Slurm accounting, so this note makes no dollar cost claim.

The frozen A0 originals were checked byte-for-byte against the copied A0 references before analysis:

```text
scores_valQ.jsonl  d8c0327d830e0e063462f59594e8eac3da28e1da7b2b6b1b2ecc474a2dab5558
scores_valT.jsonl  2bf2ead04bb2a0cae0e5c9aa8f346379a16adabb155f5e512c13c51354bcaba1
```
