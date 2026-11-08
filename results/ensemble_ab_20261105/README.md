# A+B logit-mean ensemble — 2026-11-05 (Asia/Calcutta)

Method: mean of letter probabilities p[A,B,C,D] from Stage A and Stage B score JSONL
(each already averaged over 4 TTA views). Argmax of the mean is the ensemble pred.

## Inputs
- Stage A: runs/stageA_q3vl8b_r16/scores_{val,test}.jsonl
- Stage B: runs/stageB_q3vl8b_r16_h100/scores_{val,test}.jsonl

## Outputs
- scores_val_ab_mean.jsonl (2000 rows)
- scores_test_ab_mean.jsonl (4000 rows)
- summary_val.json / summary_test.json

## Official val (labeled, clean for B)
- Stage A: 1757/2000 = 87.85%
- Stage B: 1769/2000 = 88.45%
- A+B mean: 1774/2000 = 88.70%
- vs A: net +17 (fixed/broken 45/28, McNemar p=0.0604)
- vs B: net +5 (fixed/broken 40/35, McNemar p=0.6445)
- Parent agree on val: 1840/2000 (92.00%)

## Test (unlabeled)
- Parent agree: 3683/4000 (92.07%)
- Disagree: 317 (no gold; cannot report accuracy)

## Fallback
Stage A alone remains the submission fallback (private test 3358/4000 = 83.95%).

## Notes
- Stage B trained on valQ+valT — do not gate A+B on those splits.
- tools/paired_gate.py now reports W_TG / W_CD / W_PV / W_GR slice nets (optional --require-weak).
- CUDA probe draft (NOT submitted): jobs/cuda_probe.sbatch + tools/cuda_probe_forward.py
- Pending Slurm 432730 left untouched.
