# Verified temporal-window pilot

The T1R scorer uses the frozen answer prompt and changes only the selected frame window. Human review prose is retained for audit and is not sent to the answer model.

## valQ

Accuracy: T0 6/12; T1 7/12; T1R 7/12.

| Pair | Fixes | Regressions | Wrong→wrong | Unchanged | Net |
|---|---:|---:|---:|---:|---:|
| T0_vs_T1 | 2 | 1 | 0 | 9 | +1 |
| T0_vs_T1R | 2 | 1 | 0 | 9 | +1 |
| T1_vs_T1R | 0 | 0 | 0 | 12 | +0 |

Controls: T0 6/6; T1 5/6; T1R 5/6.
Fallback offsets: T1 3; T1R 0. Boundary reviewed: 12/12; identity confirmed: 12/12. T1R bracket coverage: 24/24 offsets.

## valT

Accuracy: T0 6/12; T1 6/12; T1R 6/12.

| Pair | Fixes | Regressions | Wrong→wrong | Unchanged | Net |
|---|---:|---:|---:|---:|---:|
| T0_vs_T1 | 0 | 0 | 1 | 11 | +0 |
| T0_vs_T1R | 1 | 1 | 0 | 10 | +0 |
| T1_vs_T1R | 1 | 1 | 1 | 9 | +0 |

Controls: T0 6/6; T1 6/6; T1R 5/6.
Fallback offsets: T1 0; T1R 0. Boundary reviewed: 12/12; identity confirmed: 12/12. T1R bracket coverage: 24/24 offsets.

