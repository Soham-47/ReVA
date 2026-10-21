# Phase 2: question-aware temporal sampling results

Only Temporal Grounding predictions changed. Non-temporal predictions remain the frozen legacy baseline.

## valQ

Temporal Grounding: T0 51/69; T1 49/69; T2 47/69.
Overall with other tasks frozen: T0 498/595; T1 delta -0.34 pp; T2 delta -0.67 pp.

| Pair | Fixes | Regressions | Wrong→wrong | Unchanged | Net correct | Paired 95% CI (accuracy points) | Localized / offset views |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 → T1 | 2 | 4 | 2 | 61 | -2 | [-10.14, +4.35] | 123/138 |
| T0 → T2 | 2 | 6 | 3 | 58 | -4 | [-13.04, +1.45] | 123/138 |
| T1 → T2 | 1 | 3 | 4 | 61 | -2 | [-8.70, +2.90] | 123/138 |

## valT

Temporal Grounding: T0 99/133; T1 98/133; T2 91/133.
Overall with other tasks frozen: T0 885/1059; T1 delta -0.09 pp; T2 delta -0.76 pp.

| Pair | Fixes | Regressions | Wrong→wrong | Unchanged | Net correct | Paired 95% CI (accuracy points) | Localized / offset views |
|---|---:|---:|---:|---:|---:|---:|---:|
| T0 → T1 | 4 | 5 | 3 | 121 | -1 | [-5.26, +3.76] | 238/266 |
| T0 → T2 | 3 | 11 | 6 | 113 | -8 | [-11.28, -0.75] | 238/266 |
| T1 → T2 | 1 | 8 | 2 | 122 | -7 | [-9.77, -1.50] | 238/266 |

## Run measurements

The full paired A100 run completed in 1,608.6 seconds (26.8 minutes): 113.5 s model load, 924.0 s temporal localization across 404 calls, and 684.5 s answer scoring. The timing record is in `full/phase2_timing.json`.

## Question-form breakdown

All 202 Temporal Grounding questions were assigned one primary event form from their wording. This descriptive classification was completed after inference; it was not used to choose questions, tune prompts, or change the candidate gate. The detailed labels and per-item results are in `event_forms.json`.

| Split | Event form | Questions | T0 correct | T1 correct | T2 correct | T0→T1 net | T0→T2 net |
|---|---|---:|---:|---:|---:|---:|---:|
| valQ | First appearance | 28 | 22 | 20 | 19 | -2 | -3 |
| valQ | Duration / visibility interval | 35 | 26 | 26 | 25 | 0 | -1 |
| valQ | Event or camera onset / ordering | 6 | 3 | 3 | 3 | 0 | 0 |
| valT | First appearance | 50 | 33 | 34 | 30 | +1 | -3 |
| valT | Duration / visibility interval | 72 | 60 | 58 | 55 | -2 | -5 |
| valT | Event or camera onset / ordering | 4 | 2 | 2 | 2 | 0 | 0 |
| valT | Disappearance / exit | 5 | 3 | 3 | 3 | 0 | 0 |
| valT | Appearance and exit | 1 | 0 | 0 | 0 | 0 | 0 |
| valT | Other / non-temporal wording | 1 | 1 | 1 | 1 | 0 | 0 |

T1 has a small positive net on `valT` first-appearance questions, but a negative net on that category in `valQ` and on duration/visibility intervals in `valT`. T2 loses further answers on both first-appearance and duration questions. The disappearance and camera/event groups are small; their unchanged totals are not evidence that those failure types are solved.

## Candidate gate

A variant passes the predeclared pilot gate only with at least +3 net temporal answers on valT, nonnegative net change on valQ, and no more than a 1 pp overall drop on either split.

- T1: valT -1; valQ -2; overall condition pass; candidate not passed.
- T2: valT -8; valQ -4; overall condition pass; candidate not passed.

## Changed predictions

Every changed item, including all four aggregated option probabilities per arm and per-view selected-frame evidence, is retained in `phase2_comparison.json`. A visual review of all 34 unique changed questions across 32 clips is in `changed_answer_review/qualitative_review.md`, with the contact sheets, case selection, and video metadata in `changed_answer_review/`.

Validation sets are reused for diagnosis and are not blind test sets. The paired bootstrap intervals describe question-level variation only; clips may contribute multiple questions, so they are not cluster-corrected.
