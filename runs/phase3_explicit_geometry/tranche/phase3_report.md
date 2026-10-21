# Phase 3 explicit geometry pilot analysis

This is an error-enriched pilot summary; it is not split-wide benchmark accuracy.

## valQ

| Condition | Correct | Total | Accuracy |
|---|---:|---:|---:|
| G0 | 1 | 2 | 0.500 |
| G1 | 1 | 2 | 0.500 |
| G2 | 1 | 2 | 0.500 |

Transition counts:
- G1_vs_G0: {'fixes': 0, 'regressions': 0, 'wrong_to_wrong': 1, 'correct_to_correct': 1, 'prediction_unchanged': 2, 'prediction_changed': 0}
- G2_vs_G1: {'fixes': 0, 'regressions': 0, 'wrong_to_wrong': 1, 'correct_to_correct': 1, 'prediction_unchanged': 2, 'prediction_changed': 0}
- G2_vs_G0: {'fixes': 0, 'regressions': 0, 'wrong_to_wrong': 1, 'correct_to_correct': 1, 'prediction_unchanged': 2, 'prediction_changed': 0}

Evidence: {"G1": {"question_count": 2, "pair_count": 1, "reliable_2d_pairs": 1, "ambiguous_pairs": 0, "insufficient_pairs": 0, "unsupported_subtype_records": 1, "reliable_pair_coverage": 1.0, "records_with_reliable_evidence": 1, "records_with_reliable_evidence_fraction": 0.5, "evidence_correctness": "manual qualitative audit required; no gold spatial evidence labels are available"}, "G2": {"question_count": 2, "pair_count": 1, "reliable_2d_pairs": 1, "ambiguous_pairs": 0, "insufficient_pairs": 0, "unsupported_subtype_records": 1, "reliable_pair_coverage": 1.0, "records_with_reliable_evidence": 1, "records_with_reliable_evidence_fraction": 0.5, "evidence_correctness": "manual qualitative audit required; no gold spatial evidence labels are available"}}

## valT

| Condition | Correct | Total | Accuracy |
|---|---:|---:|---:|
| G0 | 1 | 2 | 0.500 |
| G1 | 1 | 2 | 0.500 |
| G2 | 1 | 2 | 0.500 |

Transition counts:
- G1_vs_G0: {'fixes': 0, 'regressions': 0, 'wrong_to_wrong': 1, 'correct_to_correct': 1, 'prediction_unchanged': 2, 'prediction_changed': 0}
- G2_vs_G1: {'fixes': 0, 'regressions': 0, 'wrong_to_wrong': 1, 'correct_to_correct': 1, 'prediction_unchanged': 2, 'prediction_changed': 0}
- G2_vs_G0: {'fixes': 0, 'regressions': 0, 'wrong_to_wrong': 1, 'correct_to_correct': 1, 'prediction_unchanged': 2, 'prediction_changed': 0}

Evidence: {"G1": {"question_count": 2, "pair_count": 0, "reliable_2d_pairs": 0, "ambiguous_pairs": 0, "insufficient_pairs": 0, "unsupported_subtype_records": 2, "reliable_pair_coverage": 0.0, "records_with_reliable_evidence": 0, "records_with_reliable_evidence_fraction": 0.0, "evidence_correctness": "manual qualitative audit required; no gold spatial evidence labels are available"}, "G2": {"question_count": 2, "pair_count": 0, "reliable_2d_pairs": 0, "ambiguous_pairs": 0, "insufficient_pairs": 0, "unsupported_subtype_records": 2, "reliable_pair_coverage": 0.0, "records_with_reliable_evidence": 0, "records_with_reliable_evidence_fraction": 0.0, "evidence_correctness": "manual qualitative audit required; no gold spatial evidence labels are available"}}

## Qualitative review

Changed comparison cases requiring manual inspection: 0.
Evidence correctness remains a manual audit because the dataset has no gold box or mask annotations.
