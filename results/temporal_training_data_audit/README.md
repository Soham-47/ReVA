# Temporal training-data feasibility audit

**Scope:** blind visual review of 100 train-only Temporal Grounding QA items. The audit checks whether each clip and question support trustworthy temporal supervision; it does not train a model or score model predictions.

- Dataset: `ReVA-Benchmark/ReVA` at immutable revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`.
- `train.json` SHA-256: `1d5702a698ef322d171fa16cc845c47906f3f298b8d1c9a023f60f706450da88`.
- Population: 913 eligible temporal QA rows across 468 videos after excluding all `valQ` and `valT` videos.
- Sample: 100 distinct QA IDs from 100 distinct videos; deterministic seed `20261019`.
- Strata: `{'action_or_transition': 15, 'joint_onset_exit': 15, 'other_temporal': 15, 'exit_or_offset': 15, 'onset_or_entry': 20, 'duration': 20}`.
- Dataset sources: `{'ERA_Tra': 26, 'Hawk_UAV': 36, 'UAVDT': 17, 'VisDrone': 21}`.
- Evidence views: the exact `code/common.py` legacy sample indices for offsets 0.0 and 0.5. The sampler returns 2–16 frames on short clips, so these views preserve its actual behavior. One selected clip has only six decoded frames.

## Phase 1 review results

The blind visual pass is complete. The single-reviewer disposition is **10 directly usable**, **57 conditional**, and **33 excluded** out of 100. Even if all conditional cases are repairable, only 67/100 have usable or potentially usable visual evidence, below the predeclared 80-case gate. Gold support was rated 10 supported, 51 partial/censoring-dependent, 11 not supported, and 28 not assessable. An independent 20-case review is still outstanding; do not train from this sample.

See `phase1_report.md` for method, failure taxonomy, and next action; `phase1_visual_review.json` and `phase1_gold_answer_review.json` contain case-level records. `phase1_second_review_packet.json` is a blind packet with no answers or first-review notes.

## Preliminary automated screen

Before visual review, a numeric-option check found 10/100 questions with at least one option more than one frame beyond the manifest duration. Of those, 3 gold choices cross the same limit; one case (TA028) has a gold disappearance time 3.2 seconds after a 0.200-second, six-frame clip. These are flags for human review, not confirmed relabels. Do not use them as training targets unless the source clip and label are verified. See `pre_review_checks.json`.

## Blind review order

1. Review each question with its two timestamped frame grids in the generated contact sheets. Keep the answer options and gold hidden until the visual annotation is complete.
2. Record target identity, event definition, visible-time bracket or clip-boundary censoring, whether the target is trackable at the model's legacy sampling resolution, confidence and a usable/exclude decision.
3. Reveal options and gold from `tasks/temporal_training_audit_selection.json`; record whether the supplied answer is supported. Flag out-of-clip or ambiguous options. Do not rewrite source labels.
4. Include an item in a training set only when identity and event are stable, evidence supports a bracket or event order, and the answer label is consistent or can be separately verified. Keep uncertain clips as exclusions; do not fabricate exact times.

## Predeclared gate

Proceed to training-data construction only if at least 80/100 cases have usable visual evidence and a second blinded reviewer agrees on target identity and boundary/censoring for at least 18/20 double-reviewed cases. Report answer-label consistency separately from annotation usability. Passing this gate does not establish model improvement.

## Files

- `tasks/temporal_training_audit_selection.json`: full selection, options, gold, exact frame indices and timestamps.
- `tasks/temporal_training_audit_blind_review.json`: blind review list without choices or gold.
- `tasks/temporal_training_audit_annotations_template.json`: blank review form.
- `runs/temporal_training_data_audit/review_material/`: audit contact sheets to be generated on HPC CPU from the existing clips; no GPU or model inference.
