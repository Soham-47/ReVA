# Phase 2 verified-window temporal experiment report

**Outcome:** review and preflight completed; the approved T1R inference was **not launched** because the reviewed-window coverage gate failed. The frozen Stage A baseline and prior T0/T1 outputs were left untouched.

## Scope and protocol

- Pinned dataset revision: `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`.
- Selected subset: 24 first-appearance Temporal Grounding questions, 12 from `valQ` and 12 from `valT`; 23 unique videos.
- Arms: frozen Stage A legacy frame selection (T0), already-saved question-only coarse-to-fine windows (T1), and planned T1R, which changes only the timestamp window using an independently reviewed bracket. Model, adapter, question, answer prompt, option rotation, aggregation, and 16-frame budget stay fixed.
- Review used the prepared timestamped contact sheets. The packet itself contained question text and video evidence but no answer choices, gold labels, predictions, or review-role field. The manifest template and all 24 selection IDs are retained.
- The T1R full run requires at least 20/24 reviewed and recenterable windows. The four-question tranche requires 4/4. These thresholds were not relaxed.

## Review and run gate

| Split | Questions | Qualified interior windows | Opening-censored | Ambiguous |
|---|---:|---:|---:|---:|
| valQ | 12 | 1 | 3 | 8 |
| valT | 12 | 1 | 6 | 5 |
| **Total** | **24** | **2** | **9** | **13** |

The two supported brackets are:

- `train_006985` (`valQ`, blue car): absent at 2.0833 s and first visible at 2.1667 s; bracket **[2.0833, 2.1667] s**.
- `train_000047` (`valT`, large archway with Chinese characters): absent through 4.7333 s and first visible at 4.8667 s; bracket **[4.7333, 4.8667] s**.

Preflight validated the selected records against the pinned question set, the saved T0/T1 predictions, video metadata, and saved T1 sampling windows. Both qualified brackets were recenterable, with no per-window preflight issues. Coverage still failed:

- Tranche: **1/4**, required **4/4**.
- Full set: **2/24**, required **at least 20/24**.

Therefore no Stage A model was loaded and no T1R answer forwards were run. Validation explicitly reported that no model, video decoder, or external service was invoked. No A100 GPU scoring job was started. The CPU Modal review-material job produced 23 contact sheets and the blind review packet.

## Existing paired T0/T1 results on these same 24 questions

These are comparisons of **already-saved predictions**, not new inference. Accuracy is measured against the frozen question-set answer key after completing the visual review.

| Split | n | T0 correct | T1 correct | T1 fixes | T1 regressions |
|---|---:|---:|---:|---:|---:|
| valQ | 12 | 6 (50.0%) | 7 (58.3%) | 2 | 1 |
| valT | 12 | 6 (50.0%) | 6 (50.0%) | 0 | 0 |
| **Total** | **24** | **12 (50.0%)** | **13 (54.2%)** | **2** | **1** |

Across all 24, T0→T1 yielded 2 fixes, 1 regression, 11 unchanged correct predictions, 9 unchanged wrong predictions, and 1 wrong-to-different-wrong change. Within the six sampled T0-error questions per split, T1 fixed 2/6 in valQ and 0/6 in valT. On controls, valQ had one regression (6/6 to 5/6); valT stayed at 6/6. This is a small descriptive sample; it is not evidence of a reliable overall gain. T1R has no predictions, so its fixes, regressions, accuracy, and bracket-coverage metrics are **not measured**.

## Per-question review log

Status meanings: **Qualified bracket** has a visible in-clip absence→presence boundary and confirmed target identity; **Opening-censored** means the target is already visible in the first frame, so no in-clip onset can be assigned; **Ambiguous** means target identity or onset cannot be resolved from the evidence, so it falls back to T0.

| QA ID | Split | Disposition | Reviewed bracket | Question and evidence note |
|---|---|---|---|---|
| train_009393 | valQ | Ambiguous | — | At what point in the video does the soccer field first become visible in the frame? — The broad sports field is visible from 0.000 s, but the sheet does not establish that this is specifically the queried soccer field; no defensible first-appearance bracket. |
| train_002193 | valQ | Ambiguous | — | At what time does the black car first appear in the video? — Several dark/black cars are already present at 0.000 s; the queried car cannot be uniquely tracked to a later first appearance. |
| train_001156 | valQ | Ambiguous | — | At what point in the video does a blue car first appear parked on the right side of the road? — Multiple vehicles occupy the road; the blue car parked on the right cannot be uniquely identified at a clear first-visibility boundary. |
| train_009816 | valQ | Ambiguous | — | At what point in the video does the cluster of buildings and parked cars on the right side of the road first appear? — Buildings and parked vehicles are partly in view from the opening; the queried cluster boundary is not separable from the initial view. |
| train_008659 | valQ | Ambiguous | — | At what point in the video does the first house appear in the frame? — Aerial view has possible structures near the opening and later houses become clearer; first-house identity and boundary are not unambiguous. |
| train_009187 | valQ | Ambiguous | — | When does the parking lot first become visible in the video? — The shoreline scene transitions toward a lot, but the parking area may be present earlier and no single first-visible sample is reliable. |
| train_009281 | valQ | Opening-censored | — | At what point in the video does the train track first appear in the frame? — A narrow linear track/corridor is already visible along the far tree line in the 0.000 s sample; any onset precedes the clip. |
| train_006985 | valQ | Qualified bracket | 2.0833–2.1667 s | When does the blue car first appear in the video? — Blue car absent at 2.0833 s and first visible at 2.1667 s, distinct from the white and red cars already in view. |
| train_008890 | valQ | Opening-censored | — | At what time does the road first appear in the video? — The roadway is already visible in the opening sample; no pre-clip onset can be recovered. |
| train_004581 | valQ | Opening-censored | — | When does the fire truck first become visible in the video? — The fire-truck scene already contains the truck at the opening; the start of its visibility is censored by the clip boundary. |
| train_012379 | valQ | Ambiguous | — | At what point in the video does the white car first appear on the left side of the street? — Several light-colored cars are present; the white car on the left cannot be identified as a unique first appearance. |
| train_007383 | valQ | Ambiguous | — | At what point in the video does the tennis court first become visible? — The court surface and markings are not clear enough in the sampled sheet to establish either target identity or an onset bracket. |
| train_001319 | valT | Ambiguous | — | At what time does the white bus first appear under the bridge in the video? — A small white vehicle is already visible below the bridge at 0.000 s; its bus identity cannot be confirmed against later views. |
| train_007117 | valT | Opening-censored | — | When does the red car first appear in the traffic jam? — The red car is already among the traffic at 0.000 s; onset is before the clip or censored. |
| train_007334 | valT | Opening-censored | — | At what point in the video does the residential area with houses first become visible? — The residential area and houses are visible in the opening samples; there is no in-clip onset. |
| train_008006 | valT | Ambiguous | — | When does the vehicle on the road near the bottom right of the frame first become visible? — Several road vehicles are visible from the opening and the queried bottom-right vehicle is not uniquely identifiable over time. |
| train_008007 | valT | Ambiguous | — | At what point in the video does the white car first appear in the frame? — White cars are already present in the opening scene; the queried car cannot be uniquely distinguished to assign first visibility. |
| train_008027 | valT | Opening-censored | — | At what point in the video does the large open field first appear in the frame? — The open field fills the scene from 0.000 s; its first appearance predates the clip. |
| train_000047 | valT | Qualified bracket | 4.7333–4.8667 s | At what point in the video does the large archway with Chinese characters first become visible? — Archway absent through 4.7333 s and first visible at 4.8667 s; Chinese lettering is clear in the next sample. |
| train_000079 | valT | Opening-censored | — | At what point in the video does the red truck carrying cattle first become visible in the frame? — The red livestock truck is already in the opening frame; the clip does not show its entry. |
| train_000680 | valT | Opening-censored | — | At what point in the video does the group of people on the bridge first become visible? — People are already visible on the bridge in the opening frame; their first appearance is left-censored. |
| train_001138 | valT | Ambiguous | — | At what point in the video does the white van with blue markings on its side first appear? — A white vehicle is partly present near the opening edge, but its blue markings and identity cannot be confirmed well enough for a bracket. |
| train_002260 | valT | Opening-censored | — | At what point in the video does the green tarp-covered structure on the right side of the street first become visible? — The green tarp-covered structure is visible at the right side from the opening sample; onset is left-censored. |
| train_002541 | valT | Ambiguous | — | At what point in the video does the small white truck first appear in the frame? — A small light-colored vehicle appears among dense traffic, but the queried truck cannot be uniquely followed to a reliable first-visible transition. |

## Artifacts and reproducibility

- Reviewed annotations: `tasks/phase2_temporal_verified_window_manifest.json`
- Original unreviewed manifest snapshot: `runs/phase2_temporal_verified_window/manifest_template_unreviewed.json`
- Blind packet and timestamped contact sheets: `runs/phase2_temporal_verified_window/review_material/`
- Full and tranche validation output: `runs/phase2_temporal_verified_window/preflight_full.txt` and `preflight_tranche.txt`
- Paired saved-prediction counts: `runs/phase2_temporal_verified_window/t0_t1_selected24_paired.json`
- T1R predictions/report: **not produced**; no `scores_*_T1R.jsonl` exists.

Reproduce the no-inference gate check with:

```bash
python3 code/score_temporal_verified.py --manifest tasks/phase2_temporal_verified_window_manifest.json --subset tranche
python3 code/score_temporal_verified.py --manifest tasks/phase2_temporal_verified_window_manifest.json --subset full
```

## Interpretation and next decision

The selected set is a poor fit for a *first in-clip appearance* experiment: most targets are already visible at time zero or are too ambiguous to establish an onset. Forcing their windows would introduce guessed labels and invalidate the comparison. The approved T1R hypothesis therefore remains untested. To proceed, the experiment design needs a new candidate set chosen for observable interior onsets (while preserving the 12/12 split and error/control balance), or explicit approval of a smaller feasibility-only analysis. That changes the approved subset/design and should be reviewed before another inference run.

**Review limitation:** this is a Codex visual screening pass over contact sheets, not an independent human gold annotation. The role labels were stored outside the packet in the scoring manifest and became visible during setup, so this review cannot be certified as role-blind even though the onset judgments were based on question/frame evidence. Gold answers and selected T0/T1 predictions were not inspected until after annotations were recorded. The contact sheets are sampled evidence rather than full-frame expert inspection; uncertain cases were conservatively excluded from T1R. A human, role-blind review of candidates would strengthen the onset labels before a revised run.
