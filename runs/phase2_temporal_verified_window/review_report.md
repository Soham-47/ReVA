# Phase 2 full selected-set reviewed-window inference report

**Status: completed.** T1R scoring ran on Modal A100 for all 24 selected questions: 12 from `valQ` and 12 from `valT`, across 23 unique videos. The run is [available in Modal](https://modal.com/apps/sohamsaha291/main/ap-lXW7MJVpElZ6SPgmzBpMTo).

## What was run

- **T0:** frozen Stage A baseline with legacy frame selection.
- **T1:** previously saved question-only coarse-to-fine selection.
- **T1R:** the same frozen Stage A model, adapter, question, answer prompt, option rotation, aggregation, and 16-frame budget; only the selected frame windows changed to use the reviewed brackets.
- The 24-question set contains six saved-T0 errors and six correct controls per split. It is a selected pilot, not the entire `valQ` or `valT` split.
- No training was run and no Stage A baseline or prior T0/T1 score file was changed.

## Reviewed windows and execution

The user-directed boundary rule treats a target already visible at the opening sample as first observed **within this clip**: its reviewed bracket is `[0, first nonzero sampled timestamp]`. This does not claim that the object physically entered at 0 s or rule out an earlier, pre-clip onset. **20/24** questions use this clip-boundary bracket; four have an interior absence-to-presence bracket.

| QA ID | Split | Reviewed bracket (s) |
|---|---|---:|
| `train_009393` | valQ | 0.0000–0.1335 |
| `train_002193` | valQ | 0.0000–0.1333 |
| `train_001156` | valQ | 0.0000–0.1333 |
| `train_009816` | valQ | 0.0000–0.0667 |
| `train_008659` | valQ | 0.0000–0.1335 |
| `train_009187` | valQ | 4.2042–4.3377 |
| `train_009281` | valQ | 0.0000–0.1335 |
| `train_006985` | valQ | 2.0833–2.1667 |
| `train_008890` | valQ | 0.0000–0.1335 |
| `train_004581` | valQ | 0.0000–0.1250 |
| `train_012379` | valQ | 0.0000–0.1668 |
| `train_007383` | valQ | 0.0000–0.1335 |
| `train_001319` | valT | 0.0000–0.1333 |
| `train_007117` | valT | 0.0000–0.1250 |
| `train_007334` | valT | 0.0000–0.1335 |
| `train_008006` | valT | 0.0000–0.1335 |
| `train_008007` | valT | 0.0000–0.1335 |
| `train_008027` | valT | 0.0000–0.1335 |
| `train_000047` | valT | 4.7333–4.8667 |
| `train_000079` | valT | 0.0000–0.1333 |
| `train_000680` | valT | 0.0000–0.1333 |
| `train_001138` | valT | 0.0000–0.1333 |
| `train_002260` | valT | 0.0000–0.1333 |
| `train_002541` | valT | 1.4000–1.5333 |

Four interior brackets: `train_002541` (small white truck, 1.4000–1.5333 s), `train_006985` (blue car, 2.0833–2.1667 s), `train_009187` (parking area, 4.2042–4.3377 s), and `train_000047` (archway, 4.7333–4.8667 s). The first two original high-confidence brackets (`train_006985`, `train_000047`) remain unchanged. The newly reviewed `train_002541` and `train_009187` boundaries are sample-level visual judgments and carry more uncertainty; details are recorded in the manifest.

Preflight passed for all 24 reviewed windows, with no issues. The scored files contain **12 unique records per split** and **zero T0 fallback offsets** across 48 offset selections. Forty-five offset selections recentered a saved T1 window; three selections had no usable saved T1 candidate and used the shortest local window around the reviewed bracket (at least seven frames wide): offset 0.0 for `train_009393`, and both offsets for `train_001156`. This small-window handling replaced the prior T0 fallback for those cases so the requested reviewed-window arm covered all 24.

Modal timing reported model load **113.4 s**, scoring **41.9 s** for 24 records. The model weights downloaded successfully; the log contains an unauthenticated Hugging Face rate-limit warning, but no download or scoring error.

After scoring, the manifest's reviewer description was corrected from “full-resolution” to “higher-resolution … 480×270 frame cells.” This audit-only wording change did not alter reviewed boundaries, selected frames, or predictions; output manifests and evaluation metadata were refreshed to the corrected annotation digest without rerunning the model. The final digest is `e8be1fed3145f6a045b735facf776a8f7e40ce06d55f4615767a6d1c5dc16630`.

## Results

| Split | n | T0 | T1 | T1R | T0→T1R fixes / regressions | T1→T1R fixes / regressions / wrong→wrong |
|---|---:|---:|---:|---:|---:|---:|
| valQ | 12 | 6/12 (50.0%) | 7/12 (58.3%) | 7/12 (58.3%) | 2 / 1 | 0 / 0 / 0 |
| valT | 12 | 6/12 (50.0%) | 6/12 (50.0%) | 6/12 (50.0%) | 1 / 1 | 1 / 1 / 1 |
| **Total** | **24** | **12/24 (50.0%)** | **13/24 (54.2%)** | **13/24 (54.2%)** | **3 / 2** | **1 / 1 / 1** |

T1R ties T1 overall at **13/24 (54.2%)**. Against T0, T1R produces three fixes and two regressions (net **+1 correct answer**); this is the same net result as T1. Against T1, T1R fixes one answer (`train_008007`), regresses one (`train_002541`), and changes one still-wrong answer (`train_001319`), leaving accuracy unchanged. This pilot does **not** show an overall accuracy improvement from reviewed-window centering over T1.

### Per-question predictions

Letters identify answer choices; answer text is shown to make each error auditable. `T1R vs T1` indicates whether the reviewed-window arm fixed or worsened the saved T1 result.

| Split / QA ID | Window (s) | Gold | T0 | T1 | T1R | T1R vs T1 |
|---|---:|---|---|---|---|---|
| valQ / `train_009393` | 0.0000–0.1335 | C (At 0.0 seconds) | D (At 4.0 seconds) | D (At 4.0 seconds) | D (At 4.0 seconds) | same |
| valQ / `train_002193` | 0.0000–0.1333 | D (At 1.5 seconds) | A (At 0.5 seconds) | D (At 1.5 seconds) | D (At 1.5 seconds) | same |
| valQ / `train_001156` | 0.0000–0.1333 | B (At 2.5 seconds) | D (At 1.0 seconds) | D (At 1.0 seconds) | D (At 1.0 seconds) | same |
| valQ / `train_009816` | 0.0000–0.0667 | A (At 0.0 seconds) | C (At 1.0 seconds) | A (At 0.0 seconds) | A (At 0.0 seconds) | same |
| valQ / `train_008659` | 0.0000–0.1335 | C (At 2.0 seconds) | B (At 1.0 seconds) | B (At 1.0 seconds) | B (At 1.0 seconds) | same |
| valQ / `train_009187` | 4.2042–4.3377 | A (In the first half of the video) | D (Around the middle of the video) | D (Around the middle of the video) | D (Around the middle of the video) | same |
| valQ / `train_009281` | 0.0000–0.1335 | D (At 0.0 seconds) | D (At 0.0 seconds) | D (At 0.0 seconds) | D (At 0.0 seconds) | same |
| valQ / `train_006985` | 2.0833–2.1667 | B (In the first half of the video) | B (In the first half of the video) | B (In the first half of the video) | B (In the first half of the video) | same |
| valQ / `train_008890` | 0.0000–0.1335 | A (At 2.0 seconds) | A (At 2.0 seconds) | A (At 2.0 seconds) | A (At 2.0 seconds) | same |
| valQ / `train_004581` | 0.0000–0.1250 | C (At the beginning of the video) | C (At the beginning of the video) | C (At the beginning of the video) | C (At the beginning of the video) | same |
| valQ / `train_012379` | 0.0000–0.1668 | A (At 0.0 seconds) | A (At 0.0 seconds) | A (At 0.0 seconds) | A (At 0.0 seconds) | same |
| valQ / `train_007383` | 0.0000–0.1335 | B (At 3.5 seconds) | B (At 3.5 seconds) | D (At 2.0 seconds) | D (At 2.0 seconds) | same |
| valT / `train_001319` | 0.0000–0.1333 | D (At 0.0 seconds) | B (At 3.2 seconds) | C (At 2.5 seconds) | B (At 3.2 seconds) | changed, still wrong |
| valT / `train_007117` | 0.0000–0.1250 | C (Around the middle of the video) | B (At the beginning of the video) | B (At the beginning of the video) | B (At the beginning of the video) | same |
| valT / `train_007334` | 0.0000–0.1335 | B (At 0.5 seconds) | A (At 2.0 seconds) | A (At 2.0 seconds) | A (At 2.0 seconds) | same |
| valT / `train_008006` | 0.0000–0.1335 | C (At the beginning of the video) | B (In the first half of the video) | B (In the first half of the video) | B (In the first half of the video) | same |
| valT / `train_008007` | 0.0000–0.1335 | A (At 0.0 seconds) | D (At 1.0 seconds) | D (At 1.0 seconds) | A (At 0.0 seconds) | fixed |
| valT / `train_008027` | 0.0000–0.1335 | B (At 0.0 seconds) | D (At 3.0 seconds) | D (At 3.0 seconds) | D (At 3.0 seconds) | same |
| valT / `train_000047` | 4.7333–4.8667 | D (At 3.5 seconds) | D (At 3.5 seconds) | D (At 3.5 seconds) | D (At 3.5 seconds) | same |
| valT / `train_000079` | 0.0000–0.1333 | A (At 0.0 seconds) | A (At 0.0 seconds) | A (At 0.0 seconds) | A (At 0.0 seconds) | same |
| valT / `train_000680` | 0.0000–0.1333 | D (At 0.0 seconds) | D (At 0.0 seconds) | D (At 0.0 seconds) | D (At 0.0 seconds) | same |
| valT / `train_001138` | 0.0000–0.1333 | A (At 0.0 seconds) | A (At 0.0 seconds) | A (At 0.0 seconds) | A (At 0.0 seconds) | same |
| valT / `train_002260` | 0.0000–0.1333 | C (0.0 seconds) | C (0.0 seconds) | C (0.0 seconds) | C (0.0 seconds) | same |
| valT / `train_002541` | 1.4000–1.5333 | C (At 1.5 seconds) | C (At 1.5 seconds) | C (At 1.5 seconds) | B (At 0.5 seconds) | regressed |

## Interpretation and limitations

- The reviewed-window recipe has the same aggregate score as T1. The answer changes show both a fix and a regression, not a consistent gain.
- Twenty opening-boundary windows provide little evidence about locating an event later in a clip. The experiment therefore tests the user-specified first-observation-within-clip interpretation for most items, not 24 precise event-onset localizations.
- Window annotations came from timestamped contact sheets with 48 sampled frames per clip, not frame-by-frame independent human annotation. The two new interior brackets are less certain than the adjacent-sample brackets already reviewed in the original pass.
- Some targets are small or non-unique in aerial footage. In particular, the soccer-field question shows a field with baseball-diamond-like markings; several car questions contain multiple matching vehicles. These semantic and identity uncertainties limit the strength of the numeric comparison even though all 24 were scored.
- This is a small, deliberately balanced error/control sample and should not be treated as a reliable estimate of full-split accuracy or a statistically established improvement. Review roles were not independently blinded.

## Artifacts

- Reviewed-window annotations and per-question evidence: `tasks/phase2_temporal_verified_window_manifest.json` (local-only artifact: `../../tasks/phase2_temporal_verified_window_manifest.json`).
- Full-set and tranche preflight outputs: `preflight_full.txt`, `preflight_tranche.txt`; the initial failed-gate snapshots are preserved as `preflight_full_initial.txt` and `preflight_tranche_initial.txt`.
- T1R score rows: `modal_full/scores_valQ_T1R.jsonl`, `modal_full/scores_valT_T1R.jsonl`.
- Machine-readable comparison and compact report: `modal_full/report/verified_window_comparison.json`, `modal_full/report/verified_window_report.md`.
- Modal execution log and timing: `modal_full/modal_score.log`, `modal_full/verified_window_timing.json`.
- Initial pre-run review report: `review_report_initial_pre_run.md`.

**Conclusion:** the full selected 24-question inference is complete, all 24 used reviewed windows, and the reviewed-window arm did not improve aggregate accuracy over the saved T1 arm. The strongest remaining gap is evaluation quality: a larger set of unambiguous interior events with independent, blinded temporal annotations is needed before claiming the recipe helps temporal grounding.
