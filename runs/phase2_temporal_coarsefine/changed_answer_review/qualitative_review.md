# Phase 2 changed-answer visual review

## Scope and method

The paired comparison contains 70 changed pair records, representing 34 unique questions across 32 videos. All 32 videos were inspected through contact sheets with 24 evenly spaced decoded frames per clip. The question, options, gold label, predictions, and pair-specific changes are preserved in `selection.json`; the per-arm selected timestamps and probabilities remain in `../phase2_comparison.json`.

This is a visual screening pass, not a relabeling exercise. The sheet images are small thumbnails (192 × 104 pixels inside each cell), so they are useful for broad scene and event context but cannot reliably establish the identity of a tiny vehicle or a boundary within a fraction of a second. No label was changed and no example was removed.

## Case-by-case notes

| Question ID | Visual review |
|---|---|
| `train_000747` (`valQ`) | The red bus is small among dense traffic. The sheet establishes the opening context but does not settle its exact exit time. T2 matches the label while T1 does not; treat this as a plausible, unconfirmed fix. |
| `train_001319` (`valT`) | A small white vehicle is visible near the underpass in the opening frames. All arms place first appearance later than the gold 0.0 s, despite the opening frame being available in T0. This points to target identification or answer interpretation, not missing opening coverage. |
| `train_001320` (`valT`) | This asks about the same underpass vehicle. The 24 thumbnails do not make its continuous-visibility endpoint clear. T0's 4.0 s answer matches the label; T1 and T2 shorten it to 3.5 s. |
| `train_002193` (`valQ`) | The black car is one of several small vehicles in a busy street. T1 selects the gold 1.5 s onset; T2 returns to the earlier T0 answer. Object identity remains hard to verify from thumbnails. |
| `train_002541` (`valT`) | The white truck is small among traffic. T0 and T1 match the gold 1.5 s; T2 moves earlier to 0.5 s and is wrong. |
| `train_002637` (`valT`) | The signal is tiny at the right edge of a busy intersection. The 5.5 s versus 6.0 s distinction cannot be settled from this sheet. T0 and T1 match the label; T2 changes to 5.5 s. |
| `train_004270` (`valT`) | The construction scene contains a possible ramp, but its boundary is not crisp at thumbnail size. T0 and T1 answer “for most of the video”; T2 changes to “near the end,” a regression against the label. |
| `train_007273` (`valQ`) | The drone view moves over houses and paved areas. Whether the parking lot is already visible at the start or becomes identifiable later depends on what counts as the target lot. Both modified arms move from the gold 3.0 s to 2.0 s. |
| `train_007383` (`valQ`) | The tennis court is small within a residential scene and is not confidently identifiable at this scale. T0 matches 3.5 s; T1 and T2 shift to 2.0 s. |
| `train_008027` (`valT`) | The large open field is visible in the opening frames. All three arms still choose a later time instead of the gold 0.0 s. Since T0 already samples the opening, this is a clear case where changing frame selection did not fix visible-evidence interpretation. |
| `train_008113` (`valT`) | Several small vehicles are visible on the highway from early frames; the specific white truck is difficult to isolate. T0 and T1 match 1.0 s; T2 shifts too early to 0.5 s. |
| `train_008182` (`valQ`) | The red car is too small to follow confidently throughout the clip. T0 and T1 match the 3.8 s duration; T2 shortens it to 1.2 s. |
| `train_008216` (`valT`) | Light-blue roofs are visible early, but the sheet cannot tell when the entire apartment complex becomes identifiable rather than partially visible. T0 matches 2.0 s; T1 and T2 answer 0.0 s. |
| `train_008246` (`valT`) | The freight train and track appear in the opening frames. The gold option “in the first half” overlaps with “at the beginning,” making the answer choices ambiguous. T1 and T2 match the gold wording, but this should not be counted as a confirmed visual fix. |
| `train_008248` (`valT`) | This is the same freight-train clip as `train_008246`. The train appears in the opening sheet, while this question's gold says 2.0 s. T1 matches 2.0 s and T2 moves to 1.0 s. The two questions are a label/wording consistency flag, not grounds to edit either label. |
| `train_008588` (`valT`) | Office buildings are visible from the opening. “When the camera begins to focus” describes a gradual framing change rather than a crisp first-visibility event; no arm matches the 0.5 s label. |
| `train_009656` (`valT`) | The view starts over a field and the community-center area becomes more visible as the camera advances. T1 and T2 select the gold 2.0 s rather than T0's 3.0 s; this is a plausible fix. |
| `train_009783` (`valQ`) | The parking lot becomes clearer as the camera pans and remains visible in late samples. All arms miss the gold interval of 1.0–5.5 s; the modified ranges still truncate the visible period. |
| `train_009816` (`valQ`) | The buildings and parked cars on the right are present in the opening view. T1 and T2 answer 0.0 s, matching the label where T0 answered 1.0 s; this is a plausible fix. |
| `train_009869` (`valQ`) | The baseball field is visible early and moves out of view as the camera shifts. Its exact continuous-visibility endpoint depends on a boundary judgment; the sheet impression is closer to 2.5 s than the gold 3.5 s. Flag for label review, without changing the score. |
| `train_009902` (`valT`) | The cemetery target is not clearly distinguishable from the surrounding grass, trees, and parking area in these thumbnails. T0 matches 3.5 s; both modified arms move to 2.0 s. |
| `train_010206` (`valT`) | Residential buildings are visible early, but “first becomes clearly visible” has a subjective threshold. T0 and T1 match 2.0 s; T2 shifts to 1.5 s. |
| `train_010441` (`valT`) | The blue car is small along the road, so its full visibility duration is hard to verify. T0 and T2 match 8.0 s; T1 shortens the answer to 5.0 s. |
| `train_010524` (`valQ`) | Night glare and multiple white vehicles make the van's identity and first-appearance point uncertain. T0 matches 3 s; T1 and T2 answer 5 s. |
| `train_010641` (`valQ`) | The bottom-right blue car is difficult to track in the dark, reflective intersection. All arms miss the gold 10.5 s duration. |
| `train_010654` (`valT`) | Several vehicles and strong lighting make the blue taxi difficult to distinguish. All arms miss the gold 0:02 first-appearance time. |
| `train_011596` (`valQ`) | The “JUST FURNITURE” facade/sign is visible in the opening samples, although the gold answer says it first appears at 10 s. This is a strong prompt/label interpretation flag (possibly “fully clear” versus “first visible”), not a relabeling decision. |
| `train_012885` (`valQ`) | The residential area gradually shifts as the drone moves. “Starts to recede” has no clear single boundary in the sheet; all arms miss the 7 s label. |
| `train_013713` (`valT`) | The red car is tiny among intersection traffic. T0 and T1 match 8 s; T2 shortens the duration to 6 s. The exact endpoint is difficult to establish at this scale. |
| `train_013884` (`valT`) | The white car is one of several vehicles around the intersection, making its identity and exit time uncertain. All arms miss the gold 5 s duration. |
| `train_013897` (`valT`) | The building is visible early and then the view shifts away. T0 and T1 match the 2 s label; T2 extends the answer to 3 s. |
| `train_014049` (`valT`) | A tower-like structure is present early, while a distinct church steeple appears later. The target identity is ambiguous at thumbnail scale; T0 matches 4 s while T1 and T2 answer 6 s. |
| `train_014294` (`valQ`) | The yellow roadside sign is present in the clip, but its exact visibility endpoint is hard to read at thumbnail resolution. T0 and T1 match 8 s; T2 shortens it to 5 s. |
| `train_015451` (`valT`) | The solar-panel field is visible in the opening frames. T1 and T2 answer 0.0 s, matching the label where T0 answered 3.0 s; this is a plausible fix. |

## Summary

- A few changes have visually plausible support, especially the onset answers for `train_009656`, `train_009816`, and `train_015451`, plus the T1-only onset change for `train_002193`. Tiny targets and thumbnail resolution prevent treating these as definitive object-level audits.
- Several failures persist even though the relevant scene is present in the opening frames (`train_001319`, `train_008027`, and `train_008246`/`train_008248`). More targeted frames alone are therefore not sufficient; object identity, event definition, and conversion from visible evidence to the answer option remain failure sources.
- `train_008246`/`train_008248`, `train_011596`, `train_008588`, `train_009869`, and `train_012885` merit annotation/wording review. These are flags only; no labels were altered.
- This pass confirms the quantitative no-go: T1 did not improve over T0, and T2 lost further answers relative to T1. The contact sheets are a diagnostic aid, not a full-resolution ground-truth adjudication.
