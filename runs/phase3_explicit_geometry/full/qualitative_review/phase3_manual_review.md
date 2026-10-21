# Phase 3 manual review and conclusion

## Scope and comparison

This completed pilot compares frozen Stage A baseline G0 with GroundingDINO box-derived context G1 and SAM 3 mask-derived context G2 on the same legacy frames. The selected set contains 30 geometry questions per split: 20 baseline errors and 10 correct controls in each split. This is an error-enriched sample, not split-wide benchmark accuracy. The adapter, 0/0.5 legacy offsets, and two answer rotations were unchanged; no training was run.

| Split | G0 | G1 | G2 | G1 fixes / regressions | G2 fixes / regressions | Changed answer labels |
|---|---:|---:|---:|---:|---:|---:|
| valQ | 10/30 | 10/30 | 10/30 | 0 / 0 | 0 / 0 | 0/30 for either arm |
| valT | 10/30 | 10/30 | 10/30 | 0 / 0 | 0 / 0 | 0/30 for either arm |

All 20 selected baseline errors per split remained wrong in G1 and G2; all 10 correct controls per split stayed correct. G1 and G2 also produced identical answer labels. The result does not show a benefit from this explicit geometry-text recipe.

## Evidence coverage

The strict 2D evidence gate passed only one question: 1/30 valQ and 0/30 valT. For the remaining records, context was not used: 27 questions per split were unsupported subtypes, while 2 valQ and 3 valT records had no usable entity pair. Eight of the 56 remaining questions had no extracted question entities; they generated empty caches without SAM 3 requests and correctly fell back to baseline.

## Visual audit of the one supported-looking case

Question `train_010734` asks whether the red building is to the right of the bridge. G0, G1, and G2 all answer C, which is correct for this selected control. The generated relation text says the red building is “above, inside” the bridge. The GroundingDINO bridge box spans a broad road/bridge region, and the SAM 3 bridge mask follows an oversized connected road/bridge shape; the red building itself is localized. The resulting “inside” relation is therefore a localization artifact and is not evidence for the asked right-of relation. The automatic `reliable_2d` label only reflects a repeated normalized image-plane relation across sampled frames; it does not validate object localization or question relevance.

Visual evidence: [GroundingDINO boxes](grounding_boxes_contact_sheet.png) and [SAM 3 masks](sam3_masks_contact_sheet.png).

No answers changed, so there were no changed cases to inspect. The only context actually passed to the VLM was manually checked above. This audit is qualitative: ReVA supplies no gold boxes or masks.

## Cost and execution notes

The full scope made 80 SAM 3 requests. Including 11 conservative attempts from the tranche (which included a rate-limit retry), the cumulative upper-bound estimate is 91 × $0.018 = **$1.638**, below the approved $2.50 cap. The completed retry reused all 80 full-scope masks and made zero additional Replicate calls.

The first full run completed inference and failed while creating an empty overlay for questions with no extracted entities. The overlay writer now supports a valid empty cache, allowing downstream baseline gating. A second local CLI false alarm came from re-uploading that first run’s stale error marker; the Modal runner now clears only its per-run completion/error markers before execution. The full completion sentinel and all four score files are present.

## Conclusion

This implementation did not improve paired answer accuracy, and usable geometry evidence covered only 1 of 60 questions. The supported-looking example was visually unreliable and did not encode the queried relation. These results do not justify proceeding to the larger E3 architecture or fine-tuning phases under the current evidence. A new experiment would first need approval and a revised, question-aligned entity/relation evidence design with a higher usable-evidence rate.
