# Phase 3A two-video retrieval audit

## Scope and method

Audited only the saved smoke outputs for `train_010397` and `train_009869`. Human judgments used the 640×360 source frames and SAM 3 masks, without gold or predicted QA answers. Source-frame target judgments were assigned by reviewing the proposal overlays, then propagated to candidate annotations; this is a qualitative audit, not pixel-ground-truth scoring.

The retrieval pool contains 64 candidates per query/arm/style. Strict Recall@K counts a candidate as relevant only when its proposal category matches the requested entity. Partial overlap is listed separately and excluded from strict recall. Random is the mean of 1,000 saved random top-five draws from the same candidate pool. For baseball field, recall denominator is the eight sampled frames where the target diamond is visibly present; the other eight frames contain other fields/courts or generic green areas.

## Proposal coverage and mask quality

| Entity | Proposal coverage | Correct instance / boundary judgment | Cross-frame identity |
|---|---|---|---|
| Black car | Target car is covered in 16/16 sampled frames by both question-conditioned and fixed vehicle proposals. | Category coverage is good, but each mask is a disconnected union of many cars. No candidate isolates the target car (0/16 instance-pure masks); union boxes span nearly the full frame. | Not verified; masks have no track IDs. |
| Parking lot | Question-conditioned masks overlap the lot in 16/16 frames. Fixed vocabulary has no literal parking-lot prompt; its road masks overlap only part of the lot. | Question-conditioned pavement includes aisles and adjacent access roads (mean 11.1% of frame). Fixed road proposals are not a correct lot category. | Not verified. |
| Baseball field | Question-conditioned masks returned in 15/16 frames and overlap the target in the 8/8 target-visible frames. Fixed field masks returned in 16/16 and overlap the target in 8/8 target-visible frames. | Question-conditioned mean mask area 16.9% (1.1–32.7%), with incomplete/fragmented field boundaries and spill onto nearby green/playing areas. Fixed field masks average 63.6% and substantially over-cover. Later generic fields/courts are false positives for this target. | Not verified; similar fields cannot be linked as one object. |

Visual examples: black-car proposal overlays (local-only artifact: `proposals_train_010397_call10_black_car.png`), parking-lot proposal overlays (local-only artifact: `proposals_train_010397_call11_parking_lot.png`), baseball-field overlays (local-only artifact: `proposals_train_009869_call15_baseball_field.png`). Full set: `proposals_*.png`. Ranked source-resolution crop panels show padded RGB and masked RGB candidates, rank, similarity, source frame/time, proposal prompt, and match judgment: black car (local-only artifact: `ranked_crops_black_car.png`), parking lot (local-only artifact: `ranked_crops_parking_lot.png`), baseball field (local-only artifact: `ranked_crops_baseball_field.png`). Original-pixel crop files are in `candidate_crops/`.

## Retrieval: SigLIP2 versus random

Values are Recall@1 / Recall@5. `P` = padded RGB box; `M` = masked RGB. Bold interpretation is category/frame-level only; it does not imply correct-instance retrieval.

| Entity | Proposals | Crop | SigLIP2 R@1 / R@5 | Random mean R@1 / R@5 |
|---|---|---|---:|---:|
| Black car | Fixed | P | .063 / .125 | .017 / .081 |
| Black car | Fixed | M | .000 / .063 | .016 / .077 |
| Black car | Question-conditioned | P | .063 / .313 | .030 / .154 |
| Black car | Question-conditioned | M | .063 / .313 | .032 / .157 |
| Parking lot | Fixed | P | .000 / .000 | .000 / .000 |
| Parking lot | Fixed | M | .000 / .000 | .000 / .000 |
| Parking lot | Question-conditioned | P | .000 / .000 | .032 / .157 |
| Parking lot | Question-conditioned | M | .000 / .000 | .031 / .154 |
| Baseball field | Fixed | P | .000 / .000 | .030 / .132 |
| Baseball field | Fixed | M | .125 / .250 | .027 / .131 |
| Baseball field | Question-conditioned | P | .000 / .250 | .065 / .335 |
| Baseball field | Question-conditioned | M | .125 / .250 | .067 / .334 |

### Retrieval examples and failure attribution

- **Black car:** question-conditioned SigLIP2 reaches category/frame Recall@5 .313 for both crop styles, versus .154–.157 random. Its top padded candidates include source frames 225 (7.51s) and 165 (5.51s); masked top candidates include frames 150 (5.00s), 45 (1.50s), and 0 (0.00s). These are car-containing union masks, not isolated target instances. Fixed padded top ranks instead include vehicle/building proposals at frame 225; fixed masked top ranks favor tree proposals. Candidate generation is the dominant instance-level failure; SigLIP2 improves frame selection on the question-conditioned pool but cannot recover an instance mask that does not exist.
- **Parking lot:** question-conditioned SigLIP2 has zero strict Recall@1/5, below random (.032/.157 padded; .031/.154 masked). Its top five are black-car masks, not the parking-lot masks. Fixed proposals have no exact parking-lot category, and road masks only partly overlap the lot. Both proposal vocabulary/boundaries and ranking fail; the embedding ranking is the immediate reason the available question-conditioned lot proposals are missed.
- **Baseball field:** question-conditioned SigLIP2 Recall@5 is .250 for both styles, below random (.335/.334); padded Recall@1 is zero, masked Recall@1 .125. Padded top ranks include wrong green/field regions at source frame 157 (5.24s) and target-overlap candidates at frames 78 (2.60s) and 11 (0.37s). Masked ranks include a target-overlap at frame 33 (1.10s), but then wrong regions at 146 and 123 before another target-overlap at 45. This is a combined proposal-boundary and embedding-ranking failure.

Partial-overlap sensitivity does not materially rescue the parking-lot result: fixed road candidates sometimes cover pavement within the lot, but remain road-labeled regions rather than a correctly segmented lot. No attributable native SAM 3 proposal confidence scores were present in the saved ZIPs, so native-score ranking is unavailable.

## Source mapping, execution, and cost

- The 32 audited frames map to the manifest’s original source indices and timestamps (`t = frame_index / 29.97003 fps`). Dimensions are 640×360; all masks from the primary 16 proposal calls are 640×360; all ranked crop boxes are within source-frame bounds. Original frames are linked from train_010397 (local-only artifact: `source_frames_train_010397.png`) and train_009869 (local-only artifact: `source_frames_train_009869.png`).
- Four tile-probe masks are 427×240. They were not included in the retrieval pool or the audit metrics.
- SAM 3 predictions ran remotely through Replicate. Smoke-time SigLIP2 embedding/ranking and preprocessing ran on one **H200 NVL**: 92.072 seconds, 1.66 GiB peak allocated CUDA memory, 1.69 GiB reserved, 2.60 GiB peak RSS; Slurm recorded 3.1 GiB MaxRSS with 4 CPUs and a 32 GiB allocation. This audit’s frame extraction, labeling, and contact-sheet generation ran locally on CPU; no Modal, HPC, Replicate, or additional model inference was used for this audit.
- The manifest records 20 planned prediction records, 20 successful records, and 22 client prediction attempts including two retries. It does not contain Replicate prediction IDs or account billing data, so the billable prediction count and actual charge cannot be reconciled from local artifacts. **$0.0374 is only the planning estimate for 22 attempts, not actual spend.** Replicate usage is compute-time billed; account-level usage needs to be checked in the provider account ([Replicate billing documentation](https://replicate.com/docs/topics/billing)).

## Recommendation

**Improve candidate generation before considering the retrieval-only pilot.** Prioritize per-instance separation for cars and tighter semantic boundaries for parking lots and fields. The smoke set shows that SigLIP2 can rank some question-conditioned car frames above random, but it cannot fix merged masks; it misses parking-lot regions and underperforms random on question-conditioned baseball-field Recall@5. A 32-question retrieval pilot would be hard to interpret until candidate granularity and embedding failure are separated with stronger proposals and audited target labels.

This is provisional evidence from three entities in two selected videos and 16 sampled frames per video (only eight target-visible baseball-field frames). It does not establish generalization or a numerical go/no-go gate for the 32-question cohort.
