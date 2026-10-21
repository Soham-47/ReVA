# Phase 7B manual source review

## Scope and procedure

This is a cache-only review of the six reviewed questions that map to the five
checksum-verified videos. Question wording and original source frames were used
to assess target visibility and reference ambiguity. Gold answers and baseline
correctness were not read. The plain source-only sheets were reviewed before
the detector-overlay sheets; earlier Phase 2 source-review notes were available
in this workspace, so this is not a blinded independent annotation study.

The source videos match the pinned dataset revision and the cached
Grounding-DINO inputs by SHA-256. Each cache covers 32 frames from the union of
the two original 16-frame Stage A offsets. These are sparse observations, not
the proposed 6–10 FPS dense-perception run. Cached proposals are boxes from
`IDEA-Research/grounding-dino-tiny@a2bb814dd30d776dcf7e30523b00659f4f141c71`;
the attached IDs are adjacent greedy-IoU tracks with threshold 0.3. They are
not native SAM identities or identity ground truth.

## Question-level review

| ID | Source judgment | Existing proposal evidence | Evaluation status |
|---|---|---|---|
| `train_007117` | A red car is present in the opening frame. The question gives no stable visual or spatial anchor that distinguishes one red car across the traffic scene. The requested first appearance is left-censored at 0.0 s. | The cache includes `red car` proposals, including an 18-observation greedy-IoU track. That does not establish that it is the referenced physical car. | Category presence: assessable. Correct-instance identity and first-appearance time: not assessable. |
| `train_008006` | Multiple road/parking vehicles occupy the lower portion of the aerial view; “near the bottom right” does not bind one physical vehicle consistently across the clip. | Generic vehicle/car boxes exist. | Correct-instance identity and first-appearance time: ambiguous. |
| `train_008007` | Several white cars are visible from the opening. The question supplies no attribute that distinguishes the intended car. First visibility is left-censored at 0.0 s. | White-car/category proposals exist. | Category presence: assessable. Correct-instance identity and first-appearance time: not assessable. |
| `train_008027` | The large open field is already visible in the opening source frame. First appearance is left-censored at 0.0 s. | The fixed-vocabulary text includes `field`, but no field proposal is present in the cached opening-frame detections. | Scene-region category: assessable. First-appearance time: not assessable; opening-frame proposal miss. |
| `train_000047` | A distinct archway is absent at source frame 84 (2.800 s) and recognizable at frame 90 (3.000 s). The arch structure can be identified; its Chinese characters are not legible at 640×360. | The fixed-vocabulary prompt does not include `archway`. At frame 90 the proposals cover cars/buildings elsewhere, not the arch. No target track can be formed from these proposals. | Target and onset bracket `[2.800, 3.000]` are assessable from source. Proposal coverage fails at first visible frame. |
| `train_001138` | A white van with blue side markings is visible at source frame 0.0 s. The instance is visually distinguishable at the opening, but its first appearance is left-censored by the clip boundary. Cross-frame identity is not established from the sparse source-only review. | The fixed-vocabulary prompt has no `van` phrase. Its frame-0 boxes cover other cars/buildings, not the marked van. | Instance presence at opening: assessable. First-appearance time and cross-frame identity: not assessable; target proposal miss. |

The two source-identified objects (`train_000047` archway and `train_001138`
marked van) are useful qualitative diagnostics, but both are missed by the
existing fixed-vocabulary proposals at the relevant frame. The three
question-conditioned cache matches therefore yield **zero assessable
correct-instance first-visibility cases**. The field case is an additional
scene-region proposal miss at the clip opening.

## Cache coverage and tracking limitations

| Video | Frames | Detections | Greedy-IoU tracks | Singleton tracks |
|---|---:|---:|---:|---:|
| Traffic congestion 050 | 32 | 86 | 27 | 16 |
| Hawk UAV `DJI_0028_d4_01` | 32 | 71 | 48 | 43 |
| Hawk UAV `DJI_0028_d4_02` | 32 | 59 | 33 | 22 |
| VisDrone `uav0000077_00720_v_01` | 32 | 133 | 39 | 17 |
| VisDrone `uav0000243_00001_v_04` | 32 | 97 | 78 | 69 |

The high singleton counts on both Hawk/VisDrone clips and the lack of target
boxes for the archway/van show that the existing cache cannot test persistent
instance identity. A zero label-switch proxy is not evidence of correct
identity: it only reflects the existing adjacent-frame association bookkeeping.
No mask IoU, IDF1, tracking accuracy, or Qwen result is reported.

## Gate decision

Phase 7B cache/source validation is complete, but it does **not** pass the
downstream evidence gate. The existing candidates do not supply a correct,
independently assessable target track for any question. Qwen scoring would
measure the effect of unreliable/missing evidence rather than the proposed
geometric-code and instance-aware tracking approach, so it was not run.

The HPC environment contains the frozen Stage A base/adapter and ReVA
environment, but the checked HPC account has no configured Hugging Face token
and no cached SAM 3 or DA3 checkpoints. Native SAM 3 inference therefore needs
Hugging Face access to the gated `facebook/sam3` checkpoint. DA3 geometry is
deferred until target proposals pass this gate.

## Artifacts

- Source checksum and cached-overlay manifests: `source_checksum_manifest.json`, `contact_sheet_manifest.json`.
- Annotated cached detector/tracker overlays: `contact_sheets/`.
- Plain source-only 32-frame views: `contact_sheets/source_only/`.
- Full-resolution frame checks for the van and archway: `source_review_frames/`.
- Machine-readable annotations: `manual_source_review.json`.
- Preserved initial offline report: `report_offline_snapshot.json` and `report_offline_snapshot.md`.
