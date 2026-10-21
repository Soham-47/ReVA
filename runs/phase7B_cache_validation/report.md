# Phase 7B cache validation

Offline Grounding DINO cache validation plus independent HPC source verification and cached-output overlays.

- Reviewed cases/videos: 24/23
- Policy-exact videos: 5/23
- Case statuses: `{"exact_cache_reuse": 3, "missing_cache": 18, "same_video_prompt_gap": 1, "same_video_prompt_unverified": 2}`
- HPC checksum verification: 5/5 matched at revision `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`
- Clips staged locally: 5/5
- Local source status: `{"match": 5}`

## Source verification

Lead-agent independent HPC verification: SHA-256 of each source clip at /scratch/ambpdc/soham/ReVA/data/ReVA was programmatically compared with every mapped cache video_checksum; all five matched.

The prior offline report is preserved as `report_offline_snapshot.json` and `report_offline_snapshot.md`. The current checkout has no `/scratch` HPC mount, so the attestation is recorded as provenance and is not silently presented as a local re-read.

## Overlay evidence

Contact sheets use cached Grounding DINO boxes and cached adjacent-greedy-IoU track IDs. Every tile is labelled `CACHED DINO/TRACKER OUTPUT — NOT GROUND TRUTH`; no answer, ranking, or oracle field is used.

## Gate

All 5/5 mapped caches have independently verified source checksums at the pinned revision, and all 5 source clips are staged with rendered contact sheets.

## Limitations

- Cache records do not contain video_path; same-video reuse is limited to Phase 3 G1 sidecar mappings.
- Prompt coverage is checked from question-derived entities only; no answer, gold, or baseline field is read.
- Gap and switch values are conservative observation proxies, not identity ground truth or tracking accuracy.
- HPC checksum verification is externally attested at the pinned dataset revision; this checkout could not re-read the HPC mount.
- All 5/5 independently verified source clips are staged and rendered locally.
