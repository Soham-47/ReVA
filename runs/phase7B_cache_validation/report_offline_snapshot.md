# Phase 7B cache validation

Offline validation of existing Grounding DINO caches against the canonical Phase 7 frame contract and the reviewed Phase 2 packet.

- Reviewed cases/videos: 24/23
- Video statuses: `{"exact_cache_reuse": 2, "missing_cache": 18, "same_video_prompt_gap": 1, "same_video_prompt_unverified": 2}`
- Case statuses: `{"exact_cache_reuse": 3, "missing_cache": 18, "same_video_prompt_gap": 1, "same_video_prompt_unverified": 2}`
- Policy-exact videos: 5/23
- Timeline-exact videos: 5/23
- Source checksum statuses: `{"match": 1, "unknown_source_missing": 4}`

## Gate

The cache contract is usable for a bounded evidence smoke only where a Phase 3 sidecar maps the same video, policy/timestamps/model identity are exact, and prompt coverage is present. Source checksums remain an explicit unknown for videos absent from `data/ReVA`; obtain that source evidence before treating reuse as fully reproducible.

## Limitations

- Cache records do not contain video_path; same-video reuse is limited to Phase 3 G1 sidecar mappings.
- A missing local source file yields unknown_source_missing; the cache checksum is never treated as a source checksum.
- Prompt coverage is checked from question-derived entities only; no answer, gold, or baseline field is read.
- Gap and switch values are conservative observation proxies, not identity ground truth or tracking accuracy.
