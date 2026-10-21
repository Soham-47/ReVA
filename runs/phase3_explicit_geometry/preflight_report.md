# Phase 3 explicit geometry pilot preflight

Status: **blocked_before_api_calls**.

The local baseline vectors and relation schema passed validation. No paid Replicate/SAM3 call or Modal scoring function call was made; the separate CLI parser/image check did not invoke the scoring function.

## Cost gate

The conservative full-set estimate is $1.602 for 89 SAM 3 calls, below the hard $2.50 cap. The 4-question tranche estimate is $0.162; after a successful tranche, the remaining 56 questions are estimated at $1.440 for 80 new calls.

## Checks

- Frozen G0 baseline: 60 selected questions with four probability vectors and four views each.
- Relation schema self-check: complete; all statements are bounded to normalized 2D image-plane evidence.
- Modal execution path: ready (profile names only; credential values were not read into output).
- Source videos: 54 selected, 53 absent locally; remote staging is available.
- Prior SAM 3 cache reuse: 0 reusable, 60 rejected by the exact prompt/frame-policy gate.

## Blockers

- Replicate API token is not present
- G1 GroundingDINO cache is not staged for this new 60-question manifest
- G0/G1/G2 VLM score outputs are not available locally

The run stops before spending because no Replicate credential is available to the authenticated Modal path. This is an error-enriched pilot plan, not split-wide benchmark accuracy.
