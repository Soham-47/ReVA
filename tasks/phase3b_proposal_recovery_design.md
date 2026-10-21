# Phase 3B — Proposal Recovery Design

**Status:** Design only. No Replicate prediction, Modal job, model inference, training, Qwen scoring, or pilot execution was started. Approval is required before any new inference.

## Goal and fixed conditions

Use only the same two videos (`train_010397`, `train_009869`), the same 16 source frames and timestamps per video, the pinned SigLIP2 checkpoint, and the prior annotation rules. Do not use manually drawn target coordinates, target labels, or oracle outputs as inputs to automatic proposal generation or retrieval. Keep oracle artifacts separate from automatic candidates. Do not change Stage A or Qwen3-VL.

## Offline findings

### Shared endpoint/output limitation

The saved Replicate ZIPs contain one binary mask image per frame and no per-instance IDs or attributable native scores. The inspected `lucataco/sam3-video` wrapper source OR-combines all masks in a frame before writing that image. This explains why downstream sees merged multi-car masks and cannot rank the original SAM instances separately. The [pinned version API schema](https://replicate.com/lucataco/sam3-video/versions/408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269/api) lists `visual_prompt` and `negative_prompt`; the [public wrapper source](https://github.com/lucataco/cog-sam3-video/blob/main/predict.py#L975-L1053) currently parses `visual_prompt` but explicitly leaves it unimplemented, and does not use `negative_prompt`. The source branch is not proven to be the deployed pinned container, so verify the exact version source/build before any prompt test. Treat these fields as nonfunctional until then.

### Black car

- The question identifies a particular car in the lower-right area of the parking lot; the existing `black car` prompt retrieves a category, not that instance.
- The saved question-conditioned union contains **33–38 eight-connected components per sampled frame**; the fixed `vehicle` union contains **48–54**. The black-car union occupies about 3% of each frame. This is evidence that per-instance candidates may be recoverable from cached masks by connected-component splitting, without another SAM 3 call. It is not yet proof that each component is one complete car: touching/fragmented cars need manual instance labels.
- Existing Stage A grounding output for `train_010397` contains only a truncated context with a few `car` tracks; it is not a complete detector cache on the Phase 3A frame set. `train_009869` has no grounded entities in that output. No full matching detector cache or detector weights are stored in the workspace.

### Parking lot

- The question-conditioned `parking lot` proposal overlaps the lot in **16/16** sampled frames, but its mask is a broad, disconnected pavement union with access-road spill (mean area 11.1%; tight mask bounding box reaches the image edges). So the error is not simply “no proposal exists.”
- SigLIP2 ranks black-car candidates above the existing lot candidates: question-conditioned strict Recall@5 is zero, versus random mean about .154–.157. This is a ranking failure on top of a poor scene-region representation. The fixed vocabulary also lacks an exact parking-lot prompt.
- A contextual rectangle may be a better representation than a pixel mask for the whole parking complex. No manual reference box or matching automatic region proposal is present in the audit artifacts.

### Baseball field

- The question-conditioned mask covers the target in the eight target-visible sampled frames, but masks in those frames are incomplete/fragmented and spill onto other green or playing areas. The fixed `field` masks are much broader (mean 63.6% of the frame).
- Existing four tile outputs cover only `train_010397` and are 427×240 source-native crops. They provide a 1.5× linear enlargement of the visible source content, not recovered detail beyond the original 640×360 pixels. No baseball-field tile or refined-prompt inference exists.

### Reuse and cache limits

- `hpc_report.json` records SigLIP2 scores and a cache summary, but not reusable embedding vectors for new crops. New component/box crops therefore need encoding with the same pinned SigLIP2 checkpoint on Modal; unchanged Phase 3A scores can be carried forward.
- Grounding DINO Tiny and SAM2 box-prompt segmentation are already present in the repository and prior workflow. Their code is reusable, but the available detector summaries are incomplete for these exact frames. Running them on this 16-frame set would be new inference using existing pinned models, not a new model family.

## Smallest controlled experiment

### Gate 0 — free checks and annotations

1. Verify the deployed code/image corresponding to the pinned Replicate version. Confirm that `visual_prompt` is actually applied, how coordinates are normalized, which packed-frame index it expects, and whether the output can isolate one prompted object. The public schema alone is insufficient; the currently inspectable wrapper source is a no-op for visual prompts. If the deployed implementation cannot be confirmed, make **zero visual-prompt calls**.
2. Before seeing new candidate results, annotate the target car and the parking-complex bounding region on the same 16 frames, plus the baseball-field polygon on the eight visible frames. Store these annotations as evaluation-only labels, separate from all inference inputs. Preserve an instance ID for the target car across frames. If a baseball boundary is ambiguous at source resolution, mark that ambiguity instead of inventing a sharp edge.
3. Offline, split cached `black car` and `vehicle` union masks into connected-component proposals and boxes. Also derive padded contextual crops from those components and from existing parking-lot masks. These steps require no Replicate or Modal inference.

### Gate 1 — manual black-car oracle (conditional on Gate 0)

If the pinned wrapper is confirmed functional, run two separate, manually guided oracle calls on the same packed 16-frame video: one with a verified target box on an anchor frame, one with a positive target point and negative points on neighboring cars. Use one anchor coordinate set per call; map the packed frame index through the saved source-frame manifest. These outputs are **oracle-only**, kept out of automatic candidates and ranking inputs. This diagnoses whether the model can isolate the car when given the identity directly.

If the oracle does not isolate the car, stop the SAM 3 visual-prompt route. Do not proceed to automatic visual prompting based on that path. Existing-mask connected components remain a separate, non-oracle candidate-generation hypothesis for later review.

### Gate 2 — automatic candidates on the same frames

- **Black car:** first compare cached connected-component masks/boxes against the annotated target. If components recover one target-car region per target-visible frame, rank the automatic component crops. If the target is merged with neighboring cars or missing, the next candidate is the existing Grounding DINO Tiny car boxes, optionally refined by the repository’s existing SAM2 box-mask pipeline. Use only model-produced boxes in this automatic arm; no oracle points, boxes, or target labels enter it.
- **Parking lot:** compare (a) cached SAM 3 mask boxes with fixed padding, (b) automatic Grounding DINO Tiny `parking lot` / `parking area` boxes, and (c) deterministic context-window proposals. Keep a manually annotated whole-lot box as an oracle ceiling only. SigLIP2 compares all automatic boxes with the unchanged text query `parking lot` and the same padded/masked crop processing where applicable.
- **Baseball field:** compare one predeclared refined full-frame text prompt (for example, `baseball diamond including the dirt infield and grass outfield`) against the existing `baseball field` prompt. Separately, run the existing `baseball field` prompt on the four-tile grid so the tile comparison changes resolution/crop only. The tile arm uses source-native crops with exact inverse coordinate mapping. Do not send `negative_prompt` unless the pinned wrapper implementation is verified to consume it.
- **Region fallback:** use the existing DINO→SAM2 code only if connected components and context boxes do not provide sufficient automatic candidates. The existing DINO and SAM2 checkpoints are pinned in the repository; no new segmentation or reasoning model is proposed.

### Gate 3 — scoring and comparison

Use one bounded Modal A100 job to load the existing pinned Grounding DINO Tiny, SAM2, and SigLIP2 checkpoints, build only the approved automatic candidates, and encode only new crop variants. Reuse saved SigLIP2 scores for unchanged candidates. Write all new outputs under a new `runs/phase3b_proposal_recovery/` root; never overwrite Phase 3A or Stage A artifacts.

For every arm, report:

1. **Category coverage:** target-visible frames containing at least one correctly categorized proposal.
2. **Correct-instance coverage:** target-visible frames containing a proposal matched to the annotated target instance. Keep car instance identity separate from category presence and cross-frame track identity.
3. **Boundary quality:** mask IoU and target-pixel coverage/precision for annotated object masks; box IoU and target-region coverage for parking-lot rectangles. Report the contextual-box result separately from pixel-mask results.
4. **Ranking:** SigLIP2 Recall@1/5 over all candidates and conditional on a correct proposal existing; padded RGB and masked crops; random selection from the exact same candidate pool. Report native SAM 3 scores only if the endpoint returns usable attributable scores.
5. **Failure attribution:** no correct candidate (proposal generation), correct candidate below rank (embedding ranking), wrong target identity, or boundary/annotation ambiguity.
6. **Cost/runtime:** calls and attempts per provider, actual Replicate billing if available, Modal wall/GPU time, GPU peak memory, and candidate counts.

Before scoring, freeze provisional matching rules: box IoU ≥0.5 for instance/region boxes; mask IoU ≥0.5 plus target-pixel coverage for mask proposals. Also report raw coverage/precision values so small-object cases are not hidden by one threshold. These are screening rules for this two-video experiment, not claims of general performance.

## Stop conditions

- **API stop:** if the pinned wrapper ignores visual prompts, coordinate convention cannot be established, or output mapping is unreliable, spend nothing on oracle prompting and stop that SAM 3 branch.
- **Oracle stop:** if both box and point oracle prompts fail to produce an isolated target with at least 80% target-pixel coverage and 0.5 mask IoU on the anchor frame, do not build automatic visual prompts on the same path. Report the failure and seek approval before testing a different existing pipeline.
- **Instance stop:** if mask unions or components repeatedly join the target to another car, or the target is not visible at sufficient resolution for a stable identity, report category-only recovery and do not claim correct-instance recovery.
- **Boundary stop:** if independent reviewers cannot agree on the baseball/parking boundary at 640×360, stop pixel-IoU claims for that region and retain the contextual-box metric only. Upscaling cannot restore detail absent from the source.
- **Cost/time stop:** stop at the approved attempt count or Modal timeout; do not retry, extend, or switch providers without approval.

## Minimal paid plan and estimate

This is an estimate only; it is not provider-enforced spend protection. Replicate currently lists an approximate $0.0014 per run and notes that cost varies by input ([model page](https://replicate.com/lucataco/sam3-video)). The previous smoke plan used $0.0017 per attempt as a conservative planning figure.

| Provider | Proposed work, only after approval | Bound used for estimate | Estimated cost |
|---|---|---:|---:|
| Replicate | 2 oracle calls (box and points), 1 parking-area prompt comparison, 1 baseball-diamond prompt comparison, and 4 baseball tiles | 8 planned calls; at most 1 retry (9 attempts total) | ~$0.011–$0.015 using $0.0014–$0.0017/attempt; input-dependent, not a billing cap |
| Modal | One existing-model + SigLIP2 job on A100, 4 CPU, 16 GiB memory, with a 15-minute execution timeout | 900 seconds maximum charged runtime; no persistent volume | About $0.60 at A100 40 GB list rate or $0.71 at the A100 80 GB rate, including listed CPU/memory rates; excludes any separately billed image build/egress |
| **Combined estimate** | Conditional full experiment | No retry beyond the single reserve; no extra jobs | **About $0.62–$0.73**, plus excluded image-build/egress charges and provider variation |

Modal’s current listed A100 rates are $0.000583/sec (40 GB) and $0.000694/sec (80 GB); CPU and memory are billed separately ([Modal pricing](https://modal.com/pricing)). The timeout and attempt counter are operational limits, not a guarantee of a provider account-level dollar ceiling. If the oracle implementation gate fails, make zero oracle calls and stop before the conditional Replicate plan.

## Decision requested

Approve or reject this bounded Phase 3B experiment plan. No inference will run before approval. The initial automatic baseline can be created offline from cached mask connected components; the only reason to make new Replicate calls is the conditional visual-oracle check, one parking prompt comparison, one baseball prompt comparison, and the four baseball tile calls.
