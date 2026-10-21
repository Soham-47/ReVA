# Phase 1: temporal training-data feasibility audit

**Decision: do not build a training set from this sample yet.** The audit found **10/100 directly usable**, **57/100 conditional**, and **33/100 excluded**. Even if every conditional case can be repaired, only 67/100 have usable or potentially usable visual evidence, below the predeclared 80/100 gate. This is a single-reviewer feasibility screen, not a claim about every ReVA training example.

## Scope and method

- Dataset revision: `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`; `train.json` SHA-256: `1d5702a698ef322d171fa16cc845c47906f3f298b8d1c9a023f60f706450da88`.
- Population: 913 eligible temporal QA rows across 468 videos after excluding all `valQ`/`valT` QA IDs and video paths.
- Deterministic stratified sample: 100 QA rows from 100 distinct videos, seed `20261019`; six temporal strata and four dataset sources.
- Blind pass: reviewed the question and two legacy-sampler contact sheets (offsets 0.0 and 0.5) before revealing the options/gold. Then audited the selected gold answer against the visual notes.
- Evidence limit: thumbnail-resolution sampled frames only. This does not establish exact event times from continuous or full-resolution video. No model inference, training, or GPU run was performed.

## Results

| Visual disposition | Cases | Meaning |
|---|---:|---|
| Directly usable | 10 | Target and event are clear enough for a coarse, evidence-backed temporal label. |
| Conditional | 57 | Relevant evidence exists, but target/event refinement, relabeling, or explicit clip-boundary censoring is required before training. |
| Exclude | 33 | No stable target/event or insufficient visual evidence to produce trustworthy temporal supervision. |

Gold-answer review: **10 supported**, **51 partial or censoring-dependent**, **11 not supported**, and **28 not assessable** from the reviewed sheets. These judgments are conservative and remain subject to independent review.

The preliminary numeric screen found **10/100** records with at least one time option more than one frame beyond the clip duration. **Three gold choices** exceed that limit: TA021 (4.8 s vs 4.6046 s), TA028 (3.2 s vs 0.2002 s), and TA056 (18.9 s vs 18.7333 s). These are confirmed arithmetic/range violations; the contact sheets cannot establish what the intended corrected label should be.

## Main failure patterns

1. **Untrackable targets:** repeated generic “white/black/red/blue car,” “the vehicle,” or small distant objects in scenes with several candidates.
2. **Clip-boundary censoring:** an object or action is already present at the first sampled frame, yet the question asks when it first appears/starts. These need an explicit left-censored label such as `[0, first observation]`, not a guessed pre-clip onset.
3. **Subjective events:** “fully/clearly visible,” “most prominent,” “primary focus,” and scene-reveal wording do not define a reproducible visual threshold.
4. **Static scene-duration prompts:** questions about roads, buildings, courts, flags, or landscapes often have both boundaries clipped and no in-video change; a duration answer like “most of the video” provides weak supervision.
5. **Video/answer disagreement:** beyond the three out-of-range golds, the visual review flags cases such as TA008, TA009, TA011, TA044, TA063, TA067, and TA078 for clear or likely disagreement. The source label should be checked rather than silently rewritten.
6. **Sparse visual evidence:** the legacy grids cannot reliably distinguish small objects or provide exact times; long clips have wider gaps. TA028 also contains only six decoded frames.

## Gate and next action

The predeclared gate was at least 80/100 cases with usable visual evidence plus independent agreement on at least 18/20 cases. This sample has **10 directly usable + 57 conditional = 67 evidence-bearing or potentially repairable cases**, below the 80-case threshold even if every conditional case is salvaged. The visual-evidence gate therefore fails. The independent-review gate is **not yet evaluated**; a blind packet is prepared at `phase1_second_review_packet.json`. Do not begin targeted training from this sample. First, independently review that 20-case packet, then revise the curation rules to keep only unique targets, operational event definitions, explicit left/right-censor flags, and gold labels inside the clip. Re-audit a fresh sample before considering training.

Passing a later data gate would establish annotation feasibility only. It would not show that targeted training improves temporal reasoning; that requires a separate paired, controlled model experiment and explicit approval.

## Audit files

- `phase1_visual_review.json`: blind visual disposition and case-level notes.
- `phase1_gold_answer_review.json`: revealed gold/options and separate support assessment.
- `phase1_second_review_packet.json`: 20-case independent blind packet; contains no choices, gold, or first-review notes.
- `pre_review_checks.json`: deterministic numeric-option screen.
- `review_material/`: generated contact sheets for all 100 cases.
