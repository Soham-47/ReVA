# ReVA video VLM RL recipe: evidence and pilot implications

**Scope.** Research-only handoff for a small GRPO/RL fine-tuning preview on ReVA temporal grounding and geometric/spatial reasoning. No training or provider job was run. Claims below are separated into source-backed findings and proposed implications.

## Local ReVA facts that constrain the design

- ReVA is four-way video QA: a clip, question, options A–D, and one option-letter prediction. The repository lists 15,773 labelled train questions over 1,047 clips; `fit_A` has 14,119 rows, `valT` is a video holdout with 1,059 rows, and `valQ` is a question holdout with 595 rows (`README.md` lines 12–18, 24–32, 126–138).
- The frozen Stage A reference is `Qwen/Qwen3-VL-8B-Instruct@0c351dd...`, language LoRA r16 with the vision tower frozen, 16 sampled frames, and two offsets/two option rotations. The recorded development scores are `valQ` 498/595 and `valT` 885/1,059 (`tasks/phase7_execution_plan.md` lines 11–20).
- Temporal labels are not yet safe to treat as uniformly verifiable. In the 100-item audit, 10 were directly usable, 57 conditional, and 33 excluded; the review found clip-boundary censoring, subjective “clearly visible” events, generic non-unique objects, static-scene duration questions, and out-of-range gold times (`results/temporal_training_data_audit/phase1_report.md` lines 1–3, 21–32). This is a direct reason to gate temporal rewards by label quality.
- The existing geometric pilot design keeps source-video annotation separate from automatic proposals, marks non-identifiable instances as such, and evaluates joint geometry only when all required entities and the requested relation are supported (`tasks/qcsrr_final_benchmark_design.md` lines 69–89). Those rules are suitable reward-data hygiene as well as evaluation rules.

## Primary evidence

### GRPO, verifiable rewards, and anchoring

- DeepSeek-R1 defines GRPO as group sampling from the old policy, group-relative standardized advantages, a clipped policy ratio, and a KL penalty to a reference policy. It removes the critic and estimates the baseline from group scores ([DeepSeek-R1, §2.2.1, lines 92–101](https://arxiv.org/abs/2501.12948)). This supports group rollouts and conservative policy updates, but does not establish a particular group size or learning rate for video.
- The same paper uses rule-based accuracy and format rewards and explicitly reports avoiding neural outcome/process rewards because of reward hacking and their extra training cost ([DeepSeek-R1, §2.2.2, lines 108–121](https://arxiv.org/abs/2501.12948)). This supports deterministic parsers for ReVA answer, temporal, spatial, and format terms. It is evidence against using an unconstrained learned judge as the primary reward.
- Official TRL documentation describes GRPO as online self-generated training, normalizes rewards within each group, and logs zero reward standard deviation and clipping statistics. It warns that standard-deviation scaling can introduce question-level difficulty bias and exposes `scale_rewards=False` ([TRL GRPO docs, lines 139–161, 192–198](https://huggingface.co/docs/trl/v0.19.1/grpo_trainer)). The same docs state that `num_generations` defaults to 8 and that the effective batch must be divisible by it ([TRL GRPO docs, lines 586–588](https://huggingface.co/docs/trl/v0.19.1/grpo_trainer)).
- TRL supports multiple custom reward functions, weighted sums, and `None` for a reward that is not applicable to a sample ([TRL reward-function docs, lines 405–507](https://huggingface.co/docs/trl/v0.19.1/grpo_trainer)). For mixed ReVA tasks, “not applicable/unassessable” should be excluded from that component rather than turned into a false zero.

### Video and multimodal RL evidence

- VideoChat-R1 reports task-specific data efficiency from GRPO on video MLLMs, with improvements over Qwen2.5-VL-7B on temporal grounding and object tracking, while preserving general QA in its experiments ([VideoChat-R1 abstract](https://arxiv.org/abs/2504.06958)). Its reward design is directly relevant: regex format reward, interval/box IoU reward, and exact classification accuracy reward ([VideoChat-R1, §3.1–3.2, lines 42–73](https://arxiv.org/abs/2504.06958)). This is the closest source support for combining answer correctness, temporal IoU, spatial geometry, and format.
- Video-R1 identifies a specific video failure mode: vanilla GRPO can use a single-frame shortcut when no temporal reward is present. Its T-GRPO adds temporal awareness, uses SFT cold start before RL, and trains a 7B model with 16/32/64 frames ([Video-R1 abstract and §1](https://arxiv.org/abs/2503.21776), lines 3–16). In its reported ablation, the temporal-aware model beats the no-temporal variant across the listed benchmarks; the authors used 1k RL steps for the main experiments and also report 10k-step scaling (lines 193–200, 227–240). This supports a temporal reward and an SFT initialization, but these are large research runs, not evidence that a tiny ReVA pilot will be statistically decisive.
- Visual-RFT reports task-specific verifiable visual rewards, including IoU, and a competitive result with about 100 samples, but it is primarily image-domain evidence ([Visual-RFT abstract](https://arxiv.org/abs/2503.01785)). Treat its data-efficiency claim as a motivation for a bounded pilot, not as a video compute estimate.
- The Qwen3-VL technical report states that the family includes dense 2B/4B/8B/32B variants and adds explicit textual timestamp alignment for video ([Qwen3-VL technical report](https://arxiv.org/abs/2511.21631)). A 2B or 4B arm is therefore an official family option, but the report does not show that either smaller checkpoint beats this ReVA Stage A 8B adapter. Also, the existing 8B LoRA adapter cannot be reused on a different-sized model; the smaller arm needs its own compatible SFT initialization.

## Assessment of the proposed recipe

The lead’s outline is technically sound as a **screening experiment**: initialize from supervised Stage A-style behavior, freeze a video-held-out evaluation set, use a curated train-only subset, sample groups, combine deterministic task rewards, anchor updates, and compare with the SFT baseline. The following constraints are material:

1. **Make the three-arm comparison explicit.** Use frozen Stage A 8B as the external reference; compare a smaller model’s SFT checkpoint with the same smaller model after GRPO. Keep frame count, offsets, option rotations, prompt, and evaluator fixed. A smaller-model result is not a causal improvement over 8B; it answers whether the cheaper model can retain or improve the target slices.
2. **Do not force structured rewards onto letter-only output.** Current Stage A emits one option letter. Temporal IoU requires a predicted interval, and spatial IoU/relation accuracy requires a structured geometry output or a separately supervised auxiliary task. Either (a) add a strict structured answer contract for a curated RL subset and score its fields, or (b) keep the ReVA-compatible arm answer-only and run structured temporal/spatial rewards in a separate diagnostic arm. Do not infer a temporal interval from free-form rationale or from the answer letter.
3. **Filter reward data before rollout.** Exclude or mask ambiguous identity, clip-censored onset/exit, subjective visibility thresholds, static-duration prompts, and out-of-range labels. Preserve an explicit `unassessable` status. The geometric design’s “not identifiable” rule should be applied before reward construction.
4. **Use conservative pilot controls.** Start from the smaller model’s SFT checkpoint, train LoRA with frozen vision by default, use a small group (likely 4 or 8, subject to memory), and keep one update per rollout initially. A nonzero KL coefficient plus the normal clip constraint is a defensible pilot choice even though current TRL defaults `beta=0`; monitor KL, clip ratio, reward mean/std, zero-std fraction, output length, and invalid-format rate. Group size, beta, reward weights, and step count are hypotheses requiring a pilot sweep, not source-backed constants.
5. **Treat temporal and spatial rewards as separate logged components.** Use exact answer correctness, interval IoU, relation correctness, and format validity as independently reported terms. Normalize or cap components so a broad temporal interval or verbose format cannot dominate. For mixed tasks, return no reward for inapplicable components rather than penalizing them as wrong.
6. **Evaluate against frozen SFT and Stage A.** Primary: video-held-out `valT`; secondary: `valQ`, temporal and geometric/spatial slices, invalid/no-answer rate, and per-video paired transitions. Do not use `valQ`/`valT` for reward design, prompt selection, or stopping decisions. A small curated training set can test feasibility; it cannot establish broad ReVA gains.
7. **Budget for rollout multiplication.** Each example costs roughly `group size × video forward/generation`; 16 frames and long completions multiply that cost. Video-R1’s 1k-step 7B result demonstrates feasibility of the paradigm, not affordability of this checkout. The first ReVA run should be a short, predeclared smoke/pilot with an explicit stop on parser failures, reward collapse, or baseline regression.

## Confidence, caveats, and unanswered questions

- **High confidence:** GRPO’s group-relative baseline, clip/KL structure, deterministic format/accuracy rewards, and TRL group/reward controls; local Stage A split and label-audit facts.
- **Medium confidence:** transfer from VideoChat-R1/Video-R1 to ReVA. They use different model families, labels, reward contracts, and datasets; their improvements do not predict a ReVA delta.
- **Low confidence:** the best smaller size (2B vs 4B), group size, KL coefficient, reward weights, and number of pilot steps. These need an offline reward audit and a minimal compute smoke before any training decision.

Open questions for the lead are: which pinned smaller Qwen3-VL checkpoint is available; whether its video processor and LoRA targets fit the local runtime; what exact structured interval/relation output is permitted; how many clean, independently reviewed temporal and geometry examples remain after filtering; and whether the proposed reward parser can be tested entirely offline against adversarial outputs.

## Source list

1. DeepSeek-AI, *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*, arXiv:2501.12948: <https://arxiv.org/abs/2501.12948>.
2. Hugging Face, *TRL GRPO Trainer* v0.19.1: <https://huggingface.co/docs/trl/v0.19.1/grpo_trainer>.
3. Li et al., *VideoChat-R1: Enhancing Spatio-Temporal Perception via Reinforcement Fine-Tuning*, arXiv:2504.06958: <https://arxiv.org/abs/2504.06958>.
4. Feng et al., *Video-R1: Reinforcing Video Reasoning in MLLMs*, arXiv:2503.21776: <https://arxiv.org/abs/2503.21776>.
5. Lin et al., *Visual-RFT: Visual Reinforcement Fine-Tuning*, arXiv:2503.01785: <https://arxiv.org/abs/2503.01785>.
6. Qwen Team, *Qwen3-VL Technical Report*, arXiv:2511.21631: <https://arxiv.org/abs/2511.21631>.

