# GeoVLM-R1 transfer assessment for ReVA

**Date:** 2026-10-21  
**Scope:** Research-only assessment. No weights were downloaded, jobs were launched, or ReVA experiment code/configuration was changed.

## Finding

GeoVLM-R1 is useful as a **reward and training-pattern reference**, but it is not a drop-in ReVA backbone or checkpoint. Its reported model is an image-only remote-sensing system built from `Qwen2.5-VL-3B-Instruct`, trained on Earth Observation (EO) image tasks with supervised fine-tuning (SFT) followed by task-aware GRPO. ReVA currently scores 16-frame drone videos with `Qwen3-VL-8B-Instruct`/LoRA and emits one answer letter for four-way QA. The strongest plausible transfer is to adapt the reward-contract ideas to a small, structured ReVA temporal/geometric arm; the claimed GeoVLM-R1 benchmark gains do not establish a gain on ReVA.

## What the primary sources establish

### Method and output contract

- The paper defines a two-stage pipeline: SFT over referred detection, grounding, region captioning, classification, and temporal change detection, then GRPO with task-aware rewards. GRPO samples a group of responses, uses group-relative standardized advantages, clipping, and a KL term ([paper, §3](https://arxiv.org/html/2509.25026#S3)).
- The output is explicitly structured as `<think>...</think>` followed by `<answer>...</answer>`. The format reward is 1 only when both tags are present; the total reward adds a task-specific accuracy term ([paper, §3.3](https://arxiv.org/html/2509.25026#S3.SS3)).
- Accuracy rewards are task-specific: recall for classification, Jaccard for VQA, Levenshtein similarity for captions, SBERT for region captions, IoU-based detection, and hybrid lexical/semantic plus detection rewards for grounding/change descriptions ([paper, Table 1 and §3.3](https://arxiv.org/html/2509.25026#S3.SS3)). This is the paper's most transferable idea: use a verifier that matches the output and label semantics rather than one generic reward.

### Data and compute

- The RL mixture is EO image data, not drone video: BigEarthNet (30,000 classification pairs), RSCIS (43,670 caption pairs), RSVQA-LRBEN (57,223 VQA pairs), GeoChat-Instruct (69,269 region-captioning and 73,000 referred-detection examples), xBD (4,202 referred-detection and 2,283 detection examples), and 352,825 change-caption examples from LEVIR-MCI/Dubai-CC/MUDS ([paper, Table 1](https://arxiv.org/html/2509.25026#S3.SS3)).
- The implementation details report Qwen2.5-VL-3B-Instruct, 448×448 image resizing, and normalized rotated boxes. SFT used 8 A100 GPUs for 2 epochs; GRPO used 4 A100 GPUs for 2 epochs, batch size 1, gradient accumulation 2, bf16, temperature 0.9, KL coefficient 0.04, and learning rate 1e-6 ([paper, §4.1](https://arxiv.org/html/2509.25026#S4.SS1)). These are a multi-GPU reference run, not a low-cost ReVA recipe.
- The current official IBM toolkit documents separate Python 3.11 environments, a Qwen2.5-VL-3B SFT stage, task-specific GRPO scripts, benchmark inference, and evaluation. It expects EarthDial-Instruct and GeoVLM benchmark folders and says to set GPU count/model/data paths in the run scripts ([official toolkit README](https://github.com/IBM/GeoVLM-R1-Toolkit#usage)).

### Availability and restrictions

- The toolkit is now publicly available and includes SFT, GRPO, inference, and evaluation code plus RL training data ([official toolkit README](https://github.com/IBM/GeoVLM-R1-Toolkit#scope)).
- The project page still says the GeoVLM-R1 model checkpoints will be released on Hugging Face; therefore, as of this review, the public release should be treated as an implementation/data release rather than a verified downloadable checkpoint release ([project page, News](https://mustansarfiaz.github.io/GeoVLM-R1/)). The toolkit's inference scripts require a local `GeoVLM-R1` checkpoint path, so inference cannot be assumed to work from the repository alone ([official toolkit README, Inference](https://github.com/IBM/GeoVLM-R1-Toolkit#4-inference)).
- The official toolkit is **CC BY-NC 4.0**, for non-commercial research only; commercial use requires prior written permission. Its README also states that third-party code and data retain their original licenses ([official toolkit README, license notes](https://github.com/IBM/GeoVLM-R1-Toolkit#license)). The README does not provide one blanket license for every upstream EO dataset, so each dataset's terms must be checked before copying data into ReVA.

## Transfer to ReVA

ReVA's task contract is materially different: four-way QA over drone clips, one option-letter prediction, 16 sampled frames, Qwen3-VL-8B-Instruct LoRA with a frozen vision tower, and a current Phase 8 comparison against Qwen3.8-27B and Qwen3-VL-32B ([ReVA README, lines 12–32](../README.md); [Phase 8 contract](../tasks/phase8_pretrained_scorer.md)). GeoVLM-R1 does not provide evidence for video temporal localization, timestamp prediction, or the ReVA answer-letter interface. Its xBD “temporal” results are bi-temporal remote-sensing image tasks, not continuous video; its VQA results are remote-sensing image VQA.

There is still a useful domain overlap: both settings use aerial/EO imagery, and GeoVLM-R1 reports gains on classification, VQA, temporal change assessment, and detection. That supports a **medium-confidence method-transfer hypothesis** for spatial/scene reasoning. It is weak evidence for ReVA's temporal grounding because ReVA needs identity and event evidence across video frames, and the GeoVLM-R1 training data/rewards do not encode frame intervals or event onset/exit.

The output contract is the main engineering mismatch. A one-letter ReVA scorer cannot receive an IoU, timestamp, or relation reward from its answer string. To use GeoVLM-R1's pattern, a separate structured arm would need to emit a bounded answer plus fields such as an event interval or relation, and the evaluator would need to distinguish `unassessable`/ambiguous cases from incorrect answers. This should not replace the frozen letter-only baseline or its validation protocol.

## Bounded recommendation

Do not replace the Qwen3-VL Stage A/Phase 8 backbone with GeoVLM-R1. If the lead wants one low-cost test, reuse only the **method pattern**:

1. Select a small train-only, video-held-out-from-validation cohort of independently reviewed temporal and geometric questions.
2. Keep the current ReVA model and 16-frame preprocessing fixed, and run an offline parser/reward audit first. Require strict structured output, deterministic answer-letter correctness, and one task-specific component (temporal interval IoU or geometric relation/box score) only where the label is verified; mark ambiguous cases as unassessable.
3. Compare a frozen SFT checkpoint with a short LoRA+GRPO smoke on that cohort, logging format validity, parser failures, reward variance/zero-variance groups, temporal/spatial reward components, and exact-match accuracy. Keep `valQ`/`valT` untouched until the arm is frozen.

This tests the transferable claim at low scope. It does not justify importing EarthDial or other EO training data, adopting the paper's multi-GPU settings, or claiming that GeoVLM-R1 itself improves ReVA.

## Confidence, caveats, and unanswered questions

- **High confidence:** method, base model, output format, reward families, reported compute, repository components, and CC BY-NC restriction; all are stated in the cited primary sources.
- **Medium confidence:** transfer of task-aware rewards to ReVA's spatial/temporal slices; this is a design inference, not a result reported by GeoVLM-R1.
- **Low confidence:** any expected ReVA accuracy delta. There is no apples-to-apples ReVA video experiment, no released checkpoint verified here, and no evidence that EarthDial/EO image pretraining improves Qwen3-VL video QA.

Open questions are whether the checkpoint release becomes available under compatible terms, whether the released GRPO code can be adapted to ReVA's Qwen3-VL/Transformers runtime without changing video preprocessing, and how many ReVA temporal/geometric labels remain clean enough for a non-hacking verifier.

## Sources

1. [GeoVLM-R1 project page](https://mustansarfiaz.github.io/GeoVLM-R1/)
2. [GeoVLM-R1 paper, arXiv HTML](https://arxiv.org/html/2509.25026)
3. [IBM official GeoVLM-R1 Toolkit](https://github.com/IBM/GeoVLM-R1-Toolkit)
4. [ReVA repository README](../README.md)
5. [ReVA Phase 8 pretrained-backbone scorer contract](../tasks/phase8_pretrained_scorer.md)
