# Locate Anything research handoff

**Checked:** 2026-10-29. **Scope:** identify the model meant by “Locate Anything”, verify first-party capabilities/availability, and assess fit for ReVA’s four-option drone-video QA.

## Bottom line

The name is ambiguous:

1. **NVIDIA `nvidia/LocateAnything-3B`** ([model card](https://huggingface.co/nvidia/LocateAnything-3B), [NVIDIA project page](https://research.nvidia.com/labs/lpr/locate-anything/), [official code](https://github.com/NVlabs/Eagle/tree/main/Embodied)) is a fast visual grounding/detection model. Its documented primitive is an image plus text query that returns boxes or points. It is a plausible *per-frame spatial evidence provider*, not a drop-in video-QA or tracker.
2. **MBZUAI/Oryx “Locate Anything in Videos”** ([paper](https://arxiv.org/abs/2608.28192), [code](https://github.com/mbzuai-oryx/ParallelTubeDecoding), [checkpoint](https://huggingface.co/MBZUAI/ParallelTubeDecoding-Qwen3-VL-4B)) is a separate Qwen3-VL-4B model trained for spatio-temporal grounding. It predicts one temporal interval and a spatial tube and reports zero-shot grounded-VideoQA results. This is the closer conceptual match to ReVA, but there is no UAV/drone evidence in the published training/evaluation scope.

Neither source supports assuming reliable aerial-video performance.

## NVIDIA LocateAnything-3B

**What it actually supplies.** NVIDIA describes Parallel Box Decoding (PBD): complete box coordinates or points are decoded as atomic units. The official page specifies “an image and a natural language text query” as input, and the model card lists input types as **Image and Text** and output as text containing structured box/point coordinates. The architecture is MoonViT plus a Qwen2.5-3B language decoder. Supported tasks include open-set/dense object detection, phrase/referring-expression grounding, pointing, OCR/layout, robotics, and remote sensing. The official page reports mean F1 **39.9 on VisDrone** (versus 35.8 for Rex-Omni), which is encouraging for frame-level aerial object localization but is not a ReVA QA result.

**Video is a wrapper, not demonstrated temporal reasoning.** The official Hugging Face demo accepts video, but its current `app.py` function `run_video_gpu_api` samples at most `max_video_frames` (default 4), calls `worker.generate(...)` inside a loop once per sampled frame, then draws each result back onto that frame. It does not maintain object identity, predict a temporal interval, or decode a tube. The official `LocateAnythingWorker.predict`/`detect_batch` APIs likewise take a PIL image or `(image, question)` pairs. The processor contains generic video sampling utilities (`MAX_FRAMES=64`), but the released model card, worker API, and training corpus are image-oriented; this should not be interpreted as temporal tracking or video-QA support.

**Availability and license.** Weights and code are public (released 2026-05-26), and the NVIDIA Space demo is currently running. The model is not shown as deployed by an HF Inference Provider. The model license is NVIDIA’s non-commercial/research-and-evaluation license; the official `LICENSE_MODEL` limits use to non-commercially, so check this before any deployment beyond research.

Primary source details: [NVIDIA model card](https://huggingface.co/nvidia/LocateAnything-3B), [worker API](https://github.com/NVlabs/Eagle/blob/main/Embodied/locateanything_worker.py), [video demo wrapper](https://huggingface.co/spaces/nvidia/LocateAnything/blob/main/app.py), [video processor](https://github.com/NVlabs/Eagle/blob/main/Embodied/eaglevl/utils/locany/processing_locateanything.py), [model license](https://github.com/NVlabs/Eagle/blob/main/Embodied/LICENSE_MODEL).

## MBZUAI/Oryx Parallel Tube Decoding (the other likely referent)

**What it supplies.** The paper’s Parallel Tube Decoding (PTD) first predicts a temporal interval and then generates all time-conditioned spatial boxes in parallel. The released checkpoint is `MBZUAI/ParallelTubeDecoding-Qwen3-VL-4B`; the repository quick start explicitly uploads a video and asks for an object reference, event time segment, and per-time bounding boxes. The model card documents `fps=2`, `max_num_frames=64`, a Qwen3-VL-4B backbone, and a custom PTD inference path (standard Transformers loading alone is insufficient).

**Evidence and limits.** The paper reports zero-shot ReXTime grounded-VideoQA of **73.1 answer accuracy / 47.8 mean temporal IoU**, and evaluates temporal grounding on Charades-STA and ActivityNet. The checkpoint card says training used VidSTG and HC-STVG v1/v2; Charades-STA and ActivityNet were zero-shot. It predicts **one continuous interval and one spatial tube per query**, and explicitly lists failures for subtle state changes, small/fast/occluded targets, multiple disjoint occurrences, and multiple simultaneously valid instances. Those limitations overlap ReVA’s suspected temporal and label/time ambiguity.

**Availability.** The merged FP16 checkpoint and Apache-2.0 code are public, but the model page currently shows no hosted inference provider. The release includes no training/GRPO data, so adapting to UAV footage would require separate data and substantial work.

Primary source details: [PTD paper](https://arxiv.org/abs/2608.28192) (Sections 4.4 and 5), [official repository README](https://github.com/mbzuai-oryx/ParallelTubeDecoding), [checkpoint model card](https://huggingface.co/MBZUAI/ParallelTubeDecoding-Qwen3-VL-4B).

## Fit for ReVA

ReVA is four-option drone-video QA with Qwen3-VL-8B and sparse-frame/TTA scoring ([architecture](architecture.md)). Existing grounding/segmentation pilots were not promoted and the two-video tracking feasibility gate failed ([EXP-011–014](EXPERIMENTS.md#exp-011), [EXP-027](EXPERIMENTS.md#exp-027)). Therefore:

- **NVIDIA model:** worth one narrow geometry pilot only. Run it per existing baseline frame pool for explicit object/point/box queries, then pass compact structured detections to the frozen Qwen answer model. Measure localization recall/precision and answer accuracy separately. Do not call it tracking, temporal grounding, segmentation, or a replacement for Qwen’s answer head. VisDrone is a useful prior, but it is an image detection benchmark and does not establish performance on ReVA’s camera motion, altitude, small objects, or question semantics.
- **PTD model:** the stronger candidate for the temporal cohort because it natively produces time plus boxes and has a grounded-VideoQA benchmark result. Use it as an evidence proposal/diagnostic first, not as an assumed replacement scorer: its published benchmarks are human-action videos, not UAV videos, and the single-interval/single-tube formulation may mismatch ReVA questions. Keep the final four-option decision in the existing evaluation harness so answer gains cannot be confused with better localization alone.

### Small pilot recommendation

Use a fixed, manually auditable **32-question cohort**: 12 temporal, 12 geometry/object, and 8 controls. Keep the baseline frame sampling and question text unchanged. Compare (a) frozen Qwen baseline, (b) NVIDIA per-frame box/point side-context, and (c) PTD interval/tube evidence on the temporal subset. Record answer accuracy, temporal IoU where labels exist, box/point plausibility, and regressions by question type. Stop promotion unless a model improves its intended metric on both valQ and valT-style slices without introducing broad regressions. A manual audit should specifically check identity switches, small/fast targets, and whether the evidence actually contains the answer.

## Confidence, caveats, unanswered questions

- **High confidence:** the two projects are distinct; NVIDIA’s released model is grounding/detection-first; PTD is video/STVG-first.
- **Medium confidence:** NVIDIA’s image-trained model may still be useful on ReVA frames because of its VisDrone result, but transfer to this dataset is unverified.
- **Open:** whether ReVA can legally use NVIDIA’s non-commercial model under its eventual distribution; whether PTD’s answer-generation behavior maps cleanly to ReVA’s A–D format; and whether ReVA clips contain multi-segment events or multiple valid instances, which both models handle poorly or not at all.
