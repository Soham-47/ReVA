# Training-free TG / geometry shortlist (frozen Stage A)

Research date: 2026-11-01. Scope: methods that could improve ReVA Temporal Grounding and spatial/geometric reasoning **without** training, extra frames as the sole lever, or retrials of closed EXP-003/004/008/011–014/018–019/027/031.

Frozen protocol: Qwen3-VL-8B-Instruct + language LoRA r16, 16 frames, four-view TTA. EXP-001: valQ 498/595 (83.70%), valT 885/1,059 (83.57%), official val 1,757/2,000 (87.85%). TG ≈ 73.9 / 74.4 / 75.9%. Peekaboo-VL’s **83.55% ReVA test** is a different protocol (Qwen3.5-9B, LoRA r64, 32 frames, trained 1 epoch); treat it as a ceiling check, not a Stage A win.

No ReVA scores below are invented. Protocol ≠ ReVA 16-frame TTA unless stated.

---

## Ranked candidates (max 8)

### 1. VTimeCoT progress-bar overlay (drop the agent loop)

- **Name / year / license:** VTimeCoT (Zhang et al., ICCV 2025). Paper [arXiv:2510.14672](https://arxiv.org/abs/2510.14672), [open access PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VTimeCoT_Thinking_by_Drawing_for_Video_Temporal_Grounding_and_Reasoning_ICCV_2025_paper.pdf), project [vtimecot.github.io](https://vtimecot.github.io/). No public code found; the tool is OpenCV. Paper copyright is ICCV; implementation can be re-derived.
- **Mechanism:** For each sampled frame, draw a player-style bar under the pixels: track, filled fraction \(t/T\), circle at “now”, and a **seconds** timestamp. The MLLM stays frozen and reads time as a spatial glyph (OCR/shape), not as a language token. Optional second tool: VideoCLIP-XL highlights high-similarity intervals on the bar. Full paper uses AutoGen, up to 3 reasoning steps, 32 frames. For ReVA, use **only the bar on the existing 16 frames**; do not add frames, do not run the agent loop.
- **Evidence (quoted):** Charades-STA, Qwen2VL-7B mIoU **24.34 → 43.41**, R1@0.5 **12.85 → 38.79**; GPT-4o mIoU **40.20 → 46.78**. QVHighlights GPT-4o mIoU **37.66 → 54.49**. Ablation Table 5 (GPT-4o, QVHighlights mIoU): baseline 37.66; text CoT 41.85; CoT+progress bar **49.40**; +highlight 54.49. VideoMME GPT-4o 32-frame w/o subs **61.6 → 64.2**. Authors: “average IoU improvement of 6.58% and 16.83% over GPT-4o across Charades-STA and QVHighlights.” **Protocol ≠ ReVA:** 32 frames, long indoor/web video, generated intervals not A–D letters.
- **Fit:** TG 0.5s near-misses and “when did X appear/disappear.” Visual time is the missing channel: EXP-008 already put **timestamp text** in the prompt and failed. Failure modes: bar eats bottom pixels; 16-frame quantization still bounds IoU; OCR of small bars can fail.
- **Compute:** CPU OpenCV; 0 extra frames; small extra visual tokens from concatenated bar strip. Highlight tool needs a frozen video-text encoder (extra GPU) — skip in v1.
- **Closed EXP conflict:** Not EXP-008 (text timestamps). Not EXP-003 (sample indices). Peekaboo also draws a TG timebar, but with a **trained 32-frame** model.
- **Pilot:** Frozen scorer yes. Same 16-frame TTA. Cohort: all valQ TG + matched geometry/controls (~12+12+8). Gate: TG slice **and** geometry slice beat Stage A on the same IDs; no large non-TG regression.

**Verdict: do next.**

### 2. Peekaboo-VL camera-motion sentence only (not their trained adapter)

- **Name / year / license:** Peekaboo-VL (Le, 2026). Model card [LakoreAI/peekaboo-vl](https://huggingface.co/LakoreAI/peekaboo-vl); code [github.com/LakoreAI/peekaboo-vl](https://github.com/LakoreAI/peekaboo-vl) **MIT**; adapter follows Qwen3.5-9B license. ReVA data remains non-commercial.
- **Mechanism:** CPU ORB + RANSAC on consecutive frames → one sentence in ReVA vocabulary (`Camera motion: 0.0–4.1s moving forward, slow; 4.1–15.0s panning left`) prepended to the frozen prompt. Weights unchanged. **Do not** load their LoRA, **do not** switch to 32 frames or Qwen3.5-9B.
- **Evidence (quoted from the model card):** ReVA **test 4,000: 83.55%**; development 2,078: 85.37%; ReMoSense test 80.1%; text-only floor 55.9%; TG “about 70%” on development; “most of its errors miss the right answer by about one second.” “In our tests, how the hint is worded mattered more than how precise the measurement behind it was.” **Protocol ≠ EXP-001:** trained LoRA r64, 32×640×352 frames, letter scoring + PriDe on 5% of test, Qwen3.5-9B. Their 83.55% is **not** a training-free overlay result and is **not above** EXP-001 official val 87.85%.
- **Fit:** Camera vs object motion (error-review `train_001154`-class). UAV-specific. Failure: wording in image-plane terms “hurt”; homography on nadir parking lots can look like object motion.
- **Compute:** CPU; ~1 sentence of tokens; 0 extra frames.
- **Closed EXP conflict:** None. Distinct from EXP-004/031 (frame selection) and dual-stream InternVideo (training).
- **Pilot:** Same frozen 16-frame scorer; motion sentence on/off; TG + geometry slices; gate as above.

**Verdict: do next, paired with #1 (2×2: bar × motion).**

### 3. VISER sparse rows (not dense grids, not boxes)

- **Name / year / license:** VISER (Izadi et al., NeurIPS 2025). [arXiv:2506.22146](https://arxiv.org/abs/2506.22146), [project](https://sharif-ml-lab.github.io/VISER/), [github.com/sharif-ml-lab/VISER](https://github.com/sharif-ml-lab/VISER). Paper CC BY 4.0 on OpenReview.
- **Mechanism:** Overlay **3 equidistant horizontal lines**, number rows 1–4, prepend “Scan the image sequentially based on horizontal lines…”. Frozen VLM, single query. Authors prefer sparse rows over dense grids because grids “obscure content.”
- **Evidence (quoted):** GPT-4o 2D synthetic: visual search **+25.00%**, counting **+26.83%**, spatial relations **43.00% → 52.50% (+9.50%)**; natural-image spatial **69.39% → 77.43%**. Imagining lines without drawing them is weaker (+8.5% spatial vs drawn). Qwen2.5-VL **drops 3.06%** on natural spatial scenes. **Protocol ≠ ReVA:** still images, often synthetic; not UAV video.
- **Fit:** Image-plane left/above/parallel, counting with partial visibility. Failure: lines over small UAV objects; Qwen family gains are small or negative in their natural split.
- **Compute:** CPU; 0 extra frames.
- **Closed EXP conflict:** Not EXP-011–014 (those overlay detector boxes/masks). Different from SoM.
- **Pilot:** Geometry + counting IDs only first; if TG regresses, keep VISER off for TG.

**Verdict: do next as geometry-only arm after #1–2.**

### 4. See&Trek motion reconstruction on the **fixed 16-frame pool**

- **Name / year / license:** See&Trek (Li et al., NeurIPS 2025). [arXiv:2509.16087](https://arxiv.org/abs/2509.16087), code [Hoantrbl/SeeTrek](https://github.com/Hoantrbl/SeeTrek) **Apache-2.0**.
- **Mechanism:** Visual odometry (ORB/essential matrix) plus **trajectory/point overlays** and a short motion prompt. Paper also does YOLO Balanced-TopK keyframes (MSRS). **For ReVA, disable MSRS**; paint VO cues on Stage A’s 16 frames only.
- **Evidence (quoted):** “most +3.5%” overall. InternVL3-8B VSI-Bench ablation: 40.2 → 41.8 (MSRS) → 43.2 (MSRS+motion). Qwen2.5-VL-7B VSI-Bench avg **27.3 → 29.0**. STI-Bench Qwen2.5-VL-32B **40.5 → 41.7**, with +10.8 on one displacement/path subtask. Indoor ScanNet-style videos, minutes long. **Protocol ≠ ReVA.**
- **Fit:** Ego-motion vs scene; route/order. Failure: indoor VO ≠ UAV 6-DoF; MSRS is the same family as EXP-004/031.
- **Compute:** CPU VO + drawing; 0 extra frames if MSRS is off. YOLO MSRS is extra GPU — skip.
- **Closed EXP conflict:** Full See&Trek **conflicts with EXP-004/031**. Motion overlay only does not.
- **Pilot:** Motion-overlay vs Peekaboo text-only (#2) on the same IDs; keep the cheaper text hint if overlays tie.

**Verdict: do next only as an ablation of #2, not a new sampler.**

### 5. DSE-VTG as an **option-interval decoder** (not a new sampler)

- **Name / year / license:** DSE-VTG (Cao et al., 2026). [arXiv:2609.08850](https://arxiv.org/abs/2609.08850). “Code will be released upon acceptance” — **not released**.
- **Mechanism:** Frozen video-capable embedder (they use Qwen3-VL-Embedding-8B) at 3 fps frames + 4s clips; fuse similarities (MSF); optional query offset TTA; decode \([t_s,t_e]\). For ReVA: run on the **clip**, map each A–D time option to the decoded span (point: |t−t̂|; interval: IoU), use as a **side score** fused with frozen Qwen letter logits. Do not resample Qwen frames.
- **Evidence (quoted):** Charades-STA **R@0.5 58.58, mIoU 51.30** (vs prior training-free +10.00 / +5.61). Same-backbone TFVTG 44.03 mIoU. QVHighlights 38.64 mAP. Charades-CG Novel-Word 50.86 mIoU. Default 3 fps — denser than ReVA’s ~2–2.8 fps, but the **answer model** can stay at 16 frames. **Protocol ≠ ReVA A–D.**
- **Fit:** TG options that differ by 0.5–1s. Failure: indoor actions; extra 8B embedder; no public code.
- **Compute:** Extra GPU for embedding (not Qwen SFT). 3 fps CPU/GPU feature stream, not 16-frame Qwen.
- **Closed EXP conflict:** Related to EXP-008 (temporal localization) but **different mechanism** (similarity curve vs Qwen rescoring denser windows). Not EXP-031’s adaptive 16-frame selector.
- **Pilot:** TG-only valQ; frozen Qwen still produces A–D; decoder may only break ties. Gate: TG up, geometry unchanged.

**Verdict: skip until code exists; then TG-only tie-break, not a scorer replacement.**

### 6. TFVTG query decompose + dynamic/static similarity

- **Name / year / license:** TFVTG (Zheng et al., 2024). [arXiv:2408.16219](https://arxiv.org/abs/2408.16219), [github.com/minghangz/TFVTG](https://github.com/minghangz/TFVTG).
- **Mechanism:** LLM splits the question into sub-events; BLIP-2-style VLM scores dynamic similarity rise vs static inside/outside; merge spans. Auxiliary interval → nearest A–D option.
- **Evidence (quoted):** Charades-STA zero-shot **R@0.5 49.97, R@0.7 24.32, mIoU 44.51** (Table 1 “Ours”). Beats VTG-GPT by 6.29 R@0.5. Indoor/web. **Protocol ≠ ReVA.**
- **Fit:** Compound TG language. Failure: GPT-4 class LLM cost; 3 fps; same Charades transfer risk as #5; weaker than DSE-VTG on the same task.
- **Compute:** Extra LLM + VLM GPU. 0 extra Qwen frames.
- **Closed EXP conflict:** Same family as EXP-008 (external temporal localization). Distinct from sampling changes.
- **Pilot:** Only if #5 is unavailable and TG still fails after #1.

**Verdict: skip for now (dominated by #1 and, if needed, #5).**

### 7. MarkIt **frame-index glyph** only (no Q2M masks)

- **Name / year / license:** MarkIt (2026). [arXiv:2604.25886](https://arxiv.org/html/2604.25886v1). Code/license not verified on a first-party repo in this pass.
- **Mechanism:** Full method: query→open-vocab masks + persistent frame index in a corner. **ReVA v1:** render only the **index/seconds glyph** on the 16 frames (Stage A already has timestamp **text** in the prompt; this is a **pixel** index). Skip masks.
- **Evidence (quoted):** Qwen2-VL-7B Charades-STA mIoU **7.9 → 41.1** with **full** MarkIt (masks+index). Ablation “No nouns (Q2M-Bridge disabled)” “causes a clear performance drop” — so index-only is **not** their headline result. **Protocol ≠ ReVA.**
- **Fit:** Weak TG encoding if the bar in #1 is not used. Failure: full MarkIt **is** SoM/SAM-class (EXP-012).
- **Compute:** CPU for glyphs; masks need extra GPU segmenter.
- **Closed EXP conflict:** Full MarkIt **conflicts with EXP-011–014**. Glyph-only is close to #1 and EXP-008.
- **Pilot:** Subsumed by VTimeCoT bar.

**Verdict: already covered by #1; skip standalone.**

### 8. Ego3D-VLM textual cognitive map (boxes never shown)

- **Name / year / license:** Ego3D-VLM (Gholami et al., ICLR 2026). [arXiv:2509.06266](https://arxiv.org/abs/2509.06266), [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/a72796cf8d02f11e8a0269cedb0af6fc-Paper-Conference.pdf). Uses Grounding-DINO-Base + Depth-Anything-V2-Metric.
- **Mechanism:** REC boxes + metric depth → 3D points → **text/JSON map** prepended to the frozen VLM. Difference from EXP-011: Qwen never sees painted boxes; it reads numbers. Still depends on the same detector family.
- **Evidence (quoted):** Ego3D-Bench (nuScenes/Waymo multi-view, not UAV): “12%” average multi-choice accuracy and “56%” distance RMSE relative improvement. Authors: “we did not address spatial reasoning tasks in conjunction with temporal understanding”; small models “near chance”; VILA1.5-8B “little to no improvement.” **Protocol ≠ ReVA.**
- **Fit:** “closer to / parallel to / equally distant” if depth is trustworthy. Failure: no UAV intrinsics; nadir GSD; Grounding DINO already failed to help QA in EXP-011; extra GPU.
- **Compute:** Extra GPU (DINO + depth). Few extra tokens.
- **Closed EXP conflict:** **Partial EXP-011** (same REC). Mechanistic difference is text map vs box overlay.
- **Pilot:** Geometry-only, only if VISER fails and REC quality is audited separately from QA.

**Verdict: skip as a first pilot.**

---

## Explicitly out of scope (already done or wrong lever)

| Method | Why not |
| --- | --- |
| SoM / SAM / Grounding DINO overlays | EXP-011–014, EXP-027 |
| SAHI / crops | EXP-018–019 |
| Adaptive / A.I.R. / See&Trek MSRS keyframes | EXP-004, EXP-031 |
| Coarse-to-fine Qwen resampling, timestamp **text** | EXP-008 |
| Peekaboo full LoRA + 32 frames + Qwen3.5-9B | Training + protocol change; 83.55% test is not Stage A |
| SIS-Motion optical-flow connector | Trained fusion, not training-free |
| PriDe letter prior | Stage A already rotates options in TTA; Peekaboo’s PriDe uses 5% of **test** |
| Dense Track-Then-Bind | EXP-027 |

---

## Do next / skip / already done

| # | Method | Decision |
| --- | --- | --- |
| 1 | VTimeCoT bar on 16 frames | **Do next** |
| 2 | ORB motion sentence | **Do next** (2×2 with #1) |
| 3 | VISER 3-line rows | Do next after #1–2, geometry-only |
| 4 | See&Trek VO overlay, no MSRS | Ablation of #2 |
| 5 | DSE-VTG option decoder | Skip until code |
| 6 | TFVTG | Skip |
| 7 | MarkIt full / index-only | Skip / subsumed |
| 8 | Ego3D-VLM | Skip |

### Single highest-EV experiment (frozen Qwen3-VL-8B adapter)

Keep EXP-001 sampling, prompt template except two additive channels, letter scoring, and four-view TTA. **Arm A:** OpenCV progress bar (seconds, filled fraction) concatenated under each of the 16 frames — VTimeCoT’s progress-bar tool, not their 32-frame agent. **Arm B:** Peekaboo-style ORB/RANSAC sentence in ReVA motion vocabulary. **Arm C:** both. Do not change frame count, do not load Peekaboo weights, do not paint boxes/masks. Score a fixed cohort: Temporal Grounding plus geometric-relation/layout controls. Promote only if TG **and** the geometry slice beat Stage A on the same IDs with no large residual-task drop. That is the only intervention that is (i) training-free, (ii) 16-frame-native, (iii) aimed at the two documented holes (visual time encoding vs text timestamps; ego-motion vs object motion), and (iv) not a rerun of a closed EXP.
