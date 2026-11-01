# Aerial/UAV video-understanding papers relevant to ReVA

Research date: 2026-10-29. This is a primary-source shortlist for ReVA’s four-option drone-video QA (16 sparse frames, Qwen3-VL-8B baseline). “UAV-specific” means the data or evaluation is explicitly aerial/UAV; “generic” means the method has not been shown on ReVA-like aerial footage. None of the papers below establishes an improvement in ReVA A–D accuracy without a ReVA pilot.

## ReVA anchor

**ReVA / ReMoSense (2026, UAV-specific).** The ReVA paper itself is the closest task match: 2,438 real UAV videos, 580K frames, and 21,773 QA pairs across 11 tasks. Its ReMoSense model explicitly separates inter-frame correspondence (camera/global motion) from object temporal dynamics, motivated by camera ego-motion coexisting with small object motion in aerial video. The paper’s QA pipeline also uses human verification. Sources: [paper](https://arxiv.org/html/2609.35507v1) (dataset and QA lines 56–72, motion motivation/architecture lines 171–176). This is the right local reference, but the paper is not evidence that another plug-in will fix the existing ReVA errors.

**Same-benchmark result:** The paper reports 80.04% on its ReVA test evaluation for ReMoSense, versus 73.50% for its same-backbone ablation, with larger gains on geometric relation (+6.3 points) and structural layout (+8.2 points). The controlled ablation uses a Qwen2.5-VL 7B backbone, global inter-frame correspondence tokens, and object-motion tokens; removing global correspondence tokens lowers Temporal Grounding by 9.0 points. These are the authors’ results under their own test protocol, so they are not directly comparable with this repository’s Qwen3-VL-8B official-validation/TTA score.

## Curated shortlist

### UAV-specific evidence

1. **MoRe-UAV: A Large-Scale Benchmark for Motion-Aware Visual Grounding in UAV Videos (2026).**

   - **Targets:** identity under moving cameras, viewpoint/scale changes, and motion-language queries (turning, following, passing, crossing, long-horizon interactions).
   - **Method/result:** 22,225 video–expression pairs and 7.415M frame annotations; human-in-the-loop dual review. The official benchmark includes spatio-temporal grounding, MLLM, and hybrid MLLM+tracking baselines, plus a Motion-aware Prefix Adapter and Multi-view Alignment Adapter. The authors report that existing methods remain far below human performance.
   - **Fit:** closest external match for ReVA’s small-object identity and camera-motion failures; frame-level boxes make “did the same object persist?” measurable.
   - **Limitations:** visual grounding, not multiple-choice VideoQA; self-collected distribution; no claim of transfer to ReVA’s scene-centric causal/geometry questions. Treat the adapters as hypotheses, not a drop-in model.
   - **Sources/availability:** [official project page](https://more-uav.github.io/) and [official code/data repository](https://github.com/johnbager/more-uav) (the repository says the full dataset is available through Google Drive and stores frame-level `bboxes.json`).

2. **Self in Space (SIS-Bench): Benchmarking Self-Awareness and Spatial Cognition in UAV Embodied Intelligence (2026).**

   - **Targets:** separating the UAV’s own motion/viewpoint from world/object changes; spatial memory and temporal reasoning in UAV videos.
   - **Method/result:** 4,856 QA pairs from 1,646 real UAV videos across 13 tasks and perception–memory–reasoning levels. The SIS-Motion controlled study adds an optical-flow motion encoder and lightweight fusion to a video MLLM; the paper reports gains over visual-only SFT and tests transfer to UAV navigation.
   - **Fit:** a strong diagnostic analogue for ReVA camera-motion and “self versus scene” confounds. It supports testing explicit motion tokens/flow as an ablation rather than relying on implicit frame differences.
   - **Limitations:** embodied self-awareness and navigation are not the same as ReVA scene-centric A–D QA; some sources include AirScape, UrbanVideo-Bench, and VisDrone, so it is not a pure held-out aerial domain. Report transfer only after a ReVA-controlled test.
   - **Source:** [official paper](https://arxiv.org/abs/2607.12477) (dataset/QA lines 71–84, 130–132; motion encoder and controlled study lines 193–223).

3. **SpatialUAV: Benchmarking Spatial Intelligence for Low-Altitude UAV Perception, Collaboration, and Motion (2026).**

   - **Targets:** image-plane and geometric relations, cross-view association, aerial–aerial/aerial–ground correspondence, and video motion understanding.
   - **Method/result:** 4,331 curated instances, 14 task types, seven input configurations, and nine answer formats (including option labels, region IDs, geometric values, cross-view correspondences, and free-form motion). The data construction uses detector-assisted regions, depth supervision, metadata rules, manual annotation, blind filtering, and multi-turn validation. Evaluations report large bottlenecks in aerial geometry, structured grounding, and temporal viewpoint reasoning.
   - **Fit:** closest benchmark design to ReVA’s spatial/geometry and option-label questions; useful for constructing a small per-task diagnostic slice and checking whether errors are image-plane versus depth/reference-frame ambiguity.
   - **Limitations:** low-altitude UAV settings and heterogeneous tasks; mostly a benchmark/diagnosis rather than a proven QA plug-in. It does not show that geometry labels resolve uncalibrated depth in ReVA.
   - **Sources:** [paper](https://arxiv.org/abs/2606.27876) and [official code/data repository](https://github.com/Hyu-Zhang/SpatialUAV).

4. **GrabVG: Graph-Attentive Binding for Visual Grounding in UAV Imagery (2026).**

   - **Targets:** small, dense, visually similar objects and topological ambiguity in bird’s-eye UAV imagery.
   - **Method/result:** proposal induction/text-aware filtering followed by appearance and sparse local graph/topological binding. On AerialVG, the paper reports 67.31% Top-1 with all components, versus 62.84% for the appearance-only component in its ablation; a 6-NN local graph and single early text-aware filter are important design choices.
   - **Fit:** a concrete hypothesis for ReVA’s “small red car / white vehicle / nearby distractor” identity errors: use local relational context, not only an isolated box or mask.
   - **Limitations:** image-only; no temporal tracking, event boundaries, or camera ego-motion. The AerialVG/AerialSense gains cannot be interpreted as VideoQA gains; the paper is an arXiv preprint and should be replicated on a tiny ReVA subset before any promotion.
   - **Source:** [paper](https://arxiv.org/abs/2608.18996) (method and ablations, including AerialVG results).

5. **UAVDT: The Unmanned Aerial Vehicle Benchmark: Object Detection and Tracking (2018).**

   - **Targets:** small objects, dense scenes, camera rotation/motion, occlusion, altitude and viewpoint effects, and identity tracking.
   - **Method/result:** 100 UAV videos / about 80,000 annotated frames, 0.84M boxes, and up to 14 attributes; it evaluates detection, SOT, and MOT. The paper explicitly reports degradation of then-current methods on high-density, small-object, and camera-motion cases and annotates SOT attributes such as small object, camera rotation, blur, scale variation, and occlusion.
   - **Fit:** best established UAV-specific source for stratifying ReVA failures by object size, camera motion, altitude/view, and occlusion, and for requiring identity/track metrics in addition to QA accuracy.
   - **Limitations:** detection/tracking benchmark, not language, temporal event QA, or calibrated geometry; older model results are not current MLLM evidence.
   - **Source:** [paper](https://arxiv.org/abs/1804.00518) (dataset lines 61–68; challenge factors lines 47–57 and 87–88).

6. **VisDrone: Detection and Tracking Meet Drones Challenge (2021/2022 TPAMI; dataset release 2019).**

   - **Targets:** small/crowded aerial objects, occlusion, video detection, SOT, MOT, and crowd counting.
   - **Method/result:** official release has 288 video clips / 261,908 frames and 10,209 static images from 14 Chinese cities, more than 2.6M boxes, and scene/class/occlusion attributes; tasks include video detection, SOT, and MOT.
   - **Fit:** useful for objective detection/tracking probes and difficult-case stratification. ReVA already contains VisDrone-derived clips, so this is a source/annotation baseline, not independent evidence of generalization.
   - **Limitations:** no natural-language temporal events, causal explanations, or camera pose; box/tracking success need not improve four-option QA.
   - **Source:** [official dataset/repository](https://github.com/VisDrone/VisDrone-Dataset) and its [paper citation](https://arxiv.org/abs/2001.06303).

### Generic methods with a transfer caveat

7. **UniVTG: Towards Unified Video-Language Temporal Grounding (ICCV 2023).**

   - **Targets:** temporal label/task mismatch: moment intervals, highlight curves, and point/disjoint-shot summaries.
   - **Method/result:** a unified clip representation, pseudo-labeling scheme, and grounding model trained across heterogeneous labels; it reports gains and zero-shot grounding across seven non-UAV datasets. The paper explicitly treats temporal boundary/label diversity as a central problem.
   - **Fit:** useful conceptual baseline for ReVA’s temporal-onset/visibility-boundary ambiguity and for normalizing event labels before modeling. It suggests reporting a temporal-localization metric separately from final A–D accuracy.
   - **Limitations:** trained/evaluated on web/egocentric/action datasets, not aerial video; it does not model dominant UAV ego-motion, tiny targets, or calibrated spatial relations. Do not assume zero-shot transfer.
   - **Sources:** [paper](https://arxiv.org/abs/2307.16715) and [official code](https://github.com/showlab/UniVTG).

8. **OmniGround: A Comprehensive Spatio-Temporal Grounding Benchmark for Real-World Complex Scenarios (CVPR 2026).**

   - **Targets:** dataset/label quality, small or occluded objects, multiple similar targets, complex spatial language, and the temporal-versus-spatial localization trade-off.
   - **Method/result:** 3,475 videos / 81 categories with Forward–Backward–Refinement (FBR) annotation and DeepSTG data-quality metrics. The paper reports a 10.4% average m_vIoU drop on complex scenes, especially small/occluded objects and intricate relations. Its training-free PG-TAF separates MLLM coarse temporal grounding from tracker-based fine spatial propagation; on OmniGround it reports 49.2 m_tIoU / 36.2 m_vIoU and a 25.6% / 35.6% improvement claim in the abstract.
   - **Fit:** the cleanest external argument for a two-stage ReVA pilot: question-conditioned temporal shortlist first, then local high-resolution evidence/track propagation. Its FBR/DeepSTG ideas also motivate checking label continuity, temporal boundaries, and query–video alignment.
   - **Limitations:** generic internet videos, not UAV; its tracker/segmenter assumptions may fail on tiny aerial objects and moving viewpoints. The paper’s improvements are STVG metrics, not ReVA QA accuracy.
   - **Source:** [official paper](https://arxiv.org/abs/2511.16937) (benchmark/annotation lines 3–5, 20–28; two-stage design/results lines 248–258).

## Recommended reading and small ReVA pilot

Read first: **MoRe-UAV** (moving-object identity under UAV ego-motion), **SIS-Bench** (explicit self/camera motion for UAV VideoQA), and **OmniGround** (quality-controlled temporal/spatial decomposition). Read **SpatialUAV** next if geometry/reference-frame failures remain the priority.

Recommended pilot, without changing the baseline model or evaluation math:

1. Select a frozen, stratified set of roughly 60–100 ReVA validation items: temporal boundary/visibility, small-object identity, camera motion, image-plane relation, and depth/reference-frame questions; include matched correct controls and the known label/metadata flags.
2. For each item, compare the current 16-frame sparse sample with a question-conditioned temporal shortlist plus a local high-resolution crop/track. Keep the final frame budget fixed and log whether the evidence changes the selected option.
3. Add explicit motion/context variants only as ablations: optical-flow or global camera-motion tokens (SIS/ReMoSense hypothesis), and local relational candidate context (GrabVG hypothesis). Do not train on MoRe-UAV/OmniGround during this gate.
4. Score both A–D accuracy and evidence diagnostics: temporal IoU/boundary error where timestamps are available, target identity continuity, image-plane relation accuracy, and a separate “depth/reference frame not identifiable” flag. Report results by UAV source and task, not only one aggregate.
5. Treat any improvement as credible only if it beats the fixed-sampling control on the paired subset and does not regress clear factual controls. ReVA’s current review already shows that masks cannot solve missing calibration and that some items may be clip/metadata or label issues; preserve those flags rather than silently relabeling.

## Caveats and unanswered questions

- None of the external papers demonstrates broad gains on ReVA’s four-option answer accuracy. Grounding/tracking is an auxiliary measurement, not a substitute for QA evaluation.
- “Tracking” in UAVDT/VisDrone/MoRe-UAV means boxes/tubes or referred-target localization; it is not automatically robust identity persistence under ReVA’s sparse sampling.
- Most UAV benchmarks differ in altitude, camera stabilization, object scale, geography, and source distribution. MoRe-UAV is self-collected; SpatialUAV is low-altitude; VisDrone is partly already inside ReVA.
- Before implementation, verify release licensing, exact checkpoints, input-resolution/frame-budget requirements, and whether official code supports inference on ReVA’s video formats. No GPU/HPC validation was run for this research note.
