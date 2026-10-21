# Source-video annotation protocol

Review the exact question and original sampled frames only. Do not inspect answer options, gold answers, predictions, masks, rankings, or model output.

For each entity and sampled frame, mark visibility, source-pixel XYXY box when defensible, whether identity is unique, ambiguity, confidence, and relation evidence. Use `ambiguous`/`unassessable` when the question does not identify one instance or the frame cannot resolve it. Annotations are evaluation-only and must never enter prompts, filtering, or ranking. Record timestamps from original source frame index divided by measured source FPS.

Every contact sheet preserves all 16 legacy offset-0 samples. JPEG frame images are a compact review cache at source dimensions; original videos are authoritative for fine boundary judgments.
