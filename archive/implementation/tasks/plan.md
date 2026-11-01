# Implementation Plan: ReVA dual-stream temporal grounding

Status (2026-10-30): implementation and static/CPU validation are complete in
the opt-in branch. GPU/model-runtime integration remains intentionally
unvalidated because the approved environment has no Torch/Transformers or
downloaded InternVideo2 assets; temporal resume is fail-closed until that
smoke test is approved.

## Overview

Add an opt-in Qwen3-VL research branch inspired by Grounded-VideoLLM. Preserve ReVA's current Qwen3 video path as the appearance stream, add a frozen InternVideo2 motion stream with a trainable segment projector, and add learned relative-time tokens supervised from parseable gold temporal answer options. Keep current scoring, training defaults, score files, and baseline artifacts unchanged.

## Architecture decisions

- Put the feature behind a new config; Stage A and all existing configs retain current behavior.
- Use segment-aligned motion features and small trainable projection/adaptation components. Load the temporal encoder only when enabled; do not download or initialize weights as part of local implementation checks.
- Extend the tokenizer with relative-time bins `<t000>` through `<t300>` plus explicit interval delimiters. Train the ordered answer as time marker(s) followed by the existing A–D label, and adapt inference to decode the time prefix before evaluating the answer letter.
- Derive time targets only from the correct answer option and video duration. Skip unparsable/ambiguous answers; never infer timestamps from validation or test data.
- Record whether each row received temporal supervision. Preserve validation/test evaluation math and all existing `runs/` score files.
- Do not launch GPU training/scoring or download model/data assets without separate user approval.

## Task list

### Phase 1: Data and token contract

- [x] Define robust parsing for point and interval timestamps in correct options, normalize to relative bins, and provide a mask for rows without valid temporal labels.
- [x] Define and add the temporal token vocabulary, target serialization, and output parsing contract.

### Phase 2: Dual-stream model path

- [x] Add an opt-in temporal encoder wrapper that samples dense frames by segment, extracts frozen InternVideo2 features, applies a trainable projector, and fuses segment tokens with the existing Qwen3-VL representation.
- [x] Integrate tokenizer resizing, trainable temporal-token embeddings/output rows, optimizer parameters, checkpoint save/load, and time-prefix inference without changing default model behavior.

### Phase 3: ReVA config and documentation

- [x] Add a separate dual-stream temporal config and document the required model checkpoint/dependency plus training/evaluation steps.
- [x] Run static compilation and focused CPU checks only; do not run the test suite or any GPU/download operation.

## Acceptance criteria

- Existing configurations continue through the exact existing path when the new branch is disabled.
- The opt-in path consumes appearance and motion streams, can train temporal-bin targets, and has a defined time-prefix decoding path before A–D scoring.
- Missing/ambiguous temporal labels are safely masked, and only train split records can supply targets.
- No GPU training/scoring, model download, validation/test score rewrite, or baseline artifact overwrite occurs.

## Risks

| Risk | Mitigation |
| --- | --- |
| Qwen3 multimodal generation internals differ from the paper's Phi/LLaVA code | Isolate model integration behind a wrapper and preserve default Qwen3 path. |
| Added tokens can be lost or mismatched when loading PEFT adapters | Persist tokenizer metadata and temporal embedding/head weights with the adapter bundle; fail on vocabulary mismatch. |
| Correct options include durations, vague ranges, or unparsable prose | Train timestamp tokens only for verified point/interval patterns and record masked rows. |
| Dense motion features increase storage and GPU cost | Freeze InternVideo2 and cache per-video features for later approved runs. |

## Open implementation details

- [x] Verify the official InternVideo2 checkpoint/model construction and preprocessing API before wiring its loader.
- [x] Confirm the pinned Transformers Qwen3-VL API supports the chosen multimodal embedding injection and cache-aware generation extension; runtime execution remains fail-closed until dependencies are available.
