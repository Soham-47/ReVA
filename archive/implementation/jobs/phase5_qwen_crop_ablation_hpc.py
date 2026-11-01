"""Run the frozen Phase 5 E0/E1/E2 ablation inside a Slurm GPU allocation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
ARMS = ("E0", "E1", "E2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text())


def load_records(path: Path, digest: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    raw = path.read_bytes()
    offset = 0
    lines = raw.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise
        if row.get("manifest_digest") != digest:
            raise ValueError(f"output manifest mismatch in {path}")
        qa_id = str(row["qa_id"])
        if qa_id in records:
            raise ValueError(f"duplicate score row {qa_id} in {path}")
        records[qa_id] = row
        offset += len(line)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    os.environ["WORK"] = str(input_dir)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, str(input_dir / "code"))

    import numpy as np
    import torch
    from PIL import Image
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    import common as C

    manifest = load_json(input_dir / "run_manifest.json")
    manifest_digest = sha256(input_dir / "run_manifest.json")
    if (input_dir / "run_manifest.sha256").read_text().strip() != manifest_digest:
        raise ValueError("run manifest checksum mismatch")
    crop_manifest_path = input_dir / "crop_manifest.json"
    if sha256(crop_manifest_path) != manifest["crop_manifest_sha256"]:
        raise ValueError("crop manifest checksum mismatch")
    if manifest["base_model_id"] != MODEL_ID or manifest["base_model_revision"] != MODEL_REVISION:
        raise ValueError("Phase 5 model pin differs from the approved run manifest")
    if manifest["adapter_sha256"] != sha256(input_dir / "stageA_adapter/adapter_model.safetensors"):
        raise ValueError("Stage A adapter checksum mismatch")

    qas = load_json(input_dir / "qa_rows.json")
    crop_manifest = load_json(crop_manifest_path)
    crops_by_qa = {str(row["qa_id"]): row for row in crop_manifest["questions"]}
    question_ids = [str(row["qa_id"]) for row in qas]
    if len(qas) != 12 or len(set(question_ids)) != 12 or question_ids != manifest["question_ids"]:
        raise ValueError("expected the frozen 12-question Phase 5 packet")
    if len({row["video_path"] for row in qas}) != 12:
        raise ValueError("Phase 5 questions must map to 12 distinct source videos")
    if set(question_ids) != set(crops_by_qa):
        raise ValueError("QA rows and crop manifest do not contain the same questions")
    for relative, expected in manifest["input_video_sha256"].items():
        video = input_dir / "data/ReVA" / relative
        if sha256(video) != expected:
            raise ValueError(f"source video checksum mismatch: {relative}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {arm: output_dir / f"scores_{arm}.jsonl" for arm in ARMS}
    prior = {arm: load_records(output_paths[arm], manifest_digest) for arm in ARMS}
    if any(not set(rows).issubset(question_ids) for rows in prior.values()):
        raise ValueError("existing output contains a question outside the frozen cohort")

    if not torch.cuda.is_available():
        raise RuntimeError("Slurm allocation did not expose CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one GPU, got {torch.cuda.device_count()}")
    C.configure_attention_backend("sdpa", torch)
    processor = C.load_processor()
    letter_ids = C.letter_ids(processor.tokenizer)

    # Validate the mixed 16-frame-video plus auxiliary-image processor path before loading weights.
    q0 = qas[0]
    frames0, meta0 = C.get_video(q0["video_path"], 0.0, mode="legacy")
    if len(frames0) != 16 or len(meta0.frames_indices) != 16:
        raise ValueError("legacy Stage A frame sampler no longer yields 16 frames")
    records0 = [slot["retrieved"] for slot in crops_by_qa[q0["qa_id"]]["slots"] if slot.get("retrieved")]
    images0 = [Image.open(input_dir / row["local_path"]).convert("RGB") for row in records0]
    shown0, _ = C.permute_options(q0, C.LETTERS)
    content0 = [{"type": "video"}, {"type": "text", "text": C.user_text(q0, shown0)}]
    for record, image in zip(records0, images0, strict=True):
        content0.extend([
            {"type": "text", "text": f"Auxiliary crop from source timestamp {float(record['timestamp_s']):.3f} s; padded RGB region, {image.width} by {image.height} pixels. Content is unverified; inspect the pixels directly."},
            {"type": "image"},
        ])
    preflight_messages = [
        {"role": "system", "content": [{"type": "text", "text": C.SYSTEM_PROMPT}]},
        {"role": "user", "content": content0},
    ]
    preflight_text = processor.apply_chat_template(preflight_messages, tokenize=False, add_generation_prompt=True)
    preflight = processor(
        text=[preflight_text], images=images0, videos=[frames0], video_metadata=[meta0],
        do_sample_frames=False, size=C.VIDEO_SIZE, cap_pixels_per_frame=True, return_tensors="pt",
    )
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
    if required - preflight.keys():
        raise KeyError(f"mixed processor output missing {sorted(required - preflight.keys())}")
    preflight_record = {
        "status": "ok",
        "qa_id": q0["qa_id"],
        "frame_count": len(frames0),
        "source_indices": [int(value) for value in meta0.frames_indices],
        "auxiliary_images": len(images0),
        "image_grid_thw": preflight["image_grid_thw"].tolist(),
        "input_token_count": int(preflight["input_ids"].shape[-1]),
        "processor_class": type(processor).__name__,
    }
    for image in images0:
        image.close()
    del preflight, frames0, meta0, images0
    print("processor_preflight=" + json.dumps(preflight_record), flush=True)

    model_started = time.monotonic()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    model = PeftModel.from_pretrained(model, input_dir / "stageA_adapter").merge_and_unload()
    model.to(torch.device("cuda")).eval()
    model_load_seconds = time.monotonic() - model_started
    load_allocated = int(torch.cuda.max_memory_allocated())
    load_reserved = int(torch.cuda.max_memory_reserved())
    rotations = [list(C.LETTERS[offset:] + C.LETTERS[:offset]) for offset in range(2)]
    arm_time = {arm: 0.0 for arm in ARMS}
    arm_forward_time = {arm: 0.0 for arm in ARMS}
    arm_memory = {arm: {"allocated_bytes": 0, "reserved_bytes": 0} for arm in ARMS}
    arm_tokens = {arm: {"input_tokens": 0, "auxiliary_image_tokens": 0, "video_tokens": 0, "forward_calls": 0} for arm in ARMS}
    processed_questions = []
    score_started = time.monotonic()

    def load_images(qa_crop: dict, arm: str):
        key = "retrieved" if arm == "E2" else "random"
        records = [slot[key] for slot in qa_crop["slots"] if slot.get(key)]
        images = [Image.open(input_dir / record["local_path"]).convert("RGB") for record in records]
        return records, images

    def encode_aux(q: dict, frames, meta, permutation, records, images):
        shown, perm = C.permute_options(q, permutation)
        content = [{"type": "video"}, {"type": "text", "text": C.user_text(q, shown)}]
        for record, image in zip(records, images, strict=True):
            content.extend([
                {"type": "text", "text": f"Auxiliary crop from source timestamp {float(record['timestamp_s']):.3f} s; padded RGB region, {image.width} by {image.height} pixels. Content is unverified; inspect the pixels directly."},
                {"type": "image"},
            ])
        messages = [
            {"role": "system", "content": [{"type": "text", "text": C.SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raw = processor(
            text=[text], images=images, videos=[frames], video_metadata=[meta],
            do_sample_frames=False, size=C.VIDEO_SIZE, cap_pixels_per_frame=True, return_tensors="pt",
        )
        return raw, perm

    def grid_tokens(grid, merge_size: int) -> int:
        if grid is None:
            return 0
        return sum(int(t) * int(h) * int(w) // (merge_size * merge_size) for t, h, w in grid.tolist())

    for q in qas:
        qa_id = str(q["qa_id"])
        missing_arms = [arm for arm in ARMS if qa_id not in prior[arm]]
        if not missing_arms:
            continue
        qa_crop = crops_by_qa[qa_id]
        views_by_arm = {arm: [] for arm in missing_arms}
        for offset in (0.0, 0.5):
            frames, meta = C.get_video(q["video_path"], offset, mode="legacy")
            sample_indices = [int(index) for index in meta.frames_indices]
            if len(frames) != 16 or len(sample_indices) != 16:
                raise ValueError(f"Stage A frame sampler changed for {qa_id} offset={offset}")
            for permutation in rotations:
                for arm in missing_arms:
                    records, images = load_images(qa_crop, arm) if arm != "E0" else ([], [])
                    crop_mapping = [
                        {"source_frame_index": int(record["source_frame_index"]),
                         "timestamp_s": float(record["timestamp_s"]),
                         "also_in_this_view": int(record["source_frame_index"]) in set(sample_indices)}
                        for record in records
                    ]
                    started = time.monotonic()
                    if arm == "E0":
                        encoded, perm = C.encode(processor, q, frames, meta, permutation)
                    else:
                        encoded, perm = encode_aux(q, frames, meta, permutation, records, images)
                    encode_seconds = time.monotonic() - started
                    image_tokens = 0 if arm == "E0" else grid_tokens(
                        encoded.get("image_grid_thw"), int(model.config.vision_config.spatial_merge_size)
                    )
                    video_tokens = grid_tokens(
                        encoded.get("video_grid_thw"), int(model.config.vision_config.spatial_merge_size)
                    )
                    input_tokens = int(encoded["input_ids"].shape[-1])
                    model_inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in encoded.items()}
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    forward_started = time.monotonic()
                    with torch.inference_mode():
                        logits = model(**model_inputs).logits[0, -1, list(letter_ids)].float()
                    torch.cuda.synchronize()
                    forward_seconds = time.monotonic() - forward_started
                    shown_prob = torch.softmax(logits, dim=-1).cpu().numpy()
                    original_prob = [0.0] * 4
                    for shown_index, original_letter in enumerate(perm):
                        original_prob[C.LETTERS.index(original_letter)] = float(shown_prob[shown_index])
                    arm_time[arm] += encode_seconds + forward_seconds
                    arm_forward_time[arm] += forward_seconds
                    arm_tokens[arm]["input_tokens"] += input_tokens
                    arm_tokens[arm]["auxiliary_image_tokens"] += image_tokens
                    arm_tokens[arm]["video_tokens"] += video_tokens
                    arm_tokens[arm]["forward_calls"] += 1
                    peak_allocated = int(torch.cuda.max_memory_allocated())
                    peak_reserved = int(torch.cuda.max_memory_reserved())
                    arm_memory[arm]["allocated_bytes"] = max(arm_memory[arm]["allocated_bytes"], peak_allocated)
                    arm_memory[arm]["reserved_bytes"] = max(arm_memory[arm]["reserved_bytes"], peak_reserved)
                    views_by_arm[arm].append({
                        "offset": offset, "permutation": "".join(perm),
                        "sampled_source_frame_indices": sample_indices,
                        "crop_source_mapping": crop_mapping, "p": original_prob,
                        "input_token_count": input_tokens, "auxiliary_image_count": len(images),
                        "auxiliary_image_vision_tokens": image_tokens, "video_vision_tokens": video_tokens,
                        "encode_seconds": encode_seconds, "forward_seconds": forward_seconds,
                        "peak_cuda_allocated_bytes": peak_allocated,
                        "peak_cuda_reserved_bytes": peak_reserved,
                    })
                    del model_inputs, encoded, logits
                    for image in images:
                        image.close()
        for arm in missing_arms:
            views = views_by_arm[arm]
            if len(views) != 4:
                raise ValueError(f"expected four Stage A views for {qa_id}/{arm}; got {len(views)}")
            mean_prob = np.mean([view["p"] for view in views], axis=0).tolist()
            record = {
                "qa_id": qa_id, "manifest_digest": manifest_digest, "arm": arm,
                "p": mean_prob, "pred": C.LETTERS[int(np.argmax(mean_prob))],
                "views": views, "crop_manifest_slots": qa_crop["slots"] if arm != "E0" else [],
            }
            with output_paths[arm].open("a") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            prior[arm][qa_id] = record
        processed_questions.append(qa_id)
        print(f"completed {qa_id}: {len(processed_questions)} new QA rows", flush=True)

    elapsed = time.monotonic() - score_started
    all_scores = {arm: list(load_records(output_paths[arm], manifest_digest).values()) for arm in ARMS}
    for arm in ARMS:
        all_scores[arm].sort(key=lambda row: question_ids.index(row["qa_id"]))
    summary = {
        "status": "complete" if all(len(all_scores[arm]) == 12 for arm in ARMS) else "partial",
        "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "adapter_sha256": manifest["adapter_sha256"],
        "gpu_name": torch.cuda.get_device_name(0), "gpu_count": torch.cuda.device_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scored_question_count_per_arm": {arm: len(all_scores[arm]) for arm in ARMS},
        "newly_processed_questions": processed_questions,
        "processor_preflight": preflight_record,
        "model_load_seconds": model_load_seconds,
        "model_load_peak_cuda_allocated_bytes": load_allocated,
        "model_load_peak_cuda_reserved_bytes": load_reserved,
        "peak_cuda_memory_by_arm": arm_memory,
        "inference_plus_preprocessing_seconds_by_arm": arm_time,
        "forward_seconds_by_arm": arm_forward_time,
        "token_totals_by_arm": arm_tokens,
        "gpu_scoring_elapsed_seconds": elapsed,
        "scores": all_scores,
    }
    summary_path = output_dir / "hpc_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if summary["status"] != "complete":
        raise RuntimeError(f"Phase 5 incomplete; summary saved to {summary_path}")
    print(json.dumps({key: value for key, value in summary.items() if key != "scores"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
