"""Run the paired Phase 5 Qwen crop-evidence ablation on Modal."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import modal


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/phase5_crop_ablation"
INPUT = RUN / "input"
VOLUME_NAME = "reva-phase5-qwen-crop-ablation"
MOUNT = "/mnt/reva-phase5"
REMOTE_INPUT = f"{MOUNT}/input"
REMOTE_OUTPUT = f"{MOUNT}/output"
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
ARMS = ("E0", "E1", "E2")

app = modal.App("reva-phase5-qwen-crop-ablation")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "torchvision==0.29.1",
        "transformers==5.19.0",
        "accelerate==1.15.0",
        "huggingface-hub==2.2.0",
        "peft==0.21.2",
        "av==18.1.0",
        "numpy",
        "pillow",
    )
)


def _setup_remote() -> None:
    os.environ["WORK"] = REMOTE_INPUT
    os.environ["HF_HOME"] = f"{MOUNT}/hf_cache"
    os.environ["HF_HUB_CACHE"] = f"{MOUNT}/hf_cache/hub"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.path.insert(0, f"{REMOTE_INPUT}/code")


def _load_json(path: Path):
    return json.loads(path.read_text())


def _load_records(path: Path, manifest_digest: str) -> dict[str, dict]:
    if not path.exists():
        return {}
    records = {}
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    offset = 0
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith((b"\n", b"\r")):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise
        if record.get("manifest_digest") != manifest_digest:
            raise ValueError(f"output manifest mismatch in {path}")
        qa_id = str(record["qa_id"])
        if qa_id in records:
            raise ValueError(f"duplicate score row {qa_id} in {path}")
        records[qa_id] = record
        offset += len(line)
    return records


@app.function(image=image, cpu=2, memory=8192, timeout=1800, volumes={MOUNT: volume})
def preflight():
    """Validate timestamped crops and mixed Qwen video+image processor input on CPU."""
    _setup_remote()
    import numpy as np
    from PIL import Image
    import common as C

    manifest = _load_json(Path(REMOTE_INPUT) / "run_manifest.json")
    crops = _load_json(Path(REMOTE_INPUT) / "crop_manifest.json")
    if manifest["question_ids"] != [row["qa_id"] for row in crops["questions"]]:
        raise ValueError("question order differs across input manifests")
    processor = C.load_processor()
    q = _load_json(Path(REMOTE_INPUT) / "qa_rows.json")[0]
    crop_q = crops["questions"][0]
    frames, meta = C.get_video(q["video_path"], 0.0, mode="legacy")
    images = [Image.open(Path(REMOTE_INPUT) / slot["retrieved"]["local_path"]).convert("RGB") for slot in crop_q["slots"]]
    content = [{"type": "video"}, {"type": "text", "text": C.user_text(q, q["options"])}]
    for slot, crop in zip(crop_q["slots"], images, strict=True):
        rec = slot["retrieved"]
        content.extend([
            {"type": "text", "text": f"Auxiliary crop from source timestamp {rec['timestamp_s']:.3f} s; padded RGB region, {crop.width} by {crop.height} pixels. Content is unverified; inspect the pixels directly."},
            {"type": "image"},
        ])
    messages = [
        {"role": "system", "content": [{"type": "text", "text": C.SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw = processor(
        text=[text],
        images=images,
        videos=[frames],
        video_metadata=[meta],
        do_sample_frames=False,
        size=C.VIDEO_SIZE,
        cap_pixels_per_frame=True,
        return_tensors="pt",
    )
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"}
    missing = required - raw.keys()
    if missing:
        raise KeyError(f"mixed processor output is missing {sorted(missing)}; keys={sorted(raw.keys())}")
    grid = raw["image_grid_thw"].tolist()
    token_count = sum(int(t) * int(h) * int(w) // 4 for t, h, w in grid)
    result = {
        "status": "ok",
        "qa_id": q["qa_id"],
        "processor_class": type(processor).__name__,
        "video_frame_count": int(len(frames)),
        "video_source_indices": [int(index) for index in meta.frames_indices],
        "auxiliary_image_count": len(images),
        "image_grid_thw": grid,
        "estimated_aux_image_vision_tokens": token_count,
        "input_token_count": int(raw["input_ids"].shape[-1]),
        "pixel_values_videos_shape": list(raw["pixel_values_videos"].shape),
        "pixel_values_images_shape": list(raw["pixel_values"].shape),
        "padded_random_retrieved_crop_sizes_match": all(
            slot["random"]["crop_size_pixels"] == slot["retrieved"]["crop_size_pixels"]
            for slot in crop_q["slots"]
        ),
    }
    path = Path(REMOTE_OUTPUT) / "preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, cpu=2, memory=8192, timeout=3600, volumes={MOUNT: volume})
def prefetch_model():
    """Download the pinned public checkpoint to the mounted cache without GPU billing."""
    _setup_remote()
    from huggingface_hub import snapshot_download

    started = time.monotonic()
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=os.environ["HF_HUB_CACHE"],
    )
    result = {
        "status": "ok",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "snapshot_path": path,
        "elapsed_seconds": time.monotonic() - started,
    }
    out = Path(REMOTE_OUTPUT) / "model_prefetch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, gpu="A100-40GB", cpu=8, memory=32768, timeout=1800, volumes={MOUNT: volume})
def score_all():
    """Score all three frozen arms with the Stage A adapter and exact TTA settings."""
    _setup_remote()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration
    import common as C

    torch.backends.cuda.enable_cudnn_sdp(False)
    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA device")
    run_manifest = _load_json(Path(REMOTE_INPUT) / "run_manifest.json")
    manifest_digest = Path(REMOTE_INPUT, "run_manifest.sha256").read_text().strip()
    crop_manifest = _load_json(Path(REMOTE_INPUT) / "crop_manifest.json")
    qas = _load_json(Path(REMOTE_INPUT) / "qa_rows.json")
    if len(qas) != 12 or len({q["video_path"] for q in qas}) != 12:
        raise ValueError("the Phase 5 score packet must contain 12 distinct videos")
    crops_by_qa = {str(row["qa_id"]): row for row in crop_manifest["questions"]}
    out_dir = Path(REMOTE_OUTPUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {arm: out_dir / f"scores_{arm}.jsonl" for arm in ARMS}
    prior = {arm: _load_records(output_paths[arm], manifest_digest) for arm in ARMS}

    processor = C.load_processor()
    letter_ids = C.letter_ids(processor.tokenizer)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    adapter = Path(REMOTE_INPUT) / "stageA_adapter"
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model.to(torch.device("cuda")).eval()
    model_load_memory_bytes = int(torch.cuda.max_memory_allocated())
    model_load_reserved_bytes = int(torch.cuda.max_memory_reserved())
    rotations = [list(C.LETTERS[offset:] + C.LETTERS[:offset]) for offset in range(2)]
    run_started = time.monotonic()
    arm_time = {arm: 0.0 for arm in ARMS}
    arm_forward_time = {arm: 0.0 for arm in ARMS}
    arm_memory = {arm: {"allocated_bytes": 0, "reserved_bytes": 0} for arm in ARMS}
    arm_tokens = {arm: {"total_input_tokens": 0, "auxiliary_image_tokens": 0, "total_video_tokens": 0, "forward_calls": 0} for arm in ARMS}
    processed_questions = []

    def _load_images(qa_crop, arm):
        from PIL import Image
        key = "retrieved" if arm == "E2" else "random"
        records = [slot[key] for slot in qa_crop["slots"] if slot.get(key)]
        images = [Image.open(Path(REMOTE_INPUT) / rec["local_path"]).convert("RGB") for rec in records]
        return records, images

    def _encode_aux(q, frames, meta, permutation, records, images):
        from PIL import Image
        shown, perm = C.permute_options(q, permutation)
        content = [
            {"type": "video"},
            {"type": "text", "text": C.user_text(q, shown)},
        ]
        for rec, image in zip(records, images, strict=True):
            content.extend([
                {
                    "type": "text",
                    "text": (
                        f"Auxiliary crop from source timestamp {float(rec['timestamp_s']):.3f} s; "
                        f"padded RGB region, {image.width} by {image.height} pixels. "
                        "Content is unverified; inspect the pixels directly."
                    ),
                },
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

    def _to_device(raw):
        return {key: value.to("cuda") if hasattr(value, "to") else value for key, value in raw.items()}

    def _grid_tokens(grid, merge_size):
        if grid is None:
            return 0
        return sum(int(row[0]) * int(row[1]) * int(row[2]) // (merge_size * merge_size) for row in grid.tolist())

    for q in qas:
        qa_id = str(q["qa_id"])
        if all(qa_id in prior[arm] for arm in ARMS):
            continue
        if qa_id not in crops_by_qa:
            raise KeyError(f"missing crop manifest for {qa_id}")
        qa_crop = crops_by_qa[qa_id]
        views_by_arm = {arm: [] for arm in ARMS}
        for offset in (0.0, 0.5):
            frames, meta = C.get_video(q["video_path"], offset, mode="legacy")
            sample_indices = [int(index) for index in meta.frames_indices]
            if len(frames) != 16 or len(sample_indices) != 16:
                raise ValueError(f"Stage A frame sampler changed for {qa_id} offset={offset}")
            for permutation in rotations:
                perm_label = "".join(permutation)
                for arm in ARMS:
                    if qa_id in prior[arm]:
                        continue
                    records, images = _load_images(qa_crop, arm) if arm != "E0" else ([], [])
                    sample_index_set = set(sample_indices)
                    offset_crop_mapping = [
                        {
                            "source_frame_index": int(record["source_frame_index"]),
                            "timestamp_s": float(record["timestamp_s"]),
                            "also_in_this_view": int(record["source_frame_index"]) in sample_index_set,
                        }
                        for record in records
                    ]
                    encode_started = time.monotonic()
                    if arm == "E0":
                        encoded, perm = C.encode(processor, q, frames, meta, permutation)
                    else:
                        encoded, perm = _encode_aux(q, frames, meta, permutation, records, images)
                    encode_seconds = time.monotonic() - encode_started
                    if arm == "E0":
                        extra_images = 0
                        image_tokens = 0
                    else:
                        extra_images = len(images)
                        image_tokens = _grid_tokens(encoded.get("image_grid_thw"), int(model.config.vision_config.spatial_merge_size))
                    video_tokens = _grid_tokens(encoded.get("video_grid_thw"), int(model.config.vision_config.spatial_merge_size))
                    input_tokens = int(encoded["input_ids"].shape[-1])
                    model_inputs = _to_device(encoded)
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
                    arm_tokens[arm]["total_input_tokens"] += input_tokens
                    arm_tokens[arm]["auxiliary_image_tokens"] += image_tokens
                    arm_tokens[arm]["total_video_tokens"] += video_tokens
                    arm_tokens[arm]["forward_calls"] += 1
                    arm_memory[arm]["allocated_bytes"] = max(arm_memory[arm]["allocated_bytes"], int(torch.cuda.max_memory_allocated()))
                    arm_memory[arm]["reserved_bytes"] = max(arm_memory[arm]["reserved_bytes"], int(torch.cuda.max_memory_reserved()))
                    views_by_arm[arm].append({
                        "offset": offset,
                        "perm": "".join(perm),
                        "sampled_source_frame_indices": sample_indices,
                        "crop_source_mapping": offset_crop_mapping,
                        "p": original_prob,
                        "input_token_count": input_tokens,
                        "auxiliary_image_count": extra_images,
                        "auxiliary_image_vision_tokens": image_tokens,
                        "video_vision_tokens": video_tokens,
                        "encode_seconds": encode_seconds,
                        "forward_seconds": forward_seconds,
                        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    })
                    del model_inputs, encoded, logits
                    if arm != "E0":
                        for image in images:
                            image.close()

        newly_written = []
        for arm in ARMS:
            if qa_id in prior[arm]:
                continue
            arm_views = views_by_arm[arm]
            if len(arm_views) != 4:
                raise ValueError(f"expected four Stage A views for {qa_id}/{arm}, got {len(arm_views)}")
            mean_prob = np.mean([view["p"] for view in arm_views], axis=0).tolist()
            record = {
                "qa_id": qa_id,
                "manifest_digest": manifest_digest,
                "arm": arm,
                "p": mean_prob,
                "pred": C.LETTERS[int(np.argmax(mean_prob))],
                "views": arm_views,
                "crop_manifest_slots": qa_crop["slots"] if arm != "E0" else [],
            }
            newly_written.append((arm, record))
        for arm, record in newly_written:
            with output_paths[arm].open("a") as stream:
                stream.write(json.dumps(record, separators=(",", ":")) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            prior[arm][qa_id] = record
        volume.commit()
        processed_questions.append(qa_id)
        print(f"completed {qa_id}: {len(processed_questions)} new QA rows")

    elapsed = time.monotonic() - run_started
    all_scores = {arm: list(_load_records(output_paths[arm], manifest_digest).values()) for arm in ARMS}
    summary = {
        "status": "complete" if all(len(all_scores[arm]) == 12 for arm in ARMS) else "partial",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "adapter_sha256": run_manifest["adapter_sha256"],
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "scored_question_count_per_arm": {arm: len(all_scores[arm]) for arm in ARMS},
        "newly_processed_questions": processed_questions,
        "model_load_peak_cuda_allocated_bytes": model_load_memory_bytes,
        "model_load_peak_cuda_reserved_bytes": model_load_reserved_bytes,
        "peak_cuda_memory_by_arm": arm_memory,
        "inference_plus_preprocessing_seconds_by_arm": arm_time,
        "forward_seconds_by_arm": arm_forward_time,
        "token_totals_by_arm": arm_tokens,
        "gpu_function_elapsed_seconds": elapsed,
        "scores": all_scores,
    }
    (out_dir / "modal_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    volume.commit()
    return summary


@app.function(image=image, cpu=1, memory=4096, timeout=300, volumes={MOUNT: volume})
def fetch_saved_results():
    _setup_remote()
    manifest_digest = Path(REMOTE_INPUT, "run_manifest.sha256").read_text().strip()
    output = Path(REMOTE_OUTPUT)
    scores = {arm: list(_load_records(output / f"scores_{arm}.jsonl", manifest_digest).values()) for arm in ARMS}
    preflight_path = output / "preflight.json"
    prefetch_path = output / "model_prefetch.json"
    return {
        "status": "complete" if all(len(scores[arm]) == 12 for arm in ARMS) else "partial",
        "scores": scores,
        "preflight": _load_json(preflight_path) if preflight_path.exists() else None,
        "model_prefetch": _load_json(prefetch_path) if prefetch_path.exists() else None,
    }


@app.local_entrypoint()
def main(fetch_only: bool = False):
    RUN.mkdir(parents=True, exist_ok=True)
    if not INPUT.is_dir():
        raise FileNotFoundError(f"run jobs/phase5_prepare_crop_ablation.py first: {INPUT}")
    if not fetch_only:
        with volume.batch_upload(force=True) as batch:
            batch.put_directory(str(INPUT), "/input")
        volume.commit()
        preflight_result = preflight.remote()
        print("mixed image/video processor preflight:", json.dumps(preflight_result, sort_keys=True))
        prefetch_result = prefetch_model.remote()
        print("pinned model cache:", json.dumps(prefetch_result, sort_keys=True))
        result = score_all.remote()
    else:
        result = fetch_saved_results.remote()
    (RUN / "modal_results.json").write_text(json.dumps(result, indent=2) + "\n")
    scores = result.get("scores", {})
    results_dir = RUN / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for arm, rows in scores.items():
        (results_dir / f"scores_{arm}.json").write_text(json.dumps(rows, indent=2) + "\n")
    for name in ("preflight", "model_prefetch"):
        if result.get(name) is not None:
            (RUN / f"{name}.json").write_text(json.dumps(result[name], indent=2) + "\n")
    print(json.dumps({
        "status": result.get("status"),
        "score_counts": {arm: len(rows) for arm, rows in scores.items()},
        "results_dir": str(results_dir),
    }, indent=2))
