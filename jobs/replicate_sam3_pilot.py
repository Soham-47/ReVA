"""Run an isolated Replicate SAM 3 mask pilot.

Requires `REPLICATE_API_TOKEN` in the process environment. The value is never
printed, passed as a CLI argument, or written to result files.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import urllib.request
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT
RUN = ROOT / "runs" / "sam3_pilot_stageA_replicate"
SUPPLEMENTAL_RUN = ROOT / "runs" / "sam3_remaining17_stageA_replicate"
PILOT = ROOT / "tasks" / "phase3_grounding_pilot_ids.json"
PRIOR_SUBSET = RUN / "subset_ids.json"
SPLITS = ("valQ", "valT")

import sys

sys.path.insert(0, str(ROOT / "code"))
import common as C
import replicate_sam3 as R
import sam3_overlay as M


SUBSET_PROMPTS = {
    # One short noun phrase per direct SAM 3 call.  Geometric questions include
    # every named object so the overlay can expose the relation's operands.
    "train_014717": ("parking lot", "baseball field", "main grassy field"),
    "train_002581": ("volleyball court", "basketball court"),
    "train_013068": ("red barn", "road", "river"),
    "train_003645": ("rowing boats", "yellow boat", "motorboat"),
    "train_011808": ("large parking lot", "road", "industrial buildings"),
    "train_012156": ("basketball court", "parking lot", "road"),
    "train_012502": ("red barn", "house"),
    "train_014232": ("bridge", "line of boats"),
    "train_003181": ("sports court",),
    "train_009393": ("soccer field",),
    "train_009869": ("baseball field",),
    "train_015573": ("car",),
    "train_002906": ("baseball fields",),
    "train_009901": ("soccer fields",),
    "train_010397": ("black car",),
}

# Direct noun prompts from the question text. The existence question keeps
# every listed option so a positive object mask cannot be omitted by design.
REMAINING_PROMPTS = {
    "train_008852": ("construction vehicles", "parking lot", "highway"),
    "train_003423": ("basketball court",),
    "train_003585": ("rowing boats",),
    "train_003179": ("paved pathway", "grass field", "sports court"),
    "train_007923": (
        "children's playground slides",
        "outdoor fire pits",
        "public bus stops",
        "blue tennis courts",
    ),
    "train_000976": ("red bus", "traffic light", "vehicles", "intersection"),
    "train_000794": ("vehicles", "road", "camera"),
    "train_002115": ("vehicles", "highway", "traffic flow"),
    "train_012498": ("red barn",),
    "train_004698": ("swimming pool", "house", "backyard", "driveway"),
    "train_000645": ("blue basketball court", "sports courts", "badminton courts", "red banner"),
    "train_000073": ("trucks",),
    "train_008022": ("large open field",),
    "train_015640": ("parking lot", "objects"),
    "train_004573": ("fire", "roof", "ground floor", "building"),
    "train_009140": ("water", "lake", "stone border"),
    "train_002266": ("camera", "moving objects"),
}


def _download_output(output: Any, destination: Path) -> None:
    """Persist a Replicate file output without exposing authorization data."""

    if isinstance(output, (bytes, bytearray)):
        destination.write_bytes(bytes(output))
        return
    url = output if isinstance(output, str) else getattr(output, "url", None)
    if url:
        with urllib.request.urlopen(str(url), timeout=180) as response:
            destination.write_bytes(response.read())
        return
    if hasattr(output, "read"):
        destination.write_bytes(output.read())
        return
    try:
        destination.write_bytes(b"".join(bytes(chunk) for chunk in output))
    except Exception as exc:
        raise TypeError("Replicate returned an unsupported output type") from exc


def _resolve_rows(
    prompts: dict[str, tuple[str, ...]] = SUBSET_PROMPTS,
    *,
    require_complement: bool = False,
) -> list[dict]:
    pilot = json.loads(PILOT.read_text())
    rows_by_split = {
        split: json.loads((ROOT / "data" / "splits" / f"{split}.json").read_text())["QA"]
        for split in SPLITS
    }
    for split in SPLITS:
        expected = set(pilot["splits"][split]["ids"])
        actual = {row["qa_id"] for row in rows_by_split[split]}
        if not expected <= actual:
            raise RuntimeError(f"fixed {split} pilot IDs are absent from the local split")
    pilot_ids = {
        qa_id
        for split in SPLITS
        for qa_id in pilot["splits"][split]["ids"]
    }
    prompt_ids = set(prompts)
    if prompt_ids - pilot_ids:
        raise RuntimeError("SAM 3 subset contains IDs outside the approved pilot manifest")
    if require_complement:
        prior = json.loads(PRIOR_SUBSET.read_text())
        prior_ids = {
            qa_id
            for split in SPLITS
            for qa_id in prior["splits"][split]["ids"]
        }
        if prior_ids & prompt_ids or prompt_ids != pilot_ids - prior_ids:
            raise RuntimeError("SAM 3 supplemental subset must be the approved pilot complement")
    rows = R.selected_rows(rows_by_split, prompts)
    if len(rows) != len(prompts) or len({row["video_path"] for row in rows}) != len(prompts):
        raise RuntimeError("SAM 3 subset must contain one unique video per selected ID")
    return rows


def _stage_videos(rows: list[dict], target: Path) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    target.mkdir(parents=True, exist_ok=True)
    paths = sorted({row["video_path"] for row in rows})
    snapshot_download(
        repo_id=C.DATASET_REPO,
        repo_type="dataset",
        revision=C.DATASET_REVISION,
        local_dir=str(target),
        allow_patterns=paths,
        max_workers=8,
    )
    files = {path: target / path for path in paths}
    missing = [path for path, file in files.items() if not file.is_file()]
    if missing:
        raise RuntimeError(f"source dataset download omitted selected videos: {missing}")
    return files


def _write_subset_manifest(rows: list[dict], destination: Path) -> Path:
    by_split = {
        split: [row["qa_id"] for row in rows if row["split"] == split]
        for split in SPLITS
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "tasks/phase3_grounding_pilot_ids.json",
                "model": R.REPLICATE_MODEL,
                "version": R.REPLICATE_VERSION,
                "splits": {split: {"ids": ids} for split, ids in by_split.items()},
                "prompts": {row["qa_id"]: list(row["prompts"]) for row in rows},
            },
            indent=2,
        )
        + "\n"
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--supplemental",
        action="store_true",
        help="run the exact complement of the completed 15-video pilot under runs/sam3_remaining17_stageA_replicate/",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="run the first selected video only under runs/.../smoke/; keep full-run artifacts untouched",
    )
    return parser


def _is_oom_error(error: BaseException) -> bool:
    """Classify only explicit CUDA/model out-of-memory failures for retry."""

    text = str(error).casefold()
    return "out of memory" in text or "cuda oom" in text or "cuda out-of-memory" in text


def _fallback_source_indices(video_info: dict[str, Any]) -> list[int]:
    """Return the Stage A legacy 0/0.5 union for one source clip."""

    indices = set()
    for offset in (0.0, 0.5):
        indices.update(
            C.sample_index_list(
                int(video_info["n_frames"]),
                float(video_info["fps"]),
                offset,
                mode="legacy",
            )
        )
    return sorted(indices)


def _write_fallback_video(source: Path, destination: Path, source_indices: list[int]) -> None:
    """Encode exactly the selected source frames into a temporary 4 fps MP4.

    Stage A consumes 16 frames at each of two offsets from a 4 fps target.  The
    fallback therefore keeps the selected frames at that target rate so the
    temporary clip spans the same source interval; encoding them at the source
    rate would collapse a roughly eight-second clip to about one second.
    """

    import av

    wanted = set(int(index) for index in source_indices)
    if len(wanted) != len(source_indices) or not source_indices:
        raise ValueError("fallback source indices must be a non-empty unique list")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        with av.open(str(destination), mode="w") as output_container:
            output_stream = output_container.add_stream(
                "mpeg4", rate=Fraction(str(C.FPS_TARGET))
            )
            output_stream.width = int(input_stream.width)
            output_stream.height = int(input_stream.height)
            output_stream.pix_fmt = "yuv420p"
            for source_index, frame in enumerate(input_container.decode(input_stream)):
                if source_index not in wanted:
                    continue
                video_frame = av.VideoFrame.from_ndarray(
                    frame.to_ndarray(format="rgb24"), format="rgb24"
                )
                for packet in output_stream.encode(video_frame):
                    output_container.mux(packet)
                written += 1
                if written == len(source_indices):
                    break
            for packet in output_stream.encode():
                output_container.mux(packet)
    if written != len(source_indices):
        raise RuntimeError(
            f"fallback video extracted {written} frames; expected {len(source_indices)}"
        )


def _full_input_policy(frame_count: int) -> dict[str, Any]:
    return {
        "mode": "full-clip",
        "sampling_mode": "full",
        "input_frame_count": int(frame_count),
        "source_frame_indices": None,
        "input_frame_indices": None,
    }


def _fallback_input_policy(source_indices: list[int]) -> dict[str, Any]:
    input_indices = list(range(len(source_indices)))
    return {
        "mode": "legacy-union-fallback",
        "sampling_mode": "legacy",
        "offsets": [0.0, 0.5],
        "input_frame_count": len(input_indices),
        "source_frame_indices": list(source_indices),
        "input_frame_indices": input_indices,
    }


def _mapping_record(policy: dict[str, Any]) -> dict[str, Any]:
    if policy["mode"] == "legacy-union-fallback":
        return {
            "mode": "input-to-source",
            "input_frame_indices": list(policy["input_frame_indices"]),
            "source_frame_indices": list(policy["source_frame_indices"]),
        }
    return {"mode": "identity"}


def _mask_info_matches_policy(mask_info: dict[str, Any], policy: dict[str, Any]) -> bool:
    """Reject a valid ZIP whose frame ordinals do not fit its input clip."""

    try:
        input_count = int(policy["input_frame_count"])
        indices = [int(index) for index in mask_info.get("frame_indices", [])]
    except (KeyError, TypeError, ValueError):
        return False
    return input_count > 0 and all(0 <= index < input_count for index in indices)


def _load_call_mapping(
    path: Path, *, full_frame_count: int, expected_source_indices: list[int]
) -> tuple[dict[str, Any], dict[int, int], dict[str, Any], str] | None:
    """Load durable fallback metadata; absent metadata means a full-clip call."""

    if not path.exists():
        policy = _full_input_policy(full_frame_count)
        return policy, {}, _mapping_record(policy), "complete"
    try:
        record = json.loads(path.read_text())
        status = str(record.get("status", ""))
        if record.get("schema_version") != 1 or status not in {"pending", "complete"}:
            return None
        policy = record["input_policy"]
        mapping = record["frame_index_mapping"]
        source = [int(index) for index in mapping["source_frame_indices"]]
        inputs = [int(index) for index in mapping["input_frame_indices"]]
        if (
            policy.get("mode") != "legacy-union-fallback"
            or inputs != list(range(len(source)))
            or source != expected_source_indices
            or policy.get("source_frame_indices") != source
            or policy.get("input_frame_indices") != inputs
        ):
            return None
        frame_map = dict(zip(inputs, source))
        return policy, frame_map, {
            "mode": "input-to-source",
            "input_frame_indices": inputs,
            "source_frame_indices": source,
        }, status
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_call_mapping(
    path: Path, policy: dict[str, Any], mapping: dict[str, Any], *, status: str = "complete"
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": status,
                "input_policy": policy,
                "frame_index_mapping": mapping,
            },
            indent=2,
        )
        + "\n"
    )


def run(
    *,
    smoke_only: bool = False,
    run_root: Path = RUN,
    prompts: dict[str, tuple[str, ...]] = SUBSET_PROMPTS,
    require_complement: bool = False,
) -> int:
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        raise RuntimeError("REPLICATE_API_TOKEN is required; provide it through the process environment")
    try:
        import replicate
    except ModuleNotFoundError as exc:
        raise RuntimeError("install the official Replicate Python client before running this pilot") from exc

    rows = _resolve_rows(prompts, require_complement=require_complement)
    output_root = run_root / "smoke" if smoke_only else run_root
    output_root.mkdir(parents=True, exist_ok=True)
    selected_rows = rows[:1] if smoke_only else rows
    _write_subset_manifest(
        selected_rows if smoke_only else rows,
        output_root / "subset_ids.json",
    )
    records: list[dict] = []
    # ``call_count`` is completed prompt outputs, while attempts count every
    # prediction request for cost accounting, including failed full clips.
    call_count = 0
    prediction_attempt_count = 0
    reused_count = 0
    video_manifest = C.load_manifest(ROOT)
    with tempfile.TemporaryDirectory(prefix="reva_sam3_replicate_", dir="/tmp") as temp:
        videos = _stage_videos(selected_rows, Path(temp) / "dataset")
        for ordinal, row in enumerate(selected_rows):
            video_path = videos[row["video_path"]]
            try:
                video_info = video_manifest[row["video_path"]]
            except KeyError as exc:
                raise RuntimeError(f"manifest is missing selected video: {row['video_path']}") from exc
            prompt_zips: list[
                tuple[str, Path] | tuple[str, Path, dict[int, int]]
            ] = []
            prompt_records: list[dict] = []
            fallback_indices: list[int] | None = None
            expected_fallback_indices = _fallback_source_indices(video_info)
            for prompt_index, prompt in enumerate(row["prompts"]):
                zip_path = output_root / f"{row['qa_id']}.call-{prompt_index:02d}.masks.zip"
                mapping_path = output_root / f"{row['qa_id']}.call-{prompt_index:02d}.mapping.json"
                mask_info = R.cached_mask_info(zip_path)
                loaded_mapping = _load_call_mapping(
                    mapping_path,
                    full_frame_count=int(video_info["n_frames"]),
                    expected_source_indices=expected_fallback_indices,
                )
                if mask_info is not None and loaded_mapping is not None:
                    if not _mask_info_matches_policy(mask_info, loaded_mapping[0]):
                        loaded_mapping = None
                reused = mask_info is not None and loaded_mapping is not None
                prompt_attempts = 0
                if reused:
                    input_policy, frame_map, frame_mapping, _ = loaded_mapping
                    reused_count += 1
                    print(f"reusing validated {zip_path.name}")
                else:
                    if zip_path.exists() or mapping_path.exists():
                        print(f"cached {zip_path.name} is invalid; regenerating only this call")
                    input_policy = _full_input_policy(int(video_info["n_frames"]))
                    frame_map: dict[int, int] = {}
                    frame_mapping = _mapping_record(input_policy)
                    pending_fallback = (
                        mask_info is None
                        and loaded_mapping is not None
                        and loaded_mapping[3] == "pending"
                    )

                    def prepare_fallback(*, resumed: bool) -> Any:
                        nonlocal fallback_indices, input_policy, frame_map, frame_mapping
                        if pending_fallback:
                            input_policy, frame_map, frame_mapping, _ = loaded_mapping
                            fallback_indices = list(input_policy["source_frame_indices"])
                        else:
                            if fallback_indices is None:
                                fallback_indices = _fallback_source_indices(video_info)
                            input_policy = _fallback_input_policy(fallback_indices)
                            frame_map = dict(
                                zip(
                                    input_policy["input_frame_indices"],
                                    input_policy["source_frame_indices"],
                                )
                            )
                            frame_mapping = _mapping_record(input_policy)
                        fallback_video = (
                            Path(temp)
                            / f"{row['qa_id']}.call-{prompt_index:02d}.legacy-union.mp4"
                        )
                        _write_fallback_video(video_path, fallback_video, fallback_indices)
                        if resumed:
                            print(
                                f"resuming pending fallback for {row['qa_id']} call {prompt_index} "
                                f"with {len(fallback_indices)} legacy sampled frames"
                            )
                        else:
                            print(
                                f"OOM on {row['qa_id']} call {prompt_index}; retrying with "
                                f"{len(fallback_indices)} legacy sampled frames"
                            )
                        _write_call_mapping(
                            mapping_path,
                            input_policy,
                            frame_mapping,
                            status="pending",
                        )
                        return R.prediction_input(fallback_video, prompt)

                    fallback_payload = prepare_fallback(resumed=True) if pending_fallback else None
                    if fallback_payload is None:
                        input_payload = R.prediction_input(video_path, prompt)
                        try:
                            prompt_attempts += 1
                            prediction_attempt_count += 1
                            with input_payload["video"].open("rb") as video_stream:
                                output = replicate.run(
                                    R.prediction_ref(),
                                    input={**input_payload, "video": video_stream},
                                )
                        except Exception as exc:
                            if not _is_oom_error(exc):
                                raise RuntimeError(
                                    f"Replicate SAM 3 prediction failed for {row['qa_id']} call {prompt_index}: {type(exc).__name__}"
                                ) from exc
                            fallback_payload = prepare_fallback(resumed=False)
                    try:
                        if fallback_payload is not None:
                            prompt_attempts += 1
                            prediction_attempt_count += 1
                            with fallback_payload["video"].open("rb") as video_stream:
                                output = replicate.run(
                                    R.prediction_ref(),
                                    input={**fallback_payload, "video": video_stream},
                                )
                    except Exception as retry_exc:
                        raise RuntimeError(
                            f"Replicate SAM 3 fallback failed for {row['qa_id']} call {prompt_index}: {type(retry_exc).__name__}"
                        ) from retry_exc
                    _download_output(output, zip_path)
                    mask_info = R.inspect_mask_zip(zip_path.read_bytes())
                    if not _mask_info_matches_policy(mask_info, input_policy):
                        raise ValueError(
                            f"SAM 3 output frame index exceeds {input_policy['mode']} input range"
                        )
                    if input_policy["mode"] == "legacy-union-fallback":
                        _write_call_mapping(mapping_path, input_policy, frame_mapping)
                    elif mapping_path.exists():
                        mapping_path.unlink()
                prompt_zips.append((prompt, zip_path))
                if frame_map:
                    prompt_zips[-1] = (prompt, zip_path, frame_map)
                prompt_records.append(
                    {
                        "prompt": prompt,
                        "zip": zip_path.name,
                        "reused": reused,
                        "prediction_attempts": prompt_attempts,
                        "input_policy": input_policy,
                        "frame_index_mapping": frame_mapping,
                        **mask_info,
                    }
                )
                if not reused:
                    call_count += 1
            cache_path = M.write_overlay_cache(
                output_root / "overlay_cache",
                qa_id=row["qa_id"],
                video_path=row["video_path"],
                prompt_zips=prompt_zips,
                fps=float(video_info["fps"]),
                frame_count=int(video_info["n_frames"]),
            )
            cache_info = M.load_overlay_cache(output_root / "overlay_cache", row["qa_id"])
            records.append(
                {
                    "qa_id": row["qa_id"],
                    "split": row["split"],
                    "video_path": row["video_path"],
                    "prompts": list(row["prompts"]),
                    "model": R.REPLICATE_MODEL,
                    "version": R.REPLICATE_VERSION,
                    "inputs": R.OUTPUT_REQUIRED_INPUTS,
                    "calls": prompt_records,
                    "overlay_cache": str(cache_path.relative_to(output_root)),
                    "overlay_mask_count": cache_info["mask_count"],
                }
            )
            print(
                f"completed {ordinal + 1}/{len(selected_rows)} {row['qa_id']} "
                f"calls={len(prompt_records)} masks={cache_info['mask_count']}"
            )
            if ordinal == 0:
                print("smoke output and overlay cache validated; continuing with remaining selected videos")
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "model": R.REPLICATE_MODEL,
                "version": R.REPLICATE_VERSION,
                "approx_cost_per_call_usd": R.REPLICATE_APPROX_RUN_USD,
                "call_count": call_count,
                "prediction_attempt_count": prediction_attempt_count,
                "planned_call_count": sum(len(row["prompts"]) for row in rows),
                "reused_call_count": reused_count,
                "approx_cost_usd": round(prediction_attempt_count * R.REPLICATE_APPROX_RUN_USD, 6),
                "smoke_only": smoke_only,
                "complete": not smoke_only,
                "selected_video_count": len(selected_rows),
                "full_subset_video_count": len(rows),
                "records": records,
            },
            indent=2,
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    cli_args = build_parser().parse_args()
    if cli_args.supplemental:
        raise SystemExit(
            run(
                smoke_only=cli_args.smoke_only,
                run_root=SUPPLEMENTAL_RUN,
                prompts=REMAINING_PROMPTS,
                require_complement=True,
            )
        )
    raise SystemExit(run(smoke_only=cli_args.smoke_only))
