"""Run the approved, preflight-gated two-video Phase 3A SAM 3 smoke.

The command has a dry-run mode that performs only local manifest checks.  A
real run requires a non-secret provider attestation and a token already in the
process environment; neither is read from a file or printed.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.request
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "code"))
import common as C
import phase3a_retrieval as P
import replicate_sam3 as R


def _download_output(output: Any, destination: Path) -> None:
    """Persist only model bytes; signed URLs and credentials never enter logs."""

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


def _write_video(
    source: Path,
    destination: Path,
    source_indices: list[int],
    bounds: Mapping[str, int] | None = None,
) -> tuple[int, int]:
    """Pack selected source-native pixels into a 4 fps video without resizing."""

    import av

    wanted = set(int(index) for index in source_indices)
    if len(wanted) != len(source_indices) or not source_indices:
        raise ValueError("source frame indices must be unique and non-empty")
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with av.open(str(source)) as input_container:
        input_stream = input_container.streams.video[0]
        with av.open(str(destination), mode="w") as output_container:
            output_stream = output_container.add_stream("mpeg4", rate=Fraction("4.0"))
            width = int(input_stream.width)
            height = int(input_stream.height)
            if bounds is not None:
                left, top = int(bounds["left"]), int(bounds["top"])
                right, bottom = int(bounds["right"]), int(bounds["bottom"])
                if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                    raise ValueError("tile bounds are outside source-native frame dimensions")
                width, height = right - left, bottom - top
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = "yuv420p"
            for source_index, frame in enumerate(input_container.decode(input_stream)):
                if source_index not in wanted:
                    continue
                array = frame.to_ndarray(format="rgb24")
                if bounds is not None:
                    array = array[top:bottom, left:right]
                video_frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                for packet in output_stream.encode(video_frame):
                    output_container.mux(packet)
                written += 1
                if written == len(source_indices):
                    break
            for packet in output_stream.encode():
                output_container.mux(packet)
    if written != len(source_indices):
        raise RuntimeError(f"packed {written} frames; expected {len(source_indices)}")
    return width, height


def _tile_mask_dimensions(zip_path: Path, expected: tuple[int, int]) -> None:
    from PIL import Image

    expected_width, expected_height = expected
    for entry in R.read_mask_zip(zip_path.read_bytes()):
        with Image.open(__import__("io").BytesIO(entry["payload"])) as image:
            if image.size != (expected_width, expected_height):
                raise ValueError(
                    f"tile mask dimensions {image.size} do not match tile input {(expected_width, expected_height)}"
                )


def _run_call(
    replicate: Any,
    video: Path,
    prompt: str,
    output: Path,
    *,
    max_attempts: int,
    attempts: list[int],
    retries: list[int],
) -> tuple[dict[str, Any], int]:
    """Run one pinned call with a bounded retry count and no secret-bearing errors."""

    last_error: str | None = None
    call_attempts = 0
    while True:
        if attempts[0] >= max_attempts:
            raise RuntimeError("Replicate attempt cap reached")
        attempts[0] += 1
        call_attempts += 1
        started = time.perf_counter()
        try:
            payload = R.prediction_input(video, prompt)
            with payload["video"].open("rb") as stream:
                result = replicate.run(
                    R.prediction_ref(),
                    input={**payload, "video": stream},
                )
            _download_output(result, output)
            info = R.inspect_mask_zip(output.read_bytes())
            frame_indices = [int(index) for index in info["frame_indices"]]
            if any(index < 0 or index >= 16 for index in frame_indices):
                raise ValueError("SAM 3 output frame index is outside the packed 16-frame input")
            return (
                {
                    "status": "complete",
                    "prediction_attempts": call_attempts,
                    "retry_count": call_attempts - 1,
                    "duration_seconds": round(time.perf_counter() - started, 6),
                    "mask_count": int(info["mask_count"]),
                    "mask_frame_indices": frame_indices,
                    "native_score_status": "unavailable: return_zip contained mask PNG entries without attributable per-mask scores",
                },
                attempts[0],
            )
        except Exception as exc:
            last_error = type(exc).__name__
            if retries[0] >= 5:
                raise RuntimeError(f"Replicate call failed after retry budget: {last_error}") from exc
            retries[0] += 1
            if output.exists():
                output.unlink()


def _stage_videos(rows: list[dict[str, Any]], destination: Path) -> dict[str, Path]:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    paths = sorted({str(row["row"]["video_path"]) for row in rows})
    snapshot_download(
        repo_id=C.DATASET_REPO,
        repo_type="dataset",
        revision=C.DATASET_REVISION,
        local_dir=str(destination),
        allow_patterns=paths,
        max_workers=4,
    )
    result = {path: destination / path for path in paths}
    missing = [path for path, value in result.items() if not value.is_file()]
    if missing:
        raise RuntimeError(f"selected source videos are missing: {missing}")
    return result


def run(
    root: str | Path = ROOT,
    *,
    attestation: str | Path | None = None,
    backend: str = "modal",
) -> int:
    root = Path(root).resolve()
    plan = P.load_plan(root)
    rows = P.validate_plan_rows(root, plan)
    calls = P.build_call_plan(plan)
    readiness = P.preflight(root, plan, attestation, backend=backend)
    if not readiness["ready"]:
        raise RuntimeError("Phase 3A smoke blocked: " + "; ".join(readiness["reasons"]))
    output_root = root / str(plan["artifact_root"])
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing Phase 3A artifacts: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight.json").write_text(json.dumps(readiness, indent=2) + "\n")

    import replicate

    attempts = [0]
    retries = [0]
    records: list[dict[str, Any]] = []
    source_policies: dict[str, dict[str, Any]] = {}
    tile_status = "not-started"
    with tempfile.TemporaryDirectory(prefix="reva_phase3a_", dir="/tmp") as temp:
        videos = _stage_videos(rows, Path(temp) / "dataset")
        packed: dict[str, Path] = {}
        tile_paths: dict[int, tuple[Path, dict[str, int]]] = {}
        manifest = C.load_manifest(root)
        for item in rows:
            row = item["row"]
            qa_id = str(row["qa_id"])
            info = manifest[str(row["video_path"])]
            policy = P.source_frame_policy(info)
            source_policies[qa_id] = {
                "video_path": row["video_path"],
                "video_checksum": C.path_checksum(videos[str(row["video_path"])]),
                "source_frame_indices": policy["source_frame_indices"],
                "source_timestamps": policy["source_timestamps"],
                "source_fps": float(info["fps"]),
                "source_width": int(info["width"]),
                "source_height": int(info["height"]),
            }
            packed_path = Path(temp) / f"{qa_id}.legacy-offset-0.mp4"
            _write_video(videos[str(row["video_path"])], packed_path, policy["source_frame_indices"])
            packed[qa_id] = packed_path
        tile_probe = plan["tile_probe"]
        first = source_policies[str(tile_probe["qa_id"])]
        bounds = P.tile_bounds(first["source_width"], first["source_height"], float(tile_probe["overlap_fraction"]))
        for index, bound in enumerate(bounds):
            path = Path(temp) / f"{tile_probe['qa_id']}.tile-{index:02d}.mp4"
            source_video = videos[str(first["video_path"])]
            size = _write_video(
                source_video,
                path,
                first["source_frame_indices"],
                bound,
            )
            tile_paths[index] = (path, bound)
        for call_index, call in enumerate(calls):
            if call["arm"] == "tile-probe" and tile_status == "stopped":
                continue
            if call["arm"] == "tile-probe":
                tile_status = "running"
                video, bound = tile_paths[int(call["tile"])]
            else:
                video, bound = packed[str(call["qa_id"])], None
            relative_dir = output_root / "replicate" / str(call["qa_id"])
            relative_dir.mkdir(parents=True, exist_ok=True)
            zip_path = relative_dir / f"call-{call_index:02d}.masks.zip"
            try:
                result, _ = _run_call(
                    replicate,
                    video,
                    str(call["prompt"]),
                    zip_path,
                    max_attempts=int(plan["replicate_budget"]["max_attempts"]),
                    attempts=attempts,
                    retries=retries,
                )
                if call["arm"] == "tile-probe":
                    _tile_mask_dimensions(zip_path, (bound["right"] - bound["left"], bound["bottom"] - bound["top"]))
                    result["coordinate_mapping"] = {"source_bounds": bound, "source_frame_indices": first["source_frame_indices"]}
                records.append({**call, **result, "zip": str(zip_path.relative_to(output_root))})
            except Exception as exc:
                if call["arm"] == "tile-probe":
                    tile_status = "stopped"
                    records.append({**call, "status": "stopped_after_mapping_failure", "error_type": type(exc).__name__})
                    continue
                raise
        if tile_status == "running":
            tile_status = "complete"

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "plan": plan,
        "model": plan["model"],
        "source_policies": source_policies,
        "records": records,
        "tile_probe_status": tile_status,
        "replicate_attempts": attempts[0],
        "replicate_retries": retries[0],
        "replicate_budget": plan["replicate_budget"],
        "actual_cost_usd": None,
        "actual_cost_status": "provider billing must be reconciled after the run; replicate.run does not return billed USD",
        "manual_audit_status": "pending_blind_annotation",
        "native_score_status": "unavailable unless the provider returns attributable per-mask scores",
        "ranking_status": "pending Modal SigLIP2 feature job",
        "qwen_scoring": "not-run",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attestation", help="non-secret provider-cap attestation JSON")
    parser.add_argument("--backend", choices=("modal", "hpc"), default="modal")
    parser.add_argument("--dry-run", action="store_true", help="validate local plan and print the call plan")
    parser.add_argument("--check-preflight", action="store_true", help="check gates without staging or calling providers")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = P.load_plan(ROOT)
    rows = P.validate_plan_rows(ROOT, plan)
    calls = P.build_call_plan(plan)
    if args.dry_run:
        print(json.dumps({"videos": [item["spec"]["qa_id"] for item in rows], "calls": calls}, indent=2))
        return 0
    if args.check_preflight:
        result = P.preflight(ROOT, plan, args.attestation, backend=args.backend)
        print(json.dumps(result, indent=2))
        return 0 if result["ready"] else 2
    return run(ROOT, attestation=args.attestation, backend=args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
