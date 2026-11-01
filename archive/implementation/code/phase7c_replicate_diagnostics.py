"""Render offline SAM 3 video diagnostic sheets from Phase 7C outputs.

The script consumes only the Phase 7C manifest, staged source videos, and
already cached prediction ZIPs. It makes no provider requests.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont


RUN_REL = Path("runs/phase7C_replicate_smoke")
SHEETS = {
    "train_000047": {
        "frames": [84, 90, 104, 120, 147, 179],
        "target_prompt": "large archway with Chinese characters",
        "control_prompt": "building",
    },
    "train_001138": {
        "frames": [0, 30, 49, 60, 90, 120],
        "target_prompt": "white van with blue markings on its side",
        "control_prompt": None,
    },
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _extract_frame(video_path: Path, frame_index: int) -> Image.Image:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(video_path),
        "-vf", f"select=eq(n\\,{frame_index})", "-vsync", "0", "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "png", "-",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    return Image.open(io.BytesIO(result.stdout)).convert("RGB")


def _request(manifest: dict, qa_id: str, model_key: str, prompt: str) -> dict | None:
    for request in manifest["requests"]:
        if (
            request.get("qa_id") == qa_id
            and request.get("model_key") == model_key
            and request.get("prompt") == prompt
            and request.get("status") == "succeeded"
        ):
            return request
    return None


def _read_id_mask(zip_path: Path, frame_index: int, label: int) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive:
        name = f"labels/labels_{frame_index:05d}.png"
        label_image = np.asarray(Image.open(io.BytesIO(archive.read(name))))
    return label_image == label


def _read_merged_mask(zip_path: Path, frame_index: int) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive:
        matches = [
            name for name in archive.namelist()
            if Path(name).name.casefold() in {
                f"mask_{frame_index:05d}.png", f"frame_{frame_index:05d}.png"
            }
        ]
        if len(matches) != 1:
            return np.zeros((360, 640), dtype=bool)
        values = np.asarray(Image.open(io.BytesIO(archive.read(matches[0]))))
    if values.ndim == 3:
        return np.any(values != 0, axis=2)
    return values != 0


def _overlay(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]) -> tuple[Image.Image, int, list[int] | None]:
    result = image.convert("RGBA")
    area = int(mask.sum())
    box = None
    if area:
        ys, xs = np.where(mask)
        box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        tint = Image.new("RGBA", result.size, color + (0,))
        tint.putalpha(Image.fromarray(np.where(mask, 100, 0).astype(np.uint8)))
        result = Image.alpha_composite(result, tint)
        draw = ImageDraw.Draw(result)
        draw.rectangle(box, outline=color + (255,), width=3)
    return result.convert("RGB"), area, box


def _object_label(request: dict) -> tuple[int | None, dict | None]:
    objects = (request.get("output_validation") or {}).get("objects", [])
    if not objects:
        return None, None
    # The question-conditioned endpoint output has a single object. For
    # multi-object controls, choose the largest reported instance strictly
    # for visualization; this choice is never fed back into retrieval.
    item = max(objects, key=lambda row: int(row.get("best_area", 0)))
    return int(item["label"]), item


def _make_sheet(root: Path, manifest: dict, video: dict, qa_id: str, spec: dict, output: Path) -> dict:
    frames = spec["frames"]
    target = _request(manifest, qa_id, "id_preserving_candidate", spec["target_prompt"])
    if target is None:
        raise ValueError(f"no validated question-conditioned ID masks for {qa_id}")
    target_label, target_object = _object_label(target)
    if target_label is None:
        raise ValueError(f"question-conditioned output has no object labels for {qa_id}")
    control = None
    control_label = None
    control_object = None
    if spec["control_prompt"]:
        control = _request(manifest, qa_id, "id_preserving_candidate", spec["control_prompt"])
        if control is not None:
            control_label, control_object = _object_label(control)

    cell_w, cell_h, header_h = 640, 360, 42
    rows = 3 if control and control_label is not None else 2
    sheet = Image.new("RGB", (cell_w * len(frames), (cell_h + header_h) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    row_titles = [
        "Source frame",
        f"Question prompt · SAM ID {target_label} · first frame {target_object['first_frame']}",
    ]
    if rows == 3:
        row_titles.append(f"Fixed prompt '{spec['control_prompt']}' · largest returned ID {control_label}")

    target_path = root / RUN_REL / target["output_relpath"]
    control_path = root / RUN_REL / control["output_relpath"] if control else None
    records = []
    for row in range(rows):
        for col, frame_index in enumerate(frames):
            x, y = col * cell_w, row * (cell_h + header_h)
            if row == 0:
                image = _extract_frame(root / video["source_path"], frame_index)
                area = 0
                box = None
            elif row == 1:
                mask = _read_id_mask(target_path, frame_index, target_label)
                image, area, box = _overlay(_extract_frame(root / video["source_path"], frame_index), mask, (240, 30, 30))
            else:
                mask = _read_id_mask(control_path, frame_index, control_label)
                image, area, box = _overlay(_extract_frame(root / video["source_path"], frame_index), mask, (20, 180, 255))
            sheet.paste(image, (x, y + header_h))
            if row == 0:
                timestamp = frame_index / float(video["fps"])
                text = f"frame {frame_index} · {timestamp:.2f}s"
            else:
                text = f"frame {frame_index} · mask px {area} · {area / (cell_w * cell_h):.3%}"
                records.append({"row": row, "frame_index": frame_index, "mask_area_px": area, "mask_fraction": area / (cell_w * cell_h), "bbox_xyxy": box})
            draw.text((x + 8, y + 10), text, fill="black", font=font)
        draw.text((8, row * (cell_h + header_h) + 28), row_titles[row], fill="#222222", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return {
        "qa_id": qa_id,
        "question_prompt": spec["target_prompt"],
        "target_instance": target_object,
        "timestamp_mapping_status": target["output_validation"]["timestamp_mapping_status"],
        "control_prompt": spec["control_prompt"],
        "control_instance": control_object,
        "sheet": str(output.relative_to(root)),
        "frame_mask_summary": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest or root / RUN_REL / "manifest.json"
    manifest = _json(manifest_path)
    videos = {video["qa_id"]: video for video in manifest["videos"]}
    results = []
    for qa_id, spec in SHEETS.items():
        target_request = _request(manifest, qa_id, "id_preserving_candidate", spec["target_prompt"])
        if target_request is None:
            raise ValueError(f"missing validated target output: {qa_id}")
        output = root / RUN_REL / "diagnostics" / f"{qa_id}_sam3_video_contact_sheet.jpg"
        results.append(_make_sheet(root, manifest, videos[qa_id], qa_id, spec, output))
    report = {
        "schema_version": 1,
        "kind": "phase7c_cached_sam3_visual_diagnostics",
        "provider_requests_made": 0,
        "mapping_note": "ID-preserving output maps frame indices by matching source frame count/FPS; merged-mask mapping remains unverified.",
        "examples": results,
    }
    report_path = root / RUN_REL / "diagnostics" / "visualization_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sheets": [item["sheet"] for item in results], "report": str(report_path.relative_to(root)), "provider_requests_made": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
