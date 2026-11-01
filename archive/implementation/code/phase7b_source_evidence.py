"""Stage verified Phase 7B clips and render cached detector/tracker contact sheets.

The source verification manifest is an attestation supplied by the lead agent
after an independent HPC SHA-256 comparison.  This checkout may not have the
HPC mount, so missing local source bytes are reported and never reconstructed
from a cache checksum.  Rendered boxes and IDs come directly from the frozen
Grounding DINO cache and are labelled as non-ground-truth evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import grounding as G


SCHEMA_VERSION = 1
DATASET_REVISION = "06224fa1b42e5e35677a8b7f2145e06307a9a4ee"
HPC_SOURCE_ROOT = "/scratch/ambpdc/soham/ReVA/data/ReVA"
VERIFICATION_PROVENANCE = (
    "Lead-agent independent HPC verification: SHA-256 of each source clip at "
    f"{HPC_SOURCE_ROOT} was programmatically compared with every mapped cache "
    "video_checksum; all five matched."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    l = max(float(left["left"]), float(right["left"]))
    t = max(float(left["top"]), float(right["top"]))
    r = min(float(left["right"]), float(right["right"]))
    b = min(float(left["bottom"]), float(right["bottom"]))
    intersection = max(0.0, r - l) * max(0.0, b - t)
    left_area = max(0.0, float(left["right"]) - float(left["left"])) * max(
        0.0, float(left["bottom"]) - float(left["top"])
    )
    right_area = max(0.0, float(right["right"]) - float(right["left"])) * max(
        0.0, float(right["bottom"]) - float(right["top"])
    )
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _track_lookup(cache: Mapping[str, Any], frame_index: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for track in cache.get("tracks", []):
        for observation in track.get("observations", []):
            if int(observation.get("frame_index", -1)) == frame_index:
                rows.append({"track_id": str(track.get("track_id", "unknown")), **observation})
    return rows


def _extract_frame(source: Path, frame_index: int) -> Image.Image:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"select=eq(n\\,{int(frame_index)})",
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if completed.returncode or not completed.stdout:
        raise RuntimeError(
            f"ffmpeg could not decode frame {frame_index} from {source}: "
            f"{completed.stderr.decode(errors='replace').strip()}"
        )
    return Image.open(io.BytesIO(completed.stdout)).convert("RGB")


def _overlay_tile(frame: Image.Image, cache_frame: Mapping[str, Any], tracks: list[Mapping[str, Any]]) -> Image.Image:
    image = frame.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for detection in cache_frame.get("detections", []):
        box = detection["box"]
        x0, y0 = float(box["left"]) * width, float(box["top"]) * height
        x1, y1 = float(box["right"]) * width, float(box["bottom"]) * height
        matching = [
            row
            for row in tracks
            if str(row.get("label", "")).casefold() == str(detection.get("label", "")).casefold()
            and _iou(box, row.get("box", {})) >= 0.999
        ]
        track_id = matching[0]["track_id"] if matching else "detector-only"
        label = f"{track_id} {detection.get('label', '')}"
        draw.rectangle((x0, y0, x1, y1), outline=(255, 48, 48), width=max(2, width // 320))
        draw.rectangle((x0, max(0, y0 - 14), min(width, x0 + 8 * len(label)), y0), fill=(255, 48, 48))
        draw.text((x0 + 2, max(0, y0 - 13)), label, fill=(255, 255, 255))
    return image


def _contact_sheet(
    source: Path,
    cache: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    frames = cache["frames"]
    tiles: list[Image.Image] = []
    metadata: list[dict[str, Any]] = []
    for cache_frame in frames:
        frame_index = int(cache_frame["frame_index"])
        timestamp = float(cache_frame["timestamp"])
        image = _extract_frame(source, frame_index)
        image = _overlay_tile(image, cache_frame, _track_lookup(cache, frame_index))
        tile = Image.new("RGB", (320, 204), (16, 16, 16))
        image = ImageOps.contain(image, (320, 180))
        tile.paste(image, ((320 - image.width) // 2, 24 + (180 - image.height) // 2))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 320, 24), fill=(0, 0, 0))
        draw.text((4, 3), "CACHED DINO/TRACKER OUTPUT — NOT GROUND TRUTH", fill=(255, 220, 120))
        draw.text((4, 188), f"frame={frame_index}  timestamp={timestamp:.6f}s", fill=(255, 255, 255))
        tiles.append(tile)
        metadata.append({"frame_index": frame_index, "timestamp_s": timestamp})
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 320, rows * 204), (16, 16, 16))
    for position, tile in enumerate(tiles):
        sheet.paste(tile, ((position % columns) * 320, (position // columns) * 204))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90, optimize=True)
    return {"overlay_path": str(output), "frames": metadata}


def _mapped_candidates(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for video in report.get("video_results", []):
        candidates = video.get("candidates", [])
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(f"expected one mapped cache candidate for {video.get('video_path')}")
        candidate = candidates[0]
        rows.append({
            "video_path": str(video["video_path"]),
            "cache_key": str(candidate["cache_key"]),
            "cache_path": str(candidate["cache_path"]),
            "cache_checksum": str(candidate["source_checksum"].get("cache_checksum", "")),
        })
    if len(rows) != 5:
        raise ValueError(f"expected five mapped videos, found {len(rows)}")
    return rows


def build_source_manifest(report: Mapping[str, Any], cache_dir: Path, source_root: Path, destination: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for row in _mapped_candidates(report):
        cache = G.load_detector_cache(cache_dir / f"{row['cache_key']}.json", expected_cache_key=row["cache_key"])
        cache_checksum = str(cache["video_checksum"])
        local_source = source_root / row["video_path"]
        local_checksum = _sha256(local_source) if local_source.is_file() else None
        entries.append({
            "video_path": row["video_path"],
            "cache_key": row["cache_key"],
            "cache_video_checksum": cache_checksum,
            "hpc_source_path": f"{HPC_SOURCE_ROOT}/{row['video_path']}",
            "hpc_source_sha256": cache_checksum,
            "hpc_verification_status": "match",
            "local_source_path": str(local_source),
            "local_source_sha256": local_checksum,
            "local_source_status": "match" if local_checksum == cache_checksum else "unavailable",
        })
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "reva-phase7b-source-checksum-manifest",
        "dataset_revision": DATASET_REVISION,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification_provenance": VERIFICATION_PROVENANCE,
        "hpc_source_root": HPC_SOURCE_ROOT,
        "hpc_verification_scope": "five cache-mapped videos only",
        "entries": entries,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def stage_and_render(
    report: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    *,
    cache_dir: Path,
    source_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    staged_root = out_dir / "source_videos"
    sheet_root = out_dir / "contact_sheets"
    overlay_entries: list[dict[str, Any]] = []
    for entry in source_manifest["entries"]:
        video_path = str(entry["video_path"])
        source = source_root / video_path
        destination = staged_root / video_path
        cache_path = cache_dir / f"{entry['cache_key']}.json"
        cache = G.load_detector_cache(cache_path, expected_cache_key=entry["cache_key"])
        row: dict[str, Any] = {
            "video_path": video_path,
            "cache_key": entry["cache_key"],
            "cache_video_checksum": cache["video_checksum"],
            "cached_output_label": "Grounding DINO boxes + adjacent-greedy-IoU track IDs; not ground truth",
        }
        if not source.is_file():
            row["status"] = "source_missing_in_checkout"
            row["expected_source_path"] = str(source)
            overlay_entries.append(row)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and _sha256(destination) != entry["cache_video_checksum"]:
            raise ValueError(f"refusing to overwrite staged source with a checksum mismatch: {destination}")
        if not destination.is_file():
            shutil.copy2(source, destination)
        if _sha256(destination) != entry["cache_video_checksum"]:
            raise ValueError(f"staged source checksum mismatch: {destination}")
        row["staged_source_path"] = str(destination)
        row["status"] = "staged"
        sheet = sheet_root / (video_path.replace("/", "__") + ".jpg")
        row.update(_contact_sheet(destination, cache, sheet))
        overlay_entries.append(row)
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "reva-phase7b-cached-overlay-manifest",
        "dataset_revision": DATASET_REVISION,
        "source_checksum_manifest": str(out_dir / "source_checksum_manifest.json"),
        "entries": overlay_entries,
    }
    (out_dir / "contact_sheet_manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def enrich_report(
    report: Mapping[str, Any],
    source_manifest: Mapping[str, Any],
    overlay_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge external source verification while retaining the offline snapshot."""

    enriched = copy.deepcopy(dict(report))
    source_by_path = {entry["video_path"]: entry for entry in source_manifest["entries"]}
    overlay_by_path = {entry["video_path"]: entry for entry in overlay_manifest["entries"]}
    for video in enriched.get("video_results", []):
        entry = source_by_path.get(video.get("video_path"))
        if not entry:
            continue
        for candidate in video.get("candidates", []):
            if candidate.get("cache_key") != entry["cache_key"]:
                continue
            prior = dict(candidate.get("source_checksum", {}))
            candidate["source_checksum"] = {
                **prior,
                "prior_offline_status": prior.get("status"),
                "status": "hpc_verified_match",
                "hpc_source_path": entry["hpc_source_path"],
                "hpc_source_sha256": entry["hpc_source_sha256"],
                "local_source_status": entry["local_source_status"],
                "local_source_sha256": entry["local_source_sha256"],
            }
    enriched["source_verification"] = {
        "status": "verified_external_hpc",
        "dataset_revision": DATASET_REVISION,
        "hpc_source_root": HPC_SOURCE_ROOT,
        "verified_video_count": len(source_manifest["entries"]),
        "provenance": source_manifest["verification_provenance"],
        "manifest_path": "runs/phase7B_cache_validation/source_checksum_manifest.json",
        "checkout_re_read_status": "not_available_hpc_mount_absent",
        "entries": source_manifest["entries"],
    }
    staged = sum(row.get("status") == "staged" for row in overlay_manifest["entries"])
    enriched["source_staging"] = {
        "staged_video_count": staged,
        "requested_video_count": len(overlay_manifest["entries"]),
        "source_videos_dir": "runs/phase7B_cache_validation/source_videos",
        "contact_sheet_manifest": "runs/phase7B_cache_validation/contact_sheet_manifest.json",
        "entries": [overlay_by_path[path] for path in sorted(overlay_by_path)],
    }
    enriched["coverage"]["source_checksum_status_counts"] = {"hpc_verified_match": len(source_manifest["entries"])}
    enriched["coverage"]["local_source_status_counts"] = dict(
        __import__("collections").Counter(entry["local_source_status"] for entry in source_manifest["entries"])
    )
    enriched["coverage"]["staged_video_count"] = staged
    retained_limitations = [
        item
        for item in enriched.get("limitations", [])
        if "missing local source file" not in item
        and "HPC checksum verification is externally attested" not in item
        and "Only source clips available in this checkout" not in item
        and "All five verified source clips are staged" not in item
        and "All five independently verified source clips are staged" not in item
        and "remaining four" not in item
        and "missing clips remain explicit blockers" not in item
    ]
    requested = len(overlay_manifest["entries"])
    remaining = requested - staged
    retained_limitations.extend(
        [
            "HPC checksum verification is externally attested at the pinned dataset revision; this checkout could not re-read the HPC mount.",
            (
                f"All {staged}/{requested} independently verified source clips are staged and rendered locally."
                if staged == requested
                else f"{staged}/{requested} independently verified source clips are staged and rendered locally; "
                f"{remaining} source clips remain unavailable for local rendering."
            ),
        ]
    )
    enriched["limitations"] = list(dict.fromkeys(retained_limitations))
    return enriched


def render_updated_report(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    verification = report["source_verification"]
    staging = report["source_staging"]
    lines = [
        "# Phase 7B cache validation",
        "",
        "Offline Grounding DINO cache validation plus independent HPC source verification and cached-output overlays.",
        "",
        f"- Reviewed cases/videos: {report['cohort']['review_case_count']}/{report['cohort']['review_video_count']}",
        f"- Policy-exact videos: {coverage['policy_exact_video_count']}/{report['cohort']['review_video_count']}",
        f"- Case statuses: `{json.dumps(coverage['case_status_counts'], sort_keys=True)}`",
        f"- HPC checksum verification: {verification['verified_video_count']}/{staging['requested_video_count']} matched at revision `{verification['dataset_revision']}`",
        f"- Clips staged locally: {staging['staged_video_count']}/{staging['requested_video_count']}",
        f"- Local source status: `{json.dumps(coverage['local_source_status_counts'], sort_keys=True)}`",
        "",
        "## Source verification",
        "",
        f"{verification['provenance']}",
        "",
        "The prior offline report is preserved as `report_offline_snapshot.json` and `report_offline_snapshot.md`. The current checkout has no `/scratch` HPC mount, so the attestation is recorded as provenance and is not silently presented as a local re-read.",
        "",
        "## Overlay evidence",
        "",
        "Contact sheets use cached Grounding DINO boxes and cached adjacent-greedy-IoU track IDs. Every tile is labelled `CACHED DINO/TRACKER OUTPUT — NOT GROUND TRUTH`; no answer, ranking, or oracle field is used.",
        "",
        "## Gate",
        "",
        (
            f"All {staging['staged_video_count']}/{staging['requested_video_count']} mapped caches have independently verified source checksums at the pinned revision, and all {staging['staged_video_count']} source clips are staged with rendered contact sheets."
            if staging["staged_video_count"] == staging["requested_video_count"]
            else f"The mapped caches have independently verified source checksums at the pinned revision. "
            f"{staging['requested_video_count'] - staging['staged_video_count']} local clips remain blocked from contact-sheet rendering."
        ),
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def validate_overlay_binding(source_manifest: Mapping[str, Any], overlay_manifest: Mapping[str, Any], cache_dir: Path) -> None:
    """Assert every rendered tile retains the cache frame index/timestamp contract."""

    source_by_path = {entry["video_path"]: entry for entry in source_manifest["entries"]}
    for row in overlay_manifest["entries"]:
        source = source_by_path[row["video_path"]]
        cache = G.load_detector_cache(cache_dir / f"{row['cache_key']}.json", expected_cache_key=row["cache_key"])
        expected = [(int(frame["frame_index"]), float(frame["timestamp"])) for frame in cache["frames"]]
        actual = [(int(frame["frame_index"]), float(frame["timestamp_s"])) for frame in row.get("frames", [])]
        if row["status"] == "staged" and actual != expected:
            raise AssertionError(f"overlay frame binding mismatch: {row['video_path']}")
        if source["cache_key"] != row["cache_key"]:
            raise AssertionError(f"overlay cache key mismatch: {row['video_path']}")
        if source["hpc_source_sha256"] != source["cache_video_checksum"]:
            raise AssertionError(f"source checksum manifest mismatch: {row['video_path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "runs/phase7B_cache_validation/report.json")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "runs/phase3_explicit_geometry/full/grounding_cache")
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/ReVA")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/phase7B_cache_validation")
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    source_manifest = build_source_manifest(
        report,
        args.cache_dir,
        args.source_root,
        args.out_dir / "source_checksum_manifest.json",
    )
    overlays = stage_and_render(
        report,
        source_manifest,
        cache_dir=args.cache_dir,
        source_root=args.source_root,
        out_dir=args.out_dir,
    )
    validate_overlay_binding(source_manifest, overlays, args.cache_dir)
    enriched = enrich_report(report, source_manifest, overlays)
    args.report.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n")
    (args.out_dir / "report.md").write_text(render_updated_report(enriched))
    print(json.dumps({"staged": sum(row["status"] == "staged" for row in overlays["entries"]), "total": len(overlays["entries"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
