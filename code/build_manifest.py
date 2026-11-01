"""Validate split JSONs and decode every referenced clip into a manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 15773, "val": 2000, "test": 4000}


def read_and_check(root: Path, strict_counts: bool) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
    all_ids: set[str] = set()
    for split in SPLITS:
        path = root / f"{split}.json"
        with path.open() as f:
            payload = json.load(f)
        rows = payload.get("QA")
        if not isinstance(rows, list):
            raise ValueError(f"{path} has no QA list")
        if strict_counts and len(rows) != EXPECTED_COUNTS[split]:
            raise ValueError(f"{split}: expected {EXPECTED_COUNTS[split]} rows, found {len(rows)}")
        ids = [row.get("qa_id") for row in rows]
        if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{split}: QA ids must be non-empty and unique")
        overlap = all_ids.intersection(ids)
        if overlap:
            raise ValueError(f"QA ids overlap splits: {sorted(overlap)[:3]}")
        all_ids.update(ids)
        for row in rows:
            if sorted(row.get("options", {})) != list("ABCD"):
                raise ValueError(f"{split}/{row.get('qa_id')}: options must be A-D")
            video_path = Path(row.get("video_path", ""))
            if video_path.is_absolute() or ".." in video_path.parts:
                raise ValueError(f"unsafe video_path in {split}/{row.get('qa_id')}")
            answer = row.get("correct_answer")
            if split == "test":
                if answer not in ("", None):
                    raise ValueError(f"test/{row.get('qa_id')}: test labels must be blank")
            elif answer not in set("ABCD"):
                raise ValueError(f"{split}/{row.get('qa_id')}: invalid correct_answer {answer!r}")
        splits[split] = rows
    return splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("WORK"), help="runtime tree (defaults to $WORK)")
    ap.add_argument("--allow-count-drift", action="store_true", help="check schema without fixed released counts")
    args = ap.parse_args()
    if not args.work:
        ap.error("--work or WORK is required")
    work = Path(args.work).expanduser().resolve()
    root = work / "data" / "ReVA"
    revision_path = work / "data" / "dataset_revision.json"
    if not revision_path.exists():
        raise RuntimeError("dataset_revision.json is missing; run download_data.py so the HF commit is recorded")
    with revision_path.open() as f:
        revision = json.load(f).get("revision", "unknown")
    if not revision or revision == "unknown":
        raise RuntimeError("dataset revision record is invalid")
    splits = read_and_check(root, strict_counts=not args.allow_count_drift)

    import av

    paths = sorted({row["video_path"] for rows in splits.values() for row in rows})
    manifest: dict[str, dict] = {}
    bad: list[tuple[str, str]] = []
    for relative in paths:
        path = root / relative
        try:
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate or stream.base_rate or 30.0)
                n_frames = sum(1 for _ in container.decode(stream))
                width = int(stream.width or stream.codec_context.width)
                height = int(stream.height or stream.codec_context.height)
                if n_frames < 1 or width < 1 or height < 1:
                    raise ValueError(f"invalid dimensions/frame count ({n_frames}, {width}x{height})")
                manifest[relative] = {
                    "n_frames": n_frames,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration": n_frames / fps,
                }
        except Exception as exc:  # report every file before failing
            bad.append((relative, repr(exc)))
    print(f"dataset revision: {revision} clips: {len(manifest)} bad: {len(bad)}")
    if bad:
        for item in bad[:20]:
            print("BAD", item[0], item[1])
        raise RuntimeError("manifest failed; re-download the bad clips before training")
    destination = work / "data" / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as tmp:
        json.dump(manifest, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        temporary = Path(tmp.name)
    temporary.replace(destination)
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
