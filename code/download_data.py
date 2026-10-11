"""Download ReVA JSONs and only the videos referenced by those JSONs."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import common as C

REPO = C.DATASET_REPO
REVISION = C.DATASET_REVISION
SPLITS = ("train", "val", "test")


def unreferenced_dji_files(root: Path, referenced: set[str]) -> list[Path]:
    """Return DJI MP4 files under *root* that are not referenced by the JSONs.

    Referenced Hawk_UAV clips can themselves use DJI filenames, so the guard
    must compare relative paths rather than rejecting every DJI-looking file.
    """

    referenced_paths = {Path(path).as_posix() for path in referenced}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.name.upper().startswith("DJI_")
        and path.suffix.lower() == ".mp4"
        and path.relative_to(root).as_posix() not in referenced_paths
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=os.environ.get("WORK"), help="runtime tree (defaults to $WORK)")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--expected-clips", type=int, default=1047)
    args = ap.parse_args()
    if not args.work:
        ap.error("--work or WORK is required")

    from huggingface_hub import hf_hub_download, snapshot_download

    root = Path(args.work).expanduser().resolve() / "data" / "ReVA"
    root.mkdir(parents=True, exist_ok=True)
    common = {"repo_id": REPO, "repo_type": "dataset", "local_dir": str(root)}
    for split in SPLITS:
        kwargs = dict(common, filename=f"{split}.json")
        kwargs["revision"] = args.revision
        hf_hub_download(**kwargs)

    paths: set[str] = set()
    for split in SPLITS:
        with (root / f"{split}.json").open() as f:
            rows = json.load(f)["QA"]
        paths.update(row["video_path"] for row in rows)
    paths = {p.replace("\\", "/") for p in paths}
    if len(paths) != args.expected_clips:
        raise RuntimeError(f"expected {args.expected_clips} referenced clips, found {len(paths)}")
    if any(Path(p).is_absolute() or ".." in Path(p).parts for p in paths):
        raise RuntimeError("dataset contains an unsafe video_path")

    kwargs = dict(common, allow_patterns=sorted(paths), ignore_patterns=["**/DJI_*.MP4", "DJI_*.MP4"], max_workers=8)
    kwargs["revision"] = args.revision
    snapshot_download(**kwargs)
    missing = [path for path in sorted(paths) if not (root / path).is_file()]
    raw = unreferenced_dji_files(root, paths)
    print(f"clips: {len(paths)} missing: {len(missing)} unreferenced_raw_dji_files: {len(raw)}")
    if missing:
        raise RuntimeError(f"missing referenced clips: {missing[:5]}")
    if raw:
        raise RuntimeError(f"unreferenced raw DJI originals were downloaded: {raw[:5]}")
    revision_record = root.parent / "dataset_revision.json"
    with tempfile.NamedTemporaryFile("w", dir=revision_record.parent, delete=False) as tmp:
        json.dump({"repo": REPO, "revision": args.revision, "referenced_clips": len(paths)}, tmp, indent=2)
        tmp.write("\n")
        temporary = Path(tmp.name)
    temporary.replace(revision_record)
    print(f"dataset revision: {args.revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
