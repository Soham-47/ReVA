"""Copy the reconciled motion files and write the before/after SHA256 manifests."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = Path(__file__).resolve().parent / "overlay"
BEFORE = {
    "code/common.py": "a281763626e4f6095213a071d7c75336b8929119e4d51086770534666ed1a187",
    "code/train.py": "74a4be3bb54a4e7ffdaf8e20774803d1882454372a02b7a96e7e08fab293de94",
    "code/score.py": "bca1fcccd0f54f54966c717e0aa71020149906cc5413376724f7a88bbabf826a",
    "code/configs/b1_hint16.yaml": "bd9949272ca3074254817d7c885bb35e7982ebd78eb9a6914a687e72c3629ac1",
    "code/configs/smoke_b1_hint16.yaml": "33e180c47d35b4d3b9f289440d375bd693682bfc2fc855fa90628d6524c4ce7e",
    "code/configs/b2_frames32.yaml": "a6a7f7c3e7b6af915633b224441b5198dde4ce32524c55a4561dd29d48b9cc6f",
    "code/configs/smoke_b2_frames32.yaml": "fa336a09f78c5339ee772fced818b21946ca246a5f3abc9d6ab7274d51d38d26",
}
NEW = [
    "code/specialist_data.py",
    "code/mirror_answers.py",
    "code/route_specialist.py",
    "jobs/a100_prepare.sh",
    "jobs/a100_train_smoke.sbatch",
    "jobs/a100_train_specialist.sbatch",
    "jobs/a100_score_specialist.sbatch",
    "jobs/a100_route_specialist.sh",
    "docs/a100_specialists_runbook.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if DEST.exists():
        shutil.rmtree(DEST)
    after: list[tuple[str, str]] = []
    for rel in list(BEFORE) + NEW:
        source = ROOT / rel
        if not source.is_file():
            raise SystemExit(f"missing source {source}")
        text = source.read_text(errors="replace")
        if rel.endswith((".yaml", ".yml")) and "b1_q3vl8b_hint16" in text:
            raise SystemExit(f"{rel} still writes the preserved run directory")
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        after.append((sha256(target), rel))
    parent = DEST.parent
    (parent / "MANIFEST_BEFORE.sha256").write_text(
        "".join(f"{digest}  {rel}\n" for rel, digest in BEFORE.items())
    )
    (parent / "MANIFEST_AFTER.sha256").write_text("".join(f"{digest}  {rel}\n" for digest, rel in after))
    (parent / "NEW_FILES.txt").write_text("".join(f"{rel}\n" for rel in NEW))
    print(f"wrote {len(after)} overlay files")


if __name__ == "__main__":
    main()
