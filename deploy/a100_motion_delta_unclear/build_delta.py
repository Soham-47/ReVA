"""Build the unclear-hint delta against the installed ade2841 overlay hashes."""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEST = HERE / "overlay"
CHANGED = [
    "code/specialist_data.py",
    "code/train.py",
    "code/score.py",
    "code/route_specialist.py",
    "docs/a100_specialists_runbook.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def installed_hashes() -> dict[str, str]:
    text = subprocess.check_output(
        ["git", "show", "HEAD:deploy/a100_motion_overlay/MANIFEST_AFTER.sha256"],
        cwd=ROOT,
        text=True,
    )
    found = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        found[rel] = digest
    return found


def main() -> None:
    previous = installed_hashes()
    missing = [rel for rel in CHANGED if rel not in previous]
    if missing:
        raise SystemExit(f"ade2841 manifest has no hash for {missing}")
    if DEST.exists():
        shutil.rmtree(DEST)
    after = []
    for rel in CHANGED:
        source = ROOT / rel
        target = DEST / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        after.append((sha256(target), rel))
        if sha256(target) == previous[rel] and rel != "docs/a100_specialists_runbook.md":
            raise SystemExit(f"{rel} did not change relative to the installed overlay")
    (HERE / "MANIFEST_BEFORE.sha256").write_text(
        "".join(f"{previous[rel]}  {rel}\n" for rel in CHANGED)
    )
    (HERE / "MANIFEST_AFTER.sha256").write_text(
        "".join(f"{digest}  {rel}\n" for digest, rel in after)
    )
    (HERE / "NEW_FILES.txt").write_text("")
    base = HERE.parent / "a100_motion_overlay"
    for name in ("deploy.sh", "rollback.sh"):
        text = (base / name).read_text().replace(
            "deploy/a100_motion_overlay/deploy.sh",
            "deploy/a100_motion_delta_unclear/deploy.sh",
        )
        (HERE / name).write_text(text)
        (HERE / name).chmod(0o755)
    preflight = (base / "preflight_motion.sh").read_text()
    (HERE / "preflight_motion.sh").write_text(preflight)
    (HERE / "preflight_motion.sh").chmod(0o755)
    print(f"delta files={len(after)}")


if __name__ == "__main__":
    main()
