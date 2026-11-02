"""Score Temporal Grounding with DSE-VTG similarity + gated fusion over Stage A letter logits."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import common as C  # noqa: E402
import dse_vtg as D  # noqa: E402
import reze as R  # noqa: E402
from dse_vtg_embedder import DSEEmbedder  # noqa: E402

SPLITS = {
    "valT": "data/splits/valT.json",
    "valQ": "data/splits/valQ.json",
    "val": "data/ReVA/val.json",
}


def _read_selection_file(path: str, split: str) -> set[str]:
    source = C.work_path(path)
    payload = json.loads(source.read_text())
    section = payload["splits"][split]
    values = section["ids"] if isinstance(section, dict) else section
    return set(str(v) for v in values)


def _load_baseline(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["qa_id"]] = row
    return rows


def _video_info(row: dict, work) -> tuple[float, float, int]:
    manifest = C.load_manifest(work)
    info = manifest[row["video_path"]]
    fps = float(info["fps"])
    total = int(info["n_frames"])
    duration = float(info.get("duration") or (total / fps))
    return fps, duration, total


def _query_variants(question: str) -> list[str]:
    q = question.strip()
    return [
        q,
        f"Find the time in the video that best answers: {q}",
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=sorted(SPLITS), required=True)
    ap.add_argument("--qa-ids-file", required=True)
    ap.add_argument("--baseline", required=True, help="Stage A score JSONL with letter probs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--margin-gate", type=float, default=D.DEFAULT_MARGIN_GATE)
    ap.add_argument("--alpha", type=float, default=D.DEFAULT_FUSION_ALPHA)
    ap.add_argument("--target-fps", type=float, default=D.DEFAULT_FPS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    args = ap.parse_args()

    split_path = C.work_path(SPLITS[args.split])
    gold_rows = {r["qa_id"]: r for r in json.loads(split_path.read_text())["QA"]}
    qa_ids = sorted(_read_selection_file(args.qa_ids_file, args.split))
    qa_ids = [q for i, q in enumerate(qa_ids) if i % args.nshards == args.shard]
    baseline = _load_baseline(C.work_path(args.baseline))
    missing = [q for q in qa_ids if q not in baseline]
    if missing:
        raise SystemExit(f"baseline missing {len(missing)} ids")

    out = C.work_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = C.read_completed_jsonl(out)
    stop_file = os.environ.get("REVA_STOP_FILE")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for DSE-VTG embedding")
    embedder = DSEEmbedder(device="cuda")

    manifest_path = C.score_manifest_path(out)
    expected = {
        "schema_version": 1,
        "output_kind": "dse_vtg_score",
        "split": args.split,
        "baseline": str(C.work_path(args.baseline).resolve()),
        "margin_gate": args.margin_gate,
        "alpha": args.alpha,
        "target_fps": args.target_fps,
        "embed_model": "Qwen/Qwen3-VL-Embedding-8B",
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(expected, indent=2) + "\n")

    with out.open("a") as stream:
        for qa_id in qa_ids:
            if qa_id in done:
                continue
            if stop_file and Path(stop_file).exists():
                print("stop requested")
                break
            row = gold_rows[qa_id]
            base = baseline[qa_id]
            letter_p = list(base["p"])
            fps, duration, total = _video_info(row, None)
            indices = D.sample_frame_indices(fps=fps, duration=duration, total_frames=total, target_fps=args.target_fps)
            frames = C.read_frames(row["video_path"], indices, fps=fps)
            times = [index / fps for index in indices]

            sim_curves = []
            for query in _query_variants(row["question"]):
                raw = embedder.query_frame_similarities(query, frames)
                sim_curves.append(D.normalize_similarities(raw))
            curves_msf = [D.msf_similarity(times, curve) for curve in sim_curves]
            energy = D.query_tta_energies(row, times, curves_msf, duration)
            fused_p, fuse_meta = D.gated_fuse(letter_p, energy, margin_gate=args.margin_gate, alpha=args.alpha)
            result = {
                "qa_id": qa_id,
                "p": fused_p,
                "pred": C.LETTERS[int(np.argmax(fused_p))],
                "p_letter": letter_p,
                "dse_times": times,
                "dse_sim": curves_msf[0],
                "dse_energy": energy,
                "dse_meta": fuse_meta,
            }
            stream.write(json.dumps(result, separators=(",", ":")) + "\n")
            stream.flush()
            print(f"dse-vtg {qa_id} fusion={fuse_meta['fusion_applied']} pred={result['pred']}", flush=True)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
