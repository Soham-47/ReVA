"""Tune REZE fusion weight on valQ, then score valT with the frozen alpha."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import reze as R
from analyze_visual_prompt_pilot import _load_ids, _load_scores


ALPHAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def _accuracy(ids, gold, preds) -> tuple[int, int, float]:
    ok = sum(int(preds[qa_id] == gold[qa_id]["correct_answer"]) for qa_id in ids)
    return ok, len(ids), ok / len(ids)


def _preds_from_reze(rows: dict[str, dict], alpha: float) -> dict[str, str]:
    preds = {}
    for qa_id, row in rows.items():
        letter = row.get("p_letter") or row["p"]
        energy = row["reze_energy"]
        fused = R.fuse_probs(letter, energy, alpha)
        preds[qa_id] = C.LETTERS[int(max(range(4), key=lambda i: fused[i]))]
    return preds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    parser.add_argument("--baseline-valQ", required=True)
    parser.add_argument("--baseline-valT", required=True)
    parser.add_argument("--reze-valQ", required=True)
    parser.add_argument("--reze-valT", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    ids_path = C.work_path(args.ids)
    report: dict = {"alphas": list(ALPHAS), "splits": {}}
    gold = {}
    reze_rows = {}
    baseline_rows = {}
    id_lists = {}
    for split, baseline_path, reze_path in (
        ("valQ", args.baseline_valQ, args.reze_valQ),
        ("valT", args.baseline_valT, args.reze_valT),
    ):
        ids = _load_ids(ids_path, split)
        id_lists[split] = ids
        gold[split] = {
            row["qa_id"]: row
            for row in json.loads(C.work_path(f"data/splits/{split}.json").read_text())["QA"]
        }
        baseline_rows[split] = _load_scores(C.work_path(baseline_path))
        reze_rows[split] = _load_scores(C.work_path(reze_path))
        missing = [qa_id for qa_id in ids if qa_id not in reze_rows[split] or qa_id not in baseline_rows[split]]
        if missing:
            raise SystemExit(f"{split} missing {len(missing)} ids")

    best_alpha = ALPHAS[0]
    best_correct = -1
    valq_grid = []
    for alpha in ALPHAS:
        preds = _preds_from_reze(reze_rows["valQ"], alpha)
        ok, n, acc = _accuracy(id_lists["valQ"], gold["valQ"], preds)
        valq_grid.append({"alpha": alpha, "correct": ok, "n": n, "accuracy": acc})
        if ok > best_correct:
            best_correct = ok
            best_alpha = alpha
    report["chosen_alpha"] = best_alpha
    report["valQ_grid"] = valq_grid

    for split in ("valQ", "valT"):
        ids = id_lists[split]
        base_preds = {qa_id: baseline_rows[split][qa_id]["pred"] for qa_id in ids}
        arm_preds = _preds_from_reze(reze_rows[split], best_alpha)
        letter_preds = {
            qa_id: C.LETTERS[int(max(range(4), key=lambda i: (reze_rows[split][qa_id].get("p_letter") or reze_rows[split][qa_id]["p"])[i]))]
            for qa_id in ids
        }
        b_ok, n, b_acc = _accuracy(ids, gold[split], base_preds)
        a_ok, _, a_acc = _accuracy(ids, gold[split], arm_preds)
        l_ok, _, l_acc = _accuracy(ids, gold[split], letter_preds)
        report["splits"][split] = {
            "n": n,
            "baseline": {"correct": b_ok, "accuracy": b_acc},
            "letter_only": {"correct": l_ok, "accuracy": l_acc},
            "reze": {"correct": a_ok, "accuracy": a_acc, "alpha": best_alpha},
            "net_delta": a_ok - b_ok,
        }

    out = C.work_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
