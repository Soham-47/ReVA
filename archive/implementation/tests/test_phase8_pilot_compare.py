import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import phase8_pilot_compare as P


def _score(row, *, wrong=False):
    gold = row["correct_answer"]
    pred = "ABCD"[("ABCD".index(gold) + 1) % 4] if wrong else gold
    p = [0.0] * 4
    p["ABCD".index(pred)] = 1.0
    return {
        "qa_id": row["qa_id"],
        "p": p,
        "pred": pred,
        "views": [
            {"offset": 0.0, "perm": "ABCD", "p": p},
            {"offset": 0.0, "perm": "BCDA", "p": p},
            {"offset": 0.5, "perm": "ABCD", "p": p},
            {"offset": 0.5, "perm": "BCDA", "p": p},
        ],
    }


class Phase8PilotCompareTests(unittest.TestCase):
    def test_compare_accepts_legacy_stage_a_manifest_and_writes_110_pairs(self):
        selection = json.loads((ROOT / "runs/phase8_backbone_comparison/pilot_selection.json").read_text())
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            paths = {arm: temp_path / f"{arm}_{{split}}.jsonl" for arm in P.ARMS}
            for split in P.A.SPLITS:
                rows = json.loads((ROOT / "data/splits" / f"{split}.json").read_text())["QA"]
                selected = set(selection["splits"][split]["ids"])
                selected_rows = [row for row in rows if row["qa_id"] in selected]
                for arm, path in paths.items():
                    score_path = Path(str(path).replace("{split}", split))
                    score_path.write_text(
                        "\n".join(
                            json.dumps(_score(row, wrong=arm == "fine_tuned" and index == 0))
                            for index, row in enumerate(selected_rows)
                        )
                        + "\n"
                    )
                    manifest = {
                        "schema_version": 1,
                        "output_kind": "phase8_pretrained_score",
                        "model_id": arm,
                        "model_revision": "rev",
                        "split": split,
                        "offsets": [0.0, 0.5],
                        "rots": 2,
                        "view_count": 4,
                        "frames_per_view": 16,
                        "answer_labels": {"letters": ["A", "B", "C", "D"], "token_count": 1},
                    }
                    if arm != "stageA":
                        manifest["frame_sampling"] = "legacy"
                    Path(str(score_path) + ".manifest.json").write_text(json.dumps(manifest))
            output = P.compare(
                work=ROOT,
                selection_path=ROOT / "runs/phase8_backbone_comparison/pilot_selection.json",
                stage_a=str(paths["stageA"]),
                zero_shot=str(paths["zero_shot"]),
                fine_tuned=str(paths["fine_tuned"]),
                output_prefix=temp_path / "comparison",
                bootstrap_reps=20,
                bootstrap_seed=7,
            )
            summary = json.loads(output[0].read_text())
            self.assertEqual(summary["selection"]["total_questions"], 110)
            self.assertEqual(summary["metrics"]["valQ"]["fine_tuned"]["n"], 55)
            self.assertEqual(summary["paired"]["valT"]["stageA"]["regression"], 1)
            self.assertEqual(len(output[2].read_text().splitlines()), 221)


if __name__ == "__main__":
    unittest.main()
