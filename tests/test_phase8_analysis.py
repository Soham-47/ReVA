import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import phase8_analysis as A


def _manifest(split, model_id, revision):
    return {
        "schema_version": 1,
        "output_kind": "phase8_pretrained_score",
        "model_family": "test",
        "model_id": model_id,
        "model_revision": revision,
        "split": split,
        "offsets": [0.0, 0.5],
        "rots": 2,
        "view_count": 4,
        "frame_sampling": "legacy",
        "frames_per_view": 16,
        "answer_labels": {"letters": ["A", "B", "C", "D"], "token_count": 1},
    }


def _score(qa_id, gold, *, wrong=False):
    pred = "ABCD"[("ABCD".index(gold) + 1) % 4] if wrong else gold
    p = [0.0] * 4
    p["ABCD".index(pred)] = 1.0
    views = []
    for offset, perm in A.EXPECTED_VIEWS:
        views.append({"offset": offset, "perm": perm, "p": p})
    return {"qa_id": qa_id, "p": p, "pred": pred, "views": views}


class Phase8AnalysisTests(unittest.TestCase):
    def test_paired_states_and_deltas_are_preserved(self):
        reference = {
            "q1": {"video": "v1", "task": "t", "category": "c", "gold": "A", "pred": "B", "ok": False, "confidence": 0.5, "margin": 0.1},
            "q2": {"video": "v1", "task": "t", "category": "c", "gold": "A", "pred": "A", "ok": True, "confidence": 0.8, "margin": 0.6},
        }
        candidate = {
            "q1": {**reference["q1"], "pred": "A", "ok": True, "confidence": 0.7, "margin": 0.4},
            "q2": {**reference["q2"], "confidence": 0.6, "margin": 0.2},
        }
        records = A._paired_records(candidate, reference)
        self.assertEqual([row["state"] for row in records], ["fix", "both_correct"])
        self.assertAlmostEqual(records[0]["confidence_delta"], 0.2)
        self.assertAlmostEqual(records[1]["margin_delta"], -0.4)

    def test_video_bootstrap_is_deterministic(self):
        reference = {
            "q1": {"video": "v1", "ok": False},
            "q2": {"video": "v2", "ok": True},
        }
        candidate = {
            "q1": {"video": "v1", "ok": True},
            "q2": {"video": "v2", "ok": True},
        }
        first = A._video_bootstrap(candidate, reference, seed=17, reps=100)
        second = A._video_bootstrap(candidate, reference, seed=17, reps=100)
        self.assertEqual(first, second)
        self.assertEqual(first["videos"], 2)
        self.assertEqual(first["estimate"], 0.5)

    def test_analysis_accepts_distinct_arm_revisions_and_writes_pairs(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            a0_dir = temp_path / "a0"
            candidate_dir = temp_path / "candidate"
            a0_dir.mkdir()
            candidate_dir.mkdir()
            for split in A.SPLITS:
                gold = json.loads((ROOT / "data" / "splits" / f"{split}.json").read_text())["QA"]
                a0_rows = [_score(row["qa_id"], row["correct_answer"]) for row in gold]
                candidate_rows = [
                    _score(row["qa_id"], row["correct_answer"], wrong=index == 0)
                    for index, row in enumerate(gold)
                ]
                for directory, name, rows, model, revision in (
                    (a0_dir, "a0", a0_rows, "A0/model", "a0-rev"),
                    (candidate_dir, "candidate", candidate_rows, "candidate/model", "candidate-rev"),
                ):
                    path = directory / f"scores_{name}_{split}.jsonl"
                    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
                    (Path(str(path) + ".manifest.json")).write_text(
                        json.dumps(_manifest(split, model, revision)) + "\n"
                    )
            paths = A.analyze(
                a0=str(a0_dir),
                candidates={"candidate": str(candidate_dir)},
                output_prefix=temp_path / "analysis",
                work=ROOT,
                bootstrap_reps=20,
                bootstrap_seed=5,
            )
            summary = json.loads(paths[0].read_text())
            self.assertEqual(summary["metrics"]["valQ"]["A0"]["accuracy"], 1.0)
            self.assertEqual(summary["paired"]["valQ"]["candidate"]["regression"], 1)
            self.assertEqual(
                summary["provenance"]["valQ"]["candidate"]["model_revision"],
                "candidate-rev",
            )
            self.assertEqual(len(summary["paired_records"]["valT"]["candidate"]), 1059)
            svg = temp_path / "analysis.svg"
            self.assertTrue(svg.exists())
            svg_text = svg.read_text()
            self.assertIn("data-n-a0=\"", svg_text)
            self.assertIn("data-n-candidate=\"", svg_text)
            self.assertIn("valQ", svg_text)
            self.assertIn("valT", svg_text)
            self.assertIn("task_a0_n", paths[2].read_text().splitlines()[0])

    def test_analysis_refuses_incomplete_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            score = temp_path / "scores.jsonl"
            gold = json.loads((ROOT / "data" / "splits" / "valQ.json").read_text())["QA"]
            score.write_text(json.dumps(_score(gold[0]["qa_id"], gold[0]["correct_answer"])) + "\n")
            Path(str(score) + ".manifest.json").write_text(json.dumps(_manifest("valQ", "m", "r")))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                A._load_arm(score, {row["qa_id"]: row for row in gold}, "valQ")


if __name__ == "__main__":
    unittest.main()
