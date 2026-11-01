import unittest
import tempfile
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import charades_grounding as G


class CharadesGroundingTests(unittest.TestCase):
    def test_interval_parser_is_strict_and_does_not_clip(self):
        self.assertEqual(G.parse_interval("start=1.5 end=3", 5), (1.5, 3.0))
        self.assertIsNone(G.parse_interval("start=1 end=6", 5))
        self.assertIsNone(G.parse_interval("start=1 end=3 rationale", 5))
        self.assertIsNone(G.parse_interval("start=nan end=3", 5))
        self.assertEqual(G.interval_reward("start=1 end=3", (1.0, 3.0), 5), 1.0)
        self.assertEqual(G.interval_reward("start=1 end=6", (1.0, 3.0), 5), 0.0)

    def test_canonical_rows_reject_invalid_bounds(self):
        durations = {"video_1": 10.0}
        good, reason = G.canonical_row(
            {"video_id": "video_1", "sentence": "open door", "start": 1, "end": 4},
            0,
            durations,
        )
        self.assertIsNone(reason)
        self.assertEqual(good["duration"], 10.0)
        bad, reason = G.canonical_row(
            {"video_id": "video_1", "sentence": "open door", "start": 1, "end": 11},
            1,
            durations,
        )
        self.assertIsNone(bad)
        self.assertEqual(reason, "interval outside duration")

    def test_released_annotation_format_is_parsed_and_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "charades_sta_train.txt"
            path.write_text(
                "video_1 1.25 4.50##a person is closing a door\n"
                "video_1 9 11##media validation will check this label\n"
                "malformed annotation\n"
            )
            rows, rejected = G.load_charades_sta_rows(path, {"video_1": 10.0}, "train")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["video_id"], "video_1")
        self.assertEqual(rows[0]["query"], "a person is closing a door")
        self.assertEqual((rows[0]["start"], rows[0]["end"]), (1.25, 4.5))
        self.assertEqual(rows[1]["duration"], 11.0)
        self.assertEqual(rows[1]["duration_metadata"], 10.0)
        self.assertEqual(len(rejected), 1)

    def test_video_split_is_disjoint_and_deterministic(self):
        rows = [
            {"qa_id": f"q{i}", "video_id": f"v{i // 2}", "start": 0.0, "end": 1.0, "duration": 2.0}
            for i in range(20)
        ]
        train_a, dev_a = G.split_by_video(rows, sample_rows=8, seed=7)
        train_b, dev_b = G.split_by_video(rows, sample_rows=8, seed=7)
        self.assertEqual(train_a, train_b)
        self.assertEqual(dev_a, dev_b)
        self.assertTrue({r["video_id"] for r in train_a}.isdisjoint({r["video_id"] for r in dev_a}))
        self.assertLessEqual(len(train_a), 8)

    def test_continuation_arms_share_the_same_prompt_schedule(self):
        rows = [{"qa_id": f"q{i}"} for i in range(12)]
        sft = G._scheduled_rows(rows, steps=8, seed=3)
        grpo = G._scheduled_rows(rows, steps=8, seed=3)
        self.assertEqual(sft, grpo)
        self.assertEqual(len(sft), 8)

    def test_output_guard_keeps_weights_out_of_checkout(self):
        with self.assertRaises(ValueError):
            G._safe_output(Path(__file__).resolve().parents[1] / "code" / "charades")


if __name__ == "__main__":
    unittest.main()
