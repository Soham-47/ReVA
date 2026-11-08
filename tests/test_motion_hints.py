"""Motion-hint cache must match between training dataset and scorer encode paths."""

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT / "archive" / "implementation" / "code"))

import common as C


class MotionHintParityTests(unittest.TestCase):
    def test_train_and_score_use_identical_temporal_context(self):
        video_path = "dummy/flight.mp4"
        hint = "Camera motion: 0.0s–4.0s panning left, slow"
        cache = {video_path: hint}
        row = {
            "qa_id": "train_000001",
            "video_path": video_path,
            "category": "X",
            "task": "Temporal Grounding",
            "question": "When?",
            "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "correct_answer": "A",
        }

        train_context = C.motion_hint_text(cache, row["video_path"])
        score_context = C.motion_hint_text(cache, row["video_path"])
        self.assertIsNotNone(train_context)
        self.assertEqual(train_context, score_context)
        self.assertEqual(train_context.encode("utf-8"), score_context.encode("utf-8"))

        captured: list[str | None] = []

        def fake_encode(*_args, temporal_context=None, **_kwargs):
            captured.append(temporal_context)
            return {"input_ids": [0, 1, 2, 3]}, list(C.LETTERS)

        with patch.object(C, "encode", side_effect=fake_encode):
            C.encode(None, row, None, None, C.LETTERS, temporal_context=train_context)
            C.encode(None, row, None, None, C.LETTERS, temporal_context=score_context)

        self.assertEqual(len(captured), 2)
        self.assertEqual(captured[0], hint)
        self.assertEqual(captured[1], hint)


if __name__ == "__main__":
    unittest.main()
