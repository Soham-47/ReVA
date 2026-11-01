"""CPU tests for REZE option parsing and logit fusion."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import reze as R


class RezeTests(unittest.TestCase):
    def test_parse_point_and_range(self):
        point = R.parse_option("At 2.5 seconds", duration=8.0, duration_question=False)
        self.assertEqual(point, {"kind": "point", "time": 2.5})
        span = R.parse_option("from 1.0 to 3.5s", duration=8.0, duration_question=False)
        self.assertEqual(span["kind"], "range")
        self.assertEqual(span["start"], 1.0)
        self.assertEqual(span["end"], 3.5)

    def test_qualitative_and_duration(self):
        start = R.parse_option("At the beginning of the video", duration=10.0, duration_question=False)
        self.assertEqual(start["kind"], "range")
        self.assertEqual(start["end"], 2.5)
        dur = R.parse_option("4.0 seconds", duration=10.0, duration_question=True)
        self.assertEqual(dur, {"kind": "duration", "seconds": 4.0})

    def test_point_energy_prefers_matching_peak(self):
        times = [0.0, 1.0, 2.0, 3.0]
        yes = [0.1, 0.2, 0.9, 0.2]
        row = {
            "question": "At what point does the truck appear?",
            "options": {"A": "At 0.0 seconds", "B": "At 2.0 seconds", "C": "At 3.0 seconds", "D": "At 1.0 seconds"},
        }
        energy = R.option_energy(row, times, yes, 4.0)
        self.assertEqual(energy.index(max(energy)), 1)

    def test_fuse_alpha_zero_keeps_letter_argmax(self):
        fused = R.fuse_probs([0.1, 0.6, 0.2, 0.1], [9.0, 0.0, 0.0, 0.0], 0.0)
        self.assertEqual(int(np.argmax(fused)), 1)

    def test_frame_times(self):
        meta = SimpleNamespace(fps=2.0, frames_indices=[0, 2, 4], duration=3.0)
        self.assertEqual(R.frame_times(meta), [0.0, 1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
