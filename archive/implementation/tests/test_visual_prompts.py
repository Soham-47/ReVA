"""CPU tests for progress-bar and camera-motion overlays."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import visual_prompts as V


class VisualPromptTests(unittest.TestCase):
    def test_progress_bar_appends_height_and_fills_with_time(self):
        frames = np.zeros((2, 16, 64, 3), dtype=np.uint8)
        early = V.draw_progress_bar(frames, [0.0, 1.0], 4.0, bar_height=16)
        late = V.draw_progress_bar(frames, [3.5, 4.0], 4.0, bar_height=16)
        self.assertEqual(early.shape, (2, 32, 64, 3))
        self.assertGreater(int(late[1, 22, 50].sum()), int(early[0, 22, 50].sum()))

    def test_classify_and_merge_motion_vocabulary(self):
        self.assertEqual(V.classify_pairwise_motion(0.0, 0.0, 1.0, 100), "hovering stationary")
        self.assertEqual(V.classify_pairwise_motion(3.0, 0.0, 1.0, 100), "panning left, slow")
        self.assertEqual(V.classify_pairwise_motion(0.0, 0.0, 1.12, 100), "moving forward, rapid")
        hint = V.merge_motion_segments([0.0, 2.0, 4.0], ["panning left, slow", "panning left, slow"], 5.0)
        self.assertEqual(hint, "Camera motion: 0.0s–5.0s panning left, slow")

    def test_camera_motion_hint_without_opencv_is_unavailable(self):
        frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
        meta = SimpleNamespace(fps=2.0, frames_indices=[0, 1, 2], duration=1.5, total_num_frames=3)
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.assertEqual(V.camera_motion_hint(frames, meta), V.MOTION_UNAVAILABLE)
            return
        hint = V.camera_motion_hint(frames, meta)
        self.assertTrue(hint.startswith("Camera motion:"))
        self.assertNotEqual(hint, "")


if __name__ == "__main__":
    unittest.main()
