"""Round-1 B1/B2 config pins."""

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common as C


class Round1ConfigTests(unittest.TestCase):
    def test_b1_hint16_matches_stage_a_except_motion(self):
        base = yaml.safe_load((ROOT / "code/configs/stageA.yaml").read_text())
        b1 = yaml.safe_load((ROOT / "code/configs/b1_hint16.yaml").read_text())
        self.assertEqual(b1["motion_hints"], "data/camera_motion_hints.json")
        self.assertNotIn("video", b1)
        self.assertNotIn("init_adapter", b1)
        self.assertEqual(b1["seed"], 42)
        self.assertEqual(b1["attn_implementation"], "sdpa")
        self.assertEqual(b1["train"]["dataloader_workers"], 0)
        self.assertEqual(b1["lora"], base["lora"])
        self.assertEqual(b1["train"]["epochs"], 2)
        self.assertEqual(b1["train"]["grad_accum"], 16)

    def test_b2_frames32_video_cap(self):
        b2 = yaml.safe_load((ROOT / "code/configs/b2_frames32.yaml").read_text())
        self.assertEqual(b2["video"]["frame_budget"], 32)
        self.assertEqual(b2["video"]["video_max_pixels"], C.DEFAULT_VIDEO_MAX_PIXELS)
        self.assertNotIn("motion_hints", b2)
        self.assertEqual(b2["attn_implementation"], "sdpa")
        self.assertEqual(b2["train"]["dataloader_workers"], 0)
        size = C.video_size_for(32, C.DEFAULT_VIDEO_MAX_PIXELS)
        self.assertEqual(size["longest_edge"], 32 * C.DEFAULT_VIDEO_MAX_PIXELS)

    def test_sample_index_list_honors_32_frame_budget(self):
        indices = C.sample_index_list(400, 30.0, 0.0, frame_budget=32)
        self.assertEqual(len(indices), 32)


if __name__ == "__main__":
    unittest.main()
