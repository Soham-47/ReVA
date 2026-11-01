"""CPU tests for the CROSS-style spatial contract."""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import spatial_contract as S


class SpatialContractTests(unittest.TestCase):
    def test_contract_names_image_plane_and_camera_motion(self):
        text = S.SPATIAL_CONTRACT.casefold()
        self.assertIn("image-plane", text)
        self.assertIn("camera motion", text)
        self.assertNotIn("bounding box", text)
        self.assertLess(len(S.SPATIAL_CONTRACT), 800)


if __name__ == "__main__":
    unittest.main()
