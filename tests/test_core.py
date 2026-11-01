"""Lightweight tests for the active core pipeline."""

import json
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common as C


class CoreTests(unittest.TestCase):
    def test_registry_and_experiments_json_exist(self):
        self.assertTrue((ROOT / "results" / "registry.csv").is_file())
        bundle = json.loads((ROOT / "results" / "experiments.json").read_text())
        self.assertEqual(len(bundle["experiments"]), 36)
        numbers = [row["exp_number"] for row in bundle["experiments"]]
        self.assertEqual(numbers[0], "001")
        self.assertEqual(numbers[-1], "036")
        self.assertTrue((ROOT / "experiments" / "EXP-001" / "experiment.md").is_file())
        self.assertTrue((ROOT / "experiments" / "EXP-036" / "experiment.md").is_file())

    def test_stage_a_recipe_is_pinned(self):
        config = yaml.safe_load((ROOT / "code" / "configs" / "stageA.yaml").read_text())
        self.assertEqual(config["lora"], {"r": 16, "alpha": 32, "dropout": 0.05})
        self.assertEqual(config["attn_implementation"], "sdpa")
        self.assertEqual(C.MAX_FRAMES, 16)


if __name__ == "__main__":
    unittest.main()
