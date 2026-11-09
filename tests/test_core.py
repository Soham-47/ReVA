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
        self.assertEqual(len(bundle["experiments"]), 39)
        numbers = [row["exp_number"] for row in bundle["experiments"]]
        self.assertEqual(numbers[0], "001")
        self.assertEqual(numbers[-1], "039")
        self.assertTrue((ROOT / "experiments" / "EXP-001" / "experiment.md").is_file())
        self.assertTrue((ROOT / "experiments" / "EXP-039" / "experiment.md").is_file())

    def test_stage_a_recipe_is_pinned(self):
        config = yaml.safe_load((ROOT / "code" / "configs" / "stageA.yaml").read_text())
        self.assertEqual(config["lora"], {"r": 16, "alpha": 32, "dropout": 0.05})
        self.assertEqual(config["attn_implementation"], "sdpa")
        self.assertEqual(C.MAX_FRAMES, 16)



class FastConvBackendTest(unittest.TestCase):
    def _torch(self, enabled):
        from types import SimpleNamespace
        return SimpleNamespace(backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=enabled)))

    def test_rejects_slow_conv_env(self):
        for name in C.SLOW_CONV_ENV:
            with self.assertRaises(RuntimeError):
                C.check_fast_conv_backend({name: "1"}, self._torch(True))

    def test_rejects_disabled_cudnn(self):
        with self.assertRaises(RuntimeError):
            C.check_fast_conv_backend({}, self._torch(False))

    def test_accepts_default(self):
        C.check_fast_conv_backend({}, self._torch(True))

if __name__ == "__main__":
    unittest.main()
