import re
import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import train


class Qwen38TrainingConfigTests(unittest.TestCase):
    def test_qwen38_config_matches_stage_a_recipe_and_isolated_output(self):
        config = yaml.safe_load((ROOT / "code/configs/stageA_qwen38.yaml").read_text())
        stage_a = yaml.safe_load((ROOT / "code/configs/stageA.yaml").read_text())
        self.assertEqual(config["train_json"], stage_a["train_json"])
        self.assertEqual(config["seed"], stage_a["seed"])
        self.assertEqual(config["lora"], stage_a["lora"])
        self.assertEqual(config["train"], stage_a["train"])
        self.assertNotEqual(config["output_dir"], stage_a["output_dir"])
        self.assertEqual(config["model_family"], "qwen38")
        self.assertFalse(config["chat_template"]["enable_thinking"])

    def test_qwen38_target_covers_both_attention_families_and_mlp(self):
        target = train.language_target({"model_family": "qwen38"})
        for path in (
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "model.language_model.layers.0.linear_attn.in_proj_z",
            "model.language_model.layers.0.linear_attn.in_proj_b",
            "model.language_model.layers.0.linear_attn.in_proj_a",
            "model.language_model.layers.0.linear_attn.out_proj",
            "model.language_model.layers.11.self_attn.q_proj",
            "model.language_model.layers.11.self_attn.o_proj",
            "model.language_model.layers.63.mlp.down_proj",
        ):
            self.assertIsNotNone(re.fullmatch(target, path), path)
        self.assertIsNone(re.fullmatch(target, "model.visual.blocks.0.attn.qkv"))
        self.assertIsNone(
            re.fullmatch(target, "model.language_model.layers.0.linear_attn.norm")
        )

    def test_stage_a_target_remains_unchanged(self):
        target = train.language_target({})
        self.assertIsNotNone(
            re.fullmatch(target, "model.language_model.layers.0.self_attn.q_proj")
        )
        self.assertIsNone(
            re.fullmatch(target, "model.language_model.layers.0.linear_attn.in_proj_qkv")
        )

    def test_qwen38_wrapper_is_h200_scoped_and_requires_its_config(self):
        script = (ROOT / "jobs/qwen38_train.sbatch").read_text()
        self.assertIn("#SBATCH -p gpu_h200_8", script)
        self.assertIn("#SBATCH --gres=gpu:nvidia_h200_nvl:1", script)
        self.assertIn("#SBATCH -J reva_qwen38", script)
        self.assertIn('if [[ "$CFG" != *qwen38* ]]', script)
        self.assertIn('python code/smoke.py --config "$CFG"', script)
        self.assertIn('python code/train.py "$CFG"', script)

    def test_alternate_gpu_smokes_keep_recipe_and_output_isolated(self):
        base = yaml.safe_load((ROOT / "code/configs/stageA_qwen38.yaml").read_text())
        for suffix, partition, gres in (
            ("h100", "gpu_h100_4", "nvidia_h100_80gb_hbm3"),
            ("a100", "gpu_a100_8", "nvidia_a100-sxm4-80gb"),
        ):
            config = yaml.safe_load(
                (ROOT / f"code/configs/stageA_qwen38_{suffix}_smoke.yaml").read_text()
            )
            self.assertEqual(config["train_json"], base["train_json"])
            self.assertEqual(config["seed"], base["seed"])
            self.assertEqual(config["lora"], base["lora"])
            self.assertEqual(config["train"], base["train"])
            self.assertNotEqual(config["output_dir"], base["output_dir"])
            script = (ROOT / f"jobs/qwen38_smoke_{suffix}.sbatch").read_text()
            self.assertIn(f"#SBATCH -p {partition}", script)
            self.assertIn(f"#SBATCH --gres=gpu:{gres}:1", script)
            self.assertIn('python code/smoke.py --config "$CFG"', script)
            self.assertNotIn('python code/train.py', script)

    def test_h100_training_gate_uses_passed_smoke_and_same_recipe(self):
        base = yaml.safe_load((ROOT / "code/configs/stageA_qwen38.yaml").read_text())
        config = yaml.safe_load((ROOT / "code/configs/stageA_qwen38_h100.yaml").read_text())
        self.assertEqual(config["train_json"], base["train_json"])
        self.assertEqual(config["seed"], base["seed"])
        self.assertEqual(config["lora"], base["lora"])
        self.assertEqual(config["train"], base["train"])
        self.assertNotEqual(config["output_dir"], base["output_dir"])
        self.assertNotIn("smoke_marker", config)
        script = (ROOT / "jobs/qwen38_train_h100.sbatch").read_text()
        self.assertIn("#SBATCH -p gpu_h100_4", script)
        self.assertIn("#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1", script)
        self.assertIn('SMOKE_MARKER="$WORK/runs/stageA_qwen38_27b_r16_h100_smoke/SMOKE_OK"', script)
        self.assertIn('python code/train.py "$CFG"', script)


if __name__ == "__main__":
    unittest.main()
