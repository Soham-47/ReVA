"""CPU checks for A100 specialist contracts, hint audits, and the routing gate."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common as C
import mirror_answers as M
import route_specialist as R
import specialist_data as S


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def row(qa_id: str, task: str, video: str, answer: str = "A", question: str = "What happens?") -> dict:
    return {
        "qa_id": qa_id,
        "task": task,
        "video_path": video,
        "question": question,
        "options": {"A": "left", "B": "right", "C": "neither", "D": "both"},
        "correct_answer": answer,
        "dataset_name": "VisDrone",
    }


def split(name: str, rows: list[dict]) -> dict:
    return {"metadata": {"split": name, "seed": 0}, "QA": rows}


class SpecialistContractTests(unittest.TestCase):
    def test_committed_configs_match_their_contracts(self):
        for name in (
            "smoke_b1_hint16.yaml",
            "smoke_b2_frames32.yaml",
            "b1_hint16.yaml",
            "b2_frames32.yaml",
            "b1_hint16_mirror.yaml",
        ):
            config = yaml.safe_load((ROOT / "code" / "configs" / name).read_text())
            self.assertIn("UNRESOLVED", config["output_dir"])
            S.validate_run_config(config, allow_unresolved=True)
            with self.assertRaises(ValueError):
                S.validate_run_config(config, allow_unresolved=False)

    def test_contract_rejects_full_train_and_new_architecture(self):
        config = yaml.safe_load((ROOT / "code" / "configs" / "b2_frames32.yaml").read_text())
        config["train_json"] = "data/splits/full_train.json"
        with self.assertRaises(ValueError):
            S.validate_run_config(config, allow_unresolved=True)
        config = yaml.safe_load((ROOT / "code" / "configs" / "b2_frames32.yaml").read_text())
        config["temporal_stream"] = {"enabled": True}
        with self.assertRaises(ValueError):
            S.validate_run_config(config, allow_unresolved=True)
        config = yaml.safe_load((ROOT / "code" / "configs" / "smoke_b1_hint16.yaml").read_text())
        config["train"]["max_steps"] = 4
        with self.assertRaises(ValueError):
            S.validate_run_config(config, allow_unresolved=True)

    def test_stage_a_recipe_is_unchanged(self):
        config = yaml.safe_load((ROOT / "code" / "configs" / "stageA.yaml").read_text())
        self.assertNotIn("specialist_contract", config)
        self.assertEqual(config["train"]["epochs"], 2)
        self.assertNotIn("train_tasks", config)

    def test_materialize_rewrites_only_the_stamp_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            output = work / "runs" / "specialists" / "b1_hint16" / "20261107T000000Z_1"
            dest = output / "resolved_config.yaml"
            config = S.materialize_config(
                ROOT / "code" / "configs" / "b1_hint16.yaml",
                dest,
                output,
                work,
            )
            self.assertEqual(config["output_dir"], "runs/specialists/b1_hint16/20261107T000000Z_1")
            self.assertTrue(dest.is_file())
            with self.assertRaises(FileExistsError):
                S.materialize_config(ROOT / "code" / "configs" / "b1_hint16.yaml", dest, output, work)


class EligibilityAndHintTests(unittest.TestCase):
    def make_tree(self, root: Path, *, leak_video: bool = False) -> dict[str, Path]:
        paths = {
            "fit_A": root / "fit.json",
            "valQ": root / "valQ.json",
            "valT": root / "valT.json",
            "full_train": root / "full.json",
            "val": root / "val.json",
        }
        fit_rows = [
            row("train_1", "Perspective and Viewpoint", "videos/fit.mp4"),
            row("train_2", "Change Detection", "videos/fit2.mp4"),
            row("train_3", "Temporal Grounding", "videos/fit3.mp4"),
            row("train_4", "Geometric Relation", "videos/fit4.mp4"),
            row("train_5", "Other", "videos/fit5.mp4"),
        ]
        if leak_video:
            fit_rows[0]["video_path"] = "videos/valT.mp4"
        val_q = [row("train_q", "Perspective and Viewpoint", "videos/fit.mp4")]
        val_t = [row("train_t", "Temporal Grounding", "videos/valT.mp4")]
        official = [row("val_1", "Temporal Grounding", "videos/official.mp4")]
        write_json(paths["fit_A"], split("fit_A", fit_rows))
        write_json(paths["valQ"], split("valQ", val_q))
        write_json(paths["valT"], split("valT", val_t))
        write_json(paths["full_train"], split("full_train", fit_rows + val_q + val_t))
        write_json(paths["val"], {"metadata": {"source": "official"}, "QA": official})
        return paths

    def test_audit_counts_come_from_files_and_records_stage_b_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = S.eligibility_report(self.make_tree(Path(tmp)))
        self.assertTrue(report["eligible_for_specialist_holdout"])
        self.assertTrue(report["valT_video_held_out"])
        self.assertFalse(report["valQ_video_held_out"])
        self.assertTrue(report["stageB_family_leaks_into_valQ_or_valT"])
        self.assertFalse(report["stageB_family_leaks_into_official_val"])
        self.assertEqual(report["fit_A_task_rows"]["Temporal Grounding"], 1)
        self.assertEqual(report["files"]["fit_A"]["n_qa"], 5)
        self.assertEqual(report["overlaps"]["fit_A_valQ_qa"], 0)

    def test_valT_video_overlap_blocks_specialist_holdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = S.eligibility_report(self.make_tree(Path(tmp), leak_video=True))
        self.assertFalse(report["eligible_for_specialist_holdout"])
        self.assertGreater(report["overlaps"]["fit_A_valT_videos"], 0)

    def test_missing_split_does_not_invent_counts(self):
        report = S.eligibility_report({})
        self.assertFalse(report["eligible_for_specialist_holdout"])
        self.assertEqual(report["files"], {})
        self.assertIn("fit_A", report["missing_files"])

    def test_hint_table_accepts_pixel_motion_and_rejects_answers(self):
        table = S.load_hint_table(
            {
                "videos/fit.mp4": "Camera motion: 0.0s–1.2s panning left, slow",
                "videos/fit2.mp4": "Camera motion: unavailable",
            }
        )
        rows = [row("train_1", "Perspective and Viewpoint", "videos/fit.mp4")]
        coverage = S.assert_hint_coverage(table, rows, strict_vocabulary=True)
        self.assertEqual(coverage["videos_checked"], 1)
        with self.assertRaises(ValueError):
            S.validate_hint_text("Camera motion: the answer is B", strict_vocabulary=False)
        with self.assertRaises(ValueError):
            S.load_hint_table({"train_1": "Camera motion: hovering stationary"})
        with self.assertRaises(RuntimeError):
            S.coerce_motion_text({"text": "Camera motion: hovering stationary"}, formatter=None)

    def test_object_hint_uses_hpc_formatter_when_present(self):
        def formatter(value: dict) -> str:
            return value["text"]

        text = S.coerce_motion_text({"text": "Camera motion: hovering stationary"}, formatter=formatter)
        S.validate_hint_text(text, strict_vocabulary=True)
        self.assertEqual(text, "Camera motion: hovering stationary")

    def test_preserved_frame_budget_refuses_downsampling(self):
        with self.assertRaises(RuntimeError):
            S.guard_token_budget({"train": {"max_tokens": 10}, "preserve_frame_budget": True, "frame_budget": 32}, 11, "train_1")
        S.guard_token_budget({"train": {"max_tokens": 10}}, 11, "train_1")

    def test_fixed_frame_budget_requests_that_many_frames_on_a_long_clip(self):
        self.assertEqual(len(C.sample_index_list(1000, 30.0, 0.0, frame_budget=32)), 32)
        self.assertEqual(len(C.sample_index_list(1000, 30.0, 0.0, frame_budget=16)), 16)


class MirrorTests(unittest.TestCase):
    def test_left_right_text_flips_with_the_image_and_keeps_a_unique_letter(self):
        rewritten = M.mirror_qa(
            {
                "task": "Change Detection",
                "question": "Which sequence matches?",
                "options": {
                    "A": "left then right",
                    "B": "right then left",
                    "C": "neither",
                    "D": "both stay",
                },
                "correct_answer": "A",
            }
        )
        self.assertIsNotNone(rewritten)
        self.assertEqual(rewritten["correct_answer"], "A")
        self.assertEqual(rewritten["options"]["A"], "right then left")
        self.assertEqual(rewritten["options"]["B"], "left then right")
        collided = M.mirror_qa(
            {
                "task": "Geometric Relation",
                "question": "Which side?",
                "options": {"A": "left", "B": "left", "C": "neither", "D": "both stay"},
                "correct_answer": "A",
            }
        )
        self.assertIsNone(collided)

    def test_unsafe_or_non_horizontal_language_is_not_mirrored(self):
        base = {
            "task": "Perspective and Viewpoint",
            "question": "Where is the truck?",
            "options": {"A": "left", "B": "right", "C": "center", "D": "out"},
            "correct_answer": "A",
        }
        self.assertIsNone(M.mirror_qa({**base, "question": "Does it move forward on the left?"}))
        self.assertIsNone(M.mirror_qa({**base, "options": {**base["options"], "C": "up"}}))
        self.assertIsNone(M.mirror_qa({**base, "task": "Temporal Grounding"}))
        self.assertIsNone(M.mirror_qa({**base, "question": "Did the color change?", "options": {"A": "yes", "B": "no", "C": "unknown", "D": "both"}}))

    def test_horizontal_flip_moves_the_left_pixel_to_the_right_edge(self):
        import numpy as np

        frames = np.zeros((1, 2, 3, 3), dtype=np.uint8)
        frames[0, 0, 0, 0] = 9
        flipped = M.flip_horizontal(frames)
        self.assertEqual(int(flipped[0, 0, -1, 0]), 9)
        self.assertTrue(flipped.flags["C_CONTIGUOUS"])


class RouteGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paths = EligibilityAndHintTests().make_tree(self.root)
        self.eligibility = S.eligibility_report(self.paths)
        self.gold = [
            row("val_1", "Temporal Grounding", "videos/official.mp4", "A"),
            row("val_2", "Temporal Grounding", "videos/official2.mp4", "B"),
            row("val_3", "Other", "videos/official3.mp4", "C"),
        ]
        # The eligibility fixture's official val is a different row. Build a
        # matching gold payload for the gate from this list instead.
        self.gold_path = self.root / "gold.json"
        write_json(self.gold_path, {"metadata": {"split": "val"}, "QA": self.gold})
        self.train_path = self.paths["fit_A"]

    def tearDown(self):
        self.tmp.cleanup()

    def predictions(self, letters: dict[str, str]) -> dict[str, dict]:
        return {qa_id: {"qa_id": qa_id, "pred": letter, "p": [0.25, 0.25, 0.25, 0.25]} for qa_id, letter in letters.items()}

    def test_gate_passes_only_on_the_predeclared_margin_and_routes_target_tasks(self):
        report = R.evaluate_gate(
            self.gold,
            self.predictions({"val_1": "C", "val_2": "C", "val_3": "C"}),
            self.predictions({"val_1": "A", "val_2": "B", "val_3": "A"}),
            ("Temporal Grounding",),
            {"train_3"},
            split_name="val",
            baseline_family="ensemble_ab",
            eligibility=self.eligibility,
        )
        self.assertTrue(report["passed"])
        self.assertEqual((report["n"], report["fixes"], report["regressions"]), (2, 2, 0))
        routed = self.root / "routed.jsonl"
        R.write_routed(
            routed,
            self.gold,
            self.predictions({"val_1": "C", "val_2": "C", "val_3": "C"}),
            self.predictions({"val_1": "A", "val_2": "B"}),
            ("Temporal Grounding",),
        )
        lines = [json.loads(line) for line in routed.read_text().splitlines()]
        self.assertEqual([line["route"] for line in lines], ["specialist", "specialist", "baseline"])
        self.assertEqual(lines[2]["pred"], "C")

    def test_one_fix_does_not_route(self):
        report = R.evaluate_gate(
            self.gold[:1],
            self.predictions({"val_1": "C"}),
            self.predictions({"val_1": "A"}),
            ("Temporal Grounding",),
            set(),
            split_name="val",
            baseline_family="ensemble_ab",
            eligibility=self.eligibility,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["net"], 1)

    def test_stage_b_family_cannot_use_valT_and_training_ids_cannot_overlap(self):
        with self.assertRaises(ValueError):
            R.evaluate_gate(
                self.gold,
                self.predictions({"val_1": "A", "val_2": "B"}),
                self.predictions({"val_1": "A", "val_2": "B"}),
                ("Temporal Grounding",),
                set(),
                split_name="valT",
                baseline_family="ensemble_ab",
                eligibility=self.eligibility,
            )
        with self.assertRaises(ValueError):
            R.evaluate_gate(
                self.gold,
                self.predictions({"val_1": "A", "val_2": "B"}),
                self.predictions({"val_1": "A", "val_2": "B"}),
                ("Temporal Grounding",),
                {"val_1"},
                split_name="val",
                baseline_family="stageA",
                eligibility=self.eligibility,
            )


class StaticJobTests(unittest.TestCase):
    def test_score_moves_cuda_before_adapter_merge(self):
        text = (ROOT / "code" / "score.py").read_text()
        self.assertLess(text.index("model.to(device)"), text.index("C.load_peft_for_inference"))
        self.assertIn("refusing a CPU adapter merge", text)

    def test_a100_jobs_use_the_audited_partition_and_stay_diagnostic_free(self):
        for name in (
            "a100_train_smoke.sbatch",
            "a100_train_specialist.sbatch",
            "a100_score_specialist.sbatch",
        ):
            text = (ROOT / "jobs" / name).read_text()
            self.assertIn("#SBATCH -p gpu_a100_8", text)
            self.assertIn("#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1", text)
            self.assertNotIn("CUDA_LAUNCH_BLOCKING", text)
            self.assertNotIn("CUDA_MODULE_LOADING", text)
            self.assertNotIn("scancel", text)
            self.assertNotIn("432730", text)
        self.assertIn("REVA_SDPA_MATH_ONLY=1", (ROOT / "jobs" / "a100_prepare.sh").read_text())
        self.assertIn("activate_reva.sh", (ROOT / "jobs" / "a100_prepare.sh").read_text())

    def test_math_only_sdpa_is_opt_in(self):
        class Cuda:
            def __init__(self):
                self.calls = []

            def enable_cudnn_sdp(self, value):
                self.calls.append(("cudnn", value))

            def enable_flash_sdp(self, value):
                self.calls.append(("flash", value))

            def enable_mem_efficient_sdp(self, value):
                self.calls.append(("mem", value))

            def enable_math_sdp(self, value):
                self.calls.append(("math", value))

        cuda = Cuda()
        torch_module = type("Torch", (), {})()
        torch_module.backends = type("Backends", (), {"cuda": cuda})()
        previous = os.environ.pop("REVA_SDPA_MATH_ONLY", None)
        try:
            C.configure_attention_backend("sdpa", torch_module)
            self.assertEqual(cuda.calls, [("cudnn", False)])
            os.environ["REVA_SDPA_MATH_ONLY"] = "1"
            cuda.calls.clear()
            C.configure_attention_backend("sdpa", torch_module)
            self.assertEqual(cuda.calls, [("cudnn", False), ("flash", False), ("mem", False), ("math", True)])
        finally:
            if previous is None:
                os.environ.pop("REVA_SDPA_MATH_ONLY", None)
            else:
                os.environ["REVA_SDPA_MATH_ONLY"] = previous


if __name__ == "__main__":
    unittest.main()
