import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common
import download_data
import make_submission


class PureLogicTests(unittest.TestCase):
    def test_downloader_allows_referenced_dji_and_detects_unreferenced(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            referenced = root / "Hawk_UAV" / "NJ" / "DJI_0157_d4_01.mp4"
            unreferenced = root / "Hawk_UAV" / "NJ" / "DJI_original.MP4"
            referenced.parent.mkdir(parents=True)
            referenced.write_bytes(b"referenced")
            unreferenced.write_bytes(b"unreferenced")

            self.assertEqual(
                download_data.unreferenced_dji_files(
                    root, {"Hawk_UAV/NJ/DJI_0157_d4_01.mp4"}
                ),
                [unreferenced],
            )

    def test_sampling_is_even_and_bounded(self):
        self.assertEqual(len(common.sample_index_list(180, 30, 0.33)), 16)
        for n, fps in ((1, 30), (3, 30), (180, 30), (2400, 24)):
            indices = common.sample_index_list(n, fps, 0.33)
            self.assertEqual(len(indices) % 2, 0)
            self.assertTrue(all(0 <= index < max(n, 1) for index in indices))
        with self.assertRaises(ValueError):
            common.sample_index_list(10, 30, 1.0)

    def test_fast_recipe_and_score_defaults_are_explicit(self):
        for name in ("stageA", "stageB"):
            config = yaml.safe_load((ROOT / "code" / "configs" / f"{name}.yaml").read_text())
            self.assertEqual(config["lora"], {"r": 16, "alpha": 32, "dropout": 0.05})
            self.assertEqual(config["train"]["epochs"], 2)
            self.assertEqual(config["train"]["grad_accum"], 16)
            self.assertEqual(config["attn_implementation"], "sdpa")
        self.assertEqual(common.MAX_FRAMES, 16)
        self.assertEqual(common.DEFAULT_SCORE_OFFSETS, (0.0, 0.5))
        self.assertEqual(common.DEFAULT_SCORE_ROTS, 2)

    def test_training_time_estimate_uses_optimizer_steps(self):
        self.assertAlmostEqual(common.estimate_training_hours(32, 2, 16, 3600), 4.0)
        with self.assertRaises(ValueError):
            common.estimate_training_hours(32, 2, 0, 1)

    def test_all_of_the_above_stays_last_and_maps_answer(self):
        row = {
            "options": {"A": "all of the above", "B": "two", "C": "three", "D": "four"}
        }
        shown, permutation = common.permute_options(row, ["A", "B", "C", "D"])
        self.assertEqual(permutation[-1], "A")
        self.assertEqual(shown["D"], "all of the above")

    def test_letter_ids_require_one_token_each(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return [ord(text)]

        self.assertEqual(common.letter_ids(Tokenizer()), [65, 66, 67, 68])

        class SplitTokenizer(Tokenizer):
            def encode(self, text, add_special_tokens=False):
                return [ord(text), 0] if text == "A" else super().encode(text, add_special_tokens)

        with self.assertRaises(ValueError):
            common.letter_ids(SplitTokenizer())

    def test_submission_rejects_non_normalized_probabilities(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scores.jsonl"
            path.write_text(json.dumps({"qa_id": "test_0", "p": [0.4, 0.4, 0.4, 0.4]}) + "\n")
            with self.assertRaises(ValueError):
                make_submission.read_scores([str(path)], {"test_0"})

    def test_score_resume_truncates_only_partial_final_record(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "scores.jsonl"
            path.write_bytes(b'{"qa_id":"test_0"}\n{"qa_id":"test_1"')
            self.assertEqual(common.read_completed_jsonl(path), {"test_0"})
            self.assertEqual(path.read_bytes(), b'{"qa_id":"test_0"}\n')
            path.write_text('{"qa_id":"test_0"}\n{"qa_id":"test_1"}\n')
            self.assertEqual(common.read_completed_jsonl(path), {"test_0", "test_1"})

    def test_submission_probe_is_a_valid_single_file_zip(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            data = work / "data" / "ReVA"
            data.mkdir(parents=True)
            rows = [
                {
                    "qa_id": f"test_{index:06d}",
                    "video_path": "x.mp4",
                    "options": {letter: letter for letter in "ABCD"},
                    "correct_answer": "",
                }
                for index in range(4000)
            ]
            (data / "test.json").write_text(json.dumps({"metadata": {}, "QA": rows}))
            result = subprocess.run(
                [sys.executable, str(ROOT / "code" / "make_submission.py"), "--all", "A", "--out", "subs/probe"],
                env={**dict(os.environ), "WORK": temp},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("letter histogram", result.stdout)
            import zipfile

            with zipfile.ZipFile(work / "subs" / "probe.zip") as archive:
                self.assertEqual(archive.namelist(), ["predictions.json"])
                payload = json.loads(archive.read("predictions.json"))
            self.assertEqual(len(payload["QA"]), 4000)

    def test_split_builder_keeps_video_and_question_groups_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            data = work / "data" / "ReVA"
            data.mkdir(parents=True)
            rows = []
            for video_index in range(12):
                source = ("VisDrone", "Hawk_UAV", "UAVDT")[video_index % 3]
                for question_index in range(3):
                    rows.append(
                        {
                            "qa_id": f"train_{video_index:03d}_{question_index}",
                            "video_path": f"{source}/{video_index}.mp4",
                            "dataset_name": source,
                            "task": f"task_{question_index % 2}",
                            "question": f"q-{video_index}-{question_index}",
                            "options": {letter: letter for letter in "ABCD"},
                            "correct_answer": "A",
                        }
                    )
            (data / "train.json").write_text(json.dumps({"QA": rows}))
            (data / "test.json").write_text(json.dumps({"QA": []}))
            subprocess.run(
                [sys.executable, str(ROOT / "code" / "make_splits.py"), "--work", temp, "--video-fraction", "0.2", "--question-fraction", "0.2"],
                check=True,
                capture_output=True,
                text=True,
            )
            splits = {
                name: json.loads((work / "data" / "splits" / f"{name}.json").read_text())["QA"]
                for name in ("fit_A", "valT", "valQ", "full_train")
            }
            fit_ids = {row["qa_id"] for row in splits["fit_A"]}
            self.assertFalse(fit_ids & {row["qa_id"] for row in splits["valT"] + splits["valQ"]})
            val_t_videos = {row["video_path"] for row in splits["valT"]}
            self.assertFalse(val_t_videos & {row["video_path"] for row in splits["fit_A"]})
            self.assertEqual(len(splits["full_train"]), len(rows))


if __name__ == "__main__":
    unittest.main()
