import io
import importlib.util
import json
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import common
import build_feature_cache
import build_segmentation_cache
import download_data
import grounding
import make_submission
import phase3a_retrieval
import replicate_sam3
import sam3_overlay
import segmentation


class PureLogicTests(unittest.TestCase):
    def test_phase3a_preflight_accepts_explicit_existing_credential_reuse(self):
        plan = phase3a_retrieval.load_plan(ROOT)
        attestation = {
            "replicate_credential_reuse_authorized": True,
            "replicate_endpoint_schema_verified": True,
            "replicate_version_verified": True,
            "modal_credentials_available": True,
            "modal_runtime_available": True,
        }
        with tempfile.TemporaryDirectory() as temp:
            attestation_path = Path(temp) / "attestation.json"
            attestation_path.write_text(json.dumps(attestation))
            result = phase3a_retrieval.preflight(
                ROOT,
                plan,
                attestation_path,
                environ={"REPLICATE_API_TOKEN": "test-only"},
                modal_cli="modal",
            )
        self.assertTrue(result["ready"])
        self.assertEqual(result["replicate_credential_status"], "reuse_authorized")

    def test_phase3a_preflight_keeps_rotated_credential_path_and_other_gates(self):
        plan = phase3a_retrieval.load_plan(ROOT)
        attestation = {
            "replicate_credential_rotated": True,
            "replicate_endpoint_schema_verified": True,
            "replicate_version_verified": True,
            "modal_credentials_available": True,
            "modal_runtime_available": True,
        }
        with tempfile.TemporaryDirectory() as temp:
            attestation_path = Path(temp) / "attestation.json"
            attestation_path.write_text(json.dumps(attestation))
            rotated = phase3a_retrieval.preflight(
                ROOT,
                plan,
                attestation_path,
                environ={"REPLICATE_API_TOKEN": "test-only"},
                modal_cli="modal",
            )
            attestation["replicate_credential_reuse_authorized"] = True
            attestation["replicate_endpoint_schema_verified"] = False
            attestation_path.write_text(json.dumps(attestation))
            blocked = phase3a_retrieval.preflight(
                ROOT,
                plan,
                attestation_path,
                environ={"REPLICATE_API_TOKEN": "test-only"},
                modal_cli="modal",
            )
        self.assertTrue(rotated["ready"])
        self.assertEqual(rotated["replicate_credential_status"], "rotated")
        self.assertFalse(blocked["ready"])
        self.assertIn("replicate_endpoint_schema_verified", " ".join(blocked["reasons"]))

    def test_phase3a_preflight_rejects_unauthorized_credential_reuse(self):
        plan = phase3a_retrieval.load_plan(ROOT)
        attestation = {
            "replicate_credential_rotated": False,
            "replicate_credential_reuse_authorized": False,
            "replicate_endpoint_schema_verified": True,
            "replicate_version_verified": True,
            "modal_credentials_available": True,
            "modal_runtime_available": True,
        }
        with tempfile.TemporaryDirectory() as temp:
            attestation_path = Path(temp) / "attestation.json"
            attestation_path.write_text(json.dumps(attestation))
            result = phase3a_retrieval.preflight(
                ROOT,
                plan,
                attestation_path,
                environ={"REPLICATE_API_TOKEN": "test-only"},
                modal_cli="modal",
            )
        self.assertFalse(result["ready"])
        self.assertEqual(result["replicate_credential_status"], "unverified")
        self.assertIn("replicate_credential_reuse_authorized", " ".join(result["reasons"]))

    def test_sam3_supplemental_subset_is_exact_complement_with_prompt_coverage(self):
        path = ROOT / "jobs" / "replicate_sam3_pilot.py"
        spec = importlib.util.spec_from_file_location("replicate_sam3_pilot_job", path)
        job = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(job)
        pilot = json.loads((ROOT / "tasks" / "phase3_grounding_pilot_ids.json").read_text())
        prior = json.loads((ROOT / "runs" / "sam3_pilot_stageA_replicate" / "subset_ids.json").read_text())
        pilot_ids = {
            qa_id for split in ("valQ", "valT") for qa_id in pilot["splits"][split]["ids"]
        }
        prior_ids = {
            qa_id for split in ("valQ", "valT") for qa_id in prior["splits"][split]["ids"]
        }
        self.assertEqual(set(job.REMAINING_PROMPTS), pilot_ids - prior_ids)
        self.assertFalse(set(job.REMAINING_PROMPTS) & prior_ids)
        rows = job._resolve_rows(job.REMAINING_PROMPTS, require_complement=True)
        self.assertEqual(len(rows), 17)
        self.assertEqual(len({row["video_path"] for row in rows}), 17)
        self.assertEqual({row["qa_id"] for row in rows}, set(job.REMAINING_PROMPTS))
        self.assertEqual(
            {value.casefold() for value in job.REMAINING_PROMPTS["train_007923"]},
            {value.casefold() for value in next(row for row in json.loads((ROOT / "data" / "splits" / "valQ.json").read_text())["QA"] if row["qa_id"] == "train_007923")["options"].values()},
        )

    def test_sam3_manifest_gate_derives_counts_from_subset(self):
        subset = {
            "splits": {"valQ": {"ids": ["q1"]}, "valT": {"ids": ["q2"]}},
            "prompts": {"q1": ["car"], "q2": ["road", "boat"]},
        }
        manifest = {
            "complete": True,
            "planned_call_count": 3,
            "call_count": 2,
            "reused_call_count": 1,
            "selected_video_count": 2,
            "full_subset_video_count": 2,
        }
        self.assertEqual(replicate_sam3.validate_sam3_manifest(manifest, subset), (2, 3))
        with self.assertRaises(RuntimeError):
            replicate_sam3.validate_sam3_manifest({**manifest, "planned_call_count": 28}, subset)

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

    def test_endpoint_sampling_is_opt_in_and_keeps_legacy_mapping(self):
        legacy = common.sample_index_list(180, 30, 0.0)
        endpoints = common.sample_index_list(180, 30, 0.0, mode="endpoints")
        shifted = common.sample_index_list(180, 30, 0.5, mode="endpoints")
        self.assertEqual(legacy, [min(179, (i * 180) // 16) for i in range(16)])
        self.assertEqual(len(endpoints), 16)
        self.assertEqual((endpoints[0], endpoints[-1]), (0, 179))
        self.assertEqual((shifted[0], shifted[-1]), (0, 179))
        self.assertNotEqual(endpoints, shifted)
        with self.assertRaises(ValueError):
            common.sample_index_list(180, 30, 0.0, mode="unknown")

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

    def test_sdpa_disables_cudnn_backend_but_eager_is_unchanged(self):
        class Backend:
            def __init__(self):
                self.calls = []

            def enable_cudnn_sdp(self, enabled):
                self.calls.append(enabled)

        class Torch:
            def __init__(self):
                self.backends = type("Backends", (), {"cuda": Backend()})()

        fake_torch = Torch()
        common.configure_attention_backend("sdpa", fake_torch)
        self.assertEqual(fake_torch.backends.cuda.calls, [False])
        common.configure_attention_backend("eager", fake_torch)
        self.assertEqual(fake_torch.backends.cuda.calls, [False])

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

    def test_evidence_artifacts_are_versioned_and_separate_from_legacy_scores(self):
        legacy = Path("runs/stageA/scores_valQ.jsonl")
        enriched = grounding.evidence_output_path(legacy)
        self.assertNotEqual(enriched, legacy)
        self.assertEqual(enriched.parent, legacy.parent)
        self.assertTrue(enriched.name.startswith("scores_valQ.evidence."))
        self.assertTrue(enriched.name.endswith(".jsonl"))
        self.assertEqual(grounding.evidence_output_path(enriched), enriched)
        manifest = grounding.build_evidence_manifest(
            {"schema_version": 1, "split": "valQ"},
            selector_enabled=False,
            grounding_enabled=False,
        )
        self.assertEqual(manifest["evidence_schema_version"], grounding.EVIDENCE_SCHEMA_VERSION)
        self.assertEqual(manifest["selector"]["version"], grounding.SELECTOR_VERSION)
        self.assertFalse(manifest["selector"]["enabled"])
        self.assertEqual(manifest["grounding"]["version"], grounding.GROUNDING_VERSION)
        self.assertFalse(manifest["grounding"]["enabled"])
        json.dumps(manifest)

    def test_segmentation_cache_is_keyed_and_context_is_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask_path = root / "masks" / "one.png"
            mask_path.parent.mkdir(parents=True)
            from PIL import Image

            Image.new("L", (4, 4), 255).save(mask_path)
            frame_policy = {"frame_indices": [0], "sampling_mode": "legacy"}
            mask_policy = segmentation.segmentation_mask_policy(1)
            path = segmentation.write_segmentation_cache(
                root,
                video_checksum="video-a",
                segmentation_revision="sam2-a",
                grounding_cache_key="detector-a",
                frame_policy=frame_policy,
                mask_policy=mask_policy,
                frames=[
                    {
                        "frame_index": 0,
                        "timestamp": 0.0,
                        "width": 4,
                        "height": 4,
                        "masks": [
                            {
                                "label": "boat",
                                "confidence": 0.8,
                                "area": 0.5,
                                "centroid": {"x": 0.25, "y": 0.75},
                                "box": {"left": 0.0, "top": 0.5, "right": 0.5, "bottom": 1.0},
                                "frame_index": 0,
                                "timestamp": 0.0,
                                "mask_path": "masks/one.png",
                            }
                        ],
                    }
                ],
            )
            loaded = segmentation.load_segmentation_cache(path)
            masks = segmentation.flatten_masks(loaded)
            self.assertEqual(segmentation.filter_question_masks(masks, ["boat"], max_masks=1)[0]["label"], "boat")
            context = segmentation.format_segmentation_context(masks, max_chars=80, max_masks=1)
            self.assertLessEqual(len(context), 80)
            self.assertNotEqual(
                segmentation.segmentation_cache_key("video-a", "sam2-a", "detector-a", frame_policy, mask_policy),
                segmentation.segmentation_cache_key("video-b", "sam2-a", "detector-a", frame_policy, mask_policy),
            )

    def test_segmentation_builder_and_scorer_share_mask_policy_identity(self):
        expected = {
            "algorithm": "sam2-box-prompt",
            "max_masks_per_frame": 6,
            "model_output": "single-mask",
        }
        self.assertEqual(segmentation.segmentation_mask_policy(), expected)
        self.assertEqual(
            build_segmentation_cache.S.segmentation_mask_policy(),
            segmentation.segmentation_mask_policy(),
        )

    def test_replicate_sam3_request_is_pinned_and_mask_zip_is_indexed(self):
        self.assertEqual(replicate_sam3.prediction_ref(),
                         "lucataco/sam3-video:408f82fc20c300aac5d61d2e34ddb34cd0181e810e9f609d250aed14d5f81269")
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "clip.mp4"
            video.write_bytes(b"video")
            payload = replicate_sam3.prediction_input(video, "red barn")
            self.assertEqual(payload["prompt"], "red barn")
            self.assertTrue(payload["mask_only"])
            self.assertTrue(payload["return_zip"])

            import io
            import zipfile

            archive = io.BytesIO()
            with zipfile.ZipFile(archive, "w") as output:
                output.writestr("masks/frame-0003.png", b"\x89PNG\r\n\x1a\nmask")
                output.writestr("masks/frame-0007.png", b"\x89PNG\r\n\x1a\nmask")
            info = replicate_sam3.inspect_mask_zip(archive.getvalue())
            self.assertEqual(info["mask_count"], 2)
            self.assertEqual(info["frame_indices"], [3, 7])
            self.assertEqual(
                replicate_sam3.map_frame_indices([0, 2], [7, 15, 31]),
                [7, 31],
            )
            with self.assertRaises(ValueError):
                replicate_sam3.map_frame_indices([3], [7, 15, 31])
            reusable = Path(temp) / "reusable.zip"
            reusable.write_bytes(archive.getvalue())
            self.assertEqual(replicate_sam3.cached_mask_info(reusable)["mask_count"], 2)
            invalid = Path(temp) / "invalid.zip"
            invalid.write_bytes(b"partial output")
            self.assertIsNone(replicate_sam3.cached_mask_info(invalid))

            empty_archive = io.BytesIO()
            with zipfile.ZipFile(empty_archive, "w") as output:
                output.writestr("overlay.mp4", b"valid overlay bytes")
            empty_info = replicate_sam3.inspect_mask_zip(empty_archive.getvalue())
            self.assertEqual(empty_info["mask_count"], 0)
            self.assertEqual(empty_info["frame_indices"], [])
            self.assertEqual(empty_info["overlay_members"][0]["member"], "overlay.mp4")
            empty_path = Path(temp) / "empty.zip"
            empty_path.write_bytes(empty_archive.getvalue())
            self.assertEqual(replicate_sam3.cached_mask_info(empty_path)["mask_count"], 0)

            unsafe = io.BytesIO()
            with zipfile.ZipFile(unsafe, "w") as output:
                output.writestr("../frame-0001.png", b"\x89PNG\r\n\x1a\nmask")
            with self.assertRaisesRegex(ValueError, "unsafe member path"):
                replicate_sam3.inspect_mask_zip(unsafe.getvalue())

            unsupported = io.BytesIO()
            with zipfile.ZipFile(unsupported, "w") as output:
                output.writestr("result.txt", b"no masks")
            with self.assertRaisesRegex(ValueError, "neither PNG masks nor overlay.mp4"):
                replicate_sam3.inspect_mask_zip(unsupported.getvalue())

    def test_sam3_overlay_cache_keeps_direct_prompt_masks_outside_dino_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            png = io.BytesIO()
            from PIL import Image

            Image.fromarray(np.asarray([[255, 0], [0, 0]], dtype=np.uint8)).save(png, format="PNG")
            zip_path = root / "call.zip"
            import zipfile

            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("frame-0002.png", png.getvalue())
            cache = sam3_overlay.write_overlay_cache(
                root / "cache",
                qa_id="q1",
                video_path="clip.mp4",
                prompt_zips=[("red barn", zip_path)],
                fps=10.0,
                frame_count=4,
            )
            loaded = sam3_overlay.load_overlay_cache(root / "cache", "q1")
            self.assertEqual(cache.name, "q1.json")
            self.assertEqual(loaded["prompts"], ["red barn"])
            self.assertEqual(loaded["mask_count"], 1)
            self.assertEqual(loaded["masks"][0]["frame_index"], 2)
            self.assertEqual(loaded["masks"][0]["label"], "red barn")
            mapped_cache = sam3_overlay.write_overlay_cache(
                root / "mapped-cache",
                qa_id="q-mapped",
                video_path="clip.mp4",
                prompt_zips=[("red barn", zip_path, {2: 10})],
                fps=10.0,
                frame_count=12,
            )
            mapped_loaded = sam3_overlay.load_overlay_cache(root / "mapped-cache", "q-mapped")
            self.assertEqual(mapped_cache.name, "q-mapped.json")
            self.assertEqual(mapped_loaded["masks"][0]["frame_index"], 10)
            self.assertEqual(sam3_overlay.masks_for_frame_indices(loaded["masks"], [1]), [])
            self.assertEqual(
                [mask["frame_index"] for mask in sam3_overlay.masks_for_frame_indices(loaded["masks"], [2])],
                [2],
            )
            self.assertEqual(
                sam3_overlay.score_output_path("scores.jsonl").name,
                "scores.segmentation.sam3-direct-overlay-v1.jsonl",
            )

            empty_path = root / "empty.zip"
            with zipfile.ZipFile(empty_path, "w") as archive:
                archive.writestr("overlay.mp4", b"valid overlay bytes")
            empty_cache = sam3_overlay.write_overlay_cache(
                root / "empty-cache",
                qa_id="q-empty",
                video_path="clip.mp4",
                prompt_zips=[("road", empty_path)],
                fps=10.0,
                frame_count=4,
            )
            empty_loaded = sam3_overlay.load_overlay_cache(root / "empty-cache", "q-empty")
            self.assertEqual(empty_cache.name, "q-empty.json")
            self.assertEqual(empty_loaded["mask_count"], 0)
            self.assertEqual(empty_loaded["masks"], [])

    def test_score_sam3_overlay_requires_detector_control(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "score.py"),
                "--split",
                "valQ",
                "--out",
                "scores.jsonl",
                "--sam3-overlay-dir",
                "cache",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --grounding-context", result.stderr)

    def test_segmentation_overlay_preserves_original_and_marks_selected_mask(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mask_path = root / "masks" / "one.png"
            mask_path.parent.mkdir(parents=True)
            from PIL import Image

            Image.fromarray(np.asarray([[255, 0], [0, 0]], dtype=np.uint8)).save(mask_path)
            frames = np.asarray([[[[10, 20, 30], [40, 50, 60]], [[70, 80, 90], [100, 110, 120]]]], dtype=np.uint8)
            augmented = segmentation.compose_mask_overlay_video(
                frames,
                [7],
                [{"frame_index": 7, "mask_path": "masks/one.png"}],
                root,
            )

            self.assertEqual(augmented.shape, (1, 2, 4, 3))
            np.testing.assert_array_equal(augmented[:, :, :2], frames)
            self.assertFalse(np.array_equal(augmented[0, 0, 2], frames[0, 0, 0]))
            np.testing.assert_array_equal(augmented[0, 1, 3], frames[0, 1, 1])

    def test_candidate_selection_keeps_uniform_anchors_and_spacing(self):
        candidates = [
            {"frame_index": index, "timestamp": index / 10, "score": 1.0 if index in {2, 3, 4, 5} else 0.0}
            for index in range(100)
        ]
        selected = grounding.select_candidate_frames(candidates, limit=8, anchor_count=4)
        indices = [row["frame_index"] for row in selected]
        self.assertEqual(len(indices), 8)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(indices, sorted(indices))
        self.assertTrue({0, 33, 66, 99}.issubset(indices))
        self.assertTrue(all(0 <= index < 100 for index in indices))
        self.assertGreaterEqual(min(b - a for a, b in zip(indices, indices[1:])), 6)

    def test_candidate_selection_has_stable_ties_and_drops_invalid_duplicates(self):
        candidates = [
            {"frame_index": 4, "score": 0.5},
            {"frame_index": 4, "score": 0.9},
            {"frame_index": -1, "score": 9.0},
            {"frame_index": 1, "score": 0.9},
            {"frame_index": 3, "score": 0.9},
            {"frame_index": "bad", "score": 9.0},
        ]
        selected = grounding.select_candidate_frames(candidates, limit=3, anchor_count=1, min_spacing=1)
        self.assertEqual([row["frame_index"] for row in selected], [1, 3, 4])

    def test_adaptive_feature_cache_is_keyed_and_temporally_diverse(self):
        policy = grounding.candidate_policy(rate=4.0, cap=32)
        preprocessing = {"color_space": "rgb", "dtype": "float32"}
        candidates = [
            {
                "frame_index": index,
                "timestamp": index / 4,
                "feature": [1.0, 0.0] if index in {5, 6, 7, 24, 25} else [0.0, 1.0],
            }
            for index in range(32)
        ]
        query = {"question": "Which object?", "options": {letter: letter for letter in "ABCD"}}
        with tempfile.TemporaryDirectory() as temp:
            path = grounding.write_feature_cache(
                temp,
                video_checksum="video-a",
                encoder_revision="encoder-a",
                candidate_policy=policy,
                preprocessing=preprocessing,
                candidates=candidates,
                query_features={"q1": [1.0, 0.0]},
                query_fingerprints={"q1": grounding.query_input_fingerprint(query)},
            )
            loaded = grounding.load_feature_cache(path)
            self.assertEqual(
                loaded["cache_content_digest"], grounding.feature_cache_content_digest(loaded)
            )
            selected = grounding.select_adaptive_frames(
                loaded["candidates"], loaded["query_features"]["q1"], limit=8, anchor_count=4
            )
            tampered = json.loads(path.read_text())
            tampered["candidates"][0]["frame_index"] = 31
            path.write_text(json.dumps(tampered))
            with self.assertRaises(ValueError):
                grounding.load_feature_cache(path)
        indices = [row["frame_index"] for row in selected]
        self.assertEqual(len(indices), 8)
        self.assertEqual(indices, sorted(set(indices)))
        self.assertEqual({0, 10, 21, 31}, set(indices) & {0, 10, 21, 31})
        self.assertLessEqual(len(indices), 16)
        key = grounding.feature_cache_key("video-a", "encoder-a", policy, preprocessing)
        self.assertNotEqual(key, grounding.feature_cache_key("video-b", "encoder-a", policy, preprocessing))
        self.assertNotEqual(key, grounding.feature_cache_key("video-a", "encoder-b", policy, preprocessing))
        self.assertNotEqual(key, grounding.feature_cache_key("video-a", "encoder-a", {"rate": 8.0, "cap": 32}, preprocessing))
        self.assertNotEqual(key, grounding.feature_cache_key("video-a", "encoder-a", policy, {"color_space": "gray"}))
        changed_query = {**query, "question": "Which vehicle?"}
        self.assertNotEqual(
            grounding.query_input_fingerprint(query),
            grounding.query_input_fingerprint(changed_query),
        )

    def test_adaptive_selection_shifts_offsets_and_relaxes_spacing_to_fill(self):
        candidates = [
            {"frame_index": index, "feature": [1.0, 0.0]}
            for index in range(64)
        ]
        first = grounding.select_adaptive_frames(
            candidates, [1.0, 0.0], limit=16, anchor_count=4, min_spacing=100, temporal_offset=0.0
        )
        shifted = grounding.select_adaptive_frames(
            candidates, [1.0, 0.0], limit=16, anchor_count=4, min_spacing=100, temporal_offset=0.5
        )
        first_indices = [row["frame_index"] for row in first]
        shifted_indices = [row["frame_index"] for row in shifted]
        self.assertEqual(len(first_indices), 16)
        self.assertEqual(len(shifted_indices), 16)
        self.assertNotEqual(first_indices, shifted_indices)
        self.assertEqual(
            [row["frame_index"] for row in grounding.pad_frame_selection(first[:2], 5)],
            [0, 1, 0, 1, 0],
        )

    def test_siglip_candidate_grid_and_query_cache_construction_are_deterministic(self):
        indices = grounding.candidate_index_list(180, 30, rate=4, cap=64)
        self.assertEqual(len(indices), 24)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 179)
        self.assertEqual(len(indices), len(set(indices)))
        self.assertEqual(
            build_feature_cache.query_text(
                {"question": "Which object?", "options": {"A": "car", "B": "truck"}}
            ),
            "Question: Which object?\nOptions:\nA. car\nB. truck",
        )
        revision = build_feature_cache.encoder_revision()
        self.assertEqual(
            revision,
            "google/siglip2-base-patch16-224@022b6f71160ffb0169ca4709e2d7e25be659598a",
        )
        with tempfile.TemporaryDirectory() as temp:
            path = build_feature_cache.write_video_cache(
                temp,
                video_checksum="video-a",
                candidate_indices=[0, 179],
                candidate_features=[[1.0, 0.0], [0.0, 1.0]],
                queries={"q1": {"question": "Which object?", "options": {"A": "car"}}},
                query_features={"q1": [1.0, 0.0]},
                candidate_policy=grounding.candidate_policy(),
                encoder_revision_value=revision,
            )
            loaded = grounding.load_feature_cache(path)
            self.assertEqual(loaded["cache_key"], path.stem)
            self.assertEqual(set(loaded["query_fingerprints"]), {"q1"})

    def test_siglip_cache_builder_fails_before_network_when_model_directory_is_missing(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "build_feature_cache.py"),
                "--split",
                "valQ",
                "--model-dir",
                "/definitely/missing/siglip2-snapshot",
                "--feature-cache-dir",
                "cache",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local SigLIP2 model directory does not exist", result.stderr)

    def test_siglip_model_marker_and_snapshot_integrity_are_required(self):
        self.assertEqual(
            set(build_feature_cache.SNAPSHOT_FILE_INTEGRITY),
            set(grounding.FEATURE_MODEL_REQUIRED_FILES),
        )
        with tempfile.TemporaryDirectory() as temp:
            model_dir = Path(temp)
            for name in grounding.FEATURE_MODEL_REQUIRED_FILES:
                (model_dir / name).write_text("staged")
            with self.assertRaisesRegex(RuntimeError, "missing .reva_encoder_revision"):
                build_feature_cache.validate_local_model_dir(model_dir)
            (model_dir / build_feature_cache.REVISION_MARKER).write_text("wrong\n")
            with self.assertRaisesRegex(RuntimeError, "exact pinned model revision"):
                build_feature_cache.validate_local_model_dir(model_dir)
            (model_dir / build_feature_cache.REVISION_MARKER).write_text(
                build_feature_cache.SIGLIP2_MODEL_REVISION + "\n"
            )
            fake_integrity = {
                name: {
                    "algorithm": "git-blob-sha1" if name == "config.json" else "sha256",
                    "digest": (
                        hashlib.sha1(b"blob 6\0staged").hexdigest()
                        if name == "config.json"
                        else hashlib.sha256(b"staged").hexdigest()
                    ),
                    "size": len(b"staged"),
                }
                for name in grounding.FEATURE_MODEL_REQUIRED_FILES
            }
            with mock.patch.object(build_feature_cache, "SNAPSHOT_FILE_INTEGRITY", fake_integrity):
                self.assertEqual(
                    build_feature_cache.validate_local_model_dir(model_dir), model_dir.resolve()
                )
                (model_dir / "config.json").write_text("tampered")
                with self.assertRaisesRegex(RuntimeError, "immutable HF snapshot"):
                    build_feature_cache.validate_local_model_dir(model_dir)

        policy = grounding.candidate_policy()
        expected = grounding.candidate_index_list(180, 30, policy["rate"], policy["cap"])
        self.assertEqual(policy["grid_algorithm"], "uniform-endpoints")
        self.assertEqual(policy["grid_version"], "candidate-grid-v1")
        self.assertTrue(
            grounding.candidate_grid_matches(
                [{"frame_index": index} for index in expected], expected
            )
        )
        self.assertFalse(
            grounding.candidate_grid_matches(
                [{"frame_index": index} for index in reversed(expected)], expected
            )
        )

    def test_siglip_cache_directory_has_single_writer_and_decoder_batches_preserve_order(self):
        with tempfile.TemporaryDirectory() as temp:
            cache_dir = Path(temp) / "cache"
            with build_feature_cache.cache_writer_lock(cache_dir):
                with self.assertRaisesRegex(RuntimeError, "directory is busy"):
                    with build_feature_cache.cache_writer_lock(cache_dir):
                        pass
            self.assertFalse((cache_dir / ".build_feature_cache.lock").exists())

            class FakeFrame:
                def __init__(self, number):
                    self.number = number

                def to_ndarray(self, format):
                    self.assert_format = format
                    return np.full((1, 1, 3), self.number, dtype=np.uint8)

            class FakeContainer:
                def __init__(self):
                    self.streams = type("Streams", (), {"video": [type("Stream", (), {})()]})()
                    self.decode_calls = 0

                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    return False

                def decode(self, stream):
                    self.decode_calls += 1
                    return (FakeFrame(number) for number in range(6))

            container = FakeContainer()
            fake_av = type("FakeAV", (), {"open": staticmethod(lambda path: container)})
            with mock.patch.dict(sys.modules, {"av": fake_av}):
                batches = list(common.iter_read_frames("clip.mp4", [0, 2, 4, 5], 2, temp))
            values = [int(frame[0, 0, 0]) for batch in batches for frame in batch]
            self.assertEqual(values, [0, 2, 4, 5])
            self.assertEqual(container.decode_calls, 1)

    def test_siglip_image_encoder_propagates_explicit_input_data_format(self):
        from PIL import Image

        class FakeInferenceMode:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class FakeTensor:
            def float(self):
                return self

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return [[1.0]]

        class FakeFunctional:
            @staticmethod
            def normalize(value, dim):
                assert dim == -1
                return value

        class FakeTorch:
            nn = type("NN", (), {"functional": FakeFunctional})

            @staticmethod
            def inference_mode():
                return FakeInferenceMode()

        class FakeProcessor:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                return {"pixel_values": object()}

        class FakeModel:
            @staticmethod
            def get_image_features(**inputs):
                assert set(inputs) == {"pixel_values"}
                return FakeTensor()

        processor = FakeProcessor()
        frame = np.zeros((3, 3, 3), dtype=np.uint8)
        features = build_feature_cache.encode_images(
            processor,
            FakeModel(),
            [frame],
            torch=FakeTorch(),
            device="cpu",
            batch_size=1,
            input_data_format="channels_last",
        )
        self.assertEqual(features, [[1.0]])
        self.assertEqual(processor.calls[-1]["input_data_format"], "channels_last")
        explicit_image = processor.calls[-1]["images"][0]
        self.assertIsInstance(explicit_image, np.ndarray)
        self.assertEqual(explicit_image.shape, frame.shape)
        np.testing.assert_array_equal(explicit_image, frame)

        build_feature_cache.encode_images(
            processor,
            FakeModel(),
            [frame],
            torch=FakeTorch(),
            device="cpu",
            batch_size=1,
        )
        self.assertNotIn("input_data_format", processor.calls[-1])
        self.assertIsInstance(processor.calls[-1]["images"][0], Image.Image)

    def test_phase4_failed_outputs_preserve_error_message(self):
        path = ROOT / "jobs" / "modal_phase4_qcsrr.py"
        spec = importlib.util.spec_from_file_location("phase4_qcsrr_job", path)
        phase4 = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(phase4)
        outputs = phase4._emit(
            {
                "status": "failed",
                "error_type": "ValueError",
                "error_phase": "execution",
                "error_message": "diagnostic failure",
            },
            phase4._Budget(),
        )
        manifest = json.loads(outputs["replicate_manifest.json"])
        report = json.loads(outputs["modal_report.json"])
        self.assertEqual(manifest["error_message"], "diagnostic failure")
        self.assertEqual(report["error_message"], "diagnostic failure")

    def test_score_rejects_endpoint_sampling_for_adaptive_frames(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "score.py"),
                "--split",
                "valQ",
                "--out",
                "scores.jsonl",
                "--adaptive-frames",
                "--feature-cache-dir",
                "cache",
                "--frame-sampling",
                "endpoints",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires --frame-sampling legacy", result.stderr)

        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "code" / "score.py"),
                "--split",
                "valQ",
                "--out",
                "scores.jsonl",
                "--adaptive-frames",
                "--feature-cache-dir",
                "cache",
                "--anchor-count",
                "2",
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires anchor-count in [3, 16]", result.stderr)

    def test_score_derives_versioned_output_from_baseline_canonical_path(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            split = work / "data" / "splits"
            split.mkdir(parents=True)
            (split / "valQ.json").write_text(json.dumps({"QA": []}))
            out = work / "runs" / "baseline.jsonl"
            out.parent.mkdir(parents=True)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "code" / "score.py"),
                    "--split",
                    "valQ",
                    "--out",
                    "runs/baseline.jsonl",
                    "--adaptive-frames",
                    "--feature-cache-dir",
                    "cache",
                ],
                env={**dict(os.environ), "WORK": temp},
                capture_output=True,
                text=True,
            )
            evidence = work / grounding.evidence_output_path(Path("runs/baseline.jsonl"))
            self.assertTrue(Path(str(evidence) + ".manifest.json").exists())
            self.assertFalse(out.exists())
            self.assertFalse(Path(str(out) + ".manifest.json").exists())
        self.assertNotEqual(result.returncode, 0)  # local environment has no GPU dependencies
        self.assertNotIn("cannot reuse a baseline score path", result.stderr)

    def test_question_entity_filtering_is_case_insensitive_and_confidence_aware(self):
        tracks = [
            {"track_id": "low", "label": "car", "confidence": 0.2},
            {"track_id": "truck", "label": "Truck", "confidence": 0.8},
            {"track_id": "bird", "label": "bird", "confidence": 0.99},
            {"track_id": "cart", "label": "cart", "confidence": 0.99},
        ]
        filtered = grounding.filter_question_tracks(
            tracks,
            ["truck", "car"],
            min_confidence=0.5,
            max_tracks=2,
        )
        self.assertEqual([track["track_id"] for track in filtered], ["truck"])

    def test_box_normalization_and_grounding_context_are_bounded(self):
        self.assertEqual(
            grounding.normalise_box([10, 20, 50, 80], width=100, height=200),
            {"left": 0.1, "top": 0.1, "right": 0.5, "bottom": 0.4},
        )
        tracks = [
            {
                "track_id": "car-1",
                "label": "car",
                "confidence": 0.91,
                "observations": [
                    {
                        "frame_index": index,
                        "timestamp": index / 4,
                        "box": [10, 20, 50, 80],
                        "width": 100,
                        "height": 200,
                    }
                    for index in range(10)
                ],
            }
        ]
        context = grounding.format_grounding_context(tracks, max_chars=320, max_observations=2)
        self.assertLessEqual(len(context), 320)
        self.assertIn("2D image coordinates", context)
        self.assertIn("car-1", context)
        self.assertIn("left=0.100", context)
        self.assertNotIn("depth", context.lower())


if __name__ == "__main__":
    unittest.main()
