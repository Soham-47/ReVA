import io
import json
import os
import struct
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import phase7c_replicate_runner as R


class _Response:
    def __init__(self, headers):
        self.headers = headers


class RateLimitError(Exception):
    status_code = 429

    def __init__(self, retry_after="3"):
        super().__init__("provider rate limit; do not serialize this message")
        self.response = _Response({"Retry-After": retry_after})


class FakePrediction:
    def __init__(self, prediction_id, status, output=None):
        self.id = prediction_id
        self.status = status
        self.output = output
        self.model = "pinned/model"
        self.version = "pinned-version"
        self.metrics = {"predict_time": 0.25, "total_time": 0.5}
        self.created_at = "2026-10-19T00:00:00Z"


class FakePredictions:
    def __init__(self, *, create_results=None, get_results=None):
        self.create_results = list(create_results or [])
        self.get_results = list(get_results or [])
        self.create_calls = 0
        self.get_calls = 0

    def create(self, **_kwargs):
        self.create_calls += 1
        result = self.create_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def get(self, _prediction_id):
        self.get_calls += 1
        result = self.get_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeClient:
    def __init__(self, predictions):
        self.predictions = predictions


class Phase7CReplicateRunnerTests(unittest.TestCase):
    def test_build_manifest_is_limited_to_reviewed_videos_and_question_only_prompts(self):
        manifest = R.build_run_manifest(ROOT)

        self.assertEqual(manifest["status"], "dry_run")
        self.assertEqual(manifest["selection"]["video_count"], 2)
        self.assertEqual(
            {item["qa_id"] for item in manifest["videos"]},
            {"train_000047", "train_001138"},
        )
        self.assertEqual(len(manifest["requests"]), 16)
        self.assertEqual(
            {item["model_key"] for item in manifest["requests"]},
            {"merged_mask_reference", "id_preserving_candidate"},
        )
        request_text = json.dumps(manifest["requests"]).casefold()
        self.assertNotIn("correct_answer", request_text)
        self.assertNotIn("baseline", request_text)
        self.assertNotIn("options", request_text)
        for video in manifest["videos"]:
            self.assertIn(video["question_prompt"].casefold(), video["question"].casefold())
            self.assertEqual(len(video["frame_mapping"]), 180)
            self.assertEqual(video["frame_mapping"][90]["source_frame_index"], 90)
            self.assertEqual(video["frame_mapping"][90]["timestamp_s"], 3.0)

    def test_dry_run_has_bounded_resumable_request_contract_and_no_credential_value(self):
        manifest = R.build_run_manifest(ROOT)
        self.assertEqual(manifest["credential_status"], "absent")
        execution = manifest["execution"]
        self.assertEqual(execution["max_primary_prediction_creates"], 16)
        self.assertEqual(execution["max_retry_prediction_creates"], 4)
        self.assertEqual(execution["max_prediction_create_attempts_total"], 20)
        self.assertEqual(execution["max_retries_per_request"], 1)
        self.assertEqual(execution["scheduler"], "sequential_async_predictions_create_poll")
        self.assertTrue(execution["no_parallel_submissions"])
        self.assertTrue(execution["resumable"])
        self.assertEqual(execution["runtime_dependency"]["package"], "replicate")
        self.assertEqual(execution["runtime_dependency"]["version"], "1.0.7")
        self.assertIn("replicate==1.0.7", (ROOT / R.REQUIREMENTS_RELATIVE).read_text())
        self.assertTrue(all(request["status"] == "planned" for request in manifest["requests"]))
        self.assertNotIn("REPLICATE_API_TOKEN", json.dumps(manifest))

    def test_validates_full_legacy_and_id_preserving_archives_with_mapping_contracts(self):
        manifest = R.build_run_manifest(ROOT)
        source = {**manifest["videos"][0], "n_frames": 3, "frame_mapping": manifest["videos"][0]["frame_mapping"][:3]}
        legacy = R.validate_output_zip(
            "merged_mask_reference",
            _legacy_zip(source),
            source,
        )
        self.assertEqual(legacy["output_frame_indices"], [0, 1, 2])
        self.assertEqual(legacy["frame_mapping"], [])
        self.assertEqual(legacy["timestamp_mapping_status"], "unverified_no_endpoint_metadata")

        identity = R.validate_output_zip(
            "id_preserving_candidate",
            _identity_zip(source),
            source,
            expected_prompt=source["question_prompt"],
        )
        self.assertEqual(identity["output_frame_indices"], [0, 1, 2])
        self.assertEqual(identity["frame_mapping"][2]["timestamp_s"], 0.06666666666666667)
        self.assertEqual(identity["objects"][0]["sam_id"], 1)

    def test_rejects_unnamed_sparse_and_bad_source_or_bit_depth_outputs(self):
        manifest = R.build_run_manifest(ROOT)
        source = {**manifest["videos"][0], "n_frames": 3, "frame_mapping": manifest["videos"][0]["frame_mapping"][:3]}
        with self.assertRaisesRegex(ValueError, "no frame index"):
            R.validate_output_zip(
                "merged_mask_reference",
                _zip_bytes({"masks/unnamed.png": _png_bytes(640, 360, bit_depth=8)}),
                source,
            )
        with self.assertRaisesRegex(ValueError, "sparse merged-mask output requires overlay"):
            R.validate_output_zip(
                "merged_mask_reference",
                _zip_bytes({
                    "masks/frame_00000.png": _png_bytes(640, 360, bit_depth=8),
                    "masks/frame_00002.png": _png_bytes(640, 360, bit_depth=8),
                }),
                source,
            )
        with self.assertRaisesRegex(ValueError, "source frame mapping"):
            R.validate_output_zip(
                "merged_mask_reference",
                _zip_bytes({"frame_00000.png": _png_bytes(640, 360, bit_depth=8)}),
                {**source, "frame_mapping": source["frame_mapping"][:2]},
            )
        with self.assertRaisesRegex(ValueError, "16-bit"):
            R.validate_output_zip(
                "id_preserving_candidate",
                _identity_zip(source, bit_depth=8),
                source,
            )

    def test_accepts_recovered_sparse_masks_when_overlay_matches_source_contract(self):
        manifest = R.build_run_manifest(ROOT)
        cases = (
            (
                manifest["videos"][0],
                ROOT / R.RUN_RELATIVE / "diagnostics" / "recovered_raw" /
                "train_000047--merged_mask_reference--question-conditioned--large_archway_with_Chinese_characters.zip",
                list(range(106, 180)),
            ),
            (
                manifest["videos"][1],
                ROOT / R.RUN_RELATIVE / "diagnostics" / "recovered_raw" /
                "train_001138--merged_mask_reference--question-conditioned--white_van_with_blue_markings_on_its_side.zip",
                [49],
            ),
        )
        for source, path, expected_indices in cases:
            result = R.validate_output_zip("merged_mask_reference", path.read_bytes(), source)
            self.assertEqual(result["output_frame_indices"], expected_indices)
            self.assertEqual(result["source_frame_indices"], expected_indices)
            self.assertEqual(
                result["missing_source_frame_indices"],
                [index for index in range(source["n_frames"]) if index not in expected_indices],
            )
            self.assertEqual(result["frame_mapping"][0]["source_frame_index"], expected_indices[0])
            self.assertEqual(result["timestamp_mapping_status"], "unverified_no_endpoint_metadata")
            self.assertEqual(result["overlay_metadata"]["frame_count"], source["n_frames"])
            self.assertEqual(result["overlay_metadata"]["fps"], source["fps"])
            self.assertEqual(result["overlay_metadata"]["padding"], {"width": 0, "height": 8})
            self.assertIn("endpoint timestamp metadata unavailable", result["frame_mapping_provenance"])

    def test_rejects_sparse_masks_when_overlay_metadata_does_not_match(self):
        manifest = R.build_run_manifest(ROOT)
        source = {**manifest["videos"][0], "n_frames": 3, "frame_mapping": manifest["videos"][0]["frame_mapping"][:3]}
        sparse = _zip_bytes({
            "mask_00000.png": _png_bytes(640, 360, bit_depth=8),
            "mask_00002.png": _png_bytes(640, 360, bit_depth=8),
            "overlay.mp4": b"overlay",
        })
        with patch.object(
            R,
            "_probe_overlay",
            return_value={"width": 640, "height": 368, "frame_count": 2, "fps": 30.0},
        ), self.assertRaisesRegex(ValueError, "overlay frame count"):
            R.validate_output_zip("merged_mask_reference", sparse, source)

    def test_download_helper_selects_first_zip_from_multi_file_output(self):
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / "output.zip"
            R._download_output([b"zip-payload", b"preview-payload"], destination)
            self.assertEqual(destination.read_bytes(), b"zip-payload")

    def test_output_url_fallback_never_receives_replicate_bearer_header(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b"url-payload"

        with tempfile.TemporaryDirectory() as temp, patch.object(R.urllib.request, "urlopen", return_value=Response()) as opener:
            destination = Path(temp) / "output.zip"
            R._download_output("https://replicate.delivery/example.zip", destination)
            request = opener.call_args.args[0]
            self.assertNotIn("Authorization", request.headers)
            self.assertNotIn("Bearer", str(request.headers))
            self.assertEqual(destination.read_bytes(), b"url-payload")

    def test_sdk_file_output_is_read_directly_without_url_fetch(self):
        class FileOutput:
            url = "https://replicate.delivery/example.zip"

            def read(self):
                return b"sdk-payload"

        with tempfile.TemporaryDirectory() as temp, patch.object(R.urllib.request, "urlopen") as opener:
            destination = Path(temp) / "output.zip"
            R._download_output(FileOutput(), destination)
            opener.assert_not_called()
            self.assertEqual(destination.read_bytes(), b"sdk-payload")

    def test_async_create_persists_id_and_resume_polls_without_duplicate_create(self):
        temp_root, manifest_path, manifest = self._temporary_manifest("merged_mask_reference")
        output = _legacy_zip(manifest["videos"][0])
        predictions = FakePredictions(
            create_results=[FakePrediction("pred-1", "starting")],
            get_results=[FakePrediction("pred-1", "succeeded", output)],
        )
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test-only-token"}):
            result = R.execute_manifest(manifest_path, temp_root, client=FakeClient(predictions), sleep_fn=lambda _delay: None)
        self.assertEqual(result["requests"][0]["status"], "succeeded")
        self.assertEqual(result["requests"][0]["provider_prediction_id"], "pred-1")
        self.assertEqual(predictions.create_calls, 1)
        snapshot = result["requests"][0]["provider_snapshot"]
        self.assertEqual(snapshot["status"], "succeeded")
        self.assertEqual(snapshot["metrics"]["total_time"], 0.5)
        self.assertNotIn("output", snapshot)
        self.assertNotIn("test-only-token", manifest_path.read_text())

        resumed = json.loads(manifest_path.read_text())
        resumed["requests"][0]["status"] = "polling"
        output_path = temp_root / R.RUN_RELATIVE / resumed["requests"][0]["output_relpath"]
        output_path.unlink()
        R._write_manifest(manifest_path, resumed)
        resume_predictions = FakePredictions(
            get_results=[FakePrediction("pred-1", "succeeded", output)],
        )
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test-only-token"}):
            resumed_result = R.execute_manifest(manifest_path, temp_root, client=FakeClient(resume_predictions), sleep_fn=lambda _delay: None)
        self.assertEqual(resumed_result["requests"][0]["status"], "succeeded")
        self.assertEqual(resume_predictions.create_calls, 0)
        self.assertEqual(resume_predictions.get_calls, 1)

    def test_ambiguous_create_without_id_is_unresolved_without_retry(self):
        temp_root, manifest_path, _manifest = self._temporary_manifest("merged_mask_reference")
        predictions = FakePredictions(create_results=[FakePrediction(None, "starting")])
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test-only-token"}):
            result = R.execute_manifest(manifest_path, temp_root, client=FakeClient(predictions), sleep_fn=lambda _delay: None)
        request = result["requests"][0]
        self.assertEqual(request["status"], "unresolved_creation")
        self.assertEqual(predictions.create_calls, 1)
        self.assertEqual(result["execution"]["retry_attempts"], 0)
        self.assertEqual(request["attempts"][0]["uncertainty"], "provider_response_without_prediction_id")
        self.assertIsNone(result["execution"]["paid_calls_made"])
        self.assertTrue(result["execution"]["billing_uncertainty"])

    def test_rate_limit_retries_with_server_guidance_and_shared_cap(self):
        temp_root, manifest_path, manifest = self._temporary_manifest("merged_mask_reference")
        output = _legacy_zip(manifest["videos"][0])
        predictions = FakePredictions(
            create_results=[RateLimitError("3"), FakePrediction("pred-429", "starting")],
            get_results=[FakePrediction("pred-429", "succeeded", output)],
        )
        sleeps = []
        with patch.dict(os.environ, {"REPLICATE_API_TOKEN": "test-only-token"}):
            result = R.execute_manifest(
                manifest_path,
                temp_root,
                client=FakeClient(predictions),
                sleep_fn=sleeps.append,
            )
        self.assertEqual(result["requests"][0]["status"], "succeeded")
        self.assertEqual(predictions.create_calls, 2)
        self.assertEqual(result["execution"]["prediction_create_attempts"], 2)
        self.assertEqual(result["execution"]["prediction_ids_created"], 1)
        self.assertEqual(result["execution"]["prediction_ids_observed"], 1)
        self.assertEqual(result["execution"]["successful_predictions"], 1)
        self.assertIsNone(result["execution"]["paid_calls_made"])
        self.assertEqual(result["execution"]["billable_cost_estimate_usd"], 0.0026)
        self.assertEqual(result["execution"]["retry_attempts"], 1)
        self.assertGreaterEqual(sleeps[0], 3.0)
        serialized = manifest_path.read_text()
        self.assertNotIn("test-only-token", serialized)
        self.assertNotIn("provider rate limit", serialized)

    def _temporary_manifest(self, model_key):
        temp_dir = tempfile.TemporaryDirectory()
        temp_root = Path(temp_dir.name)
        manifest = R.build_run_manifest(ROOT)
        for video in manifest["videos"]:
            source = temp_root / video["source_path"]
            source.parent.mkdir(parents=True, exist_ok=True)
            source.symlink_to(ROOT / video["source_path"])
        for index, request in enumerate(manifest["requests"]):
            if index == 0:
                request["model_key"] = model_key
            else:
                request["status"] = "blocked_cap"
        manifest["content_digest"] = R._content_digest(manifest)
        manifest_path = temp_root / R.RUN_RELATIVE / "manifest.json"
        R._write_manifest(manifest_path, manifest)
        self.addCleanup(temp_dir.cleanup)
        return temp_root, manifest_path, manifest


def _legacy_zip(source):
    return _zip_bytes({
        f"masks/frame_{index:05d}.png": _png_bytes(source["width"], source["height"], bit_depth=8)
        for index in range(int(source["n_frames"]))
    })


def _identity_zip(source, *, bit_depth=16):
    objects = {
        "width": source["width"],
        "height": source["height"],
        "frames": source["n_frames"],
        "fps": source["fps"],
        "prompt": source["question_prompt"],
        "objects": [{
            "label": 1,
            "sam_id": 1,
            "frames": 1,
            "first_frame": 0,
            "last_frame": 0,
            "mean_score": 0.8,
            "best_frame": 0,
            "best_box": [1, 2, 30, 40],
            "best_area": 100,
            "best_area_fraction": 0.01,
            "best_score": 0.9,
        }],
    }
    return _zip_bytes({
        "objects.json": json.dumps(objects).encode(),
        **{
            f"labels/labels_{index:05d}.png": _png_bytes(source["width"], source["height"], bit_depth=bit_depth)
            for index in range(int(source["n_frames"]))
        },
    })


def _zip_bytes(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    return output.getvalue()


def _png_bytes(width, height, *, bit_depth):
    # The validator intentionally checks the PNG signature/IHDR contract only;
    # a valid pixel stream is unnecessary for these unit-level archive tests.
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">IIBBBBB", width, height, bit_depth, 0, 0, 0, 0)
    )


if __name__ == "__main__":
    unittest.main()
