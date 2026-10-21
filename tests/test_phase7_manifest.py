import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import phase7_manifest as P


class Phase7ManifestTests(unittest.TestCase):
    def test_build_binds_sample_timestamps_to_source_timeline(self):
        base = {
            "clip.mp4": {
                "n_frames": 32,
                "fps": 4.0,
                "width": 640,
                "height": 360,
                "duration": 8.0,
            }
        }
        source = {
            "clip.mp4": {
                **base["clip.mp4"],
                "frame_timestamps": [round(index / 4.0, 5) for index in range(32)],
                "timestamp_source": "presentation_timestamp",
                "video_sha256": "abc",
            }
        }
        manifest = P.build_manifest(base, source_manifests=[source], video_paths=["clip.mp4"], dataset_revision="rev")
        self.assertEqual(manifest["dataset_revision"], "rev")
        record = manifest["records"]["clip.mp4"]
        self.assertEqual(record["timestamp_source"], "presentation_timestamp")
        self.assertEqual(record["video_sha256"], "abc")
        for offset, sample in record["legacy_samples"].items():
            self.assertEqual(sample["timestamps_s"], [record["frame_timestamps_s"][i] for i in sample["frame_indices"]])
        P.validate_manifest(manifest)

    def test_missing_timestamps_use_explicit_constant_fps_fallback(self):
        base = {
            "clip.mp4": {"n_frames": 5, "fps": 2.0, "width": 2, "height": 2, "duration": 2.5}
        }
        record = P.build_manifest(base)["records"]["clip.mp4"]
        self.assertEqual(record["timestamp_source"], "index_divided_by_fps")
        self.assertEqual(record["frame_timestamps_s"], [0.0, 0.5, 1.0, 1.5, 2.0])

    def test_validator_rejects_tampering_and_nonmonotone_timestamps(self):
        base = {"clip.mp4": {"n_frames": 16, "fps": 4.0, "width": 2, "height": 2, "duration": 4.0}}
        manifest = P.build_manifest(base)
        tampered = copy.deepcopy(manifest)
        tampered["records"]["clip.mp4"]["frame_timestamps_s"][3] = -1.0
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            P.validate_manifest(tampered)
        tampered = copy.deepcopy(manifest)
        tampered["records"]["clip.mp4"]["frame_timestamps_s"][3] = -1.0
        tampered["content_digest"] = P._content_digest(tampered)
        with self.assertRaisesRegex(ValueError, "not monotone"):
            P.validate_manifest(tampered)
        tampered = copy.deepcopy(manifest)
        tampered["records"]["clip.mp4"]["frame_timestamps_s"][0] = -1.0
        tampered["content_digest"] = P._content_digest(tampered)
        with self.assertRaisesRegex(ValueError, "exceed duration"):
            P.validate_manifest(tampered)

    def test_existing_reviewed_source_manifest_is_compatible(self):
        base = json.loads((ROOT / "data" / "manifest.json").read_text())
        source = json.loads((ROOT / "runs" / "phase2_temporal_verified_window" / "modal_full" / "source_manifest.json").read_text())
        manifest = P.build_manifest(
            base,
            source_manifests=[source],
            video_paths=source.keys(),
            dataset_revision="06224fa1b42e5e35677a8b7f2145e06307a9a4ee",
        )
        self.assertEqual(len(manifest["records"]), 23)
        self.assertTrue(all(record["timestamp_source"] == "presentation_timestamp" for record in manifest["records"].values()))
        P.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
