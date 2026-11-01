import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from validate_phase7_cache import compare_frame_contract, expected_frame_policy, observation_metrics


class Phase7CacheValidationTests(unittest.TestCase):
    def test_frame_policy_and_timestamps_are_bound_to_manifest(self):
        record = {
            "legacy_samples": {
                "0.0": {"frame_indices": [0, 2]},
                "0.5": {"frame_indices": [1, 3]},
            },
            "frame_timestamps_s": [0.0, 0.1, 0.2, 0.3],
        }
        cache = {
            "frame_policy": expected_frame_policy(record),
            "frames": [
                {"frame_index": index, "timestamp": timestamp}
                for index, timestamp in enumerate(record["frame_timestamps_s"])
            ],
        }
        self.assertEqual(compare_frame_contract(record, cache)["policy_status"], "exact")
        self.assertEqual(compare_frame_contract(record, cache)["timestamp_status"], "exact")

        cache["frames"][2]["timestamp"] = 0.25
        self.assertEqual(compare_frame_contract(record, cache)["timestamp_status"], "mismatch")
        cache["frames"][2]["timestamp"] = 0.2
        cache["frame_policy"]["frame_indices"] = [0, 1, 2]
        self.assertEqual(compare_frame_contract(record, cache)["policy_status"], "mismatch")

    def test_observation_metrics_keep_singletons_and_gaps_explicit(self):
        cache = {
            "frames": [
                {"frame_index": 0, "detections": []},
                {"frame_index": 1, "detections": []},
                {"frame_index": 2, "detections": []},
            ],
            "tracker": {"iou_threshold": 0.3},
            "tracks": [
                {
                    "track_id": "track-0001",
                    "uncertain": True,
                    "observations": [{"frame_index": 0, "label": "car"}],
                },
                {
                    "track_id": "track-0002",
                    "uncertain": False,
                    "observations": [
                        {"frame_index": 0, "label": "truck"},
                        {"frame_index": 2, "label": "truck", "association_iou": 0.2},
                    ],
                },
            ],
        }
        metrics = observation_metrics(cache)
        self.assertEqual(metrics["single_observation_track_count"], 1)
        self.assertEqual(metrics["uncertain_track_count"], 1)
        self.assertEqual(metrics["observation_gap_count"], 1)
        self.assertEqual(metrics["low_association_iou_proxy_count"], 1)


if __name__ == "__main__":
    unittest.main()
