"""CPU contracts for the isolated AIR selector."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from air_sampler import (  # noqa: E402
    AIRConfig,
    AIR_SIMILARITY_METHOD,
    candidate_grid,
    candidate_policy,
    find_events,
    fit_two_component_gmm,
    parse_vlm_relevance,
    select_air_frames,
)


class AirSamplerTests(unittest.TestCase):
    def test_signal_is_explicitly_labeled_as_siglip2_adaptation(self):
        self.assertEqual(AIR_SIMILARITY_METHOD, "siglip2_query_similarity_adaptation")

    def test_grid_is_endpoint_inclusive_and_unique_on_short_clip(self):
        grid = candidate_grid(5, 64, 4.0)
        self.assertEqual([item["frame_index"] for item in grid], list(range(5)))
        self.assertEqual(grid[0]["timestamp"], 0.0)
        self.assertEqual(grid[-1]["timestamp"], 1.0)

    def test_air_policy_is_exact_count_not_rate_limited(self):
        policy = candidate_policy(64)
        self.assertEqual(policy["count"], 64)
        self.assertEqual(policy["cap"], 64)
        self.assertEqual(policy["grid_version"], "air-candidate-grid-v1")

    def test_gmm_degenerate_scores_are_finite(self):
        result = fit_two_component_gmm([0.25, 0.25, 0.25])
        self.assertEqual(result.method, "degenerate")
        self.assertEqual(result.threshold, 0.25)

    def test_short_clip_events_use_candidate_spacing_not_fixed_seconds(self):
        candidates = candidate_grid(8, 8, 4.0)
        result = find_events(candidates, [0.0, 0.9, 0.9, 0.0, 0.8, 0.8, 0.0, 0.0], 0.7, merge_gap_fraction=1.5, fps=4.0)
        self.assertEqual([event["frame_indices"] for event in result], [[1, 2], [4, 5]])

    def test_malformed_vlm_output_falls_back_without_crashing(self):
        self.assertEqual(parse_vlm_relevance("not json"), {})
        self.assertEqual(parse_vlm_relevance('{"2": {"relevance": 0.8}}'), {2: 0.8})
        self.assertEqual(parse_vlm_relevance("frame 4: 0.7; frame 5=1"), {4: 0.7, 5: 1.0})

    def test_exact_budget_order_endpoints_and_reproducibility(self):
        candidates = candidate_grid(40, 16, 4.0)
        similarities = [0.1, 0.1, 0.2, 0.2, 0.9, 0.9, 0.8, 0.8, 0.2, 0.1, 0.1, 0.3, 0.3, 0.4, 0.5, 0.5]
        config = AIRConfig(candidate_count=16, final_budget=8, max_iterations=2, vlm_batch_size=3)

        def vlm(batch):
            return {str(item["frame_index"]): 0.9 for item in batch}

        first = select_air_frames(candidates, similarities, fps=4.0, budget=8, config=config, vlm_score=vlm)
        second = select_air_frames(candidates, similarities, fps=4.0, budget=8, config=config, vlm_score=vlm)
        self.assertEqual(first.frame_indices, second.frame_indices)
        self.assertEqual(first.frame_indices, sorted(first.frame_indices))
        self.assertEqual(len(first.frame_indices), 8)
        self.assertEqual(first.frame_indices[0], 0)
        self.assertEqual(first.frame_indices[-1], 39)
        self.assertEqual(json.loads(json.dumps(first.to_dict()))["version"], "reva-air-v1")
        self.assertTrue(any("interval_potentials" in step for step in first.steps if step["stage"] == "vlm_refinement"))

    def test_local_neighbor_provider_adds_original_source_frames(self):
        candidates = candidate_grid(100, 8, 4.0)
        similarities = [0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1]
        config = AIRConfig(candidate_count=8, final_budget=4, initial_budget=2, max_iterations=1, vlm_batch_size=1)

        def vlm(batch):
            return {str(batch[0]["frame_index"]): 1.0}

        def neighbors(item):
            index = int(item["frame_index"])
            return [{"frame_index": index + 1, "timestamp": (index + 1) / 4.0, "similarity": 0.8}]

        result = select_air_frames(candidates, similarities, fps=4.0, budget=4, config=config, vlm_score=vlm, neighbor_provider=neighbors)
        self.assertTrue(any(step.get("neighbor_indices") for step in result.steps if step["stage"] == "vlm_refinement"))

    def test_unmeasured_neighbor_does_not_leak_into_final_budget(self):
        candidates = candidate_grid(100, 8, 4.0)
        candidate_indices = {item["frame_index"] for item in candidates}
        config = AIRConfig(candidate_count=8, final_budget=4, initial_budget=2, max_iterations=1, vlm_batch_size=1)

        def vlm(batch):
            return {str(batch[0]["frame_index"]): 1.0}

        def neighbors(item):
            index = int(item["frame_index"])
            return [{"frame_index": index + 1, "timestamp": (index + 1) / 4.0, "similarity": 1.0}]

        result = select_air_frames(candidates, [0.1] * 8, fps=4.0, budget=4, config=config, vlm_score=vlm, neighbor_provider=neighbors)
        self.assertTrue(set(result.frame_indices).issubset(candidate_indices))
        refinement = [step for step in result.steps if step["stage"] == "vlm_refinement"]
        self.assertTrue(refinement and refinement[0]["neighbor_indices"])


if __name__ == "__main__":
    unittest.main()
