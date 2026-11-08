"""Smoke tests for tools/paired_gate.py on known EXP-001 / EXP-003 scores."""

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "runs" / "stageA_q3vl8b_r16" / "legacy"
ENDPOINT = ROOT / "runs" / "stageA_q3vl8b_r16"


@unittest.skipUnless(LEGACY.is_dir(), "baseline score JSONL not present")
class PairedGateTests(unittest.TestCase):
    def _run(self, baseline_paths, candidate_paths) -> subprocess.CompletedProcess:
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "paired_gate.py"),
            *baseline_paths,
            "--candidate",
            *candidate_paths,
            "--work",
            str(ROOT),
        ]
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)

    def test_identical_baseline_is_all_pass(self):
        paths = [
            str(LEGACY / "scores_valQ.jsonl"),
            str(LEGACY / "scores_valT.jsonl"),
            str(LEGACY / "scores_val.jsonl"),
        ]
        result = self._run(paths, paths)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)
        self.assertIn("G1=PASS", result.stdout)
        self.assertIn("net +0", result.stdout)

    def test_exp003_endpoint_regresses_valq(self):
        baseline = [
            str(LEGACY / "scores_valQ.jsonl"),
            str(LEGACY / "scores_valT.jsonl"),
            str(LEGACY / "scores_val.jsonl"),
        ]
        candidate = [
            str(ENDPOINT / "scores_valQ_endpoints.jsonl"),
            str(ENDPOINT / "scores_valT_endpoints.jsonl"),
            str(ENDPOINT / "scores_val_endpoints.jsonl"),
        ]
        if not all(Path(p).is_file() for p in candidate):
            self.skipTest("EXP-003 endpoint scores missing")
        result = self._run(baseline, candidate)
        self.assertIn("valQ baseline 498/595 candidate 495/595 net -3", result.stdout)
        self.assertIn("G1=FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main()
