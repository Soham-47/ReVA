import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import phase3a_retrieval as P

sys.path.insert(0, str(ROOT / "jobs"))
import modal_phase3a_retrieval_smoke as smoke


class Phase3AHpcTests(unittest.TestCase):
    def test_replicate_preflight_hpc_skips_modal_only_gates(self):
        plan = P.load_plan(ROOT)
        attestation = {
            "replicate_credential_reuse_authorized": True,
            "replicate_endpoint_schema_verified": True,
            "replicate_version_verified": True,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "attestation.json"
            path.write_text(json.dumps(attestation))
            result = P.preflight(
                ROOT,
                plan,
                path,
                backend="hpc",
                environ={"REPLICATE_API_TOKEN": "test-only"},
            )
        self.assertTrue(result["ready"])
        self.assertEqual(result["backend"], "HPC")
        self.assertEqual(result["replicate_credential_status"], "reuse_authorized")
        self.assertFalse(result["modal_cli_present"])

    def test_hpc_gate_and_budget_are_bounded_without_modal_attestation(self):
        plan = P.load_plan(ROOT)
        smoke._validate_hpc_gate(plan)
        budget = smoke._Budget(plan, backend="hpc")
        self.assertEqual(budget.max_wall_minutes, 30)

        invalid = json.loads(json.dumps(plan))
        invalid["modal_budget"]["memory_gib"] = 16
        with self.assertRaisesRegex(RuntimeError, "32 GiB"):
            smoke._validate_hpc_gate(invalid)

    def test_hpc_requires_actual_h200_device(self):
        class Cuda:
            def __init__(self, name):
                self.name = name

            def is_available(self):
                return True

            def get_device_name(self, index):
                return self.name

        class Torch:
            def __init__(self, name):
                self.cuda = Cuda(name)

        self.assertEqual(smoke._validate_hpc_device(Torch("NVIDIA H200 NVL")), "NVIDIA H200 NVL")
        with self.assertRaisesRegex(RuntimeError, "requires an H200"):
            smoke._validate_hpc_device(Torch("NVIDIA A100-SXM4-80GB"))

    def test_hpc_dry_run_reports_backend_and_effective_wall_limit(self):
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "jobs" / "modal_phase3a_retrieval_smoke.py"),
                "--backend",
                "hpc",
                "--dry-run",
                "--work-root",
                str(ROOT),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        payload = json.loads(result.stdout)
        self.assertEqual(payload["backend"], "HPC")
        self.assertEqual(payload["requested_resources"]["gpu"], "H200")
        self.assertEqual(payload["requested_resources"]["cpu"], 4)
        self.assertEqual(payload["requested_resources"]["max_wall_minutes"], 30)
        self.assertFalse(payload["attestation_present"])

    def test_hpc_slurm_wrapper_declares_approved_resources_and_offline_inputs(self):
        script = (ROOT / "jobs" / "hpc_phase3a_retrieval_smoke.sbatch").read_text()
        for directive in (
            "#SBATCH --partition=gpu_h200_8",
            "#SBATCH --cpus-per-task=4",
            "#SBATCH --gres=gpu:nvidia_h200_nvl:1",
            "#SBATCH --mem=32G",
            "#SBATCH --time=00:30:00",
        ):
            self.assertIn(directive, script)
        self.assertIn("export HF_HUB_OFFLINE=1", script)
        self.assertIn("--backend hpc", script)
        self.assertNotIn("PHASE3A_ATTESTATION", script)


if __name__ == "__main__":
    unittest.main()
