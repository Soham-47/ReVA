import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from phase7b_source_evidence import enrich_report, render_updated_report, validate_overlay_binding


class Phase7BSourceEvidenceTests(unittest.TestCase):
    def test_checksum_manifest_and_overlay_frames_bind_to_cache(self):
        source_manifest = json.loads(
            (ROOT / "runs/phase7B_cache_validation/source_checksum_manifest.json").read_text()
        )
        overlay_manifest = json.loads(
            (ROOT / "runs/phase7B_cache_validation/contact_sheet_manifest.json").read_text()
        )
        self.assertEqual(source_manifest["dataset_revision"], "06224fa1b42e5e35677a8b7f2145e06307a9a4ee")
        self.assertEqual(len(source_manifest["entries"]), 5)
        self.assertTrue(all(row["hpc_verification_status"] == "match" for row in source_manifest["entries"]))
        self.assertEqual(sum(row["status"] == "staged" for row in overlay_manifest["entries"]), 5)
        validate_overlay_binding(
            source_manifest,
            overlay_manifest,
            ROOT / "runs/phase3_explicit_geometry/full/grounding_cache",
        )

    def test_full_staging_report_has_no_stale_source_blocker(self):
        report = json.loads((ROOT / "runs/phase7B_cache_validation/report.json").read_text())
        source_manifest = json.loads(
            (ROOT / "runs/phase7B_cache_validation/source_checksum_manifest.json").read_text()
        )
        overlay_manifest = json.loads(
            (ROOT / "runs/phase7B_cache_validation/contact_sheet_manifest.json").read_text()
        )

        enriched = enrich_report(report, source_manifest, overlay_manifest)
        rendered = render_updated_report(enriched)

        self.assertEqual(enriched["source_staging"]["staged_video_count"], 5)
        self.assertEqual(enriched["source_staging"]["requested_video_count"], 5)
        self.assertIn("- Clips staged locally: 5/5", rendered)
        self.assertIn("All 5/5 mapped caches have independently verified source checksums", rendered)
        self.assertIn("all 5 source clips are staged with rendered contact sheets", rendered)
        self.assertNotIn("remaining four", rendered)
        self.assertNotIn("Only source clips available in this checkout", rendered)
        self.assertEqual(
            enriched["limitations"].count(
                "HPC checksum verification is externally attested at the pinned dataset revision; "
                "this checkout could not re-read the HPC mount."
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
