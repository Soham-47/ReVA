from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/phase4_qcsrr"
OUTPUT = RUN / "diagnostics/replicate_prediction_metrics.json"
app = modal.App("reva-phase4-prediction-metrics")
image = modal.Image.debian_slim(python_version="3.11")


@app.function(
    image=image,
    cpu=1,
    memory=1024,
    timeout=180,
    secrets=[modal.Secret.from_name("reva-phase3-replicate")],
)
def _get_metrics(prediction_ids: list[str]) -> list[dict]:
    import os

    token = os.environ["REPLICATE_API_TOKEN"]
    rows = []
    for prediction_id in prediction_ids:
        request = urllib.request.Request(
            f"https://api.replicate.com/v1/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            prediction = json.load(response)
        rows.append({
            "id": prediction["id"],
            "status": prediction["status"],
            "version": prediction.get("version"),
            "created_at": prediction.get("created_at"),
            "completed_at": prediction.get("completed_at"),
            "metrics": prediction.get("metrics"),
        })
    return rows


@app.local_entrypoint()
def main() -> None:
    ids = set()
    for relative in (
        "results_resumed-001/replicate_manifest.json",
        "results_resumed/replicate_manifest.json",
    ):
        manifest = json.loads((RUN / relative).read_text())
        for record in manifest["records"]:
            ids.update(record.get("prediction_ids") or ([record["prediction_id"]] if record.get("prediction_id") else []))
    ordered_ids = sorted(ids)
    with app.run():
        rows = _get_metrics.remote(ordered_ids)
    assert len(rows) == len(ordered_ids)
    assert {row["id"] for row in rows} == ids
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"prediction_count": len(rows), "predictions": rows}, indent=2) + "\n")
    print(json.dumps({"status": "complete", "prediction_count": len(rows), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
