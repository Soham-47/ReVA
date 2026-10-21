"""Run local validation for the Phase 3 pilot and record blockers safely."""

from __future__ import annotations

import json
import math
import os
import shutil
import tomllib
from pathlib import Path
from typing import Any

from geometry_relations import relation_context


ROOT = Path(__file__).resolve().parents[2]
RUN = Path(__file__).resolve().parent
MANIFEST = RUN / "selection_manifest.json"
SPLITS = ("valQ", "valT")


def _load(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _baseline_check(payload: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for split in SPLITS:
        path = ROOT / "runs" / "stageA_q3vl8b_r16" / "legacy" / f"scores_{split}.jsonl"
        by_id = {}
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                by_id[row["qa_id"]] = row
        selected = payload["selected_questions"][split]
        for row in selected:
            score = by_id.get(row["qa_id"])
            if score is None:
                raise RuntimeError(f"baseline missing {split}/{row['qa_id']}")
            if len(score.get("p", [])) != 4 or len(score.get("views", [])) != 4:
                raise RuntimeError(f"baseline vector/view count invalid for {split}/{row['qa_id']}")
            for vector in [score["p"], *(view.get("p", []) for view in score["views"])]:
                if len(vector) != 4 or any(not math.isfinite(float(value)) for value in vector):
                    raise RuntimeError(f"baseline probability vector invalid for {split}/{row['qa_id']}")
                if abs(sum(float(value) for value in vector) - 1.0) > 2e-4:
                    raise RuntimeError(f"baseline probability vector does not sum to one for {split}/{row['qa_id']}")
            checks.append({"split": split, "qa_id": row["qa_id"], "status": "complete"})
    return {"question_count": len(checks), "status": "complete", "records": checks}


def _reuse_check(payload: dict[str, Any]) -> dict[str, Any]:
    records = [row for rows in payload["selected_questions"].values() for row in rows]
    old_roots = [
        ROOT / "runs" / "sam3_pilot_stageA_replicate" / "overlay_cache",
        ROOT / "runs" / "sam3_remaining17_stageA_replicate" / "overlay_cache",
    ]
    reusable: list[str] = []
    rejected: list[dict[str, Any]] = []
    for row in records:
        candidate = next((root / f"{row['qa_id']}.json" for root in old_roots if (root / f"{row['qa_id']}.json").is_file()), None)
        if candidate is None:
            rejected.append({"qa_id": row["qa_id"], "reason": "no prior SAM3 cache"})
            continue
        cached = _load(candidate)
        reasons = []
        if cached.get("video_path") != row["video_path"]:
            reasons.append("video path mismatch")
        if list(cached.get("prompts", [])) != list(row["sam3_prompt_entities"]):
            reasons.append("question-only prompt list mismatch")
        # Prior SAM3 caches are full-clip masks. The new pilot's relation text
        # is defined on the exact legacy 0/0.5 frame views, so do not silently
        # treat a full-clip cache as an identical frame-policy cache.
        reasons.append("prior cache uses full-clip frame policy; pilot requires legacy frame policy")
        rejected.append({"qa_id": row["qa_id"], "path": str(candidate.relative_to(ROOT)), "reason": "; ".join(reasons)})
    return {"reusable_count": len(reusable), "rejected_count": len(rejected), "reusable": reusable, "rejected": rejected}


def _relation_self_check() -> dict[str, Any]:
    text, audit = relation_context(
        [
            {"label": "car", "frame_index": 0, "confidence": 0.8, "box": {"left": 0.1, "top": 0.1, "right": 0.3, "bottom": 0.3}},
            {"label": "bridge", "frame_index": 0, "confidence": 0.8, "box": {"left": 0.6, "top": 0.5, "right": 0.8, "bottom": 0.7}},
        ],
        ["car", "bridge"],
    )
    if "metric distance" not in text or audit["pairs"][0]["status"] != "reliable_2d":
        raise RuntimeError("relation context self-check failed its 2D-only contract")
    return {"status": "complete", "sample_context": text, "sample_audit": audit}


def _source_video_check(payload: dict[str, Any], *, remote_staging_supported: bool) -> dict[str, Any]:
    paths = sorted({row["video_path"] for rows in payload["selected_questions"].values() for row in rows})
    missing = [path for path in paths if not (ROOT / "data" / "ReVA" / path).is_file()]
    status = "complete" if not missing else ("remote_staging_required" if remote_staging_supported else "blocked")
    return {
        "selected_video_count": len(paths),
        "missing_video_count": len(missing),
        "missing_examples": missing[:8],
        "remote_staging_supported": remote_staging_supported,
        "status": status,
    }


def _modal_runtime_check() -> dict[str, Any]:
    """Detect a configured Modal profile without reading credential values."""

    config_candidates = []
    configured_path = os.environ.get("MODAL_CONFIG_PATH")
    if configured_path:
        config_candidates.append(Path(configured_path).expanduser())
    config_candidates.append(Path.home() / ".modal.toml")
    profile_path = next((path for path in config_candidates if path.is_file()), None)
    active_profiles: list[str] = []
    if profile_path is not None:
        try:
            config = tomllib.loads(profile_path.read_text())
        except (OSError, tomllib.TOMLDecodeError):
            config = {}
        for name, section in config.items():
            if isinstance(section, dict) and section.get("active") is True and section.get("token_id") and section.get("token_secret"):
                active_profiles.append(str(name))
    cli_candidates = [shutil.which("modal")]
    cli_candidates.extend(str(path) for path in Path.home().glob(".cache/uv/archive-v0/*/bin/modal"))
    cli = next((candidate for candidate in cli_candidates if candidate and Path(candidate).is_file()), None)
    return {
        "config_path_present": profile_path is not None,
        "active_profile_count": len(active_profiles),
        "active_profile_names": active_profiles,
        "cli_path_present": cli is not None,
        "cli_path": cli,
        "authenticated_path_ready": bool(active_profiles and cli),
    }


def run() -> dict[str, Any]:
    payload = _load(MANIFEST)
    baseline = _baseline_check(payload)
    reuse = _reuse_check(payload)
    relation = _relation_self_check()
    modal_runtime = _modal_runtime_check()
    source_videos = _source_video_check(
        payload, remote_staging_supported=modal_runtime["authenticated_path_ready"]
    )
    runtime = {
        "modal_cli": modal_runtime["cli_path"],
        "slurm_sbatch": shutil.which("sbatch"),
        "replicate_python_package": None,
        "replicate_token_present": bool(os.environ.get("REPLICATE_API_TOKEN")),
        "modal_token_id_present": bool(os.environ.get("MODAL_TOKEN_ID")),
        "modal_token_secret_present": bool(os.environ.get("MODAL_TOKEN_SECRET")),
    }
    try:
        import replicate  # noqa: F401
    except ModuleNotFoundError:
        runtime["replicate_python_package"] = False
    else:
        runtime["replicate_python_package"] = True
    blockers = [
        "Modal CLI/profile is unavailable" if not modal_runtime["authenticated_path_ready"] else None,
        "Replicate API token is not present" if not runtime["replicate_token_present"] else None,
        "G1 GroundingDINO cache is not staged for this new 60-question manifest",
        "G0/G1/G2 VLM score outputs are not available locally",
        f"{source_videos['missing_video_count']} of {source_videos['selected_video_count']} selected source videos are absent locally and remote staging is unavailable" if source_videos["missing_video_count"] and not source_videos["remote_staging_supported"] else None,
    ]
    blockers = [item for item in blockers if item]
    return {
        "schema_version": 1,
        "pilot": payload["pilot"],
        "status": "blocked_before_api_calls" if blockers else "ready_for_tranche",
        "scope": payload["scope"],
        "baseline": baseline,
        "relation_schema": relation,
        "source_videos": source_videos,
        "prior_cache_reuse": reuse,
        "runtime": runtime,
        "modal_runtime": modal_runtime,
        "cost_gate": payload["cost_gate"],
        "blockers": blockers,
        "api_calls_made": {"replicate": 0, "modal": 0},
    }


def main() -> int:
    status = run()
    (RUN / "execution_status.json").write_text(json.dumps(status, indent=2) + "\n")
    (RUN / "results_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "pilot": "phase3_explicit_geometry",
                "status": "not_run",
                "scope": "error-enriched pilot; not split-wide benchmark accuracy",
                "arms": {
                    "G0": {"status": "frozen_baseline_copy", "question_count": status["baseline"]["question_count"]},
                    "G1": {"status": "not_scored", "question_count": 0},
                    "G2": {"status": "not_scored", "question_count": 0},
                },
                "paired_analysis": None,
                "evidence_coverage": None,
                "qualitative_review": "not performed because no changed G1/G2 cases exist",
                "runtime_resource_use": {"replicate_calls": 0, "modal_calls": 0},
                "blockers": status["blockers"],
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# Phase 3 explicit geometry pilot preflight",
        "",
        f"Status: **{status['status']}**.",
        "",
        "The local baseline vectors and relation schema passed validation. No paid Replicate/SAM3 call or Modal scoring function call was made; the separate CLI parser/image check did not invoke the scoring function.",
        "",
        "## Cost gate",
        "",
        f"The conservative full-set estimate is ${status['cost_gate']['full_conservative_cost_usd']:.3f} for {status['cost_gate']['full_call_count']} SAM 3 calls, below the hard ${status['cost_gate']['cap_usd']:.2f} cap. The 4-question tranche estimate is ${status['cost_gate']['tranche_conservative_cost_usd']:.3f}; after a successful tranche, the remaining 56 questions are estimated at ${status['cost_gate']['remaining_conservative_cost_usd']:.3f} for {status['cost_gate']['remaining_call_count_after_tranche']} new calls.",
        "",
        "## Checks",
        "",
        f"- Frozen G0 baseline: {status['baseline']['question_count']} selected questions with four probability vectors and four views each.",
        f"- Relation schema self-check: {status['relation_schema']['status']}; all statements are bounded to normalized 2D image-plane evidence.",
        f"- Modal execution path: {'ready' if status['modal_runtime']['authenticated_path_ready'] else 'unavailable'} (profile names only; credential values were not read into output).",
        f"- Source videos: {status['source_videos']['selected_video_count']} selected, {status['source_videos']['missing_video_count']} absent locally; remote staging is {'available' if status['source_videos']['remote_staging_supported'] else 'unavailable'}.",
        f"- Prior SAM 3 cache reuse: {status['prior_cache_reuse']['reusable_count']} reusable, {status['prior_cache_reuse']['rejected_count']} rejected by the exact prompt/frame-policy gate.",
        "",
        "## Blockers",
        "",
    ]
    lines.extend(f"- {item}" for item in status["blockers"])
    lines += [
        "",
        "The run stops before spending because no Replicate credential is available to the authenticated Modal path. This is an error-enriched pilot plan, not split-wide benchmark accuracy.",
        "",
    ]
    (RUN / "preflight_report.md").write_text("\n".join(lines))
    print(json.dumps({"status": status["status"], "blockers": status["blockers"], "cost_gate": status["cost_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
