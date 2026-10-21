# Phase 3B Gate 1

This directory is a CPU-only, annotation-independent staging area. It uses
the saved Phase 3A source frames, the five cached mask archives needed for the
two diagnostic videos, and the Gate 0 connected-component CSV. It contains no
new model output and does not use the evaluation-only file under
`annotations/` to construct candidates.

The prepared pools contain question-conditioned and fixed-vocabulary
8-connected components, original unsplit masks, and deterministic 20%/40%
context rectangles for the parking lot and baseball field, including the
fixed-vocabulary field proposals. Each candidate has
its source frame index, timestamp, source archive/member, binary pixel mask,
proposal box, and padded crop box.

CPU-only preflight:

```bash
python3 jobs/modal_phase3b_gate1.py --dry-run
```

The preflight validates the prepared manifest, candidate masks, provenance,
exact query phrases, and the A100-40GB/4-vCPU/16-GiB/10-minute resource gate. It
never imports Modal or loads a model. The current workspace intentionally has
no local SigLIP2 snapshot; the deferred Modal function downloads only the
public commit `022b6f71160ffb0169ca4709e2d7e25be659598a` inside the job and
verifies it with `code/build_feature_cache.py` before model load. Download,
startup, model load, and ranking all count against the 10-minute cap.

After a separate approval of the remote download and paid A100 allocation, the
deferred job path requires the active Modal-enabled Python profile:

```bash
/tmp/reva-phase3a-modal-venv/bin/python jobs/modal_phase3b_gate1.py --launch
```

It writes `execution_manifest.json`, `rankings.json`,
`random_baseline.json`, `runtime.json`, `vram.json`, `storage.json`, and
`modal_report.json` under this directory without overwriting existing outputs.
Correct-instance metrics remain `uncertain/N/A` until human target identity is
resolved.
