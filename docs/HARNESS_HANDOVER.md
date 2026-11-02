# Harness handover — continue TG remediation (2026-11-01)

Single reference for a **replacement agent or automation harness** to resume ReVA work without re-deriving context from chat. Read [`AGENTS.md`](../AGENTS.md) first; this file adds **runtime paths**, **sync truth**, **what is closed**, and **exact next steps** from the last user directive.

**Last user intent (in order):**

1. Take the Gemini / PDF *Temporal Grounding Remediation Plan* seriously (papers are real even without inline URLs).
2. **Track A (training-free / fast): run on Brev** — visibility contract (already probed), PATK view reweight, CoTR timestamp burn-in, DSE-VTG-style MSF + gated fusion (needs extra embedder).
3. **Track B (training-inclusive): run on HPC** — VTG-LLM-style timestamp tokens, BiPS-style GRPO (if promoted), Charades interval SFT pilot, dual-stream InternVideo TG branch.
4. **Before executing:** this handover file exists; prior turn stopped at usage limit **before** any new Track A/B code or jobs were launched.

**Prior chat transcript:** `/home/soham/.cursor/projects/home-soham-ReVA/agent-transcripts/0611999b-b663-4cc1-98f7-2ae7838212b6/0611999b-b663-4cc1-98f7-2ae7838212b6.jsonl`

**External plan PDF (user-local):** `/home/soham/Downloads/ReVA Temporal Grounding Remediation Plan.pdf`

**TG error analysis (interactive):** `/home/soham/.cursor/projects/home-soham-ReVA/canvases/temporal-grounding-errors.canvas.tsx`

---

## Non-negotiable rules

| Rule | Detail |
| --- | --- |
| No GPU train/score at scale without **explicit user approval** | Includes full valQ/valT/val rescoring and new Slurm/Brev training. |
| Do **not** overwrite `runs/` score JSONL | New arms → new paths under `runs/<new_run_name>/`. |
| Do **not** change evaluation math | Use `code/evaluate.py`; refresh registry via `python3 code/build_evaluation_registry.py`. |
| Do **not** claim test wins without EXP-028–030 provenance | See [`docs/research_status.md`](research_status.md). |
| One-off experiment code | Lives under `archive/implementation/`; keep `code/score.py` as Stage A/B core unless user asks to extend it. |

---

## Task and baseline (EXP-001)

- **Task:** 4-option UAV VideoQA; one letter A–D per question.
- **Model:** Qwen3-VL-8B-Instruct + **language LoRA r16** (frozen for all “training-free” arms).
- **Frames:** 16 legacy sparse frames (`code/common.py` `MAX_FRAMES`, `DEFAULT_FRAME_SAMPLING=legacy`).
- **TTA:** offsets `{0, 0.5}` × **2** cyclic option rotations → **4 views**; metric uses aggregated `pred` in JSONL (`docs/evaluation.md`).
- **Adapter (canonical):** `runs/stageA_q3vl8b_r16/final_adapter` (on HPC under `$WORK/runs/...`).

| Split | Overall (TTA) | Temporal Grounding (task slice) |
| --- | --- | --- |
| valQ (595) | 498/595 (83.70%) | 51/69 (73.91%) |
| valT (1,059) | 885/1,059 (83.57%) | 99/133 (74.44%) |
| val (2,000) | 1,757/2,000 (87.85%) | 243/320 (75.94%) |

**Baseline score JSONL (local, do not overwrite):** `runs/stageA_q3vl8b_r16/legacy/scores_{valQ,valT,val}.jsonl`

---

## Why TG fails (actionable failure modes)

Use these labels when mapping interventions (full stats in TG canvas + Phase 0 review):

1. **Boundary discretization** — 16 uniform samples ≈ 0.4 s spacing vs **0.5 s** adjacent options; **~56–70%** of numeric TG errors pick the **neighboring** option.
2. **Visibility definition** — annotations lack a formal partial-vs-full visibility rule; first/last appearance questions disagree with model at boundaries.
3. **Left-censoring** — events already active at `t=0` often labeled or predicted with delayed onset.
4. **Small movers in nadir UAV** — vehicles/people TG much weaker than landmarks (~62% vs ~87% on valT in prior analysis).
5. **Low confidence on errors** — wrong TG answers tend to lower letter-logit margin (~0.58 vs ~0.83 correct).

**Do not re-run as “new ideas” without reading closed EXPs:** progress bar + ORB motion (032–034), spatial-contract prompt (035), REZE fusion (036), text timestamps (008), denser sampling (003), A.I.R. selector (031), dual-stream without new gate (archived configs only).

---

## Literature stack (verified; map to ReVA, not copy paper protocol)

| Method | arXiv / venue | ReVA use (Track) | Notes |
| --- | --- | --- | --- |
| **DSE-VTG** | [2609.08850](https://arxiv.org/abs/2609.08850) | A (+ optional B) | MSF + query TTA on similarity curves; needs **Qwen3-VL-Embedding-8B** (or reimplementation). No public code yet — implement from paper. |
| **CoTR** | [2604.22226](https://arxiv.org/abs/2604.22226) | **A (Brev)** | **Visual** `mm:ss` / `[0.xx s]` burn-in top-right on each of 16 frames — distinct from EXP-008 *text* timestamps and EXP-032 bottom progress bar. |
| **BiPS** | [2512.22120](https://arxiv.org/abs/2512.22120) | B (HPC) | Real method is **GRPO training** on evidence views — not the same as EXP-035 prompt-only “visibility contract”. |
| **PATK / phase-aligned views** | ActivityNet / VTimeLLM line | **A (CPU/Brev prep)** | Reweight the **four TTA views** by proximity of sampled frame times to each option’s numeric anchor — **no extra GPU** if computed from existing JSONL + manifest times. |
| **BoT-SORT GMC** | MMOT 2025 | A (CPU-heavy) | Ego-motion compensation for tracks/motion state; EXP-032 ORB one-liner is weaker — prefer GMC evidence if pursuing motion. |
| **SAHI** | [2202.06934](https://arxiv.org/abs/2202.06934) | A (optional) | Sliced high-res patches for small objects — geometry kill switch on same videos. |
| **VTG-LLM** | [2405.13382](https://arxiv.org/abs/2405.13382) | **B (HPC)** | Absolute-time / sequence-time tokens in **fine-tuning** — only after Track A measured; EXP-010 **gate_failed** on train label quality. |

---

## Runtime environments

### Local workspace (Cursor / dev)

| Item | Path |
| --- | --- |
| Repo | `/home/soham/ReVA` |
| Git | `main` @ **`83060b728a78f996a6a8b45595043235a2fc5b08`** (pushed to `origin/main`) |
| Data revision pin | `data/dataset_revision.json` → `06224fa1b42e5e35677a8b7f2145e06307a9a4ee`, 1047 clips |
| Data tree | `data/ReVA/`, `data/splits/`, `data/manifest.json` (gitignored) |
| Python checks | `python3 code/build_evaluation_registry.py`; `python3 -m py_compile code/*.py`; `bash tests/test_jobs.sh`; `python3 -m unittest discover -s tests -p 'test_*.py' -q` |
| Overlay unit tests (archived) | `python3 -m unittest discover -s archive/implementation/tests -p 'test_*.py' -q` |

### HPC (BITS Hyderabad)

| Item | Value |
| --- | --- |
| SSH | `ambpdc@hpc.bits-hyderabad.ac.in` (BatchMode OK) |
| **`WORK`** | `/scratch/ambpdc/soham/ReVA` |
| Venv | `$WORK/envs/reva/bin/activate` |
| Slurm partition / GRES | `gpu_h200_8`, `--gres=gpu:nvidia_h200_nvl:1` (see `jobs/train.sbatch`, `jobs/score.sbatch`) |
| Offline HF | `HF_HUB_OFFLINE=1`, `HF_HOME=$WORK/hf_cache` |
| Stage A adapter | Present: `$WORK/runs/stageA_q3vl8b_r16/final_adapter` |
| Git on cluster | **No `.git` tree** on `$WORK` — sync code via `rsync` or clone; **`dataset_revision.json` matches local** (verified 2026-11-01) |
| Commit on cluster | **Not** `83060b7` unless user re-synced after handover |

**Core Slurm (Stage A/B only in `jobs/`):**

```bash
export WORK=/scratch/ambpdc/soham/ReVA
cd "$WORK"
sbatch jobs/train.sbatch code/configs/stageA.yaml
sbatch jobs/score.sbatch runs/stageA_q3vl8b_r16/final_adapter valQ valT val
# after scores:
python code/evaluate.py runs/stageA_q3vl8b_r16/scores_valQ.jsonl valQ
```

**Track B entry points (archived — copy/sync to `$WORK` before use):**

| Goal | Script / module |
| --- | --- |
| Charades interval SFT + ReVA eval | `archive/implementation/jobs/charades_grounding_data.sbatch`, `charades_grounding_train.sbatch`, `code/charades_grounding.py` |
| Answer-only GRPO (ReMoSense-style) | `archive/implementation/jobs/reva_grpo_train.sbatch`, `reva_grpo_eval.sbatch` |
| Dual-stream InternVideo + Qwen LoRA | `archive/implementation/code/configs/dualstream_temporal_grounding.yaml`, `train.py` with `temporal_stream.enabled: true` |
| Temporal training data audit | `results/temporal_training_data_audit/`, EXP-010 **`gate_failed_no_training`** |

### Brev (NVIDIA A100 80GB)

| Item | Value |
| --- | --- |
| Instance name | `reva-temporal-tg-exp` (`brev ls`; was **STOPPED** 2026-11-01) |
| Start | `brev start reva-temporal-tg-exp` |
| SSH host alias | `reva-temporal-tg-exp` (from `~/.ssh/config` → `~/.brev/ssh_config`, port **16008**) |
| Prefer for automation | `ssh -o ControlPath=none -o ControlMaster=no -o BatchMode=yes reva-temporal-tg-exp '...'` |
| **`WORK`** | `/home/ubuntu/reva` |
| **`REPO`** | `/home/ubuntu/reva/repo` (expected layout from prior pilots) |
| Venv | `/home/ubuntu/reva/venv312` |
| HF cache | `$WORK/hf_cache` |
| Stage A adapter | `$WORK/runs/stageA_q3vl8b_r16/final_adapter` |
| Data | `$WORK/data/ReVA` (from HPC tarball `reva_data.tar` per `brev_dualstream_setup.sh`) |

**Bootstrap (dual-stream / InternVideo — if needed):** `archive/implementation/jobs/brev_dualstream_setup.sh`

---

## Filesystem sync (truth table, 2026-11-01)

| Location | Code vs `main` | Data revision | Notes |
| --- | --- | --- | --- |
| Local `/home/soham/ReVA` | **`83060b7`** | `06224fa…` | Canonical git; clean Stage A `code/score.py`. |
| HPC `$WORK` | **Unknown / not git** | **Matches local** | Adapter present; **re-sync repo** before new archived jobs. |
| Brev `$WORK` | **Unknown** (instance stopped) | Expected match if tarball unchanged | Prior runs: `visual_prompt_pilot`, `spatial_contract`, `reze_tg`, etc. under `$WORK/runs/`. |

**Recommended sync (user-approved):**

```bash
# Example: push local main to Brev repo checkout (adjust if REPO path differs)
rsync -av --exclude data --exclude runs --exclude hf_cache \
  /home/soham/ReVA/ reva-temporal-tg-exp:/home/ubuntu/reva/repo/

# HPC (no git): same from local
rsync -av --exclude data --exclude runs --exclude hf_cache \
  /home/soham/ReVA/ ambpdc@hpc.bits-hyderabad.ac.in:/scratch/ambpdc/soham/ReVA/
```

Never rsync **over** `$WORK/runs/**/*.jsonl` unless the user explicitly wants to replace scores.

---

## Scoring architecture today

### Production (`code/score.py`)

Minimal Stage A scorer — **no** `--progress-bar`, `--camera-motion-hint`, `--spatial-contract`, or `--qa-ids-file`.

```bash
python code/score.py \
  --adapter runs/stageA_q3vl8b_r16/final_adapter \
  --split valQ \
  --out runs/<NEW_ARM>/scores_valQ.jsonl
```

### Pilot overlay / prompt arms (EXP-032–036)

Implementation modules (archived):

- `archive/implementation/code/visual_prompts.py` — progress bar, ORB motion hint
- `archive/implementation/code/spatial_contract.py` — CROSS-style prompt
- `archive/implementation/code/reze.py`, `analyze_reze.py` — REZE evidence fusion
- `archive/implementation/code/analyze_visual_prompt_pilot.py` — paired reports

Brev launchers (still reference **extended** `code/score.py` flags):

- `archive/implementation/jobs/brev_score_visual_prompts.sh`
- `archive/implementation/jobs/brev_score_tg_geometry_bar.sh`
- `archive/implementation/jobs/brev_score_trend_bar.sh`
- `archive/implementation/jobs/brev_score_spatial_reze.sh`

**Harness blocker:** On `main`, those shell scripts **will fail** until you either:

1. Add a dedicated **`archive/implementation/code/score_stageA_pilots.py`** (copy of pre-cleanup scorer behavior) and point Brev jobs at it, **or**
2. Re-attach pilot flags to `code/score.py` behind explicit opt-in flags (user may prefer keeping core minimal).

Completed pilot **results** are already recorded in EXP-032–036 and local `runs/` where present; **new** Track A work should use **new** experiment IDs (EXP-037+), new artifact dirs, and updated `build_evaluation_registry.py` metadata.

### Cohort ID files (reuse for TG-only scoring)

| File | Use |
| --- | --- |
| `results/visual_prompt_pilot/valT_phase0_ids.json` | 60-item valT error-enriched pilot |
| `results/visual_prompt_pilot/tg_geometry_ids.json` | All valQ/valT TG + Geometric Relation |
| `results/visual_prompt_pilot/spatial_contract_ids.json` | TG + three spatial tasks |
| `results/visual_prompt_pilot/reze_tg_ids.json` | TG-only for REZE |
| `results/visual_prompt_pilot/trend_pattern_ids.json` | Trend and Pattern bar pilot |

---

## Closed training-free pilots (do not treat as promoted)

| EXP | Arm | valQ TG | valT TG | Decision |
| --- | --- | --- | --- | --- |
| 032 | bar / motion / both (60 valT) | — | 13/30 bar vs 10/30 base | `not_promoted` |
| 033 | progress bar all TG+geom | 50/69 | 100/133 | `not_promoted` |
| 034 | bar Trend+Pattern | — | slight mix | `not_promoted` |
| 035 | spatial contract prompt | 49/69 | 100/133 | `not_promoted` |
| 036 | REZE α grid | 49/69 (α=0) | 100/133 | `not_promoted` |

Artifacts: `runs/visual_prompt_pilot/`, `runs/visual_prompt_tg_geometry/`, `runs/visual_prompt_trend_pattern/`, `runs/spatial_contract/`, `runs/reze_tg/` (local and/or Brev).

---

## Recommended execution plan (pick up here)

### Track A — Brev (free/fast GPU scoring, CPU prep)

**Order (from remediation ranking after literature review):**

1. **CoTR visual timestamp burn-in** — new pixel channel; implement in `visual_prompts.py` or sibling module; score **TG-only** valQ → valT; compare to Stage A on `reze_tg_ids.json` / full TG denominators.
2. **PATK view reweight** — CPU: parse option times from gold JSON, load existing Stage A JSONL per-view logits if stored, else rescore once and cache view logits; reweight 4 TTA views; no training.
3. **DSE-VTG MSF + uncertainty gate** — download/run **Qwen3-VL-Embedding-8B** on Brev or HPC; fuse with letter logits only when margin low (EXP-036 showed blind fusion α=0 wins).
4. **Visibility / left-censor prompt v2** — only if substantively different from EXP-035; cite as operational T1/T3, not BiPS +7.3%.

**Gates (realistic, not PDF 14-day stacked targets):**

- Primary: **valT TG** 99/133 baseline; require **≥+2** fixes with no **≥3** regressions on TG+geometry cohort unless user accepts tradeoff.
- Secondary: official **val TG** 243/320 — run only after valT arm wins.
- Always report numerators/denominators per split; never mix pilot 60 with full valT.

### Track B — HPC (training)

Only after user approves GPU budget:

1. **Charades-STA interval pilot** — `charades_grounding_*.sbatch` (transfer + paired ReVA eval).
2. **VTG-LLM-style timestamp tokens** — new config + `train.py` changes; respect EXP-010 audit (curated subset, not naive full fit_A TG).
3. **Dual-stream** — `dualstream_temporal_grounding.yaml` + Brev or HPC config; requires InternVideo2 checkpoint and `temporal_stream.enabled`.
4. **BiPS-like GRPO** — `reva_grpo_train.sbatch` path; distinct from prompt contract.

Register new rows: add experiment folder `experiments/EXP-037/experiment.md`, extend `code/build_evaluation_registry.py` `EXPERIMENT_IDS`, run registry builder.

---

## Evaluation and registry

```bash
python3 code/evaluate.py runs/<run>/scores_valT.jsonl valT
python3 code/build_evaluation_registry.py
```

- Metrics CSV: `results/evaluations.csv`
- Human index: `experiments/INDEX.md`
- Comparisons: `results/comparisons/overall.md` (generated)

Task slices: filter JSONL by `task` field == `Temporal Grounding` when reporting TG-only numbers.

---

## Key docs in repo

| Doc | Purpose |
| --- | --- |
| [`docs/CONTEXT.md`](CONTEXT.md) | Experiments + active code map |
| [`docs/architecture.md`](architecture.md) | Pipeline diagram |
| [`docs/evaluation.md`](evaluation.md) | Splits, TTA, metric kinds |
| [`docs/research_status.md`](research_status.md) | One-paragraph status |
| [`archive/implementation/docs/training_free_tg_spatial_shortlist.md`](../archive/implementation/docs/training_free_tg_spatial_shortlist.md) | Pre-remediation shortlist |
| [`results/error_review/REVIEW.md`](../results/error_review/REVIEW.md) | Qualitative error review |
| [`results/temporal_training_data_audit/phase1_report.md`](../results/temporal_training_data_audit/phase1_report.md) | EXP-010 gate |

---

## User approval checklist (ask before doing)

- [ ] Start Brev instance (`brev start reva-temporal-tg-exp`)
- [ ] Full-split GPU scoring (valQ + valT + val) for a new arm
- [ ] Any Slurm training job (Charades, GRPO, dual-stream, VTG-LLM)
- [ ] Download new HF weights (Embedding-8B, InternVideo2) if not on `$WORK/hf_cache`
- [ ] rsync that overwrites remote `runs/` or `final_adapter`

---

## Quick status for the harness

| Work item | Status |
| --- | --- |
| Repo cleanup + EXP-001–036 docs | **Done** (`83060b7`) |
| TG deep-dive + canvas | **Done** |
| Literature verification for remediation PDF | **Done** (see table above) |
| Track A CoTR / PATK / DSE implementation | **CoTR + PATK done** (EXP-037/038); DSE not started |
| Track B HPC jobs submitted | **Charades job 428064** submitted (was PD); prior run artifacts exist under `runs/charades_grounding/` |
| Brev instance | **RUNNING** during CoTR — stop when idle: `brev stop reva-temporal-tg-exp` |
| `code/score.py` pilot flags vs Brev shell scripts | **Fixed** — use `archive/implementation/code/score_stageA_pilots.py` |

**Immediate next command for a harness (after user OK):** `brev start reva-temporal-tg-exp`, rsync repo to Brev, restore or implement pilot scorer, implement **CoTR burn-in**, score TG cohort, write `runs/cotr_tg_burnin/` + `experiments/EXP-037/experiment.md`.
