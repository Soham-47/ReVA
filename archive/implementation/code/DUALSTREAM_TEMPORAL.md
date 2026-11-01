# Opt-in dual-stream temporal branch

This branch is isolated from the production `train.py` defaults and `score.py`
CLI. It keeps Qwen3-VL as the appearance stream, adds a frozen OpenGVLab
InternVideo2 stage-2 1B encoder for dense motion segments, and trains a small
segment projector plus relative-time token rows.

## Runtime prerequisites

Provide the official InternVideo2 source locally; the branch never downloads
it implicitly. The official multi-modality package must be importable, for
example:

```bash
export PYTHONPATH=/path/to/InternVideo/InternVideo2/multi_modality:$PYTHONPATH
```

Place the approved local `InternVideo2-stage2_1b-224p-f4.pt` checkpoint at the
`temporal_stream.checkpoint` path in
`code/configs/dualstream_temporal.yaml`. The loader records and later verifies
its size and SHA-256 identity. The f4 checkpoint is interpolated to the
configured eight frames per segment; the full motion view uses 12 segments ×
8 frames = 96 frames.

## Approved training/scoring commands

After an operator has populated a runtime `WORK` tree and approved GPU use:

```bash
WORK=/runtime/reva python code/train.py code/configs/dualstream_temporal.yaml
WORK=/runtime/reva python code/score_dualstream_temporal.py \
  --adapter runs/dualstream_temporal_q3vl8b_r16/final_adapter \
  --split valT --out runs/dualstream_temporal_q3vl8b_r16/valT_dualstream.jsonl
WORK=/runtime/reva python code/evaluate.py \
  runs/dualstream_temporal_q3vl8b_r16/valT_dualstream.jsonl valT
```

The scorer is local-only and writes the existing evaluation fields (`qa_id`,
`p`, `pred`, `views`) plus decoded `time_seconds` per view. It decodes the
relative-time prefix before the shown A–D answer and then maps the answer back
through the option permutation.

## Label and validation limitations

Temporal targets are derived only from `train_*` rows and only from the correct
option. On the current `fit_A` temporal-grounding scan, the strict parser
accepted roughly 681 point and 381 interval labels and masked roughly 620
ambiguous, duration-only, or unparseable rows. Validation/test rows never
create supervision; their generated time prefix is auxiliary output.

The temporal wrapper currently fails closed if the installed Qwen generation
path drops precomputed `mm_encoder_outputs` between prefill and cached decode.
Automatic resume from an existing Trainer checkpoint is also disabled until a
runtime Transformers/PEFT integration smoke test is approved; intermediate
checkpoints still carry the temporal bundle for later validation.

## Temporal Grounding-only experiment

Use the isolated config below to train only the 1,682 `train_*` rows whose
`task` is exactly `Temporal Grounding`; the ordinary dual-stream config keeps
its broader training scope.

```bash
WORK=/runtime/reva python code/train.py \
  code/configs/dualstream_temporal_grounding.yaml
WORK=/runtime/reva python code/score_dualstream_temporal.py \
  --adapter runs/dualstream_temporal_grounding_q3vl8b_r16/final_adapter \
  --split valT \
  --task 'Temporal Grounding' \
  --out runs/dualstream_temporal_grounding_q3vl8b_r16/valT.jsonl
WORK=/runtime/reva python code/evaluate_temporal.py \
  runs/dualstream_temporal_grounding_q3vl8b_r16/valT.jsonl \
  --json-out /tmp/reva_temporal_metrics.json
```

The standalone evaluator reads only `valT` Temporal Grounding rows and the
first scorer view by default. It reports gold/prediction parse coverage, point
MAE and hit rates at 0.5/1/2 seconds, interval mean temporal IoU and recall at
IoU 0.3/0.5/0.7, plus answer accuracy. Missing or wrong-shape predictions are
coverage failures and count as misses for threshold recalls; MAE/mean-IoU are
computed only on type-matched pairs. Pass `--view-index` to inspect another
scorer view, or `--task` to audit a different task without touching
`evaluate.py`.
