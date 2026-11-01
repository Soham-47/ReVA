import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import phase8_adapters as A
import phase8_scorer as S


class _Tokenizer:
    def __init__(self, ids):
        self.ids = ids

    def encode(self, text, add_special_tokens=False):
        return list(self.ids[text])


class _Processor:
    def __init__(self):
        self.tokenizer = _Tokenizer({"A": [11], "B": [12], "C": [13], "D": [14]})
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return "rendered"


class _Tensor:
    def __init__(self, *shape):
        self.shape = shape


class _GridProducts:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class _Grid:
    def __init__(self):
        self.cpu_calls = 0
        self.prod_calls = 0

    def detach(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self

    def prod(self, dim):
        self.prod_calls += 1
        self.prod_dim = dim
        return _GridProducts([16, 32])

    def tolist(self):
        return [[1, 4, 4], [2, 4, 4]]


class _PixelValues:
    def __init__(self):
        self.type_calls = []

    def type(self, dtype):
        self.type_calls.append(dtype)
        return self


class _VisionOutput:
    pooler_output = object()


class _Vision:
    dtype = "bfloat16"
    spatial_merge_size = 2

    def __init__(self):
        self.grid = None

    def __call__(self, pixel_values, *, grid_thw, **kwargs):
        self.grid = grid_thw
        self.kwargs = kwargs
        return _VisionOutput()


class _Qwen38Inner:
    def __init__(self):
        self.visual = _Vision()

    def get_image_features(self, **kwargs):
        raise AssertionError("the Phase 8 CPU split patch was not installed")


class _Qwen38Model:
    def __init__(self):
        self.model = _Qwen38Inner()


class _TorchSplit:
    int32 = "int32"

    @staticmethod
    def split(pooler_output, split_sizes):
        return (pooler_output, list(split_sizes))

    @staticmethod
    def tensor(values, *, dtype):
        return ("tensor", list(values), dtype)


class Phase8AdapterTests(unittest.TestCase):
    def test_family_aliases_resolve_to_distinct_model_specs(self):
        q38 = A.resolve_model_spec("qwen3.8-27b")
        qvl = A.resolve_model_spec("qwen3-vl-32b")
        self.assertEqual(q38.family, "qwen38")
        self.assertEqual(q38.model_id, "Qwen/Qwen3.8-27B")
        self.assertEqual(q38.revision, "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0")
        self.assertEqual(qvl.family, "qwen3_vl_32b")
        self.assertEqual(qvl.model_id, "Qwen/Qwen3-VL-32B-Instruct")
        self.assertEqual(qvl.revision, "0cfaf48183f594c314753d30a4c4974bc75f3ccb")
        self.assertNotEqual(q38.loader, qvl.loader)
        self.assertEqual(A.resolve_model_spec("qwen38", revision="pilot-rev").revision, "pilot-rev")

    def test_qwen38_uses_non_thinking_template_and_qwen_vl_does_not(self):
        processor = _Processor()
        messages = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
        A.Qwen38Adapter(A.resolve_model_spec("qwen38")).render_chat(processor, messages)
        self.assertFalse(processor.calls[-1][1]["enable_thinking"])
        self.assertNotIn("chat_template_kwargs", processor.calls[-1][1])
        A.Qwen3VL32BAdapter(A.resolve_model_spec("qwen3vl32")).render_chat(processor, messages)
        self.assertNotIn("chat_template_kwargs", processor.calls[-1][1])

    def test_answer_labels_are_single_distinct_tokens(self):
        processor = _Processor()
        self.assertEqual(A.validate_label_tokens(processor), [11, 12, 13, 14])
        telemetry = A.label_tokenization_telemetry(processor)
        self.assertEqual(telemetry["plain"]["token_ids"], [[11], [12], [13], [14]])
        self.assertFalse(telemetry["leading_space"]["available"])
        processor.tokenizer.ids.update({" A": [21], " B": [22], " C": [23], " D": [24]})
        leading = A.label_tokenization_telemetry(processor)["leading_space"]
        self.assertEqual(leading["token_ids"], [[21], [22], [23], [24]])
        self.assertTrue(leading["distinct"])
        processor.tokenizer.ids["A"] = [11, 12]
        with self.assertRaisesRegex(ValueError, "one token"):
            A.validate_label_tokens(processor)

    def test_qwen38_split_sizes_reduce_a_cpu_grid_copy(self):
        grid = _Grid()
        self.assertEqual(A._cpu_split_sizes(grid, spatial_merge_size=2), [4, 8])
        self.assertEqual(grid.cpu_calls, 1)
        self.assertEqual(grid.prod_calls, 1)
        self.assertEqual(grid.prod_dim, -1)

    def test_qwen38_patch_keeps_encoder_grid_and_cpu_splits_afterward(self):
        model = _Qwen38Model()
        grid = _Grid()
        pixel_values = _PixelValues()
        self.assertTrue(A._patch_qwen38_vision_split(model, _TorchSplit))
        output = model.model.get_image_features(
            pixel_values, image_grid_thw=grid, return_dict=True
        )
        self.assertIs(model.model.visual.grid, grid)
        self.assertEqual(pixel_values.type_calls, ["bfloat16"])
        self.assertEqual(output.pooler_output[1], [4, 8])
        self.assertEqual(grid.cpu_calls, 2)
        self.assertEqual(
            model.model.visual.kwargs,
            {"return_dict": True, "cu_seqlens": ("tensor", [0, 16, 32, 48], "int32")},
        )

    def test_encode_remaps_shown_logits_to_original_option_order(self):
        row = {
            "category": "counting",
            "task": "which",
            "question": "Which object?",
            "options": {"A": "car", "B": "truck", "C": "boat", "D": "none"},
        }
        self.assertEqual(
            S.remap_shown_probabilities([0.1, 0.2, 0.3, 0.4], ["C", "A", "D", "B"]),
            [0.2, 0.4, 0.1, 0.3],
        )
        shown, permutation = S.build_prompt(row, ["C", "A", "D", "B"])
        self.assertEqual(permutation, ["C", "A", "D", "B"])
        self.assertIn("A. boat", shown)

    def test_source_frame_timestamps_prefer_canonical_record_and_fallback_explicitly(self):
        indices = [0, 2, 5]
        info = {"n_frames": 6, "fps": 4.0}
        self.assertEqual(
            S.frame_provenance(indices, info),
            {"frame_indices": indices, "frame_timestamps_s": [0.0, 0.5, 1.25], "timestamp_source": "index_divided_by_fps"},
        )
        canonical = {
            "n_frames": 6,
            "fps": 4.0,
            "frame_timestamps_s": [0.0, 0.2, 0.4, 0.9, 1.1, 1.6],
            "timestamp_source": "presentation_timestamp",
        }
        result = S.frame_provenance(indices, canonical)
        self.assertEqual(result["frame_timestamps_s"], [0.0, 0.4, 1.6])
        self.assertEqual(result["timestamp_source"], "presentation_timestamp")

    def test_input_telemetry_reports_only_available_shapes(self):
        telemetry = S.input_telemetry(
            {
                "input_ids": _Tensor(1, 73),
                "video_grid_thw": _Tensor(1, 16, 16),
                "pixel_values_videos": _Tensor(16, 3, 224, 224),
            }
        )
        self.assertEqual(telemetry["input_ids_shape"], [1, 73])
        self.assertEqual(telemetry["input_token_count"], 73)
        self.assertEqual(telemetry["video_grid_thw_shape"], [1, 16, 16])
        self.assertNotIn("pixel_values_shape", telemetry)

    def test_phase8_slurm_wrapper_is_h200_scoped_and_baseline_safe(self):
        script = (ROOT / "jobs" / "phase8_score.sbatch").read_text()
        for directive in (
            "#SBATCH --partition=gpu_h200_8",
            "#SBATCH --cpus-per-task=12",
            "#SBATCH --gres=gpu:nvidia_h200_nvl:1",
            "#SBATCH --time=11:50:00",
            "#SBATCH --requeue",
        ):
            self.assertIn(directive, script)
        self.assertIn("HF_HOME=", script)
        self.assertIn("code/phase8_scorer.py", script)
        self.assertIn('extra_args+=(--adapter "$PHASE8_ADAPTER")', script)
        self.assertNotIn("code/score.py", script)
        self.assertIn("phase8_*", script)

    def test_phase8_wrapper_has_explicit_pilot_and_full_paths(self):
        script = (ROOT / "jobs" / "phase8_score.sbatch").read_text()
        self.assertIn('mode="${PHASE8_MODE:-full}"', script)
        self.assertIn("phase8_backbone_comparison/pilot_selection.json", script)
        self.assertIn('selection_args=(--qa-ids-file "$pilot_selection")', script)
        self.assertIn("phase8_pilot_${family_key}", script)
        self.assertIn("phase8_pilot_*", script)
        self.assertIn('echo "Phase 8 $mode scoring complete', script)

    def test_phase8_download_is_cpu_only_and_pinned(self):
        script = (ROOT / "jobs" / "phase8_download.sbatch").read_text()
        self.assertIn("#SBATCH -p compute", script)
        self.assertNotIn("--gres=", script)
        self.assertIn("Qwen/Qwen3.8-27B", script)
        self.assertIn("1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0", script)
        self.assertIn("Qwen/Qwen3-VL-32B-Instruct", script)
        self.assertIn("0cfaf48183f594c314753d30a4c4974bc75f3ccb", script)
        self.assertIn('HF_HOME="${HF_HOME:-$WORK/hf_cache}"', script)
        self.assertIn('cache_dir = os.environ["HF_HUB_CACHE"]', script)
        self.assertIn("local_files_only=True", script)
        self.assertIn("AutoProcessor.from_pretrained", script)
        self.assertNotIn("code/score.py", script)

    def test_phase8_manifest_is_family_specific_and_resume_contract_is_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "phase8_qwen38.jsonl"
            manifest = S.expected_manifest(
                family="qwen38",
                model_id="Qwen/Qwen3.8-27B",
                revision="rev-a",
                split="valQ",
                split_checksum="digest",
                offsets=[0.0, 0.5],
                rots=2,
                frame_sampling="legacy",
                shard=0,
                nshards=1,
            )
            S.ensure_output_contract(out, manifest)
            self.assertEqual(json.loads(S.manifest_path(out).read_text())["output_kind"], "phase8_pretrained_score")
            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                S.ensure_output_contract(out, {**manifest, "model_revision": "rev-b"})

    def test_phase8_manifest_records_peft_adapter_identity(self):
        manifest = S.expected_manifest(
            family="qwen38",
            model_id="Qwen/Qwen3.8-27B",
            revision="rev-a",
            split="valQ",
            split_checksum="digest",
            offsets=[0.0, 0.5],
            rots=2,
            frame_sampling="legacy",
            shard=0,
            nshards=1,
            adapter={"kind": "peft", "path": "/tmp/adapter", "sha256": "abc"},
        )
        self.assertEqual(manifest["adapter"]["kind"], "peft")
        args = S._parser().parse_args(
            ["--family", "qwen38", "--split", "valQ", "--out", "scores.jsonl", "--adapter", "a"]
        )
        self.assertEqual(args.adapter, "a")

    def test_multi_split_parser_uses_one_template(self):
        args = S._parser().parse_args([
            "--family", "qwen38",
            "--split", "valQ", "valT",
            "--out", "runs/phase8/scores_{split}.jsonl",
        ])
        self.assertEqual(args.split, ["valQ", "valT"])
        self.assertEqual(S._latency_summary([1.0, 3.0, 2.0])["median"], 2.0)
        self.assertEqual(S.input_telemetry({})["video_grid_thw_values"], None)

    def test_h200_wrapper_invokes_one_multi_split_scorer_process(self):
        script = (ROOT / "jobs" / "phase8_score.sbatch").read_text()
        self.assertIn('--split "$@"', script)
        self.assertIn('scores_${family_key}_{split}.jsonl', script)
        self.assertEqual(script.count("code/phase8_scorer.py"), 1)

    def test_selection_and_precision_are_frozen_in_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            selection = Path(temp) / "pilot.json"
            selection.write_text('{"splits":{"valQ":{"ids":["q1"]}}}\n')
            metadata = S._selection_metadata(
                str(selection), label="QA IDs", split="valQ", selected_question_count=1
            )
            manifest = S.expected_manifest(
                family="qwen38",
                model_id="Qwen/Qwen3.8-27B",
                revision="rev-a",
                split="valQ",
                split_checksum="digest",
                offsets=[0.0, 0.5],
                rots=2,
                frame_sampling="legacy",
                shard=0,
                nshards=1,
                selection={"qa_ids_file": metadata, "video_paths_file": None},
            )
            self.assertEqual(manifest["precision"], "bfloat16")
            self.assertEqual(manifest["selection"]["qa_ids_file"]["selected_question_count"], 1)
            selection.write_text('{"splits":{"valQ":{"ids":["q2"]}}}\n')
            changed = S._selection_metadata(
                str(selection), label="QA IDs", split="valQ", selected_question_count=1
            )
            self.assertNotEqual(metadata["sha256"], changed["sha256"])
            out = Path(temp) / "scores.jsonl"
            S.ensure_output_contract(out, manifest)
            tokenization = A.label_tokenization_telemetry(_Processor())
            S.bind_manifest_tokenization(out, tokenization)
            self.assertEqual(
                json.loads(S.manifest_path(out).read_text())["answer_labels"]["tokenization"],
                tokenization,
            )
            S.ensure_output_contract(out, manifest)
            changed_manifest = {**manifest, "selection": {"qa_ids_file": changed, "video_paths_file": None}}
            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                S.ensure_output_contract(out, changed_manifest)
            with self.assertRaisesRegex(RuntimeError, "tokenization mismatch"):
                S.bind_manifest_tokenization(out, {"plain": {"token_ids": [[99], [12], [13], [14]]}})
            with self.assertRaisesRegex(RuntimeError, "manifest mismatch"):
                S.ensure_output_contract(out, {**manifest, "model_revision": "other"})


if __name__ == "__main__":
    unittest.main()
