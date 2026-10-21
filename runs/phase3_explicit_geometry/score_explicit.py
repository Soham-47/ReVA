"""Score one explicit-geometry arm with the unchanged Stage A adapter."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
import common as C


def _move(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def _read_contexts(path: Path) -> dict[str, dict[str, Any]]:
    contexts: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            qa_id = record.get("qa_id")
            if not isinstance(qa_id, str) or qa_id in contexts:
                raise ValueError(f"invalid or duplicate context record in {path}")
            if record.get("relation_schema") != "explicit-2d-image-plane-v1":
                raise ValueError(f"unsupported relation schema in {path}: {qa_id}")
            contexts[qa_id] = record
    return contexts


def _usable_context(context: dict[str, Any] | None) -> bool:
    """Allow evidence text only when every audited pair is reliable 2D evidence."""

    if not isinstance(context, dict) or not isinstance(context.get("context"), str):
        return False
    audit = context.get("audit")
    pairs = audit.get("pairs") if isinstance(audit, dict) else None
    return bool(pairs) and all(
        isinstance(pair, dict) and pair.get("status") == "reliable_2d" for pair in pairs
    )


def _self_check_context_gate() -> None:
    supported = {"context": "reliable relation", "audit": {"pairs": [{"status": "reliable_2d"}]}}
    unsupported = {"context": "fall back to baseline", "audit": {"pairs": [], "status": "unsupported_subtype"}}
    ambiguous = {"context": "ambiguous relation", "audit": {"pairs": [{"status": "ambiguous"}]}}
    assert _usable_context(supported)
    assert not _usable_context(unsupported)
    assert not _usable_context(ambiguous)


_self_check_context_gate()


def score(*, split: str, condition: str, scope: str, adapter: Path, context_path: Path | None, out: Path, selection_path: Path) -> Path:
    if condition not in {"G0", "G1", "G2"}:
        raise ValueError("condition must be G0, G1, or G2")
    if scope not in {"tranche", "full"}:
        raise ValueError("scope must be tranche or full")
    split_rows = json.loads((ROOT / "data" / "splits" / f"{split}.json").read_text())["QA"]
    selection = json.loads(selection_path.read_text())
    selected = selection["selected_questions"][split]
    if scope == "tranche":
        wanted = set(selection["tranche"]["ids"][split])
        selected = [row for row in selected if row["qa_id"] in wanted]
    by_id = {row["qa_id"]: row for row in split_rows}
    rows = [by_id[row["qa_id"]] for row in selected]
    contexts = _read_contexts(context_path) if condition != "G0" and context_path else {}
    if condition != "G0" and {row["qa_id"] for row in rows} - set(contexts):
        raise RuntimeError(f"missing {condition} context for selected questions")

    import torch
    from peft import PeftModel
    from transformers import Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage A video scoring")
    processor = C.load_processor()
    ltr_ids = C.letter_ids(processor.tokenizer)
    C.configure_attention_backend("sdpa")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID,
        revision=C.MODEL_REV,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    device = torch.device("cuda")
    model.to(device).eval()
    rotations = [list(C.LETTERS[offset:] + C.LETTERS[:offset]) for offset in range(2)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as stream, torch.inference_mode():
        for row in rows:
            context_record = contexts.get(row["qa_id"])
            context = context_record if _usable_context(context_record) else None
            views = []
            for offset in C.DEFAULT_SCORE_OFFSETS:
                frames, meta = C.get_video(row["video_path"], offset, work=os.environ.get("WORK"), mode="legacy")
                for permutation in rotations:
                    encoded, actual_perm = C.encode(
                        processor,
                        row,
                        frames,
                        meta,
                        permutation,
                        grounding_context=context["context"] if context else None,
                    )
                    logits = model(**_move(encoded, device)).logits[0, -1, ltr_ids].float()
                    shown = torch.softmax(logits, dim=-1).cpu().numpy()
                    original = [0.0] * 4
                    for index, original_letter in enumerate(actual_perm):
                        original[C.LETTERS.index(original_letter)] = float(shown[index])
                    views.append({
                        "offset": offset,
                        "perm": "".join(actual_perm),
                        "p": original,
                        "argmax_shown": C.LETTERS[int(np.argmax(shown))],
                    })
            mean = np.mean([view["p"] for view in views], axis=0).tolist()
            record = {
                "qa_id": row["qa_id"],
                "p": mean,
                "pred": C.LETTERS[int(np.argmax(mean))],
                "views": views,
                "condition": condition,
                "scope": scope,
            }
            if context_record:
                record["explicit_geometry_context_used"] = context is not None
                record["explicit_geometry_audit"] = context_record["audit"]
                record["evidence_cache_key"] = context_record["cache_key"]
                record["evidence_cache_digest"] = context_record["cache_content_digest"]
                if context is not None:
                    record["explicit_geometry_context"] = context["context"]
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("valQ", "valT"), required=True)
    parser.add_argument("--condition", choices=("G0", "G1", "G2"), required=True)
    parser.add_argument("--scope", choices=("tranche", "full"), required=True)
    parser.add_argument("--adapter", default="runs/stageA_q3vl8b_r16/final_adapter")
    parser.add_argument("--context")
    parser.add_argument("--out", required=True)
    parser.add_argument("--selection", default="runs/phase3_explicit_geometry/selection_manifest.json")
    args = parser.parse_args()
    score(
        split=args.split,
        condition=args.condition,
        scope=args.scope,
        adapter=ROOT / args.adapter,
        context_path=ROOT / args.context if args.context else None,
        out=ROOT / args.out,
        selection_path=ROOT / args.selection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
