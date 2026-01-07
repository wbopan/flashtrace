#!/usr/bin/env python3
"""Map data/math_mine.json into an exp2 cached JSONL dataset.

This script converts GSM8K-style math examples:

    {"question": "...", "answer": "... #### 18"}

into exp2's cached JSONL format (one JSON object per line) at:

    exp/exp2/data/math.jsonl

Important: exp2 expects token-level spans (NOT character spans):
  - indices_to_explain: [start_tok, end_tok] (generation-token indices, closed interval)
  - sink_span/thinking_span: token spans over tokenizer(target, add_special_tokens=False)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp2.dataset_utils import CachedExample, attach_spans_from_answer  # noqa: E402


def _load_tokenizer(tokenizer_model: str):
    tok_path = Path(tokenizer_model)
    if tok_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tok_path.as_posix(), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _split_gsm8k_answer(answer: str) -> Optional[Tuple[str, str]]:
    """Return (thinking_text, final_answer) parsed from GSM8K `answer`."""
    text = (answer or "").strip()
    if not text:
        return None
    if "####" not in text:
        return None
    thinking, final = text.rsplit("####", 1)
    thinking = thinking.strip()
    final = final.strip()
    if not final:
        return None
    return thinking, final


def _is_token_span(span: Any) -> bool:
    return isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) for x in span)


def _build_cached_example(
    *,
    question: str,
    answer: str,
    tokenizer,
    example_idx: int,
    source_path: str,
) -> Optional[CachedExample]:
    parsed = _split_gsm8k_answer(answer)
    if parsed is None:
        return None
    thinking_text, final_answer = parsed

    prompt = question.strip()
    target = f"{thinking_text}\n{final_answer}" if thinking_text else final_answer

    example = CachedExample(
        prompt=prompt,
        target=target,
        indices_to_explain=None,
        attr_mask_indices=None,
        sink_span=None,
        thinking_span=None,
        metadata={
            "dataset": "math_mine",
            "source_path": source_path,
            "example_idx": int(example_idx),
            "raw_question": question,
            "raw_answer": answer,
            "reference_answer": final_answer,
            "boxed_answer": final_answer,
        },
    )
    example = attach_spans_from_answer(example, tokenizer, final_answer)
    if not _is_token_span(example.sink_span):
        return None

    # exp2 requires token-level indices_to_explain=[start_tok,end_tok] (closed interval).
    indices_to_explain = list(example.sink_span)
    thinking_span = example.thinking_span
    if thinking_span is not None and _is_token_span(thinking_span) and indices_to_explain[0] == 0:
        # No room for "thinking" tokens; avoid overlapping spans.
        thinking_span = None

    return CachedExample(
        prompt=example.prompt,
        target=example.target,
        indices_to_explain=indices_to_explain,
        attr_mask_indices=example.attr_mask_indices,
        sink_span=indices_to_explain,
        thinking_span=thinking_span,
        metadata=example.metadata,
    )


def _write_jsonl(path: Path, *, examples) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex), ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    ap = argparse.ArgumentParser("Map data/math_mine.json to exp2 cached JSONL.")
    ap.add_argument("--in_json", type=str, default="data/math_mine.json")
    ap.add_argument("--out_jsonl", type=str, default="exp/exp2/data/math.jsonl")
    ap.add_argument(
        "--tokenizer_model",
        type=str,
        required=True,
        help="Tokenizer name or local path; must match the tokenizer used in exp2 attribution.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_json)
    out_path = Path(args.out_jsonl)
    tokenizer = _load_tokenizer(args.tokenizer_model)

    raw = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a JSON array in {in_path}, got {type(raw).__name__}.")

    total = 0
    kept = 0
    skipped_empty_q = 0
    skipped_empty_a = 0
    skipped_parse = 0
    skipped_span = 0

    examples = []
    for idx, item in enumerate(raw):
        total += 1
        if not isinstance(item, dict):
            skipped_parse += 1
            continue

        question = str(item.get("question") or "")
        answer = str(item.get("answer") or "")
        if not question.strip():
            skipped_empty_q += 1
            continue
        if not answer.strip():
            skipped_empty_a += 1
            continue

        ex = _build_cached_example(
            question=question,
            answer=answer,
            tokenizer=tokenizer,
            example_idx=idx,
            source_path=str(in_path),
        )
        if ex is None:
            # distinguish parse-vs-span failure
            parsed = _split_gsm8k_answer(answer)
            if parsed is None:
                skipped_parse += 1
            else:
                skipped_span += 1
            continue

        examples.append(ex)
        kept += 1

    written = _write_jsonl(out_path, examples=examples)
    if written != kept:
        raise SystemExit(f"Internal error: written={written} != kept={kept}")

    print(
        json.dumps(
            {
                "in_json": str(in_path),
                "out_jsonl": str(out_path),
                "tokenizer_model": args.tokenizer_model,
                "total": total,
                "kept": kept,
                "skipped_empty_question": skipped_empty_q,
                "skipped_empty_answer": skipped_empty_a,
                "skipped_parse": skipped_parse,
                "skipped_span": skipped_span,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

