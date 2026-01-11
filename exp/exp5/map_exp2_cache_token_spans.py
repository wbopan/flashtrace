#!/usr/bin/env python3
"""Map exp2 cached JSONL token spans across tokenizers (Qwen -> Llama).

Background
----------
`exp/exp2/run_exp.py` expects cached datasets to provide token-level generation spans:

  - indices_to_explain: [start_tok, end_tok] (generation-token indices; closed interval)
  - sink_span / thinking_span: same tokenizer convention as indices_to_explain

These spans are computed under a specific tokenizer (often Qwen3-8B). When switching
to a different model/tokenizer (e.g., Llama-3.1-8B-Instruct), the stored spans can
become out-of-range and crash exp2 attribution (IndexError in token-span checks).

This script remaps spans by:
  1) Tokenizing `target` with the OLD tokenizer to obtain offset_mapping
  2) Converting the OLD token span into a character span in `target`
  3) Tokenizing `target` with the NEW tokenizer and mapping the character span back
     into NEW token indices

Outputs are written under `exp/exp5/data/` by default, keeping `exp/exp2/` untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _split_args(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out


def _load_tokenizer(tokenizer_model: str):
    path = Path(tokenizer_model)
    if path.exists():
        return AutoTokenizer.from_pretrained(path.as_posix(), local_files_only=True)
    # May require network access; keep as fallback for environments that allow it.
    return AutoTokenizer.from_pretrained(tokenizer_model)


def _is_token_span(span: Any) -> bool:
    return (
        isinstance(span, list)
        and len(span) == 2
        and all(isinstance(x, int) for x in span)
        and span[0] >= 0
        and span[1] >= span[0]
    )


def _pick_old_span(obj: Dict[str, Any]) -> Optional[List[int]]:
    span = obj.get("indices_to_explain")
    if _is_token_span(span):
        return list(span)
    span = obj.get("sink_span")
    if _is_token_span(span):
        return list(span)
    return None


def _offsets_to_char_span(offsets: Any, token_span: List[int]) -> Optional[Tuple[int, int]]:
    """Convert a token span [start,end] to a character span [char_start,char_end) using offsets."""
    if offsets is None:
        return None
    if not isinstance(offsets, list):
        return None
    start_tok, end_tok = token_span
    if end_tok >= len(offsets):
        return None

    char_starts: List[int] = []
    char_ends: List[int] = []
    for idx in range(start_tok, end_tok + 1):
        off = offsets[idx]
        if off is None:
            continue
        if not (isinstance(off, (list, tuple)) and len(off) == 2):
            continue
        try:
            s, e = int(off[0]), int(off[1])
        except Exception:
            continue
        if e <= s:
            continue
        char_starts.append(s)
        char_ends.append(e)

    if not char_starts or not char_ends:
        return None
    return min(char_starts), max(char_ends)


def _char_span_to_token_span(offsets: Any, char_span: Tuple[int, int]) -> Optional[List[int]]:
    """Convert a character span [char_start,char_end) to a token span [start,end] by overlap."""
    if offsets is None:
        return None
    if not isinstance(offsets, list):
        return None
    char_start, char_end = int(char_span[0]), int(char_span[1])
    if char_end <= char_start:
        return None

    hit: List[int] = []
    for tok_idx, off in enumerate(offsets):
        if off is None:
            continue
        if not (isinstance(off, (list, tuple)) and len(off) == 2):
            continue
        try:
            s, e = int(off[0]), int(off[1])
        except Exception:
            continue
        if e <= s:
            continue
        if s < char_end and e > char_start:
            hit.append(int(tok_idx))

    if not hit:
        return None
    return [min(hit), max(hit)]


def _validate_span_with_eos(tokenizer, target: str, token_span: List[int]) -> bool:
    eos = tokenizer.eos_token or ""
    gen_ids = tokenizer(target + eos, add_special_tokens=False).input_ids
    gen_len = int(len(gen_ids))
    return 0 <= token_span[0] <= token_span[1] < gen_len


def _guess_answer_text(obj: Dict[str, Any]) -> Optional[str]:
    meta = obj.get("metadata") or {}
    if isinstance(meta, dict):
        boxed = (meta.get("boxed_answer") or "").strip()
        if boxed:
            return boxed
        ref = (meta.get("reference_answer") or "").strip()
        if ref:
            return ref
    tgt = obj.get("target")
    if isinstance(tgt, str) and tgt.strip():
        # Common exp2 cache convention: last line is the final answer.
        last_line = tgt.strip().splitlines()[-1].strip()
        return last_line or None
    return None


def _fallback_map_via_answer_text(
    obj: Dict[str, Any],
    *,
    new_tokenizer,
) -> Optional[List[int]]:
    tgt = obj.get("target")
    if not isinstance(tgt, str) or not tgt:
        return None

    from exp.exp2.dataset_utils import CachedExample, attach_spans_from_answer  # lazy import

    answer_text = _guess_answer_text(obj)
    ex = CachedExample(
        prompt=str(obj.get("prompt") or ""),
        target=tgt,
        indices_to_explain=None,
        attr_mask_indices=obj.get("attr_mask_indices"),
        sink_span=None,
        thinking_span=None,
        metadata=obj.get("metadata") or {},
    )
    out = attach_spans_from_answer(ex, new_tokenizer, answer_text)
    if out.sink_span is None:
        return None
    if not _is_token_span(out.sink_span):
        return None
    return list(out.sink_span)


def _map_one_obj(
    obj: Dict[str, Any],
    *,
    old_tokenizer,
    new_tokenizer,
    allow_fallback_answer: bool,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    target = obj.get("target")
    if not isinstance(target, str) or not target:
        return None, "missing_target"

    old_span = _pick_old_span(obj)
    if old_span is None:
        return None, "missing_old_span"

    # 1) Old token span -> char span in target.
    old_enc = old_tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    old_offsets = old_enc.get("offset_mapping")
    char_span = _offsets_to_char_span(old_offsets, old_span)
    if char_span is None:
        if not allow_fallback_answer:
            return None, "old_span_to_char_failed"
        new_span = _fallback_map_via_answer_text(obj, new_tokenizer=new_tokenizer)
        if new_span is None:
            return None, "fallback_answer_failed"
        if not _validate_span_with_eos(new_tokenizer, target, new_span):
            return None, "fallback_answer_span_invalid"
        mapped = dict(obj)
        mapped["indices_to_explain"] = new_span
        mapped["sink_span"] = new_span
        mapped["thinking_span"] = [0, new_span[0] - 1] if new_span[0] > 0 else None
        meta = mapped.get("metadata")
        if not isinstance(meta, dict):
            meta = {}
        meta = dict(meta)
        meta["exp5_span_map_method"] = "answer_text"
        mapped["metadata"] = meta
        return mapped, None

    # 2) Char span -> new token span.
    new_enc = new_tokenizer(target, add_special_tokens=False, return_offsets_mapping=True)
    new_offsets = new_enc.get("offset_mapping")
    new_span = _char_span_to_token_span(new_offsets, char_span)
    if new_span is None:
        if not allow_fallback_answer:
            return None, "char_to_new_span_failed"
        new_span = _fallback_map_via_answer_text(obj, new_tokenizer=new_tokenizer)
        if new_span is None:
            return None, "fallback_answer_failed"

    if not _validate_span_with_eos(new_tokenizer, target, new_span):
        if not allow_fallback_answer:
            return None, "new_span_invalid"
        fb = _fallback_map_via_answer_text(obj, new_tokenizer=new_tokenizer)
        if fb is None or not _validate_span_with_eos(new_tokenizer, target, fb):
            return None, "fallback_answer_span_invalid"
        new_span = fb

    mapped = dict(obj)
    mapped["indices_to_explain"] = new_span
    mapped["sink_span"] = new_span
    mapped["thinking_span"] = [0, new_span[0] - 1] if new_span[0] > 0 else None

    meta = mapped.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    meta = dict(meta)
    meta["exp5_span_map_method"] = "token_span_char_align"
    mapped["metadata"] = meta
    return mapped, None


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise RuntimeError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise RuntimeError(f"Expected JSON object per line at {path}:{line_no}.")
            yield obj


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


def _default_old_tokenizer() -> str:
    # Repo defaults used in exp2 README examples for span extraction.
    return "/opt/share/models/Qwen/Qwen3-8B"


def _default_new_tokenizer() -> str:
    return "/opt/share/models/meta-llama/Llama-3.1-8B-Instruct"


def main() -> None:
    ap = argparse.ArgumentParser("Map exp2 cache token spans from an old tokenizer to a new tokenizer.")
    ap.add_argument(
        "--in_jsonl",
        type=str,
        nargs="+",
        required=True,
        help="One or more exp2 cached JSONL files (comma-separated also accepted).",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="exp/exp5/data",
        help="Output directory for mapped JSONL files.",
    )
    ap.add_argument(
        "--old_tokenizer_model",
        type=str,
        default=_default_old_tokenizer(),
        help="Tokenizer used to produce the original token spans (default: Qwen3-8B local path).",
    )
    ap.add_argument(
        "--new_tokenizer_model",
        type=str,
        default=_default_new_tokenizer(),
        help="Tokenizer to map spans into (default: Llama-3.1-8B-Instruct local path).",
    )
    ap.add_argument("--strict", action="store_true", help="Fail on the first example that cannot be mapped.")
    ap.add_argument(
        "--allow_fallback_answer",
        action="store_true",
        help=(
            "If span alignment fails, try to recompute spans by locating metadata.boxed_answer in target "
            "(useful when caches were not built with the assumed old tokenizer)."
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    args = ap.parse_args()

    in_paths = [Path(p) for p in _split_args(args.in_jsonl)]
    out_dir = Path(args.out_dir)

    old_tok = _load_tokenizer(str(args.old_tokenizer_model))
    new_tok = _load_tokenizer(str(args.new_tokenizer_model))

    # exp2 convention: ensure a pad token exists for downstream perturbation.
    if new_tok.pad_token is None and new_tok.eos_token is not None:
        new_tok.pad_token = new_tok.eos_token

    summary: Dict[str, Any] = {
        "old_tokenizer_model": str(args.old_tokenizer_model),
        "new_tokenizer_model": str(args.new_tokenizer_model),
        "datasets": [],
    }

    for in_path in in_paths:
        if not in_path.exists():
            raise SystemExit(f"Missing input JSONL: {in_path}")
        out_path = out_dir / in_path.name
        if out_path.exists() and not bool(args.overwrite):
            raise SystemExit(f"Refusing to overwrite existing output: {out_path} (use --overwrite)")

        total = 0
        mapped_ok = 0
        dropped = 0
        errors: Dict[str, int] = {}

        mapped_rows: List[Dict[str, Any]] = []
        for obj in _read_jsonl(in_path):
            total += 1
            mapped, err = _map_one_obj(
                obj,
                old_tokenizer=old_tok,
                new_tokenizer=new_tok,
                allow_fallback_answer=bool(args.allow_fallback_answer),
            )
            if err is not None or mapped is None:
                errors[err or "unknown_error"] = errors.get(err or "unknown_error", 0) + 1
                if bool(args.strict):
                    raise SystemExit(f"Failed to map {in_path} example #{total}: {err}")
                dropped += 1
                continue
            mapped_ok += 1
            mapped_rows.append(mapped)

        written = _write_jsonl(out_path, mapped_rows)
        if written != mapped_ok:  # pragma: no cover
            raise SystemExit(f"Internal error: written={written} != mapped_ok={mapped_ok}")

        record = {
            "in_jsonl": str(in_path),
            "out_jsonl": str(out_path),
            "total": int(total),
            "mapped_ok": int(mapped_ok),
            "dropped": int(dropped),
            "errors": errors,
        }
        summary["datasets"].append(record)
        print(json.dumps(record, ensure_ascii=False))

    # Human-readable compact summary at end.
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
