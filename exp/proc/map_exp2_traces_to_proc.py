#!/usr/bin/env python3
"""Map exp2 trace artifacts into a collaborator-friendly per-sample NPZ format.

Input: an exp2 trace run directory produced by `exp/exp2/run_exp.py --save_hop_traces`,
e.g.:

  exp/exp2/output/traces/exp/exp2/data/morehopqa.jsonl/qwen-8B/ifr_all_positions_mfaithfulness_gen_95ex/

This directory contains:
  - manifest.jsonl (one JSON object per sample)
  - ex_*.npz (per-sample vectors and scores)

Output: per-sample NPZ files under `exp/proc/output/` (or a user-provided output path),
each containing only:
  - attr: row attribution vector over [input + CoT + output] tokens, with chat template and EOS removed
  - hop: per-hop vectors (FT-IFR only), aligned to attr (optional)
  - tok: tokenized text pieces aligned to attr/hop (no chat template, no EOS)
  - span_in/span_cot/span_out: inclusive ranges for input/CoT/output in the above vectors
  - rise/mas: row faithfulness scores (RISE, MAS)
  - recovery: row Recovery@10% score (NaN when unavailable)

This script is intentionally self-contained under exp/proc/ and does not modify exp2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from transformers import AutoTokenizer


FT_IFR_ATTR_FUNCS: set[str] = {
    "ifr_in_all_gen",
    "ifr_multi_hop_stop_words",
    "ifr_multi_hop_both",
    "ifr_multi_hop_split_hop",
}


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_tokenizer(tokenizer_model: str):
    tok_path = Path(tokenizer_model)
    if tok_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tok_path.as_posix(), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    if tokenizer.eos_token is None:
        raise SystemExit("Tokenizer is missing eos_token; cannot match exp2 generation tokenization.")
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _decode_text_into_tokens(tokenizer, text: str) -> List[str]:
    """Mirror llm_attr.LLMAttribution.decode_text_into_tokens (offset-slice tokens)."""
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc.get("input_ids")
    offsets = enc.get("offset_mapping")
    if ids is None or offsets is None:
        raise ValueError("Tokenizer must provide input_ids and offset_mapping for exact exp2 token alignment.")
    if len(ids) != len(offsets):
        raise ValueError("Tokenizer returned mismatched input_ids vs offset_mapping lengths.")
    tokens: List[str] = []
    for start, end in offsets:
        tokens.append(text[int(start) : int(end)])
    return tokens


@dataclass(frozen=True)
class DatasetEntry:
    prompt: str
    target: str


def _index_dataset_by_sha1(dataset_jsonl: Path) -> Dict[Tuple[str, str], DatasetEntry]:
    """Build (prompt_sha1, target_sha1) -> (prompt, target) for cache lookup."""
    index: Dict[Tuple[str, str], DatasetEntry] = {}
    collisions: Dict[Tuple[str, str], int] = {}

    with dataset_jsonl.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt") or "")
            target = obj.get("target")
            if target is None:
                # exp2 trace matching requires cached targets.
                continue
            target = str(target)

            key = (_sha1_text(prompt), _sha1_text(target))
            if key in index:
                collisions[key] = collisions.get(key, 1) + 1
                continue
            index[key] = DatasetEntry(prompt=prompt, target=target)

    if collisions:
        raise SystemExit(
            "Dataset cache contains duplicate (prompt,target) pairs; cannot uniquely match by sha1. "
            f"Example collision count={next(iter(collisions.values()))}. "
            f"dataset_jsonl={dataset_jsonl}"
        )

    if not index:
        raise SystemExit(
            "No usable (prompt,target) pairs found in dataset cache. "
            "Ensure you pass the exp2 cached JSONL used for attribution (with target filled)."
        )

    return index


def _infer_trace_suffix(trace_dir: Path) -> Optional[Path]:
    parts = list(trace_dir.parts)
    if "traces" not in parts:
        return None
    idx = parts.index("traces")
    suffix_parts = parts[idx + 1 :]
    if not suffix_parts:
        return None
    return Path(*suffix_parts)


def _parse_manifest(manifest_path: Path) -> List[dict]:
    records: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    if not records:
        raise SystemExit(f"Empty manifest.jsonl: {manifest_path}")
    return records


def _read_span(npz: np.lib.npyio.NpzFile, key: str) -> Optional[Tuple[int, int]]:
    if key not in npz.files:
        return None
    arr = npz[key]
    if arr.shape != (2,):
        raise ValueError(f"Expected {key} to have shape (2,), got {arr.shape}.")
    return int(arr[0]), int(arr[1])


def _span_or_empty(span: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if span is None:
        return -1, -1
    return int(span[0]), int(span[1])


def _tokenize_for_exp2_alignment(
    tokenizer,
    *,
    prompt: str,
    target: str,
    expected_prompt_len: int,
    expected_gen_len: int,
) -> List[str]:
    prompt_text = " " + (prompt or "")
    prompt_tokens = _decode_text_into_tokens(tokenizer, prompt_text)
    if len(prompt_tokens) != int(expected_prompt_len):
        raise ValueError(f"Prompt token length mismatch: expected {expected_prompt_len}, got {len(prompt_tokens)}.")

    gen_ids = tokenizer(target + tokenizer.eos_token, add_special_tokens=False).input_ids
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    gen_tokens = _decode_text_into_tokens(tokenizer, gen_text)
    if len(gen_tokens) != int(expected_gen_len):
        raise ValueError(f"Generation token length mismatch: expected {expected_gen_len}, got {len(gen_tokens)}.")

    gen_tokens_no_eos = gen_tokens[:-1] if gen_tokens else []
    return prompt_tokens + gen_tokens_no_eos


def _clamp_span(span: Optional[Tuple[int, int]], *, max_index: int) -> Optional[Tuple[int, int]]:
    if span is None:
        return None
    start, end = int(span[0]), int(span[1])
    if max_index < 0:
        return None
    if end < 0 or start > max_index:
        return None
    start = max(0, start)
    end = min(max_index, end)
    if end < start:
        return None
    return start, end


def _proc_one(
    *,
    trace_npz_path: Path,
    record: dict,
    dataset_index: Dict[Tuple[str, str], DatasetEntry],
    tokenizer,
    out_path: Path,
    overwrite: bool,
    allow_missing_ft_hops: bool,
) -> None:
    prompt_sha1 = str(record.get("prompt_sha1") or "")
    target_sha1 = str(record.get("target_sha1") or "")
    if not prompt_sha1 or not target_sha1:
        raise ValueError("manifest record missing prompt_sha1/target_sha1; cannot match dataset.")

    entry = dataset_index.get((prompt_sha1, target_sha1))
    if entry is None:
        raise ValueError(
            "Failed to match manifest sha1 to dataset_jsonl. "
            "Ensure --dataset_jsonl points to the exact cached JSONL used for this trace run."
        )

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path} (use --overwrite).")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(trace_npz_path, allow_pickle=False) as f:
        prompt_len = int(np.asarray(f.get("prompt_len")).item())
        gen_len = int(np.asarray(f.get("gen_len")).item())
        total_len = prompt_len + gen_len
        gen_no_eos = max(0, gen_len - 1)
        L = prompt_len + gen_no_eos

        v_row_all = f.get("v_row_all")
        if v_row_all is None:
            raise ValueError("Missing v_row_all in trace npz; cannot build row attribution vector.")
        v_row_all = np.asarray(v_row_all, dtype=np.float32)
        if v_row_all.ndim != 1 or int(v_row_all.shape[0]) != int(total_len):
            raise ValueError(f"v_row_all shape mismatch: expected ({total_len},), got {tuple(v_row_all.shape)}.")
        attr = v_row_all[:L]

        indices_to_explain = _read_span(f, "indices_to_explain_gen")
        sink_span_gen = _read_span(f, "sink_span_gen") or indices_to_explain
        if sink_span_gen is None:
            raise ValueError("Missing sink_span_gen/indices_to_explain_gen; cannot define output span.")
        thinking_span_gen = _read_span(f, "thinking_span_gen")
        if thinking_span_gen is None:
            sink_start = int(sink_span_gen[0])
            think_end = sink_start - 1
            thinking_span_gen = (0, think_end) if think_end >= 0 else None

        sink_span_gen = _clamp_span(sink_span_gen, max_index=gen_no_eos - 1)
        thinking_span_gen = _clamp_span(thinking_span_gen, max_index=gen_no_eos - 1)

        span_in = (0, prompt_len - 1) if prompt_len > 0 else (-1, -1)
        span_cot = (
            (prompt_len + thinking_span_gen[0], prompt_len + thinking_span_gen[1])
            if thinking_span_gen is not None
            else (-1, -1)
        )
        span_out = (
            (prompt_len + sink_span_gen[0], prompt_len + sink_span_gen[1]) if sink_span_gen is not None else (-1, -1)
        )

        tokens = _tokenize_for_exp2_alignment(
            tokenizer,
            prompt=entry.prompt,
            target=entry.target,
            expected_prompt_len=prompt_len,
            expected_gen_len=gen_len,
        )
        if len(tokens) != int(L):
            raise ValueError(f"Token length mismatch after EOS drop: expected {L}, got {len(tokens)}.")

        # Scores: row = index 1.
        rise = float("nan")
        mas = float("nan")
        faith = f.get("faithfulness_scores")
        if faith is not None:
            faith = np.asarray(faith, dtype=np.float64)
            if faith.shape != (3, 3):
                raise ValueError(f"faithfulness_scores shape mismatch: expected (3,3), got {tuple(faith.shape)}.")
            rise = float(faith[1, 0])
            mas = float(faith[1, 1])

        recovery = float("nan")
        rec = f.get("recovery_scores")
        if rec is not None:
            rec = np.asarray(rec, dtype=np.float64)
            if rec.shape != (3,):
                raise ValueError(f"recovery_scores shape mismatch: expected (3,), got {tuple(rec.shape)}.")
            recovery = float(rec[1])

        out_payload = {
            "attr": np.asarray(attr, dtype=np.float32),
            "tok": np.asarray(tokens, dtype=np.str_),
            "span_in": np.asarray(span_in, dtype=np.int64),
            "span_cot": np.asarray(span_cot, dtype=np.int64),
            "span_out": np.asarray(span_out, dtype=np.int64),
            "rise": np.asarray(rise, dtype=np.float64),
            "mas": np.asarray(mas, dtype=np.float64),
            "recovery": np.asarray(recovery, dtype=np.float64),
        }

        attr_func = str(record.get("attr_func") or "")
        want_hops = attr_func in FT_IFR_ATTR_FUNCS
        if want_hops:
            vh = f.get("vh")
            if vh is None:
                if not allow_missing_ft_hops:
                    raise ValueError(
                        f"FT-IFR method '{attr_func}' requires per-hop vectors but trace npz is missing 'vh'. "
                        "Re-run exp2 with --save_hop_traces using the updated code."
                    )
            else:
                vh = np.asarray(vh, dtype=np.float32)
                if vh.ndim != 2 or int(vh.shape[1]) != int(total_len):
                    raise ValueError(
                        f"vh shape mismatch: expected (H,{total_len}), got {tuple(vh.shape)} for {trace_npz_path}."
                    )
                out_payload["hop"] = vh[:, :L]

        np.savez_compressed(out_path, **out_payload)


def main() -> None:
    ap = argparse.ArgumentParser("Map exp2 trace folder -> exp/proc/output per-sample npz files.")
    ap.add_argument("--trace_dir", type=str, required=True, help="Path to an exp2 trace run directory (contains manifest.jsonl).")
    ap.add_argument("--dataset_jsonl", type=str, default=None, help="Path to the exp2 cached dataset JSONL used for this trace.")
    ap.add_argument(
        "--tokenizer_model",
        type=str,
        required=True,
        help="Tokenizer model name or local path (must match exp2 attribution tokenizer).",
    )
    ap.add_argument("--out_root", type=str, default="exp/proc/output", help="Root directory for proc outputs.")
    ap.add_argument("--out_dir", type=str, default=None, help="Optional explicit output directory (overrides --out_root).")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if present.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples to process (debug).")
    ap.add_argument(
        "--allow_missing_ft_hops",
        action="store_true",
        help="Allow producing FT-IFR outputs even when per-hop vectors (vh) are missing (not recommended).",
    )
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    if not trace_dir.exists() or not trace_dir.is_dir():
        raise SystemExit(f"Missing trace_dir: {trace_dir}")
    manifest_path = trace_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.jsonl: {manifest_path}")

    dataset_jsonl: Optional[Path] = Path(args.dataset_jsonl) if args.dataset_jsonl else None
    if dataset_jsonl is None:
        suffix = _infer_trace_suffix(trace_dir)
        if suffix is not None and len(suffix.parts) >= 3:
            # suffix = <dataset_name...>/<model_tag>/<run_tag>
            inferred_dataset = Path(*suffix.parts[:-2])
            if inferred_dataset.exists() and inferred_dataset.is_file():
                dataset_jsonl = inferred_dataset
    if dataset_jsonl is None:
        raise SystemExit("Please pass --dataset_jsonl (could not infer it from --trace_dir).")
    if not dataset_jsonl.exists():
        raise SystemExit(f"Missing --dataset_jsonl: {dataset_jsonl}")

    tokenizer = _load_tokenizer(str(args.tokenizer_model))
    dataset_index = _index_dataset_by_sha1(dataset_jsonl)
    records = _parse_manifest(manifest_path)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        suffix = _infer_trace_suffix(trace_dir)
        out_dir = Path(args.out_root) / suffix if suffix is not None else Path(args.out_root) / trace_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(records)
    limit = args.limit
    if limit is not None:
        if limit <= 0:
            raise SystemExit("--limit must be a positive integer.")
        total = min(total, int(limit))

    processed = 0
    for record in records[:total]:
        file_name = str(record.get("file") or "")
        if not file_name:
            raise SystemExit("manifest record missing 'file' field.")
        trace_npz_path = trace_dir / file_name
        if not trace_npz_path.exists():
            raise SystemExit(f"Missing trace npz referenced by manifest: {trace_npz_path}")

        out_path = out_dir / file_name
        try:
            _proc_one(
                trace_npz_path=trace_npz_path,
                record=record,
                dataset_index=dataset_index,
                tokenizer=tokenizer,
                out_path=out_path,
                overwrite=bool(args.overwrite),
                allow_missing_ft_hops=bool(args.allow_missing_ft_hops),
            )
        except Exception as exc:
            raise SystemExit(f"Failed processing {trace_npz_path}: {exc}") from exc
        processed += 1

    print(f"Wrote {processed} proc samples -> {out_dir}")


if __name__ == "__main__":
    main()
