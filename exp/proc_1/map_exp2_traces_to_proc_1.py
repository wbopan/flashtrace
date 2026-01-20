#!/usr/bin/env python3
"""Map exp2 trace artifacts into a collaborator-friendly per-sample NPZ format (proc_1).

This is a lightweight variant of `exp/proc/map_exp2_traces_to_proc.py`:
- Removes `tok` (per-token text pieces).
- Adds `length` with three components [in, cot, out], aligned to span_in/span_cot/span_out.
- Saves `hop` only when the trace sample contains `vh` (default strategy).
- Can process a single exp2 trace run directory or all run directories under a traces root.

Input: an exp2 trace run directory produced by `exp/exp2/run_exp.py --save_hop_traces`, e.g.:

  exp/exp2/output/traces/exp/exp2/data/math.jsonl/qwen-8B/ifr_multi_hop_both_n1_mfaithfulness_gen_100ex/

This directory contains:
  - manifest.jsonl (one JSON object per sample)
  - ex_*.npz (per-sample vectors and scores)

Output: per-sample NPZ files under `exp/proc_1/output/` (or a user-provided output path),
each containing only:
  - attr: row attribution vector over [input + CoT + output] tokens, with EOS removed
  - hop: per-hop vectors (optional; only if `vh` exists in the trace npz), aligned to attr
  - span_in/span_cot/span_out: inclusive ranges for input/CoT/output in the above vectors
  - length: int64[3] = [in, cot, out], derived strictly from spans
  - rise/mas: row faithfulness scores (RISE, MAS)
  - recovery: row Recovery@10% score (NaN when unavailable)

This script is intentionally self-contained under exp/proc_1/ and does not modify exp2.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np


def _infer_trace_suffix(trace_dir: Path) -> Optional[Path]:
    parts = list(trace_dir.parts)
    if "traces" not in parts:
        return None
    idx = parts.index("traces")
    suffix_parts = parts[idx + 1 :]
    if not suffix_parts:
        return None
    return Path(*suffix_parts)


def _iter_run_dirs(traces_root: Path) -> List[Path]:
    runs = {p.parent for p in traces_root.rglob("manifest.jsonl") if p.is_file()}
    return sorted(runs)


def _parse_manifest(manifest_path: Path) -> List[dict]:
    records: List[dict] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def _read_span(npz: np.lib.npyio.NpzFile, key: str) -> Optional[Tuple[int, int]]:
    if key not in npz.files:
        return None
    arr = npz[key]
    if arr.shape != (2,):
        raise ValueError(f"Expected {key} to have shape (2,), got {arr.shape}.")
    return int(arr[0]), int(arr[1])


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


def _span_len(span: Tuple[int, int]) -> int:
    start, end = int(span[0]), int(span[1])
    if start < 0 or end < 0 or end < start:
        return 0
    return int(end - start + 1)


@dataclass(frozen=True)
class ProcOneResult:
    wrote: bool
    has_hop: bool


def _proc_one(
    *,
    trace_npz_path: Path,
    record: dict,
    out_path: Path,
    overwrite: bool,
) -> ProcOneResult:
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

        length = np.asarray([_span_len(span_in), _span_len(span_cot), _span_len(span_out)], dtype=np.int64)

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
            "span_in": np.asarray(span_in, dtype=np.int64),
            "span_cot": np.asarray(span_cot, dtype=np.int64),
            "span_out": np.asarray(span_out, dtype=np.int64),
            "length": np.asarray(length, dtype=np.int64),
            "rise": np.asarray(rise, dtype=np.float64),
            "mas": np.asarray(mas, dtype=np.float64),
            "recovery": np.asarray(recovery, dtype=np.float64),
        }

        has_hop = False
        vh = f.get("vh")
        if vh is not None:
            vh = np.asarray(vh, dtype=np.float32)
            if vh.ndim != 2 or int(vh.shape[1]) != int(total_len):
                raise ValueError(f"vh shape mismatch: expected (H,{total_len}), got {tuple(vh.shape)} for {trace_npz_path}.")
            out_payload["hop"] = vh[:, :L]
            has_hop = True

        np.savez_compressed(out_path, **out_payload)
        _ = record
        return ProcOneResult(wrote=True, has_hop=has_hop)


def _resolve_out_dir_for_trace_dir(*, trace_dir: Path, out_root: Path, out_dir: Optional[Path]) -> Path:
    if out_dir is not None:
        return out_dir
    suffix = _infer_trace_suffix(trace_dir)
    return (out_root / suffix) if suffix is not None else (out_root / trace_dir.name)


def _process_trace_dir(
    *,
    trace_dir: Path,
    out_root: Path,
    out_dir: Optional[Path],
    overwrite: bool,
    limit: Optional[int],
    skip_empty_manifest: bool,
) -> Tuple[int, int]:
    manifest_path = trace_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest.jsonl: {manifest_path}")

    records = _parse_manifest(manifest_path)
    if not records:
        if skip_empty_manifest:
            print(f"[skip] empty manifest: {manifest_path}")
            return 0, 0
        raise SystemExit(f"Empty manifest.jsonl: {manifest_path}")

    total = len(records)
    if limit is not None:
        if limit <= 0:
            raise SystemExit("--limit must be a positive integer.")
        total = min(total, int(limit))

    resolved_out_dir = _resolve_out_dir_for_trace_dir(trace_dir=trace_dir, out_root=out_root, out_dir=out_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    wrote_with_hop = 0
    for record in records[:total]:
        file_name = str(record.get("file") or "")
        if not file_name:
            raise SystemExit("manifest record missing 'file' field.")
        trace_npz_path = trace_dir / file_name
        if not trace_npz_path.exists():
            raise SystemExit(f"Missing trace npz referenced by manifest: {trace_npz_path}")

        out_path = resolved_out_dir / file_name
        try:
            res = _proc_one(trace_npz_path=trace_npz_path, record=record, out_path=out_path, overwrite=overwrite)
        except Exception as exc:
            raise SystemExit(f"Failed processing {trace_npz_path}: {exc}") from exc
        wrote += int(res.wrote)
        wrote_with_hop += int(res.has_hop)

    print(f"[ok] wrote {wrote} samples ({wrote_with_hop} with hop) -> {resolved_out_dir}")
    return wrote, wrote_with_hop


def main() -> None:
    ap = argparse.ArgumentParser("Map exp2 trace folder(s) -> exp/proc_1/output per-sample npz files.")
    ap.add_argument(
        "--trace_dir",
        type=str,
        default=None,
        help="Path to a single exp2 trace run directory (contains manifest.jsonl).",
    )
    ap.add_argument(
        "--traces_root",
        type=str,
        default=None,
        help="Path to traces root; processes all run dirs under it (each with a manifest.jsonl).",
    )
    ap.add_argument("--out_root", type=str, default="exp/proc_1/output", help="Root directory for proc_1 outputs.")
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional explicit output directory (only valid with --trace_dir; overrides --out_root).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output files if present.")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit on number of samples per run (debug).")
    ap.add_argument(
        "--fail_on_empty_manifest",
        action="store_true",
        help="Fail (instead of skipping) when encountering an empty manifest.jsonl.",
    )
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir) if args.trace_dir else None
    traces_root = Path(args.traces_root) if args.traces_root else None
    if (trace_dir is None) == (traces_root is None):
        raise SystemExit("Please pass exactly one of --trace_dir or --traces_root.")

    out_root = Path(args.out_root)
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir is not None and trace_dir is None:
        raise SystemExit("--out_dir is only valid with --trace_dir (for --traces_root use --out_root).")

    skip_empty_manifest = not bool(args.fail_on_empty_manifest)

    if trace_dir is not None:
        if not trace_dir.exists() or not trace_dir.is_dir():
            raise SystemExit(f"Missing trace_dir: {trace_dir}")
        _process_trace_dir(
            trace_dir=trace_dir,
            out_root=out_root,
            out_dir=out_dir,
            overwrite=bool(args.overwrite),
            limit=args.limit,
            skip_empty_manifest=skip_empty_manifest,
        )
        return

    assert traces_root is not None
    if not traces_root.exists() or not traces_root.is_dir():
        raise SystemExit(f"Missing traces_root: {traces_root}")

    run_dirs = _iter_run_dirs(traces_root)
    if not run_dirs:
        raise SystemExit(f"No run directories found under traces_root={traces_root} (expected manifest.jsonl).")

    total_written = 0
    total_with_hop = 0
    for run_dir in run_dirs:
        wrote, wrote_with_hop = _process_trace_dir(
            trace_dir=run_dir,
            out_root=out_root,
            out_dir=None,
            overwrite=bool(args.overwrite),
            limit=args.limit,
            skip_empty_manifest=skip_empty_manifest,
        )
        total_written += wrote
        total_with_hop += wrote_with_hop

    print(f"[done] total wrote {total_written} samples ({total_with_hop} with hop) under out_root={out_root}")


if __name__ == "__main__":
    main()

