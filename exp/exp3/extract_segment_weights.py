#!/usr/bin/env python3
"""
Extract CoT/output segment attribution weights from exp3 trace artifacts.

Background
----------
exp/exp3/run_exp.py saves per-sample trace npz files that contain token-level
importance vectors over the FULL (prompt + generation) token sequence:
  - v_seq_all: sum over rows of seq attribution matrix (shape [P+G])
  - v_row_all: row attribution vector for indices_to_explain (shape [P+G])
  - v_rec_all: recursive attribution vector for indices_to_explain (shape [P+G])

For exp3 cached samples, we also have generation-token spans:
  - thinking_span_gen: CoT span [start,end] in generation-token coordinates
  - sink_span_gen: output span [start,end] in generation-token coordinates

This script slices v_*_all into:
  - cot: tokens in thinking_span_gen (offset by prompt_len)
  - output: tokens in sink_span_gen (offset by prompt_len)

and reports segment sums/fractions (and optionally writes a JSON summary).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TracePaths:
    dataset: str
    model_tag: str
    run_tag: str
    npz_path: Path


def _pick_latest_subdir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    subs = [p for p in path.iterdir() if p.is_dir()]
    if not subs:
        return None
    subs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subs[0]


def _resolve_trace_paths(
    *,
    output_root: Path,
    dataset: str,
    model_tag: Optional[str],
    run_tag: Optional[str],
    example_idx: int,
) -> TracePaths:
    base = output_root / "traces" / dataset
    if not base.exists():
        raise FileNotFoundError(f"Trace dataset dir not found: {base}")

    if model_tag is None:
        model_dirs = [p for p in base.iterdir() if p.is_dir()]
        if not model_dirs:
            raise FileNotFoundError(f"No model subdir under: {base}")
        if len(model_dirs) != 1:
            raise SystemExit(f"Multiple model dirs under {base}; pass --model_tag. Found: {[p.name for p in model_dirs]}")
        model_dir = model_dirs[0]
        model_tag = model_dir.name
    else:
        model_dir = base / model_tag
        if not model_dir.exists():
            raise FileNotFoundError(f"Trace model dir not found: {model_dir}")

    if run_tag is None:
        run_dir = _pick_latest_subdir(model_dir)
        if run_dir is None:
            raise FileNotFoundError(f"No run subdir under: {model_dir}")
        run_tag = run_dir.name
    else:
        run_dir = model_dir / run_tag
        if not run_dir.exists():
            raise FileNotFoundError(f"Trace run dir not found: {run_dir}")

    npz_name = f"ex_{int(example_idx):06d}.npz"
    npz_path = run_dir / npz_name
    if not npz_path.exists():
        raise FileNotFoundError(f"Trace npz not found: {npz_path}")

    return TracePaths(dataset=dataset, model_tag=model_tag, run_tag=run_tag, npz_path=npz_path)


def _as_span(arr: Any) -> Optional[Tuple[int, int]]:
    if arr is None:
        return None
    try:
        a = np.asarray(arr).reshape(-1).tolist()
    except Exception:
        return None
    if len(a) != 2:
        return None
    try:
        start = int(a[0])
        end = int(a[1])
    except Exception:
        return None
    if start < 0 or end < start:
        return None
    return start, end


def _segment_stats(v: np.ndarray, start: int, end: int) -> Dict[str, float]:
    if end < start:
        return {"sum": 0.0, "mean": 0.0, "max": 0.0}
    seg = v[start : end + 1]
    if seg.size == 0:
        return {"sum": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "sum": float(seg.sum()),
        "mean": float(seg.mean()),
        "max": float(seg.max()),
    }


def _slice_segment(v: np.ndarray, start: int, end: int) -> List[float]:
    if end < start:
        return []
    seg = v[start : end + 1]
    return [float(x) for x in seg.tolist()]


def extract_one(npz_path: Path) -> Dict[str, Any]:
    d = np.load(npz_path)
    required = ["prompt_len", "gen_len", "v_seq_all", "v_row_all", "v_rec_all"]
    for k in required:
        if k not in d:
            raise KeyError(f"Missing key in trace npz {npz_path}: {k}")

    prompt_len = int(np.asarray(d["prompt_len"]).item())
    gen_len = int(np.asarray(d["gen_len"]).item())
    total_len = prompt_len + gen_len

    v_seq_all = np.asarray(d["v_seq_all"], dtype=np.float64).reshape(-1)
    v_row_all = np.asarray(d["v_row_all"], dtype=np.float64).reshape(-1)
    v_rec_all = np.asarray(d["v_rec_all"], dtype=np.float64).reshape(-1)
    for name, v in [("v_seq_all", v_seq_all), ("v_row_all", v_row_all), ("v_rec_all", v_rec_all)]:
        if int(v.size) != int(total_len):
            raise ValueError(f"{name} length mismatch: expected {total_len}, got {int(v.size)}")

    sink_span_gen = _as_span(d.get("sink_span_gen"))
    thinking_span_gen = _as_span(d.get("thinking_span_gen"))
    if sink_span_gen is None:
        raise KeyError("Trace missing sink_span_gen; cannot define output span.")
    if thinking_span_gen is None:
        # Best-effort: infer thinking span as [0, sink_start-1].
        sink_start, _ = sink_span_gen
        thinking_span_gen = (0, max(0, sink_start - 1))

    think_start_g, think_end_g = thinking_span_gen
    sink_start_g, sink_end_g = sink_span_gen

    cot_start = prompt_len + think_start_g
    cot_end = min(prompt_len + think_end_g, total_len - 1)
    out_start = prompt_len + sink_start_g
    out_end = min(prompt_len + sink_end_g, total_len - 1)

    def pack(v: np.ndarray) -> Dict[str, Any]:
        total = float(v.sum())
        cot = _segment_stats(v, cot_start, cot_end)
        out = _segment_stats(v, out_start, out_end)
        denom = cot["sum"] + out["sum"]
        return {
            "total_sum": total,
            "cot": {
                "start_abs": int(cot_start),
                "end_abs": int(cot_end),
                "len": int(max(0, cot_end - cot_start + 1)),
                **cot,
                "fraction_of_total": float(cot["sum"] / total) if total > 0 else float("nan"),
                "fraction_of_cot_plus_output": float(cot["sum"] / denom) if denom > 0 else float("nan"),
            },
            "output": {
                "start_abs": int(out_start),
                "end_abs": int(out_end),
                "len": int(max(0, out_end - out_start + 1)),
                **out,
                "fraction_of_total": float(out["sum"] / total) if total > 0 else float("nan"),
                "fraction_of_cot_plus_output": float(out["sum"] / denom) if denom > 0 else float("nan"),
            },
            "cot_weights": _slice_segment(v, cot_start, cot_end),
            "output_weights": _slice_segment(v, out_start, out_end),
        }

    return {
        "prompt_len": int(prompt_len),
        "gen_len": int(gen_len),
        "total_len": int(total_len),
        "thinking_span_gen": [int(think_start_g), int(think_end_g)],
        "sink_span_gen": [int(sink_start_g), int(sink_end_g)],
        "seq": pack(v_seq_all),
        "row": pack(v_row_all),
        "rec": pack(v_rec_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser("Extract CoT/output weights from exp3 traces.")
    parser.add_argument("--output_root", type=str, default="exp/exp3/output")
    parser.add_argument("--dataset_tag", type=str, default="niah_mq_q2")
    parser.add_argument("--model_tag", type=str, default=None, help="If omitted, auto-detect when unique.")
    parser.add_argument("--run_tag", type=str, default=None, help="If omitted, picks the latest run subdir.")
    parser.add_argument("--example_idx", type=int, default=0)
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    datasets = [f"{args.dataset_tag}_short_cot", f"{args.dataset_tag}_long_cot"]

    results: List[Dict[str, Any]] = []
    for ds_name in datasets:
        paths = _resolve_trace_paths(
            output_root=output_root,
            dataset=ds_name,
            model_tag=args.model_tag,
            run_tag=args.run_tag,
            example_idx=args.example_idx,
        )
        out = extract_one(paths.npz_path)
        out["dataset"] = paths.dataset
        out["model_tag"] = paths.model_tag
        out["run_tag"] = paths.run_tag
        out["npz_path"] = str(paths.npz_path)
        results.append(out)

    text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote -> {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()

