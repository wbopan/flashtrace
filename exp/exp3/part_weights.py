#!/usr/bin/env python3
"""
Compute attribution mass on (input, cot, output) segments from exp3 trace npz files.

Definitions (token-level, aligned with exp2/exp3 runners):
- input  : prompt-side tokens (user prompt), indices [0, prompt_len)
- cot    : generation tokens in thinking span, indices [prompt_len + t0, prompt_len + t1]
- output : generation tokens in sink span (answer), indices [prompt_len + s0, prompt_len + s1]

The trace stores token-importance vectors:
  - v_seq_all, v_row_all, v_rec_all  (length = prompt_len + gen_len)

This script sums those vectors over each segment and reports both absolute sums
and fractions of the total sum.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class TraceRun:
    dataset: str
    model: str
    run_dir: Path


def _pick_single_subdir(parent: Path) -> Path:
    subdirs = [p for p in parent.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found under {parent}")
    if len(subdirs) == 1:
        return subdirs[0]
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs[0]


def _resolve_run(
    trace_root: Path,
    *,
    dataset: str,
    model: Optional[str],
    run_tag: Optional[str],
) -> TraceRun:
    ds_dir = trace_root / dataset
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset trace directory not found: {ds_dir}")

    if model is None:
        model_dir = _pick_single_subdir(ds_dir)
    else:
        model_dir = ds_dir / model
        if not model_dir.exists():
            raise FileNotFoundError(f"Model trace directory not found: {model_dir}")

    if run_tag is None:
        run_dir = _pick_single_subdir(model_dir)
    else:
        run_dir = model_dir / run_tag
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

    return TraceRun(dataset=dataset, model=model_dir.name, run_dir=run_dir)


def _iter_manifest(run_dir: Path) -> Iterable[dict]:
    manifest = run_dir / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _as_span(arr: np.ndarray, *, name: str) -> Tuple[int, int]:
    if arr is None:
        raise ValueError(f"Missing {name} in trace npz.")
    a = np.asarray(arr).reshape(-1)
    if a.size != 2:
        raise ValueError(f"Expected {name} to have 2 ints, got shape {a.shape}.")
    return int(a[0]), int(a[1])


def _segment_sums(
    v: np.ndarray,
    *,
    prompt_len: int,
    gen_len: int,
    thinking_span_gen: Optional[Tuple[int, int]],
    sink_span_gen: Optional[Tuple[int, int]],
) -> Dict[str, float]:
    total_len = int(prompt_len) + int(gen_len)
    if int(v.shape[0]) != total_len:
        raise ValueError(f"Vector length mismatch: len(v)={int(v.shape[0])} vs prompt_len+gen_len={total_len}.")

    v = np.asarray(v, dtype=np.float64).reshape(-1)
    prompt_len = int(prompt_len)
    gen_len = int(gen_len)

    # Default: no cot/output when spans missing (should not happen in exp3).
    think_start, think_end = (0, -1) if thinking_span_gen is None else thinking_span_gen
    sink_start, sink_end = (0, -1) if sink_span_gen is None else sink_span_gen

    # Clamp spans into [0, gen_len-1].
    def _clamp_span(a: int, b: int) -> Tuple[int, int]:
        a = max(0, min(int(a), gen_len - 1))
        b = max(0, min(int(b), gen_len - 1))
        if b < a:
            return 0, -1
        return a, b

    think_start, think_end = _clamp_span(think_start, think_end)
    sink_start, sink_end = _clamp_span(sink_start, sink_end)

    mask = np.zeros((total_len,), dtype=bool)
    # input = all prompt tokens
    input_slice = slice(0, prompt_len)
    mask[input_slice] = True

    cot_slice = slice(prompt_len + think_start, prompt_len + think_end + 1) if think_end >= think_start else slice(0, 0)
    output_slice = slice(prompt_len + sink_start, prompt_len + sink_end + 1) if sink_end >= sink_start else slice(0, 0)
    mask[cot_slice] = True
    mask[output_slice] = True

    input_sum = float(v[input_slice].sum())
    cot_sum = float(v[cot_slice].sum()) if think_end >= think_start else 0.0
    output_sum = float(v[output_slice].sum()) if sink_end >= sink_start else 0.0
    other_sum = float(v[~mask].sum())
    total_sum = float(v.sum())

    return {
        "total": total_sum,
        "input": input_sum,
        "cot": cot_sum,
        "output": output_sum,
        "other": other_sum,
    }


def _with_fracs(sums: Dict[str, float]) -> Dict[str, float]:
    total = float(sums.get("total") or 0.0)
    if total <= 0.0:
        return {**sums, "input_frac": float("nan"), "cot_frac": float("nan"), "output_frac": float("nan"), "other_frac": float("nan")}
    return {
        **sums,
        "input_frac": float(sums["input"]) / total,
        "cot_frac": float(sums["cot"]) / total,
        "output_frac": float(sums["output"]) / total,
        "other_frac": float(sums["other"]) / total,
    }


def _analyze_npz(npz_path: Path) -> Dict[str, dict]:
    d = np.load(npz_path)
    prompt_len = int(np.asarray(d["prompt_len"]).item())
    gen_len = int(np.asarray(d["gen_len"]).item())
    thinking_span_gen = _as_span(d["thinking_span_gen"], name="thinking_span_gen") if "thinking_span_gen" in d.files else None
    sink_span_gen = _as_span(d["sink_span_gen"], name="sink_span_gen") if "sink_span_gen" in d.files else None

    out: Dict[str, dict] = {"prompt_len": prompt_len, "gen_len": gen_len}
    for key in ("v_seq_all", "v_row_all", "v_rec_all"):
        if key not in d.files:
            raise ValueError(f"Missing {key} in trace npz: {npz_path}")
        sums = _segment_sums(
            d[key],
            prompt_len=prompt_len,
            gen_len=gen_len,
            thinking_span_gen=thinking_span_gen,
            sink_span_gen=sink_span_gen,
        )
        out[key] = _with_fracs(sums)
    out["thinking_span_gen"] = list(thinking_span_gen) if thinking_span_gen is not None else None
    out["sink_span_gen"] = list(sink_span_gen) if sink_span_gen is not None else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser("Summarize input/cot/output attribution mass from exp3 traces.")
    parser.add_argument("--trace_root", type=str, default="exp/exp3/output/traces")
    parser.add_argument("--dataset_tag", type=str, default="niah_mq_q2", help="Base tag; expands to <tag>_short_cot and <tag>_long_cot.")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names (overrides --dataset_tag expansion).")
    parser.add_argument("--model", type=str, default=None, help="Model directory name under traces (default: auto if single).")
    parser.add_argument("--run_tag", type=str, default=None, help="Run tag directory (default: auto pick newest/single).")
    args = parser.parse_args()

    trace_root = Path(args.trace_root)
    if not trace_root.exists():
        raise SystemExit(f"trace_root not found: {trace_root}")

    if args.datasets:
        datasets = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    else:
        datasets = [f"{args.dataset_tag}_short_cot", f"{args.dataset_tag}_long_cot"]

    for ds in datasets:
        run = _resolve_run(trace_root, dataset=ds, model=args.model, run_tag=args.run_tag)
        records = list(_iter_manifest(run.run_dir))
        if not records:
            raise SystemExit(f"Empty manifest: {run.run_dir/'manifest.jsonl'}")
        for rec in records:
            npz_path = run.run_dir / str(rec["file"])
            analysis = _analyze_npz(npz_path)
            print(
                json.dumps(
                    {
                        "dataset": run.dataset,
                        "model": run.model,
                        "run_dir": str(run.run_dir),
                        "example_idx": int(rec.get("example_idx", -1)),
                        **analysis,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()

