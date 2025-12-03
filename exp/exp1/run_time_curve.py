#!/usr/bin/env python3
"""
Measure wall-clock time and GPU memory for attribution methods across
different context lengths using a single synthetic RULER-style example.

This script stays self-contained under exp/exp1 and reuses the attribution
implementations in the repo (IG, perturbation, attention*IG, IFR/FlashTrace).
The goal is to populate the time-vs-length table; correctness of the task
content is not important, only matching token lengths and running 3 repeats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_attr
from evaluations.attribution_coverage import load_model

DEFAULT_LENGTHS = [10, 2000, 10000]
DEFAULT_ATTRS = [
    "IG",
    "attention_I_G",
    "perturbation_all",
    "perturbation_REAGENT",
    "perturbation_CLP",
    "ifr_all_positions",
    "ifr_multi_hop",
]
DEFAULT_RULER_FILE = REPO_ROOT / "data" / "ruler_multihop" / "8192" / "vt_h10_c1" / "validation.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("FlashTrace time/memory curve.")
    parser.add_argument("--model", type=str, required=True, help="Model name or HF repo id.")
    parser.add_argument("--model_path", type=str, default=None, help="Optional local model path.")
    parser.add_argument("--cuda", type=str, default=None, help='CUDA devices, e.g. "0,1" or "0".')
    parser.add_argument("--cuda_num", type=int, default=0, help="Single GPU index if --cuda is not set.")
    parser.add_argument(
        "--attr_funcs",
        type=str,
        default=",".join(DEFAULT_ATTRS),
        help="Comma-separated attribution methods.",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default=",".join(str(x) for x in DEFAULT_LENGTHS),
        help="Comma-separated target prompt token lengths (only prompt, not including target).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per cell.")
    parser.add_argument("--output_dir", type=str, default="exp/exp1/out", help="Output directory.")
    parser.add_argument(
        "--ruler_file",
        type=str,
        default=str(DEFAULT_RULER_FILE),
        help="RULER jsonl file providing a long base passage.",
    )
    parser.add_argument(
        "--chunk_tokens",
        type=int,
        default=128,
        help="IFR chunk_tokens override when context is long.",
    )
    parser.add_argument(
        "--sink_chunk_tokens",
        type=int,
        default=32,
        help="IFR sink_chunk_tokens override when context is long.",
    )
    parser.add_argument(
        "--target_text",
        type=str,
        default=" The answer is 42.",
        help="Fixed target generation to avoid variability.",
    )
    return parser.parse_args()


def resolve_device(cuda: Optional[str], cuda_num: int) -> str:
    if cuda is not None and "," in cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        return "auto"
    if cuda is not None and cuda.strip():
        try:
            idx = int(cuda)
        except Exception:
            idx = 0
        return f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    return f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"


def load_ruler_base(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    with path.open() as f:
        for line in f:
            try:
                record = json.loads(line)
                if "input" in record:
                    return record["input"]
            except json.JSONDecodeError:
                continue
    return fallback


def build_prompt_to_length(tokenizer, base_text: str, target_tokens: int) -> Tuple[str, int]:
    """
    Build a prompt whose tokenized length (without special tokens) is ~target_tokens.
    If base_text is shorter, we repeat it; if longer, we truncate.
    """
    if target_tokens <= 0:
        short = base_text.split("\n")[0]
        ids = tokenizer(short, add_special_tokens=False).input_ids[:1]
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False), len(ids)

    base_ids = tokenizer(base_text, add_special_tokens=False).input_ids
    if not base_ids:
        base_ids = [tokenizer.eos_token_id]

    tiled: List[int] = []
    while len(tiled) < target_tokens:
        tiled.extend(base_ids)
    tiled = tiled[:target_tokens]
    prompt = tokenizer.decode(tiled, clean_up_tokenization_spaces=False)
    return prompt, len(tiled)


def exceeds_model_ctx(tokenizer, prompt: str, target: str, max_ctx: Optional[int]) -> bool:
    if max_ctx is None:
        return False
    total = len(tokenizer(prompt + target, add_special_tokens=False).input_ids)
    return total > max_ctx


def maybe_reset_cuda(device_str: str) -> None:
    if torch.cuda.is_available() and device_str != "cpu":
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def measure(method_fn, device_str: str) -> Tuple[str, Optional[float], Optional[float]]:
    status = "ok"
    wall: Optional[float] = None
    mem: Optional[float] = None
    try:
        if torch.cuda.is_available() and device_str != "cpu":
            torch.cuda.synchronize()
        t0 = time.time()
        method_fn()
        if torch.cuda.is_available() and device_str != "cpu":
            torch.cuda.synchronize()
            mem = torch.cuda.max_memory_allocated() / 1e9
        wall = time.time() - t0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            status = "oom"
        else:
            status = f"runtime_error: {e}"
    except Exception as e:
        status = f"error: {e}"
    return status, wall, mem


def make_attr_runner(
    attr_func: str,
    model: Any,
    tokenizer: Any,
    chunk_tokens: int,
    sink_chunk_tokens: int,
    batch_size: int,
    prompt: str,
    target: str,
):
    lf = attr_func.lower()
    if lf == "ig":
        llm_attributor = llm_attr.LLMGradientAttribtion(model, tokenizer)

        def fn():
            return llm_attributor.calculate_IG_per_generation(
                prompt, steps=20, baseline=tokenizer.eos_token_id, batch_size=batch_size, target=target
            )

        return fn

    if lf == "attention_i_g":
        llm_attn = llm_attr.LLMAttentionAttribution(model, tokenizer)
        llm_ig = llm_attr.LLMGradientAttribtion(model, tokenizer)

        def fn():
            attn = llm_attn.calculate_attention_attribution(prompt, target=target)
            ig = llm_ig.calculate_IG_per_generation(
                prompt, steps=20, baseline=tokenizer.eos_token_id, batch_size=batch_size, target=target
            )
            attn.attribution_matrix = attn.attribution_matrix * ig.attribution_matrix
            return attn

        return fn

    if lf == "perturbation_all":
        llm_attrtor = llm_attr.LLMPerturbationAttribution(model, tokenizer)

        def fn():
            return llm_attrtor.calculate_feature_ablation_sentences(
                prompt, baseline=tokenizer.eos_token_id, measure="log_loss", target=target
            )

        return fn

    if lf == "perturbation_clp":
        llm_attrtor = llm_attr.LLMPerturbationAttribution(model, tokenizer)

        def fn():
            return llm_attrtor.calculate_feature_ablation_sentences(
                prompt, baseline=tokenizer.eos_token_id, measure="KL", target=target
            )

        return fn

    if lf == "perturbation_reagent":
        llm_attrtor = llm_attr.LLMPerturbationAttribution(model, tokenizer)

        def fn():
            return llm_attrtor.calculate_feature_ablation_sentences_mlm(prompt, target=target)

        return fn

    if lf == "ifr_all_positions":
        llm_attrtor = llm_attr.LLMIFRAttribution(
            model, tokenizer, chunk_tokens=chunk_tokens, sink_chunk_tokens=sink_chunk_tokens
        )

        def fn():
            return llm_attrtor.calculate_ifr_for_all_positions(prompt, target=target)

        return fn

    if lf == "ifr_multi_hop":
        llm_attrtor = llm_attr.LLMIFRAttribution(
            model, tokenizer, chunk_tokens=chunk_tokens, sink_chunk_tokens=sink_chunk_tokens
        )

        def fn():
            return llm_attrtor.calculate_ifr_multi_hop(prompt, target=target)

        return fn

    raise ValueError(f"Unsupported attr_func {attr_func}")


def compute_batch_size(tokenizer, prompt: str, target: str, max_input_len: int) -> int:
    denom = len(tokenizer(prompt + target, add_special_tokens=False).input_ids)
    return max(1, math.floor((max_input_len - 100) / max(1, denom)))


def aggregate_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int], Dict[str, List[float]]] = defaultdict(lambda: {"time": [], "mem": []})
    statuses: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for row in rows:
        key = (row["attr_func"], row["context_tokens"])
        statuses[key].append(row["status"])
        if row.get("time_sec") is not None:
            grouped[key]["time"].append(row["time_sec"])
        if row.get("mem_gb") is not None:
            grouped[key]["mem"].append(row["mem_gb"])

    summary = []
    for key, vals in grouped.items():
        attr_func, ctx = key
        times = vals["time"]
        mems = vals["mem"]
        summary.append(
            {
                "attr_func": attr_func,
                "context_tokens": ctx,
                "time_mean": np.mean(times) if times else None,
                "time_std": np.std(times) if times else None,
                "mem_mean": np.mean(mems) if mems else None,
                "mem_std": np.std(mems) if mems else None,
                "statuses": statuses[key],
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    device = resolve_device(args.cuda, args.cuda_num)
    attr_funcs = [a.strip() for a in args.attr_funcs.split(",") if a.strip()]
    target_lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model, tokenizer = load_model(args.model if args.model_path is None else args.model_path, device)
    max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)

    base_text = load_ruler_base(Path(args.ruler_file), fallback="RULER fallback text. ")
    all_rows: List[Dict[str, Any]] = []

    for ctx_tokens in target_lengths:
        prompt, actual_prompt_len = build_prompt_to_length(tokenizer, base_text, ctx_tokens)
        if exceeds_model_ctx(tokenizer, prompt, args.target_text, max_ctx):
            for attr in attr_funcs:
                for rep in range(args.repeats):
                    all_rows.append(
                        {
                            "attr_func": attr,
                            "context_tokens": ctx_tokens,
                            "actual_prompt_tokens": actual_prompt_len,
                            "status": "skipped_model_ctx",
                            "time_sec": None,
                            "mem_gb": None,
                            "repeat": rep,
                        }
                    )
            continue

        batch_size = compute_batch_size(
            tokenizer, prompt=prompt, target=args.target_text, max_input_len=max_ctx or 200000
        )

        for attr in attr_funcs:
            for rep in range(args.repeats):
                maybe_reset_cuda(device)
                runner = make_attr_runner(
                    attr,
                    model=model,
                    tokenizer=tokenizer,
                    chunk_tokens=args.chunk_tokens,
                    sink_chunk_tokens=args.sink_chunk_tokens,
                    batch_size=batch_size,
                    prompt=prompt,
                    target=args.target_text,
                )
                status, wall, mem = measure(runner, device_str=device)
                all_rows.append(
                    {
                        "attr_func": attr,
                        "context_tokens": ctx_tokens,
                        "actual_prompt_tokens": actual_prompt_len,
                        "status": status,
                        "time_sec": wall,
                        "mem_gb": mem,
                        "repeat": rep,
                    }
                )

    summary = aggregate_results(all_rows)

    jsonl_path = out_dir / "time_curve_runs.jsonl"
    with jsonl_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    summary_path = out_dir / "time_curve_summary.csv"
    with summary_path.open("w") as f:
        f.write("attr_func,context_tokens,time_mean,time_std,mem_mean,mem_std,statuses\n")
        for row in summary:
            f.write(
                "{},{},{},{},{},{},{}\n".format(
                    row["attr_func"],
                    row["context_tokens"],
                    "" if row["time_mean"] is None else f"{row['time_mean']:.4f}",
                    "" if row["time_std"] is None else f"{row['time_std']:.4f}",
                    "" if row["mem_mean"] is None else f"{row['mem_mean']:.4f}",
                    "" if row["mem_std"] is None else f"{row['mem_std']:.4f}",
                    "|".join(row["statuses"]),
                )
            )

    print(f"Wrote per-run records to {jsonl_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
