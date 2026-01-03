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


def _early_set_cuda_visible_devices() -> None:
    """Parse --cuda early to set CUDA_VISIBLE_DEVICES before torch import."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.cuda and "," in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


_early_set_cuda_visible_devices()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_attr

DEFAULT_INPUT_LENGTHS = [1024, 4096, 8192]
DEFAULT_OUTPUT_LENGTHS = [32, 256, 512]
DEFAULT_ATTRS = [
    "IG",
    "perturbation_all",
    "attention_I_G",
    "perturbation_REAGENT",
    "ifr_all_positions",
    "perturbation_CLP",
    "ifr_multi_hop",
    "attnlrp",
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

    length_group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--output_lengths",
        type=str,
        default=",".join(str(x) for x in DEFAULT_OUTPUT_LENGTHS),
        help="Comma-separated target output token lengths (sink/output segment).",
    )
    length_group.add_argument(
        "--input_lengths",
        type=str,
        default=",".join(str(x) for x in DEFAULT_INPUT_LENGTHS),
        help="Comma-separated target input/prompt token lengths (user prompt only; excludes chat template).",
    )
    length_group.add_argument(
        "--total_lengths",
        "--lengths",
        dest="total_lengths",
        type=str,
        default=None,
        help="Deprecated. Target total token lengths (prompt + output). Use --input_lengths instead.",
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
        "--catch_oom",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, treat CUDA OOM as status=oom and continue; if false, let OOM raise.",
    )
    parser.add_argument(
        "--target_text",
        type=str,
        default=" The answer is 42.",
        help="Base text to tile when constructing outputs of a given length.",
    )
    return parser.parse_args()


def parse_csv_ints(value: str) -> List[int]:
    return [int(x) for x in value.split(",") if x.strip()]


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
        return "", 0

    base_ids = tokenizer(base_text, add_special_tokens=False).input_ids
    if not base_ids:
        base_ids = [tokenizer.eos_token_id]

    tiled: List[int] = []
    while len(tiled) < target_tokens:
        tiled.extend(base_ids)
    tiled = tiled[:target_tokens]
    prompt = tokenizer.decode(tiled, clean_up_tokenization_spaces=False)
    return prompt, len(tiled)


def build_output_to_length(tokenizer, base_text: str, target_tokens: int) -> Tuple[str, int]:
    """
    Build a target/output string of ~target_tokens using a base snippet.
    """
    if target_tokens <= 0:
        return "", 0

    base_ids = tokenizer(base_text, add_special_tokens=False).input_ids
    if not base_ids:
        base_ids = [tokenizer.eos_token_id]

    tiled: List[int] = []
    while len(tiled) < target_tokens:
        tiled.extend(base_ids)
    tiled = tiled[:target_tokens]
    text = tokenizer.decode(tiled, clean_up_tokenization_spaces=False)
    return text, len(tiled)


def build_formatted_prompt(tokenizer, prompt: str) -> str:
    user_prompt = " " + prompt
    modified_prompt = llm_attr.DEFAULT_PROMPT_TEMPLATE.format(context=user_prompt, query="")
    formatted_prompt = [{"role": "user", "content": modified_prompt}]
    return tokenizer.apply_chat_template(
        formatted_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def estimate_model_lengths(tokenizer, prompt: str, target: str) -> Dict[str, int]:
    user_prompt = " " + prompt
    formatted_prompt = build_formatted_prompt(tokenizer, prompt)

    user_prompt_len = len(tokenizer(user_prompt, add_special_tokens=False).input_ids)
    formatted_prompt_len = len(tokenizer(formatted_prompt, add_special_tokens=False).input_ids)
    generation_len = len(tokenizer(target + tokenizer.eos_token, add_special_tokens=False).input_ids)

    return {
        "user_prompt_tokens": user_prompt_len,
        "formatted_prompt_tokens": formatted_prompt_len,
        "generation_tokens": generation_len,
        "total_tokens": formatted_prompt_len + generation_len,
    }


def exceeds_model_ctx(tokenizer, prompt: str, target: str, max_ctx: Optional[int]) -> bool:
    if max_ctx is None:
        return False
    return estimate_model_lengths(tokenizer, prompt, target)["total_tokens"] > max_ctx


def load_model_balanced(model_name: str, device: str):
    """Load model with an explicit balanced device_map when multi-GPU is requested."""
    if device == "auto":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="balanced",
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
    elif isinstance(device, str) and device.startswith("cuda:"):
        try:
            gpu_idx = int(device.split(":")[1])
        except Exception:
            gpu_idx = 0
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map={"": gpu_idx},
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="eager",
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def collect_device_indices(device_str: str, model: Any) -> List[int]:
    """
    Infer the CUDA device indices that should be tracked for memory stats.
    Prefers the model's device map; otherwise falls back to all visible devices
    or the single requested device.
    """
    if not torch.cuda.is_available():
        return []

    devices: set[int] = set()
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for dev in device_map.values():
            if dev is None:
                continue
            idx: Optional[int] = None
            if isinstance(dev, torch.device):
                idx = dev.index if dev.index is not None else (0 if dev.type == "cuda" else None)
            elif isinstance(dev, str):
                try:
                    d = torch.device(dev)
                    idx = d.index if d.index is not None else (0 if d.type == "cuda" else None)
                except Exception:
                    idx = None
            elif isinstance(dev, int):
                idx = dev
            if idx is not None:
                devices.add(idx)

    if not devices:
        if device_str == "auto":
            devices.update(range(torch.cuda.device_count()))
        elif isinstance(device_str, str) and device_str.startswith("cuda:"):
            try:
                devices.add(int(device_str.split(":")[1]))
            except Exception:
                pass
        else:
            devices.update(range(torch.cuda.device_count()))

    return sorted(devices)


def maybe_reset_cuda(device_indices: List[int]) -> None:
    if not torch.cuda.is_available() or not device_indices:
        return
    for idx in device_indices:
        try:
            torch.cuda.reset_peak_memory_stats(device=idx)
        except Exception:
            pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def measure(
    method_fn,
    device_indices: List[int],
    *,
    catch_oom: bool,
) -> Tuple[str, Optional[float], Optional[float], Optional[float], Dict[int, Dict[str, float]]]:
    status = "ok"
    wall: Optional[float] = None
    mem_alloc: Optional[float] = None
    mem_reserved: Optional[float] = None
    mem_by_device: Dict[int, Dict[str, float]] = {}
    try:
        if torch.cuda.is_available() and device_indices:
            for idx in device_indices:
                torch.cuda.synchronize(device=idx)
        t0 = time.time()
        method_fn()
        if torch.cuda.is_available() and device_indices:
            for idx in device_indices:
                torch.cuda.synchronize(device=idx)
        wall = time.time() - t0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            status = "oom"
            if not catch_oom:
                raise
        else:
            status = f"runtime_error: {e}"
            if not catch_oom:
                raise
    except Exception as e:
        status = f"error: {e}"
        if not catch_oom:
            raise
    finally:
        if torch.cuda.is_available() and device_indices:
            try:
                total_alloc = 0.0
                total_reserved = 0.0
                for idx in device_indices:
                    alloc_bytes = torch.cuda.max_memory_allocated(device=idx)
                    reserved_bytes = torch.cuda.max_memory_reserved(device=idx)
                    total_alloc += alloc_bytes
                    total_reserved += reserved_bytes
                    mem_by_device[idx] = {
                        "allocated_gb": alloc_bytes / 1e9,
                        "reserved_gb": reserved_bytes / 1e9,
                    }
                mem_alloc = total_alloc / 1e9
                mem_reserved = total_reserved / 1e9
            except Exception:
                pass
    return status, wall, mem_alloc, mem_reserved, mem_by_device


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
            model, tokenizer, chunk_tokens=chunk_tokens, sink_chunk_tokens=1
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

    if lf == "ifr_multi_hop_both":
        import ft_ifr_improve

        llm_attrtor = ft_ifr_improve.LLMIFRAttributionBoth(
            model, tokenizer, chunk_tokens=chunk_tokens, sink_chunk_tokens=sink_chunk_tokens
        )

        def fn():
            return llm_attrtor.calculate_ifr_multi_hop_both(prompt, target=target)

        return fn

    if lf == "attnlrp":
        llm_attrtor = llm_attr.LLMLRPAttribution(model, tokenizer)

        def fn():
            return llm_attrtor.calculate_attnlrp(prompt, target=target)

        return fn

    raise ValueError(f"Unsupported attr_func {attr_func}")


def compute_batch_size(sequence_length: int, max_input_len: int) -> int:
    denom = int(sequence_length)
    return max(1, math.floor((max_input_len - 100) / max(1, denom)))


def aggregate_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, int], Dict[str, List[float]]] = defaultdict(lambda: {"time": [], "mem": []})
    statuses: Dict[Tuple[str, int, int], List[str]] = defaultdict(list)
    for row in rows:
        key = (row["attr_func"], row["target_input_tokens"], row["target_output_tokens"])
        statuses[key].append(row["status"])
        if row.get("time_sec") is not None:
            grouped[key]["time"].append(row["time_sec"])
        if row.get("peak_mem_gb") is not None:
            grouped[key]["mem"].append(row["peak_mem_gb"])

    summary = []
    for key, vals in grouped.items():
        attr_func, input_tokens, output_tokens = key
        total_tokens = input_tokens + output_tokens
        times = vals["time"]
        mems = vals["mem"]
        summary.append(
            {
                "attr_func": attr_func,
                "target_input_tokens": input_tokens,
                "target_total_tokens": total_tokens,
                "target_output_tokens": output_tokens,
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
    target_output_lengths = parse_csv_ints(args.output_lengths)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model_name = args.model if args.model_path is None else args.model_path
    model, tokenizer = load_model_balanced(model_name, device)
    device_indices = collect_device_indices(device, model)
    max_ctx = getattr(getattr(model, "config", None), "max_position_embeddings", None)

    base_text = load_ruler_base(Path(args.ruler_file), fallback="RULER fallback text. ")
    target_base = args.target_text
    all_rows: List[Dict[str, Any]] = []

    using_deprecated_total = args.total_lengths is not None
    if using_deprecated_total:
        target_total_lengths = parse_csv_ints(args.total_lengths)
        length_grid: List[Tuple[int, int, int]] = []
        for total_tokens in target_total_lengths:
            for output_tokens in target_output_lengths:
                length_grid.append((total_tokens - output_tokens, output_tokens, total_tokens))
    else:
        target_input_lengths = parse_csv_ints(args.input_lengths)
        length_grid = []
        for input_tokens in target_input_lengths:
            for output_tokens in target_output_lengths:
                length_grid.append((input_tokens, output_tokens, input_tokens + output_tokens))

    for input_tokens, output_tokens, total_tokens in length_grid:
        if input_tokens <= 0:
            for attr in attr_funcs:
                for rep in range(args.repeats):
                    all_rows.append(
                        {
                            "attr_func": attr,
                            "target_input_tokens": input_tokens,
                            "target_output_tokens": output_tokens,
                            "target_total_tokens": total_tokens,
                            "actual_input_tokens": None,
                            "actual_output_tokens": None,
                            "actual_total_tokens_raw": None,
                            "actual_user_prompt_tokens": None,
                            "actual_formatted_prompt_tokens": None,
                            "actual_generation_tokens": None,
                            "actual_total_tokens": None,
                            "status": "skipped_nonpositive_input",
                            "time_sec": None,
                            "peak_mem_gb": None,
                            "peak_mem_reserved_gb": None,
                            "repeat": rep,
                            "used_deprecated_total_lengths": using_deprecated_total,
                        }
                    )
            continue

        prompt, actual_input_len = build_prompt_to_length(tokenizer, base_text, input_tokens)
        target, actual_output_len = build_output_to_length(tokenizer, target_base, output_tokens)
        actual_total_tokens_raw = len(tokenizer(prompt + target, add_special_tokens=False).input_ids)
        model_lens = estimate_model_lengths(tokenizer, prompt, target)

        if max_ctx is not None and model_lens["total_tokens"] > max_ctx:
            for attr in attr_funcs:
                for rep in range(args.repeats):
                    all_rows.append(
                        {
                            "attr_func": attr,
                            "target_input_tokens": input_tokens,
                            "target_output_tokens": output_tokens,
                            "target_total_tokens": total_tokens,
                            "actual_input_tokens": actual_input_len,
                            "actual_output_tokens": actual_output_len,
                            "actual_total_tokens_raw": actual_total_tokens_raw,
                            "actual_user_prompt_tokens": model_lens["user_prompt_tokens"],
                            "actual_formatted_prompt_tokens": model_lens["formatted_prompt_tokens"],
                            "actual_generation_tokens": model_lens["generation_tokens"],
                            "actual_total_tokens": model_lens["total_tokens"],
                            "status": "skipped_model_ctx",
                            "time_sec": None,
                            "peak_mem_gb": None,
                            "peak_mem_reserved_gb": None,
                            "repeat": rep,
                            "used_deprecated_total_lengths": using_deprecated_total,
                        }
                    )
            continue

        batch_size = compute_batch_size(model_lens["total_tokens"], max_input_len=max_ctx or 200000)

        for attr in attr_funcs:
            for rep in range(args.repeats):
                maybe_reset_cuda(device_indices)
                runner = make_attr_runner(
                    attr,
                    model=model,
                    tokenizer=tokenizer,
                    chunk_tokens=args.chunk_tokens,
                    sink_chunk_tokens=args.sink_chunk_tokens,
                    batch_size=batch_size,
                    prompt=prompt,
                    target=target,
                )
                status, wall, mem_alloc, mem_reserved, mem_by_device = measure(
                    runner, device_indices=device_indices, catch_oom=args.catch_oom
                )
                all_rows.append(
                    {
                        "attr_func": attr,
                        "target_input_tokens": input_tokens,
                        "target_output_tokens": output_tokens,
                        "target_total_tokens": total_tokens,
                        "actual_input_tokens": actual_input_len,
                        "actual_output_tokens": actual_output_len,
                        "actual_total_tokens_raw": actual_total_tokens_raw,
                        "actual_user_prompt_tokens": model_lens["user_prompt_tokens"],
                        "actual_formatted_prompt_tokens": model_lens["formatted_prompt_tokens"],
                        "actual_generation_tokens": model_lens["generation_tokens"],
                        "actual_total_tokens": model_lens["total_tokens"],
                        "status": status,
                        "time_sec": wall,
                        "peak_mem_gb": mem_reserved if mem_reserved is not None else mem_alloc,
                        "peak_mem_reserved_gb": mem_reserved,
                        "peak_mem_by_device_gb": mem_by_device if mem_by_device else None,
                        "repeat": rep,
                        "used_deprecated_total_lengths": using_deprecated_total,
                    }
                )

    summary = aggregate_results(all_rows)

    jsonl_path = out_dir / "time_curve_runs.jsonl"
    with jsonl_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    summary_path = out_dir / "time_curve_summary.csv"
    with summary_path.open("w") as f:
        f.write(
            "attr_func,target_input_tokens,target_output_tokens,target_total_tokens,time_mean,time_std,peak_mem_mean,peak_mem_std,statuses\n"
        )
        for row in summary:
            f.write(
                "{},{},{},{},{},{},{},{},{}\n".format(
                    row["attr_func"],
                    row["target_input_tokens"],
                    row["target_output_tokens"],
                    row["target_total_tokens"],
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
