#!/usr/bin/env python3
"""
Experiment 2 runner: token-level faithfulness (generation perturbation).

Math is intentionally rejected. AT2 is omitted.
"""

from __future__ import annotations

import argparse
import os
import sys
from itertools import islice
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Early CUDA mask handling: set CUDA_VISIBLE_DEVICES before importing torch.
def _early_set_cuda_visible_devices():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=str, default=None)
    # parse_known_args keeps the full argv for later parsing by the main parser
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.cuda and "," in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


_early_set_cuda_visible_devices()

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

# ensure repo root on path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_attr
import llm_attr_eval
from attribution_datasets import AttributionExample
from exp.exp2 import dataset_utils as ds_utils

utils.logging.set_verbosity_error()


def load_model(model_name: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device == "auto" else {"": int(device.split(":")[1])} if device.startswith("cuda:") else None,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def resolve_device(args) -> str:
    if args.cuda is not None and "," in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        return "auto"
    if args.cuda is not None and args.cuda.strip():
        return f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    return f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"


def run_attribution(testing_dict, example: ds_utils.CachedExample, target: Optional[str]) -> List[torch.Tensor]:
    model = testing_dict["model"]
    tokenizer = testing_dict["tokenizer"]
    attr_func = testing_dict["attr_func"]

    indices_to_explain = example.indices_to_explain
    if not (isinstance(indices_to_explain, list) and len(indices_to_explain) == 2):
        raise ValueError(
            "exp2 requires token-span indices_to_explain=[start_tok,end_tok]. "
            "Please re-sample or run exp/exp2/migrate_indices_to_explain_token_span.py on your cache."
        )

    llm_attributor = None
    if "IG" in attr_func:
        llm_attributor = llm_attr.LLMGradientAttribtion(model, tokenizer)
        attr = llm_attributor.calculate_IG_per_generation(
            example.prompt,
            20,
            tokenizer.eos_token_id,
            batch_size=testing_dict["batch_size"],
            target=target,
        )
    elif "perturbation" in attr_func:
        llm_attributor = llm_attr.LLMPerturbationAttribution(model, tokenizer)
        if attr_func == "perturbation_all":
            attr = llm_attributor.calculate_feature_ablation_sentences(
                example.prompt, baseline=tokenizer.eos_token_id, measure="log_loss", target=target
            )
        elif attr_func == "perturbation_CLP":
            attr = llm_attributor.calculate_feature_ablation_sentences(
                example.prompt, baseline=tokenizer.eos_token_id, measure="KL", target=target
            )
        elif attr_func == "perturbation_REAGENT":
            attr = llm_attributor.calculate_feature_ablation_sentences_mlm(example.prompt, target=target)
        else:
            raise ValueError(f"Unsupported perturbation attr_func {attr_func}")
    elif "attention" in attr_func:
        llm_attributor = llm_attr.LLMAttentionAttribution(model, tokenizer)
        llm_attributor_ig = llm_attr.LLMGradientAttribtion(model, tokenizer)
        attr = llm_attributor.calculate_attention_attribution(example.prompt, target=target)
        attr_b = llm_attributor_ig.calculate_IG_per_generation(
            example.prompt, 20, tokenizer.eos_token_id, batch_size=testing_dict["batch_size"], target=target
        )
        attr.attribution_matrix = attr.attribution_matrix * attr_b.attribution_matrix
    elif attr_func == "ifr_all_positions":
        llm_attributor = llm_attr.LLMIFRAttribution(
            model,
            tokenizer,
            chunk_tokens=testing_dict["chunk_tokens"],
            sink_chunk_tokens=testing_dict["sink_chunk_tokens"],
        )
        attr = llm_attributor.calculate_ifr_for_all_positions(example.prompt, target=target)
    elif attr_func == "ifr_multi_hop":
        llm_attributor = llm_attr.LLMIFRAttribution(
            model,
            tokenizer,
            chunk_tokens=testing_dict["chunk_tokens"],
            sink_chunk_tokens=testing_dict["sink_chunk_tokens"],
        )
        attr = llm_attributor.calculate_ifr_multi_hop(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict["n_hops"],
        )
    elif attr_func == "attnlrp":
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        attr = llm_attributor.calculate_attnlrp(example.prompt, target=target)
    elif attr_func in ("ft_attnlrp", "attnlrp_aggregated_multi_hop"):
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        attr = llm_attributor.calculate_attnlrp_aggregated_multi_hop(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict["n_hops"],
        )
    elif attr_func == "basic":
        llm_attributor = llm_attr.LLMBasicAttribution(model, tokenizer)
        attr = llm_attributor.calculate_basic_attribution(example.prompt, target=target)
    else:
        raise ValueError(f"Unsupported attr_func {attr_func}")

    seq_attr, row_attr, rec_attr = attr.get_all_token_attrs(indices_to_explain)
    return [seq_attr, row_attr, rec_attr]


def faithfulness_generation(testing_dict, example: ds_utils.CachedExample, target: str, llm_evaluator) -> np.ndarray:
    prompt = example.prompt
    generation = target

    attr_list = run_attribution(testing_dict, example, target)
    seq_attr = attr_list[0]
    prompt_len = int(seq_attr.shape[1] - seq_attr.shape[0])  # cols=(P+G), rows=G

    results = []
    for attr in attr_list:
        # Only use prompt-side attribution, matching evaluations/faithfulness.py
        attr_prompt = attr[:, :prompt_len]
        scores = llm_evaluator.faithfulness_test(attr_prompt, prompt, generation)
        results.append(scores)

    return np.array(results)


def evaluate_dataset(args, dataset_name: str, examples: List[ds_utils.CachedExample], testing_dict):
    tokenizer = testing_dict["tokenizer"]
    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(testing_dict["model"], tokenizer)
    results = []
    durations: List[float] = []
    total = min(len(examples), args.num_examples)
    iterator = islice(examples, total)
    for ex in iterator:
        # Determine generation/target once
        target = ex.target
        if target is None:
            generation, full_output = llm_evaluator.response(ex.prompt)
            target = generation
            response_len = len(tokenizer(full_output).input_ids)
        else:
            response_len = len(tokenizer(llm_evaluator.format_prompt(" " + ex.prompt) + target).input_ids)

        # Estimate batch size (align with evaluations/coverage.py & faithfulness.py)
        testing_dict["batch_size"] = max(1, math.floor((testing_dict["max_input_len"] - 100) / max(1, response_len)))

        sample_start = time.perf_counter()
        scores = faithfulness_generation(testing_dict, ex, target, llm_evaluator)
        durations.append(time.perf_counter() - sample_start)
        results.append(scores)

    if not results:
        return None

    scores = np.array(results)
    mean = scores.mean(0)
    std = scores.std(0)
    avg_time = float(np.mean(durations)) if durations else 0.0
    return mean, std, avg_time


def main():
    parser = argparse.ArgumentParser("Experiment 2 runner (math skipped, AT2 skipped).")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated names or paths.")
    parser.add_argument("--attr_funcs", type=str, required=True, help="Comma-separated attr funcs (no AT2).")
    parser.add_argument("--model", type=str, default=None, help="HF repo id (required unless --model_path set).")
    parser.add_argument("--model_path", type=str, default=None, help="Local path; overrides --model for loading.")
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--cuda_num", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--mode", type=str, default="faithfulness_gen", choices=["faithfulness_gen"])
    parser.add_argument("--sample", type=int, default=None, help="Optional subsample before num_examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_tokens", type=int, default=128)
    parser.add_argument("--sink_chunk_tokens", type=int, default=32)
    parser.add_argument("--n_hops", type=int, default=3)
    parser.add_argument("--data_root", type=str, default="exp/exp2/data", help="Filtered dataset cache directory.")
    parser.add_argument("--output_root", type=str, default="exp/exp2/output", help="Directory to store evaluation outputs.")
    args = parser.parse_args()

    if args.model_path:
        model_name = args.model_path
    elif args.model:
        model_name = args.model
    else:
        raise SystemExit("Please set --model or --model_path.")
    model_tag = args.model if args.model else Path(args.model_path).name

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    attr_funcs = [a.strip() for a in args.attr_funcs.split(",") if a.strip()]

    device = resolve_device(args)
    model, tokenizer = load_model(model_name, device)

    max_input_len = {
        "llama-1B": 5500,
        "llama-3B": 4800,
        "llama-8B": 3500,
        "qwen-1.7B": 5500,
        "qwen-4B": 3500,
        "qwen-8B": 5000,
        "qwen-32B": 1500,
        "gemma-12B": 1500,
        "gemma-27B": 2000,
    }.get(args.model, 2000)

    for ds_name in datasets:
        if ds_name.startswith("math"):
            raise SystemExit("Math is skipped by design for exp2.")

        # Resolve dataset (prefer prepared cache under data_root)
        cached_path = Path(args.data_root) / f"{ds_name}.jsonl"
        if cached_path.exists():
            examples = ds_utils.load_cached(cached_path, sample=args.sample, seed=args.seed)
        else:
            # allow direct cached path or raw loader
            p = Path(ds_name)
            if p.exists():
                examples = ds_utils.load_cached(p, sample=args.sample, seed=args.seed)
            else:
                raise SystemExit(
                    f"Missing exp2 cache for '{ds_name}'. "
                    f"Expected {cached_path}; please run exp/exp2/sample_and_filter.py first (or pass an explicit cached JSONL path)."
                )

        for attr_func in attr_funcs:
            if attr_func.lower() == "at2":
                print("Skipping AT2 as requested.")
                continue

            testing_dict: Dict[str, any] = {
                "model": model,
                "tokenizer": tokenizer,
                "attr_func": attr_func,
                "max_input_len": max_input_len,
                "chunk_tokens": args.chunk_tokens,
                "sink_chunk_tokens": args.sink_chunk_tokens,
                "n_hops": args.n_hops,
                "device": device,
                "batch_size": 1,
            }
            result = evaluate_dataset(args, ds_name, examples, testing_dict)
            if result is None:
                print(f"No results for {ds_name} with {attr_func} (likely missing attr_mask).")
                continue
            mean, std, avg_time = result

            # Save
            out_dir = Path(args.output_root) / "faithfulness" / ds_name / model_tag
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{attr_func}_{args.num_examples}_examples.csv"
            with open(out_dir / filename, "w") as f:
                f.write("Method,RISE,MAS,RISE+AP\n")
                f.write(",".join(["Seq Attr Scores Mean"] + [str(x) for x in mean[0].tolist()]) + "\n")
                f.write(",".join(["Row Attr Scores Mean"] + [str(x) for x in mean[1].tolist()]) + "\n")
                f.write(",".join(["Recursive Attr Scores Mean"] + [str(x) for x in mean[2].tolist()]) + "\n")
                f.write(",".join(["Seq Attr Scores Var"] + [str(x) for x in std[0].tolist()]) + "\n")
                f.write(",".join(["Row Attr Scores Var"] + [str(x) for x in std[1].tolist()]) + "\n")
                f.write(",".join(["Recursive Attr Scores Var"] + [str(x) for x in std[2].tolist()]) + "\n")
                f.write(f"Avg Sample Time (s),{avg_time}\n")
            print(f"[{ds_name}] {attr_func} -> {out_dir/filename} (avg sample time: {avg_time:.2f}s)")


if __name__ == "__main__":
    main()
