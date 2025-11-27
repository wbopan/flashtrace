#!/usr/bin/env python3
"""
Run IG and FlashTrace (ifr_multi_hop) on the first k math samples and dump full
attribution artifacts for case study. This reuses the same logic and defaults
as the evaluation scripts, but keeps everything self-contained under exp/case_study.
"""

import argparse
import json
import math
import os
import sys
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_attr
import llm_attr_eval
from attribution_datasets import MathAttributionDataset
from evaluations.attribution_coverage import load_model


MODEL_MAX_INPUT_LEN = {
    "llama-1B": 5500,
    "llama-3B": 4800,
    "llama-8B": 3500,
    "qwen-1.7B": 5500,
    "qwen-4B": 3500,
    "qwen-8B": 3000,
    "qwen-32B": 1500,
    "gemma-12B": 1500,
    "gemma-27B": 2000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Case runner for IG and FlashTrace on math.")
    parser.add_argument("--model", type=str, default="qwen-8B")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--cuda_num", type=int, default=0)
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default="./data/math_mine.json")
    parser.add_argument("--k", type=int, default=5, help="Number of math samples to run.")
    parser.add_argument(
        "--attr_funcs",
        type=str,
        default="IG,ifr_multi_hop",
        help="Comma-separated attribution methods to run.",
    )
    parser.add_argument(
        "--coverage_mode",
        type=str,
        default="prompt",
        choices=["prompt", "all", "input"],
        help="Which columns to use when computing coverage.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="exp/case_study/out",
        help="Directory to write JSONL case files.",
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=None,
        help="Override max input length; defaults to evaluation preset for model.",
    )
    return parser.parse_args()


def resolve_device(cuda: Optional[str], cuda_num: int) -> str:
    if cuda is not None and isinstance(cuda, str) and "," in cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda
        return "auto"
    if cuda is not None and isinstance(cuda, str) and cuda.strip() != "":
        try:
            idx = int(cuda)
        except Exception:
            idx = 0
        return f"cuda:{idx}" if torch.cuda.is_available() else "cpu"
    return f"cuda:{cuda_num}" if torch.cuda.is_available() else "cpu"


def resolve_max_input_len(model: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    return MODEL_MAX_INPUT_LEN.get(model, 2000)


def compute_batch_size(
    tokenizer: Any, llm_eval: llm_attr_eval.LLMAttributionEvaluator, prompt: str, target: Optional[str], max_input_len: int
) -> int:
    if target is None:
        _, full_output = llm_eval.response(prompt)
        denom = len(tokenizer(full_output).input_ids)
    else:
        denom = len(tokenizer(llm_eval.format_prompt(prompt) + target).input_ids)
    return max(1, math.floor((max_input_len - 100) / denom))


def run_attr(
    attr_func: str,
    model: Any,
    tokenizer: Any,
    prompt: str,
    target: Optional[str],
    batch_size: int,
) -> llm_attr.LLMAttributionResult:
    attr_key = attr_func.lower()
    if attr_key == "ig":
        llm_attributor = llm_attr.LLMGradientAttribtion(model, tokenizer)
        return llm_attributor.calculate_IG_per_generation(
            prompt,
            steps=20,
            baseline=tokenizer.eos_token_id,
            batch_size=batch_size,
            captum_version=False,
            target=target,
        )
    if attr_key == "ifr_multi_hop":
        llm_attributor = llm_attr.LLMIFRAttribution(model, tokenizer)
        return llm_attributor.calculate_ifr_multi_hop(prompt, target=target)
    raise ValueError(f"Unsupported attr_func {attr_func}")


def tensor_to_list(t: Optional[torch.Tensor]) -> Optional[List[Any]]:
    if t is None:
        return None
    return t.detach().cpu().tolist()


def sanitize_ifr_meta(meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if meta is None:
        return None
    cleaned: Dict[str, Any] = {}
    for key, value in meta.items():
        if key == "raw":
            continue
        if key == "per_hop_projected":
            cleaned[key] = [tensor_to_list(v) for v in value]
            continue
        if key == "observation_projected":
            obs = {}
            for obs_k, obs_v in value.items():
                if isinstance(obs_v, list):
                    obs[obs_k] = [tensor_to_list(v) for v in obs_v]
                elif torch.is_tensor(obs_v):
                    obs[obs_k] = tensor_to_list(obs_v)
                else:
                    obs[obs_k] = obs_v
            cleaned[key] = obs
            continue
        if torch.is_tensor(value):
            cleaned[key] = tensor_to_list(value)
        else:
            cleaned[key] = value
    return cleaned


def coverage_full_attr(seq_attr: torch.Tensor, prompt_sentences: Sequence[str], coverage_mode: str) -> torch.Tensor:
    if coverage_mode in ("prompt", "input"):
        return seq_attr[:, : len(prompt_sentences)]
    if coverage_mode == "all":
        return seq_attr
    raise ValueError(f"Unsupported coverage mode {coverage_mode}")


def make_record_base(
    example_idx: int,
    attr_func: str,
    example: Any,
    attr_result: llm_attr.LLMAttributionResult,
    tokenizer: Any,
) -> Dict[str, Any]:
    generation_text = "".join(attr_result.generation_tokens) if attr_result.generation_tokens is not None else ""
    if tokenizer.eos_token:
        generation_text = generation_text.replace(tokenizer.eos_token, "")
    return {
        "example_idx": example_idx,
        "attr_func": attr_func,
        "prompt": example.prompt,
        "target": example.target,
        "indices_to_explain": example.indices_to_explain,
        "attr_mask_indices": example.attr_mask_indices,
        "generation": generation_text,
        "prompt_tokens": attr_result.prompt_tokens,
        "generation_tokens": attr_result.generation_tokens,
        "prompt_sentences": attr_result.prompt_sentences,
        "generation_sentences": attr_result.generation_sentences,
        "all_sentences": attr_result.all_sentences,
    }


def run_cases(args: argparse.Namespace) -> None:
    device = resolve_device(args.cuda, args.cuda_num)
    max_input_len = resolve_max_input_len(args.model, args.max_input_len)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = args.model_path if args.model_path is not None else args.model
    model, tokenizer = load_model(model_name, device)

    dataset = MathAttributionDataset(args.dataset_path, tokenizer)
    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(model, tokenizer)

    attr_funcs = [a.strip() for a in args.attr_funcs.split(",") if a.strip()]
    examples: Iterable[Any] = islice(dataset, args.k)
    examples = list(examples)

    for attr_func in attr_funcs:
        coverage_path = output_dir / f"coverage_math_{args.model}_{attr_func}.jsonl"
        faithfulness_path = output_dir / f"faithfulness_math_{args.model}_{attr_func}.jsonl"
        with coverage_path.open("w", encoding="utf-8") as f_cov, faithfulness_path.open("w", encoding="utf-8") as f_faith:
            for idx, example in enumerate(examples):
                batch_size = compute_batch_size(tokenizer, llm_evaluator, example.prompt, example.target, max_input_len)
                attr_result = run_attr(attr_func, model, tokenizer, example.prompt, example.target, batch_size)

                seq_attr, row_attr, rec_attr = attr_result.get_all_sentence_attrs(
                    example.indices_to_explain if example.indices_to_explain is not None else [-2]
                )
                prompt_sentences = llm_attr_eval.create_sentences(" " + example.prompt, tokenizer)

                coverage_scores: Optional[Tuple[float, float, float]] = None
                if example.attr_mask_indices is not None:
                    full_attr = coverage_full_attr(seq_attr, prompt_sentences, args.coverage_mode)
                    partial_attr = seq_attr[:, example.attr_mask_indices]
                    coverage_raw = llm_evaluator.evaluate_attr_coverage(full_attr, partial_attr)
                    coverage_scores = tuple(float(x) for x in coverage_raw)

                generation_for_faith = example.target
                if generation_for_faith is None:
                    generation_for_faith = "".join(attr_result.generation_tokens) if attr_result.generation_tokens is not None else ""
                    if tokenizer.eos_token:
                        generation_for_faith = generation_for_faith.replace(tokenizer.eos_token, "")
                faithfulness_raw = llm_evaluator.faithfulness_test(
                    seq_attr[:, : len(prompt_sentences)],
                    prompt_sentences,
                    generation_for_faith,
                )
                faithfulness_scores = tuple(float(x) for x in faithfulness_raw)

                base_record = make_record_base(idx, attr_func, example, attr_result, tokenizer)
                common_payload = {
                    "seq_attr": tensor_to_list(seq_attr),
                    "row_attr": tensor_to_list(row_attr),
                    "rec_attr": tensor_to_list(rec_attr),
                    "ifr_meta": sanitize_ifr_meta(attr_result.metadata.get("ifr") if attr_result.metadata else None),
                }

                cov_record = base_record | common_payload | {"coverage": coverage_scores}
                f_cov.write(json.dumps(cov_record, ensure_ascii=False) + "\n")

                faith_record = base_record | common_payload | {
                    "faithfulness": {
                        "RISE": faithfulness_scores[0],
                        "MAS": faithfulness_scores[1],
                        "RISE_AP": faithfulness_scores[2],
                    }
                }
                f_faith.write(json.dumps(faith_record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    run_cases(args)


if __name__ == "__main__":
    main()
