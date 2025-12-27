#!/usr/bin/env python3
"""
Experiment 2 runner: token-level faithfulness (generation perturbation).

Math is intentionally rejected. AT2 is omitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from itertools import islice
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _infer_attnlrp_spans_from_hops(
    raw_attributions: Any,
    *,
    gen_len: int,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    if not raw_attributions:
        return (0, max(0, gen_len - 1)), (0, max(0, gen_len - 1))
    sink_span = tuple(int(x) for x in raw_attributions[0].sink_range)
    if len(raw_attributions) >= 2:
        thinking_span = tuple(int(x) for x in raw_attributions[1].sink_range)
    else:
        thinking_span = sink_span
    return sink_span, thinking_span


def _build_hop_trace_payload(
    attr_func: str,
    attr: Any,
    *,
    indices_to_explain: List[int],
) -> Optional[Dict[str, np.ndarray]]:
    """Extract per-hop vectors (postprocessed) and minimal span metadata."""
    prompt_len = int(len(getattr(attr, "prompt_tokens", []) or []))
    gen_len = int(len(getattr(attr, "generation_tokens", []) or []))
    total_len = prompt_len + gen_len
    if total_len <= 0:
        return None

    hop_vectors: List[torch.Tensor] = []
    sink_span_gen: Optional[Tuple[int, int]] = None
    thinking_span_gen: Optional[Tuple[int, int]] = None
    attnlrp_neg_handling: str = ""
    attnlrp_norm_mode: str = ""
    attnlrp_ratio_enabled: int = -1

    if attr_func == "ifr_multi_hop":
        meta = (getattr(attr, "metadata", None) or {}).get("ifr") or {}
        per_hop = meta.get("per_hop_projected") or []
        if not per_hop:
            return None
        hop_vectors = [torch.as_tensor(v, dtype=torch.float32) for v in per_hop]
        sink_span_gen = meta.get("sink_span_generation")
        thinking_span_gen = meta.get("thinking_span_generation")
        if sink_span_gen is not None:
            sink_span_gen = tuple(int(x) for x in sink_span_gen)
        if thinking_span_gen is not None:
            thinking_span_gen = tuple(int(x) for x in thinking_span_gen)

    elif attr_func in ("ft_attnlrp", "attnlrp_aggregated_multi_hop"):
        meta = getattr(attr, "metadata", None) or {}
        attnlrp_neg_handling = str(meta.get("neg_handling") or "")
        attnlrp_norm_mode = str(meta.get("norm_mode") or "")
        if meta.get("ratio_enabled") is not None:
            attnlrp_ratio_enabled = int(bool(meta.get("ratio_enabled")))
        multi_hop = meta.get("multi_hop_result")
        if multi_hop is None:
            return None
        raw_attributions = getattr(multi_hop, "raw_attributions", None) or []
        if not raw_attributions:
            return None
        hop_vectors = [
            torch.as_tensor(getattr(hop, "token_importance_total"), dtype=torch.float32)
            for hop in raw_attributions
        ]
        sink_span_gen, thinking_span_gen = _infer_attnlrp_spans_from_hops(raw_attributions, gen_len=gen_len)
        sink_override = meta.get("sink_span")
        thinking_override = meta.get("thinking_span")
        if sink_override is not None:
            sink_span_gen = tuple(int(x) for x in sink_override)
        if thinking_override is not None:
            thinking_span_gen = tuple(int(x) for x in thinking_override)

    else:
        return None

    if sink_span_gen is None:
        sink_span_gen = (0, max(0, gen_len - 1))
    if thinking_span_gen is None:
        thinking_span_gen = sink_span_gen

    stacked = torch.stack([v.reshape(-1) for v in hop_vectors], dim=0)
    if stacked.shape[1] != total_len:
        raise ValueError(
            f"Hop vector length mismatch for {attr_func}: expected T={total_len}, got {stacked.shape[1]}."
        )

    return {
        "vh": stacked.detach().cpu().numpy().astype(np.float32, copy=False),
        "prompt_len": np.asarray(prompt_len, dtype=np.int64),
        "gen_len": np.asarray(gen_len, dtype=np.int64),
        "sink_span_gen": np.asarray(sink_span_gen, dtype=np.int64),
        "thinking_span_gen": np.asarray(thinking_span_gen, dtype=np.int64),
        "indices_to_explain_gen": np.asarray(indices_to_explain, dtype=np.int64),
        "attnlrp_neg_handling": np.asarray(attnlrp_neg_handling, dtype="U16"),
        "attnlrp_norm_mode": np.asarray(attnlrp_norm_mode, dtype="U16"),
        "attnlrp_ratio_enabled": np.asarray(attnlrp_ratio_enabled, dtype=np.int64),
    }


def _write_hop_trace(
    trace_dir: Path,
    *,
    example_idx: int,
    attr_func: str,
    prompt: str,
    target: Optional[str],
    payload: Dict[str, np.ndarray],
    manifest_handle,
) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    npz_name = f"ex_{example_idx:06d}.npz"
    npz_path = trace_dir / npz_name
    np.savez_compressed(npz_path, **payload)

    record = {
        "example_idx": int(example_idx),
        "attr_func": attr_func,
        "file": npz_name,
        "prompt_sha1": _sha1_text(prompt),
        "target_sha1": _sha1_text(target) if target is not None else None,
        "prompt_len": int(payload["prompt_len"].item()),
        "gen_len": int(payload["gen_len"].item()),
        "n_hops_plus_one": int(payload["vh"].shape[0]),
        "total_len": int(payload["vh"].shape[1]),
        "sink_span_gen": payload["sink_span_gen"].tolist(),
        "thinking_span_gen": payload["thinking_span_gen"].tolist(),
        "indices_to_explain_gen": payload["indices_to_explain_gen"].tolist(),
        "attnlrp_neg_handling": str(payload["attnlrp_neg_handling"].item()),
        "attnlrp_norm_mode": str(payload["attnlrp_norm_mode"].item()),
        "attnlrp_ratio_enabled": int(payload["attnlrp_ratio_enabled"].item()),
    }
    manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_handle.flush()


def _faithfulness_test_with_user_prompt_indices(
    llm_evaluator: llm_attr_eval.LLMAttributionEvaluator,
    attribution: torch.Tensor,
    prompt: str,
    generation: str,
    *,
    user_prompt_indices: List[int],
) -> Tuple[float, float, float]:
    """Token-level MAS/RISE faithfulness via guided deletion using provided prompt indices.

    This mirrors llm_attr_eval.LLMAttributionEvaluator.faithfulness_test, but avoids
    locating the user prompt span via token-id subsequence matching (which may fail
    for some tokenizers due to non-compositional BPE merges at template boundaries).
    """

    def auc(arr: np.ndarray) -> float:
        return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / max(1, (arr.shape[0] - 1))

    pad_token_id = llm_evaluator._ensure_pad_token_id()

    user_prompt = " " + prompt
    formatted_prompt = llm_evaluator.format_prompt(user_prompt)
    formatted_ids = llm_evaluator.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids

    prompt_ids = formatted_ids.to(llm_evaluator.device)
    prompt_ids_perturbed = prompt_ids.clone()
    generation_ids = llm_evaluator.tokenizer(
        generation + llm_evaluator.tokenizer.eos_token,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(llm_evaluator.device)

    attr_cpu = attribution.detach().cpu()
    w = attr_cpu.sum(0)
    sorted_attr_indices = torch.argsort(w, descending=True)
    attr_sum = float(w.sum().item())

    P = int(w.numel())
    if len(user_prompt_indices) != P:
        raise ValueError(
            "user_prompt_indices length does not match prompt-side attribution length: "
            f"indices P={len(user_prompt_indices)}, attr P={P}."
        )
    if P == 0:
        return 0.0, 0.0, 0.0

    if max(user_prompt_indices) >= int(prompt_ids_perturbed.shape[1]):
        raise ValueError("user_prompt_indices contains an out-of-bounds index for formatted prompt ids.")

    scores = np.zeros(P + 1, dtype=np.float64)
    density = np.zeros(P + 1, dtype=np.float64)

    scores[0] = (
        llm_evaluator.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
    )
    density[0] = 1.0

    if attr_sum <= 0:
        density = np.linspace(1.0, 0.0, P + 1)

    for i, idx in enumerate(sorted_attr_indices):
        j = int(idx.item())
        abs_pos = int(user_prompt_indices[j])
        prompt_ids_perturbed[0, abs_pos] = pad_token_id
        scores[i + 1] = (
            llm_evaluator.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
        )
        if attr_sum > 0:
            density[i + 1] = density[i] - (float(w[j].item()) / attr_sum)

    min_normalized_pred = 1.0
    normalized_model_response = scores.copy()
    for i in range(len(scores)):
        normalized_pred = (normalized_model_response[i] - scores[-1]) / (abs(scores[0] - scores[-1]))
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        min_normalized_pred = min(min_normalized_pred, normalized_pred)
        normalized_model_response[i] = min_normalized_pred

    alignment_penalty = np.abs(normalized_model_response - density)
    corrected_scores = normalized_model_response + alignment_penalty
    corrected_scores = corrected_scores.clip(0.0, 1.0)
    corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))

    if np.isnan(corrected_scores).any():
        corrected_scores = np.linspace(1.0, 0.0, len(scores))

    return auc(normalized_model_response), auc(corrected_scores), auc(normalized_model_response + alignment_penalty)


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


def run_attribution(
    testing_dict, example: ds_utils.CachedExample, target: Optional[str]
) -> Tuple[List[torch.Tensor], Optional[Dict[str, np.ndarray]], Optional[List[int]]]:
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
    elif attr_func == "ifr_all_positions_output_only":
        llm_attributor = llm_attr.LLMIFRAttribution(
            model,
            tokenizer,
            chunk_tokens=testing_dict["chunk_tokens"],
            sink_chunk_tokens=testing_dict["sink_chunk_tokens"],
        )
        sink_span = tuple(example.sink_span) if example.sink_span else tuple(indices_to_explain)
        attr = llm_attributor.calculate_ifr_for_all_positions_output_only(
            example.prompt,
            target=target,
            sink_span=sink_span,
        )
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
        attr = llm_attributor.calculate_attnlrp_ft_hop0(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            neg_handling=str(testing_dict.get("attnlrp_neg_handling", "drop")),
            norm_mode=str(testing_dict.get("attnlrp_norm_mode", "norm")),
        )
    elif attr_func in ("ft_attnlrp", "attnlrp_aggregated_multi_hop"):
        llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
        attr = llm_attributor.calculate_attnlrp_aggregated_multi_hop(
            example.prompt,
            target=target,
            sink_span=tuple(example.sink_span) if example.sink_span else None,
            thinking_span=tuple(example.thinking_span) if example.thinking_span else None,
            n_hops=testing_dict["n_hops"],
            neg_handling=str(testing_dict.get("attnlrp_neg_handling", "drop")),
            norm_mode=str(testing_dict.get("attnlrp_norm_mode", "norm")),
        )
    elif attr_func == "basic":
        llm_attributor = llm_attr.LLMBasicAttribution(model, tokenizer)
        attr = llm_attributor.calculate_basic_attribution(example.prompt, target=target)
    else:
        raise ValueError(f"Unsupported attr_func {attr_func}")

    seq_attr, row_attr, rec_attr = attr.get_all_token_attrs(indices_to_explain)
    hop_payload = None
    if bool(testing_dict.get("save_hop_traces", False)):
        try:
            hop_payload = _build_hop_trace_payload(attr_func, attr, indices_to_explain=indices_to_explain)
        except Exception as exc:
            print(f"[warn] hop trace extraction failed for {attr_func}: {exc}")
            hop_payload = None

    user_prompt_indices = getattr(llm_attributor, "user_prompt_indices", None)
    if isinstance(user_prompt_indices, list):
        user_prompt_indices = [int(x) for x in user_prompt_indices]
    else:
        user_prompt_indices = None

    return [seq_attr, row_attr, rec_attr], hop_payload, user_prompt_indices


def faithfulness_generation(
    testing_dict, example: ds_utils.CachedExample, target: str, llm_evaluator
) -> Tuple[np.ndarray, Optional[Dict[str, np.ndarray]]]:
    prompt = example.prompt
    generation = target

    attr_list, hop_payload, user_prompt_indices = run_attribution(testing_dict, example, target)
    seq_attr = attr_list[0]
    prompt_len = int(seq_attr.shape[1] - seq_attr.shape[0])  # cols=(P+G), rows=G

    results = []
    for attr in attr_list:
        # Only use prompt-side attribution, matching evaluations/faithfulness.py
        attr_prompt = attr[:, :prompt_len]
        if user_prompt_indices is not None:
            scores = _faithfulness_test_with_user_prompt_indices(
                llm_evaluator,
                attr_prompt,
                prompt,
                generation,
                user_prompt_indices=user_prompt_indices,
            )
        else:
            scores = llm_evaluator.faithfulness_test(attr_prompt, prompt, generation)
        results.append(scores)

    return np.array(results), hop_payload


def evaluate_dataset(args, dataset_name: str, examples: List[ds_utils.CachedExample], testing_dict):
    tokenizer = testing_dict["tokenizer"]
    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(testing_dict["model"], tokenizer)
    results = []
    durations: List[float] = []
    total = min(len(examples), args.num_examples)
    iterator = islice(examples, total)

    attr_func = testing_dict["attr_func"]
    save_traces = bool(getattr(args, "save_hop_traces", False)) and attr_func in (
        "ifr_multi_hop",
        "ft_attnlrp",
        "attnlrp_aggregated_multi_hop",
    )

    manifest_handle = None
    trace_dir: Optional[Path] = None
    if save_traces:
        model_tag = str(testing_dict.get("model_tag", "model"))
        out_dir = Path(args.output_root) / "faithfulness" / dataset_name / model_tag
        trace_dir = out_dir / "hop_traces" / f"{attr_func}_n{int(testing_dict.get('n_hops', 0))}_{total}ex"
        trace_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = trace_dir / "manifest.jsonl"
        manifest_handle = open(manifest_path, "w", encoding="utf-8")

    try:
        for example_idx, ex in enumerate(iterator):
            # Determine generation/target once
            target = ex.target
            if target is None:
                generation, full_output = llm_evaluator.response(ex.prompt)
                target = generation
                response_len = len(tokenizer(full_output).input_ids)
            else:
                response_len = len(tokenizer(llm_evaluator.format_prompt(" " + ex.prompt) + target).input_ids)

            # Estimate batch size (align with evaluations/attribution_recovery.py & faithfulness.py)
            testing_dict["batch_size"] = max(
                1, math.floor((testing_dict["max_input_len"] - 100) / max(1, response_len))
            )

            sample_start = time.perf_counter()
            scores, hop_payload = faithfulness_generation(testing_dict, ex, target, llm_evaluator)
            durations.append(time.perf_counter() - sample_start)
            results.append(scores)

            if manifest_handle is not None and trace_dir is not None and hop_payload is not None:
                try:
                    _write_hop_trace(
                        trace_dir,
                        example_idx=example_idx,
                        attr_func=attr_func,
                        prompt=ex.prompt,
                        target=target,
                        payload=hop_payload,
                        manifest_handle=manifest_handle,
                    )
                except Exception as exc:
                    print(f"[warn] hop trace save failed for {attr_func} ex={example_idx}: {exc}")
    finally:
        if manifest_handle is not None:
            try:
                manifest_handle.close()
            except Exception:
                pass

    if not results:
        return None

    scores = np.array(results)
    mean = scores.mean(0)
    std = scores.std(0)
    avg_time = float(np.mean(durations)) if durations else 0.0
    return mean, std, avg_time


def evaluate_dataset_recovery_ruler(args, dataset_name: str, examples: List[ds_utils.CachedExample], testing_dict):
    tokenizer = testing_dict["tokenizer"]
    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(testing_dict["model"], tokenizer)

    results: List[np.ndarray] = []
    durations: List[float] = []
    skipped = 0

    total = min(len(examples), args.num_examples)
    iterator = islice(examples, total)

    for ex in iterator:
        needle_spans = (ex.metadata or {}).get("needle_spans")
        if not isinstance(needle_spans, list) or not needle_spans:
            raise SystemExit(
                "recovery_ruler requires RULER samples with metadata.needle_spans; "
                f"dataset={dataset_name} has missing/empty needle_spans."
            )
        if ex.target is None:
            raise SystemExit(
                "recovery_ruler requires cached targets (CoT+answer) so row/rec attribution is well-defined. "
                f"dataset={dataset_name} has target=None; run exp/exp2/sample_and_filter.py first."
            )

        response_len = len(tokenizer(llm_evaluator.format_prompt(" " + ex.prompt) + ex.target).input_ids)
        testing_dict["batch_size"] = max(1, math.floor((testing_dict["max_input_len"] - 100) / max(1, response_len)))

        gold_prompt = ds_utils.ruler_gold_prompt_token_indices(ex, tokenizer)
        if not gold_prompt:
            skipped += 1
            continue

        sample_start = time.perf_counter()
        attr_list, _, _ = run_attribution(testing_dict, ex, ex.target)
        durations.append(time.perf_counter() - sample_start)

        seq_attr = attr_list[0]
        prompt_len = int(seq_attr.shape[1] - seq_attr.shape[0])  # cols=(P+G), rows=G
        if prompt_len <= 0:
            skipped += 1
            continue

        scores = [
            llm_evaluator.evaluate_attr_recovery(
                attr,
                prompt_len=prompt_len,
                gold_prompt_token_indices=gold_prompt,
                top_fraction=0.1,
            )
            for attr in attr_list
        ]
        results.append(np.asarray(scores, dtype=np.float64))

    if not results:
        return None

    scores = np.stack(results, axis=0)  # [N, 3]
    mean = scores.mean(0)
    std = scores.std(0)
    avg_time = float(np.mean(durations)) if durations else 0.0
    used = int(scores.shape[0])
    return mean, std, avg_time, used, int(skipped)


def main():
    parser = argparse.ArgumentParser("Experiment 2 runner (math skipped, AT2 skipped).")
    parser.add_argument("--datasets", type=str, required=True, help="Comma-separated names or paths.")
    parser.add_argument("--attr_funcs", type=str, required=True, help="Comma-separated attr funcs (no AT2).")
    parser.add_argument("--model", type=str, default=None, help="HF repo id (required unless --model_path set).")
    parser.add_argument("--model_path", type=str, default=None, help="Local path; overrides --model for loading.")
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--cuda_num", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument(
        "--mode",
        type=str,
        default="faithfulness_gen",
        choices=["faithfulness_gen", "recovery_ruler"],
        help="faithfulness_gen: token-level RISE/MAS; recovery_ruler: Recall@10pct of RULER needle tokens in top-attributed prompt tokens.",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional subsample before num_examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_tokens", type=int, default=128)
    parser.add_argument("--sink_chunk_tokens", type=int, default=32)
    parser.add_argument("--n_hops", type=int, default=3)
    parser.add_argument(
        "--attnlrp_neg_handling",
        type=str,
        choices=["drop", "abs"],
        default="drop",
        help="FT-AttnLRP: how to handle negative values after each hop (drop=clamp>=0, abs=absolute value).",
    )
    parser.add_argument(
        "--attnlrp_norm_mode",
        type=str,
        choices=["norm", "no_norm"],
        default="norm",
        help="FT-AttnLRP: norm enables per-hop global+thinking normalization + ratios; no_norm disables all three.",
    )
    parser.add_argument("--data_root", type=str, default="exp/exp2/data", help="Filtered dataset cache directory.")
    parser.add_argument("--output_root", type=str, default="exp/exp2/output", help="Directory to store evaluation outputs.")
    parser.add_argument(
        "--save_hop_traces",
        action="store_true",
        help="Save per-hop token vectors (vh) for ifr_multi_hop and ft_attnlrp under output_root.",
    )
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
                "model_tag": model_tag,
                "tokenizer": tokenizer,
                "attr_func": attr_func,
                "max_input_len": max_input_len,
                "chunk_tokens": args.chunk_tokens,
                "sink_chunk_tokens": args.sink_chunk_tokens,
                "n_hops": args.n_hops,
                "attnlrp_neg_handling": args.attnlrp_neg_handling,
                "attnlrp_norm_mode": args.attnlrp_norm_mode,
                "device": device,
                "batch_size": 1,
                "save_hop_traces": bool(args.save_hop_traces),
            }
            if args.mode == "faithfulness_gen":
                result = evaluate_dataset(args, ds_name, examples, testing_dict)
                if result is None:
                    print(f"No results for {ds_name} with {attr_func}.")
                    continue
                mean, std, avg_time = result

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

            elif args.mode == "recovery_ruler":
                if ds_name == "morehopqa":
                    raise SystemExit("recovery_ruler only supports RULER datasets (with needle_spans), not morehopqa.")
                result = evaluate_dataset_recovery_ruler(args, ds_name, examples, testing_dict)
                if result is None:
                    print(f"No recovery results for {ds_name} with {attr_func}.")
                    continue
                mean, std, avg_time, used, skipped = result

                out_dir = Path(args.output_root) / "recovery" / ds_name / model_tag
                out_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{attr_func}_{args.num_examples}_examples.csv"
                with open(out_dir / filename, "w") as f:
                    f.write("Method,Recovery@10%\n")
                    f.write(f"Seq Attr Recovery Mean,{mean[0]}\n")
                    f.write(f"Row Attr Recovery Mean,{mean[1]}\n")
                    f.write(f"Recursive Attr Recovery Mean,{mean[2]}\n")
                    f.write(f"Seq Attr Recovery Std,{std[0]}\n")
                    f.write(f"Row Attr Recovery Std,{std[1]}\n")
                    f.write(f"Recursive Attr Recovery Std,{std[2]}\n")
                    f.write(f"Examples Used,{used}\n")
                    f.write(f"Examples Skipped,{skipped}\n")
                    f.write(f"Avg Sample Time (s),{avg_time}\n")
                print(
                    f"[{ds_name}] {attr_func} -> {out_dir/filename} "
                    f"(used={used} skipped={skipped} avg sample time: {avg_time:.2f}s)"
                )
            else:
                raise SystemExit(f"Unsupported mode {args.mode}")


if __name__ == "__main__":
    main()
