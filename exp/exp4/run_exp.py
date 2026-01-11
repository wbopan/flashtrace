#!/usr/bin/env python3
"""
Experiment 4 runner: Aider token-level attribution faithfulness.

Evaluates only:
- IFR: ifr_all_positions
  - sink = last meaningful code line (excluding fences)
  - sink = last token of that code line
- FlashTrace: ifr_multi_hop_both
  - sink = full output (excluding appended EOS)

Outputs only row-level faithfulness scores (RISE, MAS). No sample-level traces.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _early_set_cuda_visible_devices() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda", type=str, default=None)
    args, _ = parser.parse_known_args(sys.argv[1:])
    if args.cuda and "," in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda


_early_set_cuda_visible_devices()

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, utils

# Ensure repo root on path for `import llm_attr`, `import ft_ifr_improve`, etc.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ft_ifr_improve
import llm_attr
import llm_attr_eval

utils.logging.set_verbosity_error()


@dataclass(frozen=True)
class AiderExample:
    prompt: str
    target: str
    metadata: Dict[str, Any]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def load_aider(path: Path) -> List[AiderExample]:
    rows = _read_jsonl(path)
    examples: List[AiderExample] = []
    for row in rows:
        prompt = str(row.get("input") or "")
        target = str(row.get("output") or "")
        examples.append(AiderExample(prompt=prompt, target=target, metadata={"length": row.get("length")}))
    return examples


def _token_span_full_output(tokenizer, target: str) -> List[int]:
    ids = tokenizer(target, add_special_tokens=False).input_ids
    if not ids:
        return [0, 0]
    return [0, int(len(ids) - 1)]


def _last_meaningful_code_line_char_span(target: str) -> Optional[Tuple[int, int]]:
    lines = target.splitlines(keepends=True)
    pos = 0
    spans: List[Tuple[int, int, str]] = []
    for line in lines:
        start = pos
        pos += len(line)
        spans.append((start, pos, line))

    for start, end, line in reversed(spans):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if start == 0 and stripped.endswith(".py"):
            return None

        line_no_nl = line.rstrip("\r\n")
        end_no_nl = start + len(line_no_nl)
        if end_no_nl <= start:
            continue
        return start, end_no_nl

    return None


def _char_span_to_token_span(tokenizer, text: str, span: Tuple[int, int]) -> Optional[List[int]]:
    start_char, end_char = int(span[0]), int(span[1])
    if end_char <= start_char:
        return None

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc.get("offset_mapping")
    if offsets is None:
        raise ValueError("Tokenizer does not provide offset_mapping; cannot map char spans to tokens.")

    tok_indices: List[int] = []
    for idx, off in enumerate(offsets):
        if off is None:
            continue
        s, e = int(off[0]), int(off[1])
        if s < end_char and e > start_char:
            tok_indices.append(int(idx))
    if not tok_indices:
        return None
    return [min(tok_indices), max(tok_indices)]


def _last_meaningful_code_line_token_span(tokenizer, target: str) -> List[int]:
    full_span = _token_span_full_output(tokenizer, target)
    span_chars = _last_meaningful_code_line_char_span(target)
    if span_chars is None:
        return full_span

    span_toks = _char_span_to_token_span(tokenizer, target, span_chars)
    if span_toks is None:
        return full_span

    span_toks[0] = max(int(span_toks[0]), int(full_span[0]))
    span_toks[1] = min(int(span_toks[1]), int(full_span[1]))
    if span_toks[1] < span_toks[0]:
        return full_span
    return span_toks


def _last_token_span(token_span: Sequence[int]) -> List[int]:
    if not (isinstance(token_span, Sequence) and len(token_span) == 2):
        return [0, 0]
    end = int(token_span[1])
    return [end, end]


def resolve_device(args) -> str:
    if args.cuda is not None and "," in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        return "auto"
    if args.cuda is not None and args.cuda.strip():
        return f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    return f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"


def load_model_and_tokenizer(args) -> tuple[Any, Any]:
    model_id = args.model_path or args.model
    if not model_id:
        raise SystemExit("Provide --model_path (local) or --model (HF repo id).")

    tokenizer_id = args.tokenizer_path or model_id
    device = resolve_device(args)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto" if device == "auto" else {"": int(device.split(":")[1])} if device.startswith("cuda:") else None,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def _faithfulness_test_with_user_prompt_indices(
    llm_evaluator: llm_attr_eval.LLMAttributionEvaluator,
    attribution: torch.Tensor,
    prompt: str,
    generation: str,
    *,
    user_prompt_indices: List[int],
    k: int = 20,
) -> Tuple[float, float, float]:
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

    steps = int(k) if k is not None else 0
    if steps <= 0:
        steps = 1
    steps = min(steps, P)

    scores = np.zeros(steps + 1, dtype=np.float64)
    density = np.zeros(steps + 1, dtype=np.float64)

    scores[0] = (
        llm_evaluator.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
    )
    density[0] = 1.0

    if attr_sum <= 0:
        density = np.linspace(1.0, 0.0, steps + 1)

    base = P // steps
    remainder = P % steps
    start = 0
    for step in range(steps):
        size = base + (1 if step < remainder else 0)
        group = sorted_attr_indices[start : start + size]
        start += size

        for idx in group:
            j = int(idx.item())
            abs_pos = int(user_prompt_indices[j])
            prompt_ids_perturbed[0, abs_pos] = pad_token_id
        scores[step + 1] = (
            llm_evaluator.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
        )
        if attr_sum > 0:
            dec = float(w.index_select(0, group).sum().item()) / attr_sum
            density[step + 1] = density[step] - dec

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


def _row_faithfulness_scores(
    *,
    llm_evaluator: llm_attr_eval.LLMAttributionEvaluator,
    attribution_prompt: torch.Tensor,
    prompt: str,
    generation: str,
    user_prompt_indices: Optional[List[int]],
    keep_prompt_token_indices: Optional[Sequence[int]] = None,
    k: int = 20,
) -> Tuple[float, float]:
    if keep_prompt_token_indices is not None:
        rise, mas, _ = ft_ifr_improve.faithfulness_test_skip_tokens(
            llm_evaluator,
            attribution_prompt,
            prompt,
            generation,
            keep_prompt_token_indices=keep_prompt_token_indices,
            user_prompt_indices=user_prompt_indices,
            k=int(k),
        )
        return float(rise), float(mas)
    if user_prompt_indices is not None:
        rise, mas, _ = _faithfulness_test_with_user_prompt_indices(
            llm_evaluator,
            attribution_prompt,
            prompt,
            generation,
            user_prompt_indices=user_prompt_indices,
            k=int(k),
        )
        return float(rise), float(mas)

    rise, mas, _ = llm_evaluator.faithfulness_test(attribution_prompt, prompt, generation, k=int(k))
    return float(rise), float(mas)


def _model_tag(args) -> str:
    if args.model:
        return str(args.model)
    if args.model_path:
        return Path(args.model_path).name
    return "model"


def main() -> None:
    parser = argparse.ArgumentParser("Experiment 4 runner: aider faithfulness (row-only).")
    parser.add_argument("--data_path", type=str, default="exp/exp4/data/aider.jsonl")
    parser.add_argument("--output_root", type=str, default="exp/exp4/output")
    parser.add_argument("--model", type=str, default=None, help="HF repo id (required unless --model_path set).")
    parser.add_argument("--model_path", type=str, default=None, help="Local path; overrides --model for loading.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional tokenizer path/id (defaults to model).")
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--cuda_num", type=int, default=0)
    parser.add_argument("--num_examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42, help="Reserved for future use; exp4 runs in file order.")
    parser.add_argument("--chunk_tokens", type=int, default=128)
    parser.add_argument("--sink_chunk_tokens", type=int, default=32)
    parser.add_argument("--n_hops", type=int, default=3)
    parser.add_argument("--k", type=int, default=20, help="Perturbation steps for MAS/RISE.")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise SystemExit(f"Missing Aider JSONL: {data_path}")

    model, tokenizer = load_model_and_tokenizer(args)
    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(model, tokenizer)

    examples = load_aider(data_path)
    total = min(len(examples), int(args.num_examples))
    iterator = islice(examples, total)

    ifr = llm_attr.LLMIFRAttribution(
        model,
        tokenizer,
        chunk_tokens=int(args.chunk_tokens),
        sink_chunk_tokens=int(args.sink_chunk_tokens),
    )
    flashtrace = ft_ifr_improve.LLMIFRAttributionBoth(
        model,
        tokenizer,
        chunk_tokens=int(args.chunk_tokens),
        sink_chunk_tokens=int(args.sink_chunk_tokens),
    )

    results: Dict[Tuple[str, str], List[Tuple[float, float]]] = {
        ("ifr_all_positions", "last_line"): [],
        ("ifr_all_positions", "last_token"): [],
        ("ifr_multi_hop_both", "full_output"): [],
    }
    skipped: Dict[Tuple[str, str], int] = {k: 0 for k in results}
    sample_times: Dict[Tuple[str, str], List[float]] = {k: [] for k in results}

    for example_idx, ex in enumerate(iterator):
        prompt = ex.prompt
        target = ex.target

        full_span = _token_span_full_output(tokenizer, target)
        last_line_span = _last_meaningful_code_line_token_span(tokenizer, target)
        last_token_span = _last_token_span(last_line_span)

        attr_all = None
        attr_all_time_s = 0.0
        user_prompt_indices_all: Optional[List[int]] = None
        prompt_len_all = 0
        try:
            t_attr = time.perf_counter()
            attr_all = ifr.calculate_ifr_for_all_positions(prompt, target=target)
            attr_all_time_s = float(time.perf_counter() - t_attr)
            user_prompt_indices_all = list(getattr(ifr, "user_prompt_indices", []) or [])
            prompt_len_all = int(len(attr_all.prompt_tokens))
        except Exception as exc:
            skipped[("ifr_all_positions", "last_line")] += 1
            skipped[("ifr_all_positions", "last_token")] += 1
            print(f"[warn] ifr_all_positions attribution failed ex={example_idx}: {exc}")

        if attr_all is not None and user_prompt_indices_all is not None and prompt_len_all >= 0:
            for sink_name, span in (("last_line", last_line_span), ("last_token", last_token_span)):
                key = ("ifr_all_positions", sink_name)
                try:
                    t_faith = time.perf_counter()
                    row = attr_all.get_all_token_attrs(list(span))[1]
                    rise, mas = _row_faithfulness_scores(
                        llm_evaluator=llm_evaluator,
                        attribution_prompt=row[:, :prompt_len_all],
                        prompt=prompt,
                        generation=target,
                        user_prompt_indices=user_prompt_indices_all,
                        k=int(args.k),
                    )
                    faith_time_s = float(time.perf_counter() - t_faith)
                    results[key].append((rise, mas))
                    sample_times[key].append(attr_all_time_s + faith_time_s)
                except Exception as exc:
                    skipped[key] += 1
                    print(f"[warn] ifr_all_positions {sink_name} failed ex={example_idx}: {exc}")

        try:
            t_attr = time.perf_counter()
            attr_ft = flashtrace.calculate_ifr_multi_hop_both(
                prompt,
                target=target,
                sink_span=None,
                thinking_span=None,
                n_hops=int(args.n_hops),
            )
            attr_ft_time_s = float(time.perf_counter() - t_attr)
            user_prompt_indices_ft = list(getattr(flashtrace, "user_prompt_indices", []) or [])
            prompt_len_ft = int(len(attr_ft.prompt_tokens))
            keep_prompt_token_indices = ft_ifr_improve.keep_token_indices(list(attr_ft.prompt_tokens))

            t_faith = time.perf_counter()
            row_full = attr_ft.get_all_token_attrs(full_span)[1]
            rise, mas = _row_faithfulness_scores(
                llm_evaluator=llm_evaluator,
                attribution_prompt=row_full[:, :prompt_len_ft],
                prompt=prompt,
                generation=target,
                user_prompt_indices=user_prompt_indices_ft,
                keep_prompt_token_indices=keep_prompt_token_indices,
                k=int(args.k),
            )
            faith_time_s = float(time.perf_counter() - t_faith)
            results[("ifr_multi_hop_both", "full_output")].append((rise, mas))
            sample_times[("ifr_multi_hop_both", "full_output")].append(attr_ft_time_s + faith_time_s)
        except Exception as exc:
            skipped[("ifr_multi_hop_both", "full_output")] += 1
            print(f"[warn] ifr_multi_hop_both failed ex={example_idx}: {exc}")

    model_tag = _model_tag(args)
    out_dir = Path(args.output_root) / "faithfulness" / "aider" / model_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"row_only_{total}_examples.csv"

    with out_path.open("w", encoding="utf-8") as f:
        f.write("Method,Sink,Row_RISE_Mean,Row_RISE_Std,Row_MAS_Mean,Row_MAS_Std,Used,Skipped,Avg_Sample_Time_s\n")
        for (method, sink), vals in results.items():
            arr = np.asarray(vals, dtype=np.float64)
            used = int(arr.shape[0])
            if used == 0:
                rise_mean = float("nan")
                rise_std = float("nan")
                mas_mean = float("nan")
                mas_std = float("nan")
            else:
                rise_mean = float(arr[:, 0].mean())
                rise_std = float(arr[:, 0].std())
                mas_mean = float(arr[:, 1].mean())
                mas_std = float(arr[:, 1].std())
            times = sample_times.get((method, sink)) or []
            avg_time = float(np.mean(times)) if times else 0.0
            f.write(
                f"{method},{sink},{rise_mean},{rise_std},{mas_mean},{mas_std},{used},{int(skipped[(method, sink)])},{avg_time}\n"
            )

    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
