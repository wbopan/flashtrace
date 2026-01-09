#!/usr/bin/env python3
"""
Experiment 3 runner: long-vs-short CoT case study (AttnLRP hop0, Recovery@10%).

This runner is intentionally minimal:
  - Only reads two cached samples produced by exp/exp3/sample_and_filter.py:
      <dataset_tag>_short_cot.jsonl
      <dataset_tag>_long_cot.jsonl
  - Only runs attribution method: attnlrp (hop0 path, aligned with exp2).
  - Only computes token-level recovery (Recall@10%) using RULER needle_spans.
  - Always saves per-sample trace artifacts under exp/exp3/output/traces/.

All outputs are written under exp/exp3/output/ (configurable via --output_root).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import llm_attr
import llm_attr_eval
from exp.exp2 import dataset_utils as ds_utils

utils.logging.set_verbosity_error()


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _token_importance_vector(attr: torch.Tensor) -> np.ndarray:
    w = torch.nan_to_num(attr.sum(0).to(dtype=torch.float32), nan=0.0).clamp(min=0.0)
    return w.detach().cpu().numpy().astype(np.float32, copy=False)


def _trace_run_tag(*, neg_handling: str, norm_mode: str, total: int) -> str:
    return f"attnlrp_neg{neg_handling}_norm{norm_mode}_recovery_{int(total)}ex"


def _build_sample_trace_payload(
    example: ds_utils.CachedExample,
    *,
    seq_attr: torch.Tensor,
    row_attr: torch.Tensor,
    rec_attr: torch.Tensor,
    prompt_len: int,
    user_prompt_indices: Optional[List[int]],
    gold_prompt_token_indices: Optional[List[int]],
    recovery_scores: Optional[np.ndarray],
    time_attr_s: Optional[float],
    time_recovery_s: Optional[float],
) -> Dict[str, np.ndarray]:
    gen_len = int(seq_attr.shape[0])

    v_seq_all = _token_importance_vector(seq_attr)
    v_row_all = _token_importance_vector(row_attr)
    v_rec_all = _token_importance_vector(rec_attr)

    payload: Dict[str, np.ndarray] = {
        "v_seq_all": v_seq_all,
        "v_row_all": v_row_all,
        "v_rec_all": v_rec_all,
        "v_seq_prompt": v_seq_all[:prompt_len],
        "v_row_prompt": v_row_all[:prompt_len],
        "v_rec_prompt": v_rec_all[:prompt_len],
        "prompt_len": np.asarray(int(prompt_len), dtype=np.int64),
        "gen_len": np.asarray(int(gen_len), dtype=np.int64),
        "indices_to_explain_gen": np.asarray(list(example.indices_to_explain or []), dtype=np.int64),
    }

    if example.sink_span is not None:
        payload["sink_span_gen"] = np.asarray(list(example.sink_span), dtype=np.int64)
    if example.thinking_span is not None:
        payload["thinking_span_gen"] = np.asarray(list(example.thinking_span), dtype=np.int64)

    if user_prompt_indices is not None:
        payload["user_prompt_indices"] = np.asarray(list(user_prompt_indices), dtype=np.int64)
    if gold_prompt_token_indices is not None:
        payload["gold_prompt_token_indices"] = np.asarray(list(gold_prompt_token_indices), dtype=np.int64)

    if recovery_scores is not None:
        payload["recovery_scores"] = np.asarray(recovery_scores, dtype=np.float64)

    if time_attr_s is not None:
        payload["time_attr_s"] = np.asarray(float(time_attr_s), dtype=np.float64)
    if time_recovery_s is not None:
        payload["time_recovery_s"] = np.asarray(float(time_recovery_s), dtype=np.float64)

    return payload


def _write_sample_trace(
    trace_dir: Path,
    *,
    example_idx: int,
    prompt: str,
    target: str,
    payload: Dict[str, np.ndarray],
    manifest_handle,
    neg_handling: str,
    norm_mode: str,
    recovery_skipped_reason: Optional[str],
) -> None:
    trace_dir.mkdir(parents=True, exist_ok=True)
    npz_name = f"ex_{example_idx:06d}.npz"
    npz_path = trace_dir / npz_name
    np.savez_compressed(npz_path, **payload)

    prompt_len = int(np.asarray(payload.get("prompt_len", 0)).item())
    gen_len = int(np.asarray(payload.get("gen_len", 0)).item())
    record: Dict[str, Any] = {
        "example_idx": int(example_idx),
        "attr_func": "attnlrp",
        "file": npz_name,
        "prompt_sha1": _sha1_text(prompt),
        "target_sha1": _sha1_text(target),
        "prompt_len": prompt_len,
        "gen_len": gen_len,
        "indices_to_explain_gen": payload.get("indices_to_explain_gen").tolist()
        if payload.get("indices_to_explain_gen") is not None
        else None,
        "sink_span_gen": payload.get("sink_span_gen").tolist() if payload.get("sink_span_gen") is not None else None,
        "thinking_span_gen": payload.get("thinking_span_gen").tolist()
        if payload.get("thinking_span_gen") is not None
        else None,
        "gold_prompt_token_indices": payload.get("gold_prompt_token_indices").tolist()
        if payload.get("gold_prompt_token_indices") is not None
        else None,
        "recovery_scores": payload.get("recovery_scores").tolist() if payload.get("recovery_scores") is not None else None,
        "recovery_skipped_reason": recovery_skipped_reason,
        "time_attr_s": float(np.asarray(payload.get("time_attr_s")).item()) if payload.get("time_attr_s") is not None else None,
        "time_recovery_s": float(np.asarray(payload.get("time_recovery_s")).item())
        if payload.get("time_recovery_s") is not None
        else None,
        "attnlrp_neg_handling": str(neg_handling),
        "attnlrp_norm_mode": str(norm_mode),
    }
    manifest_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest_handle.flush()


def resolve_device(args) -> str:
    if args.cuda is not None and "," in args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
        return "auto"
    if args.cuda is not None and str(args.cuda).strip():
        return f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    return f"cuda:{args.cuda_num}" if torch.cuda.is_available() else "cpu"


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


def _evaluate_one_dataset(
    *,
    dataset_name: str,
    examples: List[ds_utils.CachedExample],
    model,
    tokenizer,
    output_root: Path,
    model_tag: str,
    neg_handling: str,
    norm_mode: str,
    top_fraction: float,
    num_examples: int,
) -> Tuple[np.ndarray, np.ndarray, float, int, int]:
    llm_evaluator = llm_attr_eval.LLMAttributionEvaluator(model, tokenizer)

    results: List[np.ndarray] = []
    durations: List[float] = []
    skipped = 0

    total = min(len(examples), int(num_examples))
    iterator = islice(examples, total)

    run_tag = _trace_run_tag(neg_handling=neg_handling, norm_mode=norm_mode, total=total)
    trace_dir = output_root / "traces" / dataset_name / model_tag / run_tag
    trace_dir.mkdir(parents=True, exist_ok=True)
    manifest_handle = open(trace_dir / "manifest.jsonl", "w", encoding="utf-8")

    try:
        for example_idx, ex in enumerate(iterator):
            time_recovery_s: Optional[float] = None
            recovery_scores: Optional[np.ndarray] = None

            needle_spans = (ex.metadata or {}).get("needle_spans")
            if not isinstance(needle_spans, list) or not needle_spans:
                raise SystemExit(
                    "exp3 recovery requires RULER samples with metadata.needle_spans; "
                    f"dataset={dataset_name} has missing/empty needle_spans."
                )
            if ex.target is None:
                raise SystemExit(
                    "exp3 recovery requires cached targets (CoT+answer) so row/rec attribution is well-defined. "
                    f"dataset={dataset_name} has target=None; run exp/exp3/sample_and_filter.py first."
                )
            if not (isinstance(ex.indices_to_explain, list) and len(ex.indices_to_explain) == 2):
                raise SystemExit(
                    "exp3 expects indices_to_explain=[start_tok,end_tok] in generation-token coordinates; "
                    f"dataset={dataset_name} has indices_to_explain={ex.indices_to_explain!r}; "
                    "run exp/exp3/sample_and_filter.py first."
                )

            gold_prompt = ds_utils.ruler_gold_prompt_token_indices(ex, tokenizer)
            recovery_skip_reason: Optional[str] = None

            sample_start = time.perf_counter()
            llm_attributor = llm_attr.LLMLRPAttribution(model, tokenizer)
            attr_result = llm_attributor.calculate_attnlrp_ft_hop0(
                ex.prompt,
                target=ex.target,
                sink_span=tuple(ex.sink_span) if ex.sink_span else None,
                thinking_span=tuple(ex.thinking_span) if ex.thinking_span else None,
                neg_handling=str(neg_handling),
                norm_mode=str(norm_mode),
            )
            seq_attr, row_attr, rec_attr = attr_result.get_all_token_attrs(list(ex.indices_to_explain))
            time_attr_s = time.perf_counter() - sample_start
            durations.append(float(time_attr_s))

            prompt_len = int(seq_attr.shape[1] - seq_attr.shape[0])
            if prompt_len <= 0:
                recovery_skip_reason = "empty_prompt_len"
            elif not gold_prompt:
                recovery_skip_reason = "empty_gold_prompt"
            else:
                t2 = time.perf_counter()
                recovery_scores = np.asarray(
                    [
                        llm_evaluator.evaluate_attr_recovery(
                            a,
                            prompt_len=prompt_len,
                            gold_prompt_token_indices=gold_prompt,
                            top_fraction=top_fraction,
                        )
                        for a in (seq_attr, row_attr, rec_attr)
                    ],
                    dtype=np.float64,
                )
                time_recovery_s = time.perf_counter() - t2
                if np.isnan(recovery_scores).any():
                    recovery_scores = None
                    recovery_skip_reason = "nan_recovery"

            if recovery_scores is None and recovery_skip_reason is not None:
                skipped += 1
            elif recovery_scores is not None:
                results.append(recovery_scores)

            payload = _build_sample_trace_payload(
                ex,
                seq_attr=seq_attr,
                row_attr=row_attr,
                rec_attr=rec_attr,
                prompt_len=prompt_len,
                user_prompt_indices=getattr(llm_attributor, "user_prompt_indices", None),
                gold_prompt_token_indices=gold_prompt,
                recovery_scores=recovery_scores,
                time_attr_s=time_attr_s,
                time_recovery_s=time_recovery_s,
            )
            _write_sample_trace(
                trace_dir,
                example_idx=example_idx,
                prompt=ex.prompt,
                target=str(ex.target),
                payload=payload,
                manifest_handle=manifest_handle,
                neg_handling=str(neg_handling),
                norm_mode=str(norm_mode),
                recovery_skipped_reason=recovery_skip_reason,
            )
    finally:
        try:
            manifest_handle.close()
        except Exception:
            pass

    scores = np.stack(results, axis=0) if results else np.zeros((0, 3), dtype=np.float64)
    used = int(scores.shape[0])
    mean = scores.mean(0) if used else np.full((3,), np.nan, dtype=np.float64)
    std = scores.std(0) if used else np.full((3,), np.nan, dtype=np.float64)
    avg_time = float(np.mean(durations)) if durations else 0.0
    return mean, std, avg_time, used, int(skipped)


def main() -> None:
    parser = argparse.ArgumentParser("Experiment 3 runner (attnlrp hop0, recovery only).")
    parser.add_argument("--dataset_tag", type=str, default="niah_mq_q2", help="Base tag for exp3 caches.")
    parser.add_argument("--data_root", type=str, default="exp/exp3/data")
    parser.add_argument("--output_root", type=str, default="exp/exp3/output")
    parser.add_argument("--num_examples", type=int, default=1, help="How many examples to evaluate per dataset (default 1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None, help="HF repo id (required unless --model_path set).")
    parser.add_argument("--model_path", type=str, default=None, help="Local path; overrides --model for loading.")
    parser.add_argument("--cuda_num", type=int, default=0)
    parser.add_argument("--cuda", type=str, default=None)
    parser.add_argument("--top_fraction", type=float, default=0.1, help="Top fraction of prompt tokens used for recovery.")
    parser.add_argument(
        "--attnlrp_neg_handling",
        type=str,
        choices=["drop", "abs"],
        default="drop",
        help="AttnLRP hop0: how to handle negative values (drop=clamp>=0, abs=absolute value).",
    )
    parser.add_argument(
        "--attnlrp_norm_mode",
        type=str,
        choices=["norm", "no_norm"],
        default="norm",
        help="AttnLRP hop0: norm enables internal normalization; no_norm disables it.",
    )
    args = parser.parse_args()

    if args.model_path:
        model_name = args.model_path
    elif args.model:
        model_name = args.model
    else:
        raise SystemExit("Please set --model or --model_path.")
    model_tag = args.model if args.model else Path(args.model_path).name

    device = resolve_device(args)
    model, tokenizer = load_model(model_name, device)

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    short_name = f"{args.dataset_tag}_short_cot"
    long_name = f"{args.dataset_tag}_long_cot"
    dataset_names = [short_name, long_name]

    summary_rows: List[Dict[str, Any]] = []

    for ds_name in dataset_names:
        cache_path = data_root / f"{ds_name}.jsonl"
        if not cache_path.exists():
            raise SystemExit(f"Missing exp3 cache: {cache_path}. Run exp/exp3/sample_and_filter.py first.")
        examples = ds_utils.load_cached(cache_path, sample=None, seed=args.seed)

        mean, std, avg_time, used, skipped = _evaluate_one_dataset(
            dataset_name=ds_name,
            examples=examples,
            model=model,
            tokenizer=tokenizer,
            output_root=output_root,
            model_tag=model_tag,
            neg_handling=args.attnlrp_neg_handling,
            norm_mode=args.attnlrp_norm_mode,
            top_fraction=float(args.top_fraction),
            num_examples=int(args.num_examples),
        )

        out_dir = output_root / "recovery" / ds_name / model_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"attnlrp_{int(args.num_examples)}_examples.csv"
        with (out_dir / filename).open("w", encoding="utf-8") as f:
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

        print(f"[{ds_name}] attnlrp -> {out_dir/filename} (used={used} skipped={skipped} avg {avg_time:.2f}s)")
        summary_rows.append(
            {
                "dataset": ds_name,
                "model": model_tag,
                "neg_handling": args.attnlrp_neg_handling,
                "norm_mode": args.attnlrp_norm_mode,
                "seq_recovery@10%": float(mean[0]) if used else float("nan"),
                "row_recovery@10%": float(mean[1]) if used else float("nan"),
                "rec_recovery@10%": float(mean[2]) if used else float("nan"),
                "used": int(used),
                "skipped": int(skipped),
            }
        )

    # Lightweight combined summary for quick comparison.
    summary_path = output_root / "recovery" / f"summary_{model_tag}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
