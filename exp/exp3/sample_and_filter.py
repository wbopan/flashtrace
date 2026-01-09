#!/usr/bin/env python3
"""
Experiment 3 sampler: long-vs-short CoT case study (RULER niah_mq_q2, 1024).

This script searches the raw RULER JSONL for a *single* prompt where BOTH:
  - a short-CoT generation and a long-CoT generation
  - follow the strict format: "<thinking text> + final \\box{...} answer" with
    nothing after the box
  - pass a judge model verifying the boxed answer matches the reference answer
  - satisfy length constraints (short <= max_short_thinking_tokens,
    long >= min_long_thinking_tokens)

It writes two exp2-compatible cache JSONLs to exp/exp3/data/:
  - <dataset_tag>_short_cot.jsonl
  - <dataset_tag>_long_cot.jsonl

Each JSONL line matches exp/exp2/dataset_utils.CachedExample schema and keeps
RULER metadata. The output caches are intended to be consumed by exp/exp3/run_exp.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp2 import dataset_utils as ds_utils
from exp.exp2.dataset_utils import CachedExample, attach_spans_from_answer, split_boxed_generation


class RateLimitError(RuntimeError):
    """Raised when API returns 429; carries a suggested wait time."""

    def __init__(self, wait_seconds: float, detail: str) -> None:
        super().__init__(detail)
        self.wait_seconds = wait_seconds


SHORT_COT_SYSTEM_PROMPT = (
    "You are a reasoning assistant. "
    "Before answering, engage in a brief chain of thought. "
    "Process this freely and naturally without using specific headers or strict formatting. "
    "When you reach the conclusion, wrap the entire final sentence containing the answer inside \\box{}. "
    "Ensure the box wraps the **sentence** that naturally delivers the answer. "
    "Do not add anything after the box."
)

LONG_COT_SYSTEM_PROMPT = (
    "You are a careful reasoning assistant. "
    "Before answering, engage in an extremely detailed and exhaustive chain of thought. "
    "Do not skip any logical steps, even if they seem obvious. "
    "Process this freely and naturally without using specific headers or strict formatting. "
    "When you reach the conclusion, wrap the entire final sentence containing the answer inside \\box{}. "
    "Ensure the box wraps the **sentence** that naturally delivers the answer. "
    "Do not add anything after the box."
)

JUDGE_SYSTEM_PROMPT = (
    "You verify whether the model's boxed answer matches the reference answer. "
    "Reply strictly with True or False and nothing else."
)


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def call_chat_api(
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    timeout: int,
    max_tokens: int,
    temperature: float,
    cache_ttl: int,
    cache_namespace: Optional[str],
    rate_limit_delay: Optional[float] = None,
) -> str:
    url = api_base.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if cache_ttl > 0:
        cache_obj: Dict[str, Any] = {"ttl": cache_ttl}
        if cache_namespace:
            cache_obj["namespace"] = cache_namespace
        payload["cache"] = cache_obj

    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            resp_bytes = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
        if e.code == 429:
            retry_after = None
            if hasattr(e, "headers") and e.headers:
                retry_after_header = e.headers.get("Retry-After")
                if retry_after_header:
                    try:
                        retry_after = float(retry_after_header)
                    except ValueError:
                        retry_after = None
            wait = retry_after or rate_limit_delay or 5.0
            raise RateLimitError(wait, f"API HTTP 429: {detail}") from e
        raise RuntimeError(f"API HTTP error {e.code}: {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"API request failed: {e}") from e

    try:
        response = json.loads(resp_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to decode API response: {resp_bytes!r}") from e

    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError(f"Empty choices from API: {response}")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Empty content from API: {response}")
    return content.strip()


def _call_with_retries(
    *,
    api_base: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    timeout: int,
    max_tokens: int,
    temperature: float,
    cache_ttl: int,
    cache_namespace: Optional[str],
    rate_limit_delay: float,
    retries: int,
    retry_delay: float,
) -> str:
    for attempt in range(retries + 1):
        try:
            return call_chat_api(
                api_base,
                api_key,
                model,
                messages,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
                cache_ttl=cache_ttl,
                cache_namespace=cache_namespace,
                rate_limit_delay=rate_limit_delay,
            )
        except RateLimitError as e:
            if attempt >= retries:
                raise
            time.sleep(e.wait_seconds)
        except Exception:  # noqa: BLE001
            if attempt >= retries:
                raise
            time.sleep(retry_delay)
    raise RuntimeError("Unreachable")


def build_gen_messages(prompt: str, system_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]


def build_judge_messages(reference_answer: str, candidate_answer: str) -> List[Dict[str, str]]:
    user = (
        "Decide if the model's boxed answer matches the reference answer.\n"
        f"Reference answer: {reference_answer}\n"
        f"Model boxed answer (only the content inside \\box{{}}): {candidate_answer}\n"
        "Output only True if they are semantically consistent; otherwise output False."
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_bool(text: str) -> bool:
    first = text.strip().splitlines()[0].strip().lower()
    if first in {"true", "yes"}:
        return True
    if first in {"false", "no"}:
        return False
    if "true" in first and "false" not in first:
        return True
    if "false" in first:
        return False
    raise ValueError(f"Cannot parse boolean from: {text!r}")


def write_cache(out_path: Path, examples: Iterable[CachedExample]) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            obj: Dict[str, Any] = {
                "prompt": ex.prompt,
                "target": ex.target,
                "indices_to_explain": ex.indices_to_explain,
                "attr_mask_indices": ex.attr_mask_indices,
                "sink_span": ex.sink_span,
                "thinking_span": ex.thinking_span,
                "metadata": ex.metadata,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


@dataclass(frozen=True)
class AcceptedGeneration:
    thinking_text: str
    boxed_answer: str
    target_text: str
    thinking_tokens: int
    generation_text: str
    judge_response: str


def _infer_reference_answer(example: CachedExample) -> str:
    meta = example.metadata or {}
    ref = str(meta.get("reference_answer") or "").strip()
    if ref:
        return ref
    outputs = meta.get("outputs") or []
    if isinstance(outputs, list) and outputs:
        return ", ".join(str(x) for x in outputs)
    tgt = str(example.target or "").strip()
    return tgt


def _infer_dataset_tag(dataset_path: Path) -> str:
    if dataset_path.name.endswith(".jsonl") and dataset_path.name != "validation.jsonl":
        return dataset_path.stem
    if dataset_path.name == "validation.jsonl":
        return dataset_path.parent.name
    return dataset_path.stem


def _count_tokens(tokenizer, text: str) -> int:
    return int(len(tokenizer(text, add_special_tokens=False).input_ids))


def _generate_one_style(
    *,
    prompt: str,
    reference_answer: str,
    tokenizer,
    style: str,
    system_prompt: str,
    api_base: str,
    api_key: str,
    generator_model: str,
    judge_model: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
    cache_ttl: int,
    cache_namespace: Optional[str],
    rate_limit_delay: float,
    retries: int,
    retry_delay: float,
    request_interval: float,
    judge_interval: float,
    min_long_thinking_tokens: int,
    max_short_thinking_tokens: int,
) -> Optional[AcceptedGeneration]:
    gen_messages = build_gen_messages(prompt, system_prompt)
    generation = _call_with_retries(
        api_base=api_base,
        api_key=api_key,
        model=generator_model,
        messages=gen_messages,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
        cache_ttl=cache_ttl,
        cache_namespace=cache_namespace,
        rate_limit_delay=rate_limit_delay,
        retries=retries,
        retry_delay=retry_delay,
    )
    if request_interval > 0:
        time.sleep(request_interval)

    parsed = split_boxed_generation(generation)
    if not parsed:
        return None
    thinking_text, _boxed_segment, boxed_answer = parsed
    thinking_tokens = _count_tokens(tokenizer, thinking_text)

    if style == "short":
        if max_short_thinking_tokens > 0 and thinking_tokens > max_short_thinking_tokens:
            return None
    elif style == "long":
        if min_long_thinking_tokens > 0 and thinking_tokens < min_long_thinking_tokens:
            return None
    else:
        raise ValueError(f"Unsupported style: {style}")

    judge_messages = build_judge_messages(reference_answer, boxed_answer)
    judge_resp = _call_with_retries(
        api_base=api_base,
        api_key=api_key,
        model=judge_model,
        messages=judge_messages,
        timeout=timeout,
        max_tokens=64,
        temperature=0.0,
        cache_ttl=cache_ttl,
        cache_namespace=cache_namespace,
        rate_limit_delay=rate_limit_delay,
        retries=retries,
        retry_delay=retry_delay,
    )
    if judge_interval > 0:
        time.sleep(judge_interval)
    ok = parse_bool(judge_resp)
    if not ok:
        return None

    target_text = f"{thinking_text}\n{boxed_answer}" if thinking_text else boxed_answer
    return AcceptedGeneration(
        thinking_text=thinking_text,
        boxed_answer=boxed_answer,
        target_text=target_text,
        thinking_tokens=thinking_tokens,
        generation_text=generation,
        judge_response=judge_resp,
    )


def main() -> None:
    parser = argparse.ArgumentParser("Sample short-CoT and long-CoT cases for exp3 (independently).")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/ruler_multihop/1024/niah_mq_q2/validation.jsonl",
        help="Raw RULER JSONL path (default: niah_mq_q2 1024 validation).",
    )
    parser.add_argument("--dataset_tag", type=str, default=None, help="Output tag; default inferred from dataset_path.")
    parser.add_argument(
        "--max_pairs",
        type=int,
        default=1,
        help="Deprecated alias for --max_short and --max_long (kept for convenience).",
    )
    parser.add_argument("--max_short", type=int, default=None, help="How many short-CoT samples to keep (default: --max_pairs).")
    parser.add_argument("--max_long", type=int, default=None, help="How many long-CoT samples to keep (default: --max_pairs).")
    parser.add_argument("--max_raw_examples", type=int, default=None, help="Optional cap on raw examples to try.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--api_base", type=str, default="http://localhost:4000/v1", help="Chat API base URL.")
    parser.add_argument("--api_key", type=str, default=None, help="API key; defaults to FLASHTRACE_API_KEY/OPENAI_API_KEY.")
    parser.add_argument("--generator_model", type=str, default="qwen3-235b-a22b-2507")
    parser.add_argument("--judge_model", type=str, default="deepseek-v3-1-terminus")
    parser.add_argument("--api_timeout", type=int, default=300)
    parser.add_argument("--api_temperature", type=float, default=0.0)
    parser.add_argument("--api_cache_ttl", type=int, default=600)
    parser.add_argument("--api_cache_namespace", type=str, default="flashtrace-exp3")
    parser.add_argument("--retry_delay", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=2, help="Additional retries on API failure.")
    parser.add_argument("--request_interval", type=float, default=1.0, help="Sleep seconds between generation calls.")
    parser.add_argument("--judge_interval", type=float, default=1.0, help="Sleep seconds between judge calls.")
    parser.add_argument("--rate_limit_delay", type=float, default=5.0, help="Seconds to wait on HTTP 429 before retrying.")
    parser.add_argument(
        "--api_max_tokens_short",
        type=int,
        default=2048,
        help="Max tokens for the short-CoT generation call.",
    )
    parser.add_argument(
        "--api_max_tokens_long",
        type=int,
        default=8192,
        help="Max tokens for the long-CoT generation call.",
    )
    parser.add_argument(
        "--min_long_thinking_tokens",
        type=int,
        default=512,
        help="Minimum tokenizer tokens required in the long-CoT thinking segment.",
    )
    parser.add_argument(
        "--max_short_thinking_tokens",
        type=int,
        default=256,
        help="Maximum tokenizer tokens allowed in the short-CoT thinking segment.",
    )
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        default=None,
        help="Tokenizer path for span extraction & length constraints (default: generator_model).",
    )
    parser.add_argument("--data_root", type=str, default="exp/exp3/data", help="Output directory for exp3 caches.")
    parser.add_argument("--out_short", type=str, default=None, help="Optional explicit output path (short JSONL).")
    parser.add_argument("--out_long", type=str, default=None, help="Optional explicit output path (long JSONL).")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("FLASHTRACE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set --api_key or FLASHTRACE_API_KEY/OPENAI_API_KEY for API access.")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file not found: {dataset_path}")
    dataset_tag = str(args.dataset_tag or _infer_dataset_tag(dataset_path))

    tok_name = args.tokenizer_model or args.generator_model
    tok_path = Path(tok_name)
    if tok_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(tok_path.as_posix(), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer.pad_token = tokenizer.eos_token

    raw_examples = ds_utils.load_ruler(dataset_path, sample=None, seed=args.seed)
    if not raw_examples:
        raise SystemExit("No examples loaded from the RULER JSONL.")

    max_short = int(args.max_short) if args.max_short is not None else int(args.max_pairs)
    max_long = int(args.max_long) if args.max_long is not None else int(args.max_pairs)
    if max_short < 0 or max_long < 0:
        raise SystemExit("--max_short/--max_long must be >= 0.")

    kept_short: List[CachedExample] = []
    kept_long: List[CachedExample] = []

    total = len(raw_examples)
    attempted = 0

    for idx, ex in enumerate(tqdm(raw_examples, total=total, desc="Scanning raw RULER"), 1):
        attempted = idx
        if args.max_raw_examples is not None and idx > int(args.max_raw_examples):
            break
        if len(kept_short) >= max_short and len(kept_long) >= max_long:
            break

        reference_answer = _infer_reference_answer(ex)
        prompt = ex.prompt

        sample_id = _sha1_text(prompt)
        base_meta = dict(ex.metadata or {})
        base_meta["reference_answer"] = reference_answer
        base_meta["sample_id"] = sample_id
        base_meta["pair_id"] = sample_id  # backward-compatible name (may not be paired)
        base_meta["source_dataset_path"] = str(dataset_path)
        base_meta["prompt_sha1"] = sample_id

        if len(kept_short) < max_short:
            short_gen = _generate_one_style(
                prompt=prompt,
                reference_answer=reference_answer,
                tokenizer=tokenizer,
                style="short",
                system_prompt=SHORT_COT_SYSTEM_PROMPT,
                api_base=args.api_base,
                api_key=api_key,
                generator_model=args.generator_model,
                judge_model=args.judge_model,
                timeout=args.api_timeout,
                max_tokens=args.api_max_tokens_short,
                temperature=args.api_temperature,
                cache_ttl=args.api_cache_ttl,
                cache_namespace=args.api_cache_namespace,
                rate_limit_delay=args.rate_limit_delay,
                retries=args.retries,
                retry_delay=args.retry_delay,
                request_interval=args.request_interval,
                judge_interval=args.judge_interval,
                min_long_thinking_tokens=args.min_long_thinking_tokens,
                max_short_thinking_tokens=args.max_short_thinking_tokens,
            )
            if short_gen is not None:
                short_meta = dict(base_meta)
                short_meta.update(
                    {
                        "cot_style": "short",
                        "generator_model": args.generator_model,
                        "judge_model": args.judge_model,
                        "judge_response": short_gen.judge_response,
                        "boxed_answer": short_gen.boxed_answer,
                        "thinking_tokens": int(short_gen.thinking_tokens),
                    }
                )
                short_ex = CachedExample(
                    prompt=prompt,
                    target=short_gen.target_text,
                    indices_to_explain=None,
                    attr_mask_indices=ex.attr_mask_indices,
                    sink_span=None,
                    thinking_span=None,
                    metadata=short_meta,
                )
                short_ex = attach_spans_from_answer(short_ex, tokenizer, short_gen.boxed_answer)
                if isinstance(short_ex.sink_span, list) and len(short_ex.sink_span) == 2:
                    short_ex = CachedExample(
                        prompt=short_ex.prompt,
                        target=short_ex.target,
                        indices_to_explain=short_ex.sink_span,
                        attr_mask_indices=short_ex.attr_mask_indices,
                        sink_span=short_ex.sink_span,
                        thinking_span=short_ex.thinking_span,
                        metadata=short_ex.metadata,
                    )
                    kept_short.append(short_ex)
                    print(
                        f"[kept short] raw_idx={idx}/{total} thinking_tokens={short_gen.thinking_tokens} "
                        f"sample_id={sample_id[:8]} kept={len(kept_short)}/{max_short}"
                    )

        if len(kept_long) < max_long:
            long_gen = _generate_one_style(
                prompt=prompt,
                reference_answer=reference_answer,
                tokenizer=tokenizer,
                style="long",
                system_prompt=LONG_COT_SYSTEM_PROMPT,
                api_base=args.api_base,
                api_key=api_key,
                generator_model=args.generator_model,
                judge_model=args.judge_model,
                timeout=args.api_timeout,
                max_tokens=args.api_max_tokens_long,
                temperature=args.api_temperature,
                cache_ttl=args.api_cache_ttl,
                cache_namespace=args.api_cache_namespace,
                rate_limit_delay=args.rate_limit_delay,
                retries=args.retries,
                retry_delay=args.retry_delay,
                request_interval=args.request_interval,
                judge_interval=args.judge_interval,
                min_long_thinking_tokens=args.min_long_thinking_tokens,
                max_short_thinking_tokens=args.max_short_thinking_tokens,
            )
            if long_gen is not None:
                long_meta = dict(base_meta)
                long_meta.update(
                    {
                        "cot_style": "long",
                        "generator_model": args.generator_model,
                        "judge_model": args.judge_model,
                        "judge_response": long_gen.judge_response,
                        "boxed_answer": long_gen.boxed_answer,
                        "thinking_tokens": int(long_gen.thinking_tokens),
                    }
                )
                long_ex = CachedExample(
                    prompt=prompt,
                    target=long_gen.target_text,
                    indices_to_explain=None,
                    attr_mask_indices=ex.attr_mask_indices,
                    sink_span=None,
                    thinking_span=None,
                    metadata=long_meta,
                )
                long_ex = attach_spans_from_answer(long_ex, tokenizer, long_gen.boxed_answer)
                if isinstance(long_ex.sink_span, list) and len(long_ex.sink_span) == 2:
                    long_ex = CachedExample(
                        prompt=long_ex.prompt,
                        target=long_ex.target,
                        indices_to_explain=long_ex.sink_span,
                        attr_mask_indices=long_ex.attr_mask_indices,
                        sink_span=long_ex.sink_span,
                        thinking_span=long_ex.thinking_span,
                        metadata=long_ex.metadata,
                    )
                    kept_long.append(long_ex)
                    print(
                        f"[kept long] raw_idx={idx}/{total} thinking_tokens={long_gen.thinking_tokens} "
                        f"sample_id={sample_id[:8]} kept={len(kept_long)}/{max_long}"
                    )

    data_root = Path(args.data_root)
    out_short = Path(args.out_short) if args.out_short else data_root / f"{dataset_tag}_short_cot.jsonl"
    out_long = Path(args.out_long) if args.out_long else data_root / f"{dataset_tag}_long_cot.jsonl"

    n_short = write_cache(out_short, kept_short)
    n_long = write_cache(out_long, kept_long)
    print(
        f"Wrote short={n_short} -> {out_short}\n"
        f"Wrote long ={n_long} -> {out_long}\n"
        f"Attempted {attempted} / {total}"
    )

    missing: List[str] = []
    if len(kept_short) < max_short:
        missing.append(f"short({len(kept_short)}/{max_short})")
    if len(kept_long) < max_long:
        missing.append(f"long({len(kept_long)}/{max_long})")
    if missing:
        raise SystemExit(f"Could not find enough samples: {', '.join(missing)} (attempted {attempted} / {total}).")


if __name__ == "__main__":
    main()
