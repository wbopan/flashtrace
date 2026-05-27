from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable


def _bootstrap_local_flashtrace() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "flashtrace").is_dir():
            sys.path.insert(0, str(parent))
            return


_bootstrap_local_flashtrace()

from demo.live.qwen_generation import format_chat_prompt, generate_with_qwen
from demo.live.token_overlay import (
    TokenRecord,
    build_token_records,
    build_token_records_from_ids,
    detect_sections,
)
from demo.live.token_document import build_document_views

DEFAULT_MODEL = os.environ.get("FLASHTRACE_DEMO_MODEL", "Qwen/Qwen3-4B-Thinking-2507")
DEFAULT_PROMPT = """Context:
Paris is the capital of France.
Berlin is the capital of Germany.
Madrid is the capital of Spain.

Question: What is the capital of France?"""
MAX_PROMPT_CHARS = int(os.environ.get("FLASHTRACE_DEMO_MAX_PROMPT_CHARS", "4000"))

# Model objects live on the GPU and are evicted when idle; tokenizers are
# CPU-only and stay cached so the prompt view keeps working without the model.
# All model load / inference / unload paths are funneled through the demo's
# single-worker executor, which serializes them — so no extra lock is needed.
_MODEL_CACHE: dict[tuple[str, str, str], tuple[object, object]] = {}
_TOKENIZER_CACHE: dict[str, object] = {}
_loaded_model_key: tuple[str, str, str] | None = None
_last_active_at: float | None = None


def _touch_active() -> None:
    global _last_active_at
    _last_active_at = time.monotonic()


def _free_cuda_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def clear_model_cache() -> None:
    """Drop every cached model and tokenizer and free GPU memory."""
    global _loaded_model_key, _last_active_at
    _MODEL_CACHE.clear()
    _TOKENIZER_CACHE.clear()
    _loaded_model_key = None
    _last_active_at = None
    _free_cuda_memory()


def unload_model() -> bool:
    """Evict the loaded model from GPU memory; keep tokenizers cached.

    Returns True if a model was actually resident and got unloaded.
    """
    global _loaded_model_key, _last_active_at
    had_model = bool(_MODEL_CACHE)
    _MODEL_CACHE.clear()
    _loaded_model_key = None
    _last_active_at = None
    if had_model:
        _free_cuda_memory()
    return had_model


def is_model_loaded() -> bool:
    return bool(_MODEL_CACHE)


def should_unload(idle_timeout: float) -> bool:
    """Whether an idle model has exceeded the unload timeout."""
    if idle_timeout <= 0 or not _MODEL_CACHE or _last_active_at is None:
        return False
    return (time.monotonic() - _last_active_at) >= idle_timeout


def model_status(idle_timeout: float | None = None) -> dict[str, Any]:
    loaded = bool(_MODEL_CACHE)
    idle_seconds = None
    seconds_until_unload = None
    if loaded and _last_active_at is not None:
        idle_seconds = max(0.0, time.monotonic() - _last_active_at)
        if idle_timeout is not None and idle_timeout > 0:
            seconds_until_unload = max(0.0, idle_timeout - idle_seconds)
    return {
        "loaded": loaded,
        "model": _loaded_model_key[0] if _loaded_model_key else None,
        "idle_seconds": idle_seconds,
        "idle_timeout": idle_timeout,
        "seconds_until_unload": seconds_until_unload,
    }


def parse_optional_span(value: str | None) -> tuple[int, int] | None:
    text = (value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError("Span must use START:END format.")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError("Span bounds must be integers.") from exc
    if start < 0 or end < start:
        raise ValueError("Span must satisfy 0 <= START <= END.")
    return start, end


_parse_optional_span = parse_optional_span


def _load_cached_model(
    model_name: str,
    *,
    device_map: str,
    dtype: str,
    loader: Callable | None = None,
):
    if loader is None:
        from flashtrace import load_model_and_tokenizer

        loader = load_model_and_tokenizer
    cache_key = (model_name, device_map, dtype)
    global _loaded_model_key
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = loader(model_name, device_map=device_map, dtype=dtype)
        _loaded_model_key = cache_key
        # The model loader also yields a tokenizer; reuse it for prompt views.
        _TOKENIZER_CACHE.setdefault(model_name, _MODEL_CACHE[cache_key][1])
    _touch_active()
    return _MODEL_CACHE[cache_key]


def _load_cached_tokenizer(
    model_name: str,
    *,
    device_map: str,
    dtype: str,
    tokenizer_loader: Callable | None = None,
    loader: Callable | None = None,
):
    """Load just the tokenizer (CPU only) so prompt tokenization never touches
    the GPU. Falls back to the injected model ``loader`` for tests/back-compat.
    """
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    if tokenizer_loader is not None:
        tokenizer = tokenizer_loader(model_name)
    elif loader is not None:
        _model, tokenizer = loader(model_name, device_map=device_map, dtype=dtype)
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _TOKENIZER_CACHE[model_name] = tokenizer
    return tokenizer


def _format_span(span: tuple[int, int] | None) -> str:
    if span is None:
        return ""
    return f"{span[0]}:{span[1]}"


def _clamp_span(span: tuple[int, int] | None, max_index: int) -> tuple[int, int] | None:
    if span is None or max_index < 0:
        return None
    start, end = span
    start = max(0, min(int(start), int(max_index)))
    end = max(0, min(int(end), int(max_index)))
    if end < start:
        return None
    return start, end


def _validate_model_and_prompt(model_name: str, prompt: str) -> tuple[str, str]:
    model_id = model_name.strip()
    prompt_text = prompt.strip()
    if not model_id:
        raise ValueError("Model is required.")
    if not prompt_text:
        raise ValueError("Prompt is required.")
    if len(prompt_text) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt must be at most {MAX_PROMPT_CHARS} characters.")
    return model_id, prompt_text


def _prompt_text_for_document(prompt: str, tokenizer) -> str:
    return format_chat_prompt(prompt, tokenizer)


def _build_prompt_records(prompt: str, tokenizer) -> list[TokenRecord]:
    return build_token_records(
        text=_prompt_text_for_document(prompt, tokenizer),
        tokenizer=tokenizer,
        section="prompt",
        role="user",
    )


def _strip_single_trailing_eos_text(text: str, tokenizer) -> str:
    eos = getattr(tokenizer, "eos_token", None)
    if eos and text.endswith(eos):
        return text[: -len(eos)]
    return text


def _validate_generation_span(
    *,
    span: tuple[int, int],
    generated_text: str,
    tokenizer,
) -> None:
    records = build_token_records(
        text=generated_text,
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )
    max_index = len(records) - 1
    if max_index < 0:
        raise ValueError("Generated text has no tokens to trace.")
    if span[1] > max_index:
        raise ValueError(f"Span end {span[1]} is outside generated token range 0:{max_index}.")


def run_prompt_document_phase(
    *,
    model_name: str,
    prompt: str,
    device_map: str,
    dtype: str,
    loader: Callable | None = None,
    tokenizer_loader: Callable | None = None,
) -> dict[str, Any]:
    model_id, prompt_text = _validate_model_and_prompt(model_name, prompt)
    # Tokenization only needs the tokenizer, so browsing the prompt view never
    # pulls the model onto the GPU (keeps it cold until Generate/Trace).
    tokenizer = _load_cached_tokenizer(
        model_id,
        device_map=device_map,
        dtype=dtype,
        tokenizer_loader=tokenizer_loader,
        loader=loader,
    )
    prompt_records = _build_prompt_records(prompt_text, tokenizer)
    return {
        "render_model": build_document_views(phase="prompt", prompt_records=prompt_records),
        "status": f"Prompt tokenized into {len(prompt_records)} tokens.",
    }


def run_generate_document_phase(
    *,
    model_name: str,
    prompt: str,
    device_map: str,
    dtype: str,
    max_new_tokens: int,
    loader: Callable | None = None,
) -> dict[str, Any]:
    model_id, prompt_text = _validate_model_and_prompt(model_name, prompt)
    model, tokenizer = _load_cached_model(
        model_id, device_map=device_map, dtype=dtype, loader=loader
    )
    output = generate_with_qwen(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_new_tokens=int(max_new_tokens),
    )
    sections = detect_sections(text=output.text, tokenizer=tokenizer)
    prompt_records = _build_prompt_records(prompt_text, tokenizer)
    generation_records, generated_text = build_token_records_from_ids(
        token_ids=output.token_ids,
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )
    document = build_document_views(
        phase="generated",
        prompt_records=prompt_records,
        generation_records=generation_records,
        answer_token_span=sections.answer_token_span,
    )
    target_span = document["target_span"]
    target_span_tuple = tuple(target_span) if target_span is not None else None
    max_index = target_span_tuple[1] if target_span_tuple is not None else -1
    reasoning_span = _clamp_span(sections.thinking_token_span, max_index)
    status = (
        f"Generated {len(generation_records)} tokens. "
        f"Default target span: {_format_span(target_span_tuple) or 'empty'}."
    )
    if sections.thinking_token_span is not None and reasoning_span is None:
        status += " Reasoning span was outside the attributable generation range and was dropped."
    if target_span_tuple is None:
        status += " Trace is disabled because there are no selectable generation tokens."
    _touch_active()
    return {
        "render_model": document,
        "generated_text": generated_text,
        "target_span": _format_span(target_span_tuple),
        "reasoning_span": _format_span(reasoning_span),
        "status": status,
    }


def run_trace_document_phase(
    *,
    model_name: str,
    prompt: str,
    generated_text: str,
    target_span: str,
    reasoning_span: str,
    method: str,
    hops: int,
    device_map: str,
    dtype: str,
    chunk_tokens: int,
    sink_chunk_tokens: int,
    loader: Callable | None = None,
    tracer_cls: type | None = None,
) -> dict[str, Any]:
    model_id, prompt_text = _validate_model_and_prompt(model_name, prompt)
    if not generated_text:
        raise ValueError("Generate a response before tracing.")
    output_span = parse_optional_span(target_span)
    if output_span is None:
        raise ValueError("Select at least two target endpoints before tracing.")
    reasoning_span_tuple = parse_optional_span(reasoning_span)
    if tracer_cls is None:
        from flashtrace import FlashTrace

        tracer_cls = FlashTrace

    model, tokenizer = _load_cached_model(
        model_id, device_map=device_map, dtype=dtype, loader=loader
    )
    target_text = _strip_single_trailing_eos_text(generated_text, tokenizer)
    _validate_generation_span(span=output_span, generated_text=target_text, tokenizer=tokenizer)
    tracer = tracer_cls(
        model,
        tokenizer,
        chunk_tokens=int(chunk_tokens),
        sink_chunk_tokens=int(sink_chunk_tokens),
        use_chat_template=True,
    )
    result = tracer.trace(
        prompt=prompt_text,
        target=target_text,
        output_span=output_span,
        reasoning_span=reasoning_span_tuple,
        hops=int(hops),
        method=method,
    )
    prompt_records = _build_prompt_records(prompt_text, tokenizer)
    generation_records = build_token_records(
        text=target_text,
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )
    document = build_document_views(
        phase="traced",
        result=result,
        prompt_records=prompt_records,
        generation_records=generation_records,
    )
    # Views are [Select target, Hop 1..N, Aggregate].
    hop_count = max(0, len(document["views"]) - 2)
    if hop_count:
        status = (
            f"Trace complete with Select target, {hop_count} hop view(s), "
            f"and Aggregate for {method}."
        )
    else:
        status = f"Trace complete with Select target and Aggregate views for {method}."
    _touch_active()
    return {
        "render_model": document,
        "trace_json": result.to_dict(),
        "status": status,
    }
