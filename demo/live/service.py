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
DEFAULT_PROMPT = 'Melanie Richards Griffith (born August 9, 1957) is an American film, stage, and television actress, and film producer.She began her career in the 1970s, appearing in several independent thriller films before achieving mainstream success in the mid-1980s.Born in New York City to actress Tippi Hedren and advertising executive Peter Griffith, she was raised mainly in Los Angeles, where she graduated from the Hollywood Professional School at age sixteen.In 1975, a then seventeen-year-old Griffith appeared opposite Gene Hackman in Arthur Penn\'s film noir "Night Moves".She later rose to prominence for her role portraying a pornographic actress in Brian De Palma\'s thriller "Body Double" (1984), which earned her a National Society of Film Critics Award for Best Supporting Actress.Griffith\'s subsequent performance in the comedy "Something Wild" (1986) garnered critical acclaim before she was cast in 1988\'s "Working Girl", which earned her a nomination for the Academy Award for Best Actress and won her a Golden Globe.The 1990s saw Griffith in a series of roles which received varying critical reception: she received Golden Globe nominations for her performances in "Buffalo Girls" (1995), and as Marion Davies in "RKO 281" (1999), while also earning a Golden Raspberry Award for Worst Actress for her performances in "Shining Through" (1992), as well as receiving nominations for "Crazy in Alabama" (1999) and John Waters\' cult film "Cecil B. Demented" (2000).Other credits include John Schlesinger\'s "Pacific Heights" (1990), "Milk Money" (1994), the neo-noir film "Mulholland Falls" (1996), as Charlotte Haze in Adrian Lyne\'s "Lolita" (1997), and "Another Day in Paradise" (1998).She later starred as Barbara Marx in "The Night We Called It a Day" (2003), and spent the majority of the 2000s appearing on such television series as "Nip/TuckRaising Hope", and "Hawaii Five-0".After acting on stage in London, in 2003 she made her Broadway debut in a revival of the musical "Chicago", receiving celebratory reviews.In the 2010s, Griffith returned to film, starring opposite then-husband Antonio Banderas in the science fiction film "Autómata" (2014) and as an acting coach in James Franco\'s "The Disaster Artist" (2017). Dakota Mayi Johnson (born October 4, 1989) is an American actress and model.The daughter of actors Don Johnson and Melanie Griffith, she made her film debut at age ten with a minor appearance in "Crazy in Alabama" (1999), a dark comedy film starring her mother.Johnson was discouraged from pursuing acting further until she completed high school, after which she began auditioning for roles in Los Angeles.She was cast in a minor part in "The Social Network" (2010), and subsequently had supporting roles in the comedy "21 Jump Street", the independent comedy "Goats", and the romantic comedy "The Five-Year Engagement" (all 2012).In 2015, Johnson had her first starring role as Anastasia Steele in the "Fifty Shades" film series (2015–18).For her performance in the series, she received a BAFTA Rising Star Award nomination in 2016.Following "Fifty Shades", Johnson appeared in the biographical crime film "Black Mass" (2015) and Luca Guadagnino\'s drama "A Bigger Splash" (2015).She reunited with Guadagnino, portraying the lead role in "Suspiria" (2018), a supernatural horror film based on the 1977 film by Dario Argento.That same year, she appeared in an ensemble cast in the thriller film "Bad Times at the El Royale" (2018).In 2019, Johnson had a starring role in the psychological horror film "Wounds" and the comedy-drama film "The Peanut Butter Falcon".\nWho is the maternal grandmother of Dakota Johnson?'
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
