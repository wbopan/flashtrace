from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from demo.live.token_overlay import TokenRecord
from flashtrace.viz import _score_color

Phase = Literal["prompt", "generated", "traced"]
Region = Literal["prompt", "generation"]

TOKEN_DOCUMENT_ELEM_ID = "flashtrace-token-document"


def _span_to_list(span: tuple[int, int] | list[int] | None) -> list[int] | None:
    if span is None:
        return None
    if len(span) != 2:
        return None
    return [int(span[0]), int(span[1])]


def _is_target(gen_index: int | None, target_span: list[int] | None) -> bool:
    if gen_index is None or target_span is None:
        return False
    return int(target_span[0]) <= int(gen_index) <= int(target_span[1])


def _record_token(
    record: TokenRecord,
    *,
    document_index: int,
    region: Region,
    gen_index: int | None,
    selectable: bool,
    score: float | None = None,
    color: str | None = None,
    is_target: bool = False,
) -> dict[str, Any]:
    return {
        "i": int(document_index),
        "text": record.display_text or record.token_text,
        "kind": record.kind,
        "region": region,
        "gen_index": gen_index,
        "selectable": bool(selectable),
        "score": None if score is None else float(score),
        "color": color,
        "is_target": bool(is_target),
    }


def _text_token(
    *,
    text: str,
    document_index: int,
    region: Region,
    gen_index: int | None,
    score: float | None,
    color: str | None,
    is_target: bool = False,
) -> dict[str, Any]:
    return {
        "i": int(document_index),
        "text": str(text),
        "kind": "content",
        "region": region,
        "gen_index": gen_index,
        "selectable": False,
        "score": None if score is None else float(score),
        "color": color,
        "is_target": bool(is_target),
    }


def _default_target_span(generation_records: Sequence[TokenRecord]) -> list[int] | None:
    selectable = [
        int(record.token_index)
        for record in generation_records
        if record.kind == "content" and bool(record.selectable)
    ]
    if not selectable:
        return None
    return [min(selectable), max(selectable)]


def _build_prompt_generated_view(
    *,
    name: str,
    prompt_records: Sequence[TokenRecord],
    generation_records: Sequence[TokenRecord],
    interactive: bool,
    target_span: list[int] | None,
) -> dict[str, Any]:
    tokens: list[dict[str, Any]] = []
    for record in prompt_records:
        tokens.append(
            _record_token(
                record,
                document_index=len(tokens),
                region="prompt",
                gen_index=None,
                selectable=False,
                color=None,
            )
        )
    for record in generation_records:
        selectable = bool(record.selectable and record.kind == "content")
        tokens.append(
            _record_token(
                record,
                document_index=len(tokens),
                region="generation",
                gen_index=int(record.token_index),
                selectable=selectable,
                color=None,
                is_target=_is_target(int(record.token_index), target_span),
            )
        )
    return {
        "name": name,
        "interactive": bool(interactive),
        "target_span": target_span,
        "tokens": tokens,
    }


def _build_trace_view(
    *,
    name: str,
    prompt_tokens: Sequence[str],
    generation_tokens: Sequence[str],
    scores: Sequence[float],
    generation_scores: Sequence[float] | None = None,
    target_span: list[int] | None = None,
) -> dict[str, Any]:
    tokens: list[dict[str, Any]] = []
    max_score = max((abs(float(score)) for score in scores), default=0.0)
    generation_values = list(generation_scores or [])
    max_generation_score = max((abs(float(score)) for score in generation_values), default=0.0)
    for index, token in enumerate(prompt_tokens):
        score = float(scores[index]) if index < len(scores) else 0.0
        tokens.append(
            _text_token(
                text=token,
                document_index=len(tokens),
                region="prompt",
                gen_index=None,
                score=score,
                color=_score_color(score, max_score),
                is_target=False,
            )
        )
    for gen_index, token in enumerate(generation_tokens):
        if gen_index < len(generation_values):
            generation_score: float | None = float(generation_values[gen_index])
            generation_color: str | None = _score_color(generation_score, max_generation_score)
        else:
            generation_score = None
            generation_color = None
        tokens.append(
            _text_token(
                text=token,
                document_index=len(tokens),
                region="generation",
                gen_index=gen_index,
                score=generation_score,
                color=generation_color,
                is_target=_is_target(gen_index, target_span),
            )
        )
    return {
        "name": name,
        "interactive": False,
        "target_span": target_span,
        "tokens": tokens,
    }


def build_document_views(
    *,
    phase: Phase,
    prompt_records: Sequence[TokenRecord] | None = None,
    generation_records: Sequence[TokenRecord] | None = None,
    result: Any | None = None,
    target_span: tuple[int, int] | list[int] | None = None,
    active_view: int = 0,
) -> dict[str, Any]:
    if phase == "traced":
        if result is None:
            raise ValueError("result is required for traced phase")
        model_target = _span_to_list(result.output_span)
        views = [
            _build_trace_view(
                name="Aggregate",
                prompt_tokens=result.prompt_tokens,
                generation_tokens=result.generation_tokens,
                scores=result.scores,
                generation_scores=getattr(result, "generation_scores", []),
                target_span=model_target,
            )
        ]
        per_hop_target_spans = list(getattr(result, "per_hop_target_spans", []) or [])
        per_hop_generation_scores = list(getattr(result, "per_hop_generation_scores", []) or [])
        for hop_index, hop_scores in enumerate(result.per_hop_scores or [], start=1):
            target_for_hop = (
                _span_to_list(per_hop_target_spans[hop_index - 1])
                if hop_index - 1 < len(per_hop_target_spans)
                else model_target
            )
            generation_for_hop = (
                per_hop_generation_scores[hop_index - 1]
                if hop_index - 1 < len(per_hop_generation_scores)
                else []
            )
            views.append(
                _build_trace_view(
                    name=f"Hop {hop_index}",
                    prompt_tokens=result.prompt_tokens,
                    generation_tokens=result.generation_tokens,
                    scores=hop_scores,
                    generation_scores=generation_for_hop,
                    target_span=target_for_hop,
                )
            )
        return {
            "phase": phase,
            "active_view": int(active_view),
            "views": views,
            "target_span": model_target,
        }

    prompt = list(prompt_records or [])
    generation = list(generation_records or [])
    model_target = _span_to_list(target_span)
    if phase == "generated" and model_target is None:
        model_target = _default_target_span(generation)
    interactive = phase == "generated" and model_target is not None
    return {
        "phase": phase,
        "active_view": int(active_view),
        "views": [
            _build_prompt_generated_view(
                name="Document",
                prompt_records=prompt,
                generation_records=generation,
                interactive=interactive,
                target_span=model_target,
            )
        ],
        "target_span": model_target,
    }
