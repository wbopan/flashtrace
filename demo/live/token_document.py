from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from demo.live.token_overlay import TokenRecord
from flashtrace.viz import _score_color

Phase = Literal["prompt", "generated", "traced"]
Region = Literal["prompt", "generation"]

TOKEN_DOCUMENT_ELEM_ID = "flashtrace-token-document"


def _generation_color(score: float, max_score: float) -> str:
    """Blue-toned gradient for generation-side attribution weights.

    Mirrors ``flashtrace.viz._score_color`` (which is warm/orange and used for
    the prompt region) but renders the generation region in blue so the two
    sides are visually distinct. Normalisation is per-view (caller passes the
    view's own max-abs score).
    """
    if max_score <= 0.0:
        return "rgba(245,245,245,0.75)"
    ratio = min(1.0, abs(float(score)) / (max_score + 1e-12))
    red = int(226 - 158 * ratio)
    green = int(240 - 86 * ratio)
    blue = 255
    alpha = 0.22 + 0.58 * ratio
    return f"rgba({red},{green},{blue},{alpha:.3f})"


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


def _default_target_span(
    generation_records: Sequence[TokenRecord],
    *,
    answer_span: tuple[int, int] | list[int] | None = None,
) -> list[int] | None:
    """Pick the default target span over selectable generation content tokens.

    When ``answer_span`` is given (the answer-section token span — i.e. content
    after ``</think>`` and before ``<|im_end|>``), the default is restricted to
    the selectable content tokens inside it. Otherwise it spans every selectable
    content token.
    """
    selectable = [
        int(record.token_index)
        for record in generation_records
        if record.kind == "content" and bool(record.selectable)
    ]
    if not selectable:
        return None
    if answer_span is not None and len(answer_span) == 2:
        low, high = int(answer_span[0]), int(answer_span[1])
        restricted = [index for index in selectable if low <= index <= high]
        if restricted:
            return [min(restricted), max(restricted)]
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
            generation_color: str | None = _generation_color(
                generation_score, max_generation_score
            )
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
    answer_token_span: tuple[int, int] | list[int] | None = None,
    active_view: int | None = None,
) -> dict[str, Any]:
    if phase == "traced":
        if result is None:
            raise ValueError("result is required for traced phase")
        model_target = _span_to_list(result.output_span)
        aggregate_view = _build_trace_view(
            name="Aggregate",
            prompt_tokens=result.prompt_tokens,
            generation_tokens=result.generation_tokens,
            scores=result.scores,
            generation_scores=getattr(result, "generation_scores", []),
            target_span=model_target,
        )
        per_hop_target_spans = list(getattr(result, "per_hop_target_spans", []) or [])
        per_hop_generation_scores = list(getattr(result, "per_hop_generation_scores", []) or [])
        hop_views: list[dict[str, Any]] = []
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
            hop_views.append(
                _build_trace_view(
                    name=f"Hop {hop_index}",
                    prompt_tokens=result.prompt_tokens,
                    generation_tokens=result.generation_tokens,
                    scores=hop_scores,
                    generation_scores=generation_for_hop,
                    target_span=target_for_hop,
                )
            )
        # The "Select target" tab re-uses the interactive prompt/generation view
        # so the user can pick a fresh span and trace again. It is the default
        # (left-most) tab; "Aggregate" sits at the far right after the hops.
        select_view = _build_prompt_generated_view(
            name="Select target",
            prompt_records=list(prompt_records or []),
            generation_records=list(generation_records or []),
            interactive=True,
            target_span=model_target,
        )
        views = [select_view, *hop_views, aggregate_view]
        resolved_active = int(active_view) if active_view is not None else len(views) - 1
        return {
            "phase": phase,
            "active_view": resolved_active,
            "views": views,
            "target_span": model_target,
        }

    prompt = list(prompt_records or [])
    generation = list(generation_records or [])
    model_target = _span_to_list(target_span)
    if phase == "generated" and model_target is None:
        model_target = _default_target_span(generation, answer_span=answer_token_span)
    interactive = phase == "generated" and model_target is not None
    return {
        "phase": phase,
        "active_view": int(active_view) if active_view is not None else 0,
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
