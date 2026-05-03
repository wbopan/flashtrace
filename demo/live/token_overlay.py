from __future__ import annotations

from collections.abc import Collection
from typing import Literal

Section = Literal["prompt", "thinking", "answer", "other"]
TokenKind = Literal["content", "whitespace", "special", "template", "control"]

_TEMPLATE_MARKERS = {
    "<|im_start|>",
    "<|im_end|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|endoftext|>",
}

_CONTROL_MARKERS = {
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<reasoning>",
    "</reasoning>",
}


def classify_token_kind(
    *,
    token_text: str,
    token_id: int | None,
    special_ids: Collection[int],
) -> TokenKind:
    # Precedence: whitespace > template > control > special_ids > content.
    # Template/control beat special_ids because Qwen's chat-template tokens appear in both.
    if token_text and not token_text.strip():
        return "whitespace"
    if token_text in _TEMPLATE_MARKERS:
        return "template"
    if token_text in _CONTROL_MARKERS:
        return "control"
    if token_id is not None and token_id in special_ids:
        return "special"
    return "content"


def char_span_to_token_span(
    offsets: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    indices = [
        i
        for i, (start, end) in enumerate(offsets)
        if start < end and start < end_char and end > start_char
    ]
    if not indices:
        raise ValueError(
            f"No tokenizer tokens overlap the selected character span "
            f"[{start_char}, {end_char})"
        )
    return min(indices), max(indices)
