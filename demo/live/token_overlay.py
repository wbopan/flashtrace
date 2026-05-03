from __future__ import annotations

from typing import Iterable, Literal

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
    special_ids: Iterable[int],
) -> TokenKind:
    if token_text and not token_text.strip():
        return "whitespace"
    if token_text in _TEMPLATE_MARKERS:
        return "template"
    if token_text in _CONTROL_MARKERS:
        return "control"
    if token_id is not None and token_id in set(special_ids):
        return "special"
    return "content"
