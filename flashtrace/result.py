from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TraceResult:
    """Public attribution result returned by FlashTrace."""

    prompt_tokens: list[str]
    generation_tokens: list[str]
    scores: list[float]
