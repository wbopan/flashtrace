from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_REASON_RE = re.compile(r"<reason>(.*?)</reason>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Matches \boxed{simple content}. Does not match nested braces
# (e.g., \boxed{\frac{1}{2}}); such inputs fall through to parse_last_paragraph.
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_SENT_END_RE = re.compile(r"[.!?](?=\s|$)")


@dataclass(frozen=True)
class ParseResult:
    thinking_char_span: Optional[tuple[int, int]]
    answer_char_span: tuple[int, int]
    parser: str


def parse_think_answer(text: str) -> Optional[ParseResult]:
    answer_match = _ANSWER_RE.search(text)
    if not answer_match:
        return None
    think_match = _THINK_RE.search(text)
    parser = "think_answer"
    thinking_span: Optional[tuple[int, int]] = None
    if think_match is not None:
        thinking_span = think_match.span(1)
    else:
        reason_match = _REASON_RE.search(text)
        if reason_match is not None:
            thinking_span = reason_match.span(1)
            parser = "reason_answer"
    return ParseResult(
        thinking_char_span=thinking_span,
        answer_char_span=answer_match.span(1),
        parser=parser,
    )


def parse_boxed_answer(text: str) -> Optional[ParseResult]:
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    thinking_span: Optional[tuple[int, int]] = None
    if last.start() > 0:
        pre = text[: last.start()]
        sent_matches = list(_SENT_END_RE.finditer(pre))
        if sent_matches:
            thinking_end = sent_matches[-1].end()
        else:
            thinking_end = last.start()
        thinking_span = (0, thinking_end)
    return ParseResult(
        thinking_char_span=thinking_span,
        answer_char_span=last.span(1),
        parser="boxed",
    )


def parse_last_paragraph(text: str) -> Optional[ParseResult]:
    if not text.strip():
        return None
    blocks = [b for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        stripped = text.strip()
        start = text.find(stripped)
        return ParseResult(
            thinking_char_span=None,
            answer_char_span=(start, start + len(stripped)),
            parser="last_paragraph",
        )
    final_block = blocks[-1]
    answer_start_relative = text.rfind(final_block)
    answer_stripped = final_block.strip()
    answer_offset = final_block.find(answer_stripped)
    answer_start = answer_start_relative + answer_offset
    answer_end = answer_start + len(answer_stripped)

    thinking_text = text[:answer_start_relative].rstrip()
    thinking_start = 0
    while thinking_start < len(thinking_text) and thinking_text[thinking_start].isspace():
        thinking_start += 1
    thinking_span: Optional[tuple[int, int]] = None
    if thinking_start < len(thinking_text):
        thinking_span = (thinking_start, len(thinking_text))

    return ParseResult(
        thinking_char_span=thinking_span,
        answer_char_span=(answer_start, answer_end),
        parser="last_paragraph",
    )


_PARSER_CHAIN: list[Callable[[str], Optional[ParseResult]]] = [
    parse_think_answer,
    parse_boxed_answer,
    parse_last_paragraph,
]


def run_parser_chain(text: str) -> ParseResult:
    for parser in _PARSER_CHAIN:
        result = parser(text)
        if result is not None:
            return result
    raise ValueError("No parser produced a span for the given text.")
