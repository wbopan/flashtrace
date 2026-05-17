from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Literal

from demo.live.span_parsers import run_parser_chain

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
    "<reason>",
    "</reason>",
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


@dataclass(frozen=True)
class TokenRecord:
    section: Section
    token_index: int
    token_id: int
    token_text: str
    char_start: int
    char_end: int
    kind: TokenKind
    selectable: bool
    role: str


def build_token_records(
    *,
    text: str,
    tokenizer,
    section: Section,
    role: str,
) -> list[TokenRecord]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids: list[int] = list(encoded["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in encoded["offset_mapping"]]
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    records: list[TokenRecord] = []
    for index, (token_id, (start, end)) in enumerate(zip(token_ids, offsets)):
        token_text = tokenizer.convert_ids_to_tokens(int(token_id))
        if isinstance(token_text, bytes):
            token_text = token_text.decode("utf-8", errors="replace")
        kind = classify_token_kind(
            token_text=token_text,
            token_id=int(token_id),
            special_ids=special_ids,
        )
        selectable = kind == "content"
        records.append(
            TokenRecord(
                section=section,
                token_index=index,
                token_id=int(token_id),
                token_text=token_text,
                char_start=int(start),
                char_end=int(end),
                kind=kind,
                selectable=selectable,
                role=role,
            )
        )
    return records


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def build_token_records_from_ids(
    *,
    token_ids: list[int] | tuple[int, ...],
    tokenizer,
    section: Section,
    role: str,
) -> tuple[list[TokenRecord], str]:
    """Build token records from model-emitted token IDs without re-tokenizing text."""
    ids = [int(token_id) for token_id in token_ids]
    decoded = _decode_token_ids(tokenizer, ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    records: list[TokenRecord] = []
    for index, token_id in enumerate(ids):
        current_text = _decode_token_ids(tokenizer, ids[: index + 1])
        single_text = _decode_token_ids(tokenizer, [token_id])
        if single_text and current_text.endswith(single_text):
            char_start = len(current_text) - len(single_text)
            char_end = len(current_text)
        else:
            char_start = records[-1].char_end if records else 0
            char_end = char_start + len(single_text)
        token_text = tokenizer.convert_ids_to_tokens(token_id)
        if isinstance(token_text, bytes):
            token_text = token_text.decode("utf-8", errors="replace")
        kind = classify_token_kind(
            token_text=token_text,
            token_id=token_id,
            special_ids=special_ids,
        )
        records.append(
            TokenRecord(
                section=section,
                token_index=index,
                token_id=token_id,
                token_text=token_text,
                char_start=int(char_start),
                char_end=int(char_end),
                kind=kind,
                selectable=kind == "content",
                role=role,
            )
        )
    return records, decoded


@dataclass(frozen=True)
class WordBox:
    box_id: str
    section: Section
    text: str
    char_start: int
    char_end: int
    token_indices: tuple[int, ...]
    kind: TokenKind
    selectable: bool
    score: float | None = None


def group_into_word_boxes(
    records: list[TokenRecord],
    *,
    source_text: str,
) -> list[WordBox]:
    boxes: list[WordBox] = []
    pending: list[TokenRecord] = []

    def flush_pending() -> None:
        if not pending:
            return
        char_start = pending[0].char_start
        char_end = pending[-1].char_end
        if source_text:
            if char_end > len(source_text):
                raise ValueError(
                    f"source_text length {len(source_text)} cannot satisfy "
                    f"char_end {char_end}; records must come from this text"
                )
            text = source_text[char_start:char_end]
        else:
            text = "".join(r.token_text for r in pending)
        section = pending[0].section
        boxes.append(
            WordBox(
                box_id=f"{section}-w-{len(boxes)}",
                section=section,
                text=text,
                char_start=char_start,
                char_end=char_end,
                token_indices=tuple(r.token_index for r in pending),
                kind="content",
                selectable=True,
            )
        )
        pending.clear()

    for record in records:
        if record.kind == "content":
            if pending and (
                record.char_start != pending[-1].char_end
                or record.section != pending[-1].section
            ):
                flush_pending()
            pending.append(record)
            continue

        flush_pending()
        section = record.section
        boxes.append(
            WordBox(
                box_id=f"{section}-t-{len(boxes)}",
                section=section,
                text=record.token_text,
                char_start=record.char_start,
                char_end=record.char_end,
                token_indices=(record.token_index,),
                kind=record.kind,
                selectable=record.selectable,
            )
        )

    flush_pending()
    return boxes


@dataclass(frozen=True)
class GenerationSections:
    generation_text: str
    thinking_char_span: tuple[int, int] | None
    answer_char_span: tuple[int, int]
    thinking_token_span: tuple[int, int] | None
    answer_token_span: tuple[int, int]
    parser: str


def detect_sections(*, text: str, tokenizer) -> GenerationSections:
    """Run the parser chain on `text`, then map char spans to token spans.

    The tokenizer must be a fast HF tokenizer that supports
    `return_offsets_mapping=True` (i.e. `PreTrainedTokenizerFast`).
    Use the same `tokenizer` and `text` here as in `build_token_records`
    so token indices line up across both calls.
    """
    parse = run_parser_chain(text)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in encoded:
        raise ValueError(
            "tokenizer did not return 'offset_mapping'; pass a fast HF tokenizer "
            "(PreTrainedTokenizerFast) so detect_sections can align spans."
        )
    offsets = [tuple(o) for o in encoded["offset_mapping"]]

    answer_token_span = char_span_to_token_span(
        offsets, parse.answer_char_span[0], parse.answer_char_span[1]
    )
    thinking_token_span: tuple[int, int] | None = None
    if parse.thinking_char_span is not None:
        try:
            thinking_token_span = char_span_to_token_span(
                offsets, parse.thinking_char_span[0], parse.thinking_char_span[1]
            )
        except ValueError:
            thinking_token_span = None

    return GenerationSections(
        generation_text=text,
        thinking_char_span=parse.thinking_char_span,
        answer_char_span=parse.answer_char_span,
        thinking_token_span=thinking_token_span,
        answer_token_span=answer_token_span,
        parser=parse.parser,
    )
