# Qwen Token Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an end-to-end Qwen-backed live demo that generates reasoning + answer, classifies every tokenizer token, groups them into word boxes, lets the user inspect/select answer and reasoning spans, and feeds those spans into FlashTrace.

**Architecture:** Three new helper modules in `demo/live/` cleanly separate concerns: `token_overlay.py` owns the data model + tokenizer-offset math + word grouping, `span_parsers.py` parses think/answer/boxed/last-paragraph formats, and `qwen_generation.py` wraps deterministic Qwen generation. The existing `app.py` is extended with a Generate button, inspector tables, and auto-populated span fields that flow into the existing `tracer.trace()` call. UI uses native Gradio components (`HighlightedText`, `Dataframe`, `Textbox`) - no custom JS.

**Tech Stack:** Python 3.11+, transformers (AutoTokenizer with `return_offsets_mapping=True`), Gradio 5.x, pytest.

---

## File Structure

**New files:**
- `demo/live/token_overlay.py` — Dataclasses (`TokenRecord`, `WordBox`, `GenerationSections`), token kind classification, char↔token span conversion, word grouping, token-record builders.
- `demo/live/span_parsers.py` — Parser chain for `<think>/<answer>`, `\boxed{}`, last-paragraph fallback. Pure string functions returning char spans.
- `demo/live/qwen_generation.py` — Deterministic generation wrapper that returns generated text + token ids.
- `tests/test_token_overlay.py` — Unit tests for the data model and word grouping.
- `tests/test_span_parsers.py` — Unit tests for each parser + the chain.
- `tests/test_qwen_generation.py` — Tests for generation wrapper using mocked model.

**Modified files:**
- `demo/live/app.py` — Add Generate button, sections inspector, raw-token Dataframe, auto-populate span textboxes, pipe selected spans into existing `run_trace`.
- `tests/test_live_demo.py` — Add e2e test that exercises generate→parse→trace flow against the smoke model.

---

## Task 1: Token Kind Classification

**Files:**
- Create: `demo/live/token_overlay.py`
- Test: `tests/test_token_overlay.py`

- [ ] **Step 1.1: Write the failing test**

Create `tests/test_token_overlay.py` with:

```python
from __future__ import annotations

from demo.live.token_overlay import classify_token_kind


def test_classify_whitespace_only():
    assert classify_token_kind(token_text=" ", token_id=10, special_ids=set()) == "whitespace"
    assert classify_token_kind(token_text="\n\t", token_id=11, special_ids=set()) == "whitespace"


def test_classify_special_id_marks_special():
    assert classify_token_kind(token_text="<eos>", token_id=2, special_ids={2}) == "special"


def test_classify_chat_template_marker_is_template():
    assert classify_token_kind(token_text="<|im_start|>", token_id=151644, special_ids={151644}) == "template"
    assert classify_token_kind(token_text="<|im_end|>", token_id=151645, special_ids={151645}) == "template"


def test_classify_think_marker_is_control():
    assert classify_token_kind(token_text="<think>", token_id=200, special_ids={200}) == "control"
    assert classify_token_kind(token_text="</answer>", token_id=201, special_ids={201}) == "control"


def test_classify_regular_word_is_content():
    assert classify_token_kind(token_text="Paris", token_id=42, special_ids=set()) == "content"
    assert classify_token_kind(token_text="!", token_id=43, special_ids=set()) == "content"
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `pytest tests/test_token_overlay.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'demo.live.token_overlay'`

- [ ] **Step 1.3: Implement `classify_token_kind`**

Create `demo/live/token_overlay.py`:

```python
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
```

- [ ] **Step 1.4: Run tests, verify pass**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 5 PASS

- [ ] **Step 1.5: Commit**

```bash
git add demo/live/token_overlay.py tests/test_token_overlay.py
git commit -m "feat(live): add token kind classification"
```

---

## Task 2: Char Span to Token Span Conversion

**Files:**
- Modify: `demo/live/token_overlay.py`
- Test: `tests/test_token_overlay.py`

- [ ] **Step 2.1: Append failing tests**

Add to `tests/test_token_overlay.py`:

```python
import pytest

from demo.live.token_overlay import char_span_to_token_span


def test_char_span_picks_overlapping_tokens():
    offsets = [(0, 5), (5, 6), (6, 11), (11, 12), (12, 16)]
    assert char_span_to_token_span(offsets, 0, 11) == (0, 2)
    assert char_span_to_token_span(offsets, 6, 16) == (2, 4)


def test_char_span_includes_partial_overlap():
    offsets = [(0, 5), (5, 10), (10, 15)]
    assert char_span_to_token_span(offsets, 3, 12) == (0, 2)


def test_char_span_raises_on_no_overlap():
    offsets = [(0, 5), (5, 10)]
    with pytest.raises(ValueError, match="No tokenizer tokens overlap"):
        char_span_to_token_span(offsets, 20, 30)


def test_char_span_skips_zero_length_tokens():
    offsets = [(0, 5), (5, 5), (5, 10)]
    assert char_span_to_token_span(offsets, 5, 10) == (2, 2)
```

- [ ] **Step 2.2: Run tests, verify failure**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 4 new tests FAIL with `ImportError`

- [ ] **Step 2.3: Implement `char_span_to_token_span`**

Append to `demo/live/token_overlay.py`:

```python
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
```

- [ ] **Step 2.4: Run tests, verify pass**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 9 PASS total

- [ ] **Step 2.5: Commit**

```bash
git add demo/live/token_overlay.py tests/test_token_overlay.py
git commit -m "feat(live): convert character spans to token spans via overlap"
```

---

## Task 3: TokenRecord and `build_token_records`

**Files:**
- Modify: `demo/live/token_overlay.py`
- Test: `tests/test_token_overlay.py`

- [ ] **Step 3.1: Append failing test**

Add to `tests/test_token_overlay.py`:

```python
from tests.helpers import make_tiny_qwen2_model_and_tokenizer

from demo.live.token_overlay import TokenRecord, build_token_records


def test_build_token_records_uses_tokenizer_offsets():
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    records = build_token_records(
        text="t10 t20 t30",
        tokenizer=tokenizer,
        section="prompt",
        role="user",
    )

    assert len(records) == 3
    assert all(isinstance(r, TokenRecord) for r in records)
    assert [r.token_text for r in records] == ["t10", "t20", "t30"]
    assert [r.token_index for r in records] == [0, 1, 2]
    assert records[0].char_start == 0 and records[0].char_end == 3
    assert records[1].char_start == 4 and records[1].char_end == 7
    assert all(r.section == "prompt" for r in records)
    assert all(r.role == "user" for r in records)
    assert all(r.kind == "content" for r in records)
    assert all(r.selectable for r in records)


def test_build_token_records_marks_specials_unselectable():
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    eos_id = tokenizer.eos_token_id

    records = build_token_records(
        text="t10 t1",
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )

    eos_records = [r for r in records if r.token_id == eos_id]
    assert eos_records, "expected an EOS-id token in records"
    assert all(r.kind == "special" and not r.selectable for r in eos_records)
```

- [ ] **Step 3.2: Run, verify failure**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 2 new tests FAIL with `ImportError`

- [ ] **Step 3.3: Implement `TokenRecord` and `build_token_records`**

Append to `demo/live/token_overlay.py`:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class TokenRecord:
    section: Section
    token_index: int
    token_id: int | None
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
```

- [ ] **Step 3.4: Run tests, verify pass**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 11 PASS total

- [ ] **Step 3.5: Commit**

```bash
git add demo/live/token_overlay.py tests/test_token_overlay.py
git commit -m "feat(live): build TokenRecord list from tokenizer offsets"
```

---

## Task 4: WordBox Grouping

**Files:**
- Modify: `demo/live/token_overlay.py`
- Test: `tests/test_token_overlay.py`

- [ ] **Step 4.1: Append failing tests**

Add to `tests/test_token_overlay.py`:

```python
from demo.live.token_overlay import WordBox, group_into_word_boxes


def _record(
    *,
    index: int,
    text: str,
    char_start: int,
    char_end: int,
    kind: str = "content",
    selectable: bool = True,
    section: str = "answer",
) -> TokenRecord:
    return TokenRecord(
        section=section,
        token_index=index,
        token_id=index,
        token_text=text,
        char_start=char_start,
        char_end=char_end,
        kind=kind,
        selectable=selectable,
        role="assistant",
    )


def test_group_into_word_boxes_joins_subwords():
    records = [
        _record(index=0, text="Par", char_start=0, char_end=3),
        _record(index=1, text="is", char_start=3, char_end=5),
        _record(index=2, text=" ", char_start=5, char_end=6, kind="whitespace", selectable=False),
        _record(index=3, text="rocks", char_start=6, char_end=11),
    ]

    boxes = group_into_word_boxes(records, source_text="Paris rocks")

    content_boxes = [b for b in boxes if b.kind == "content"]
    assert len(content_boxes) == 2
    assert content_boxes[0].text == "Paris"
    assert content_boxes[0].token_indices == (0, 1)
    assert content_boxes[1].text == "rocks"
    assert content_boxes[1].token_indices == (3,)


def test_group_keeps_special_tokens_as_their_own_box():
    records = [
        _record(index=0, text="Hi", char_start=0, char_end=2),
        _record(index=1, text="<eos>", char_start=2, char_end=2, kind="special", selectable=False),
    ]

    boxes = group_into_word_boxes(records, source_text="Hi")

    assert any(b.kind == "special" and b.text == "<eos>" for b in boxes)
    assert any(b.kind == "content" and b.text == "Hi" for b in boxes)


def test_group_assigns_unique_box_ids():
    records = [
        _record(index=0, text="a", char_start=0, char_end=1),
        _record(index=1, text=" ", char_start=1, char_end=2, kind="whitespace", selectable=False),
        _record(index=2, text="b", char_start=2, char_end=3),
    ]

    boxes = group_into_word_boxes(records, source_text="a b")

    ids = [b.box_id for b in boxes]
    assert len(set(ids)) == len(ids)
```

- [ ] **Step 4.2: Run, verify failure**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 3 new tests FAIL with `ImportError`

- [ ] **Step 4.3: Implement `WordBox` and `group_into_word_boxes`**

Append to `demo/live/token_overlay.py`:

```python
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
        text = source_text[char_start:char_end] if source_text else "".join(
            r.token_text for r in pending
        )
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
            if pending and record.char_start != pending[-1].char_end:
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
```

- [ ] **Step 4.4: Run tests, verify pass**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 14 PASS total

- [ ] **Step 4.5: Commit**

```bash
git add demo/live/token_overlay.py tests/test_token_overlay.py
git commit -m "feat(live): group token records into word boxes"
```

---

## Task 5: Span Parsers (think/answer, boxed, last paragraph)

**Files:**
- Create: `demo/live/span_parsers.py`
- Test: `tests/test_span_parsers.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/test_span_parsers.py`:

```python
from __future__ import annotations

from demo.live.span_parsers import (
    ParseResult,
    parse_boxed_answer,
    parse_last_paragraph,
    parse_think_answer,
    run_parser_chain,
)


def test_think_answer_extracts_both_spans():
    text = "<think>\nfirst step\n</think>\n<answer>\nParis\n</answer>"

    result = parse_think_answer(text)

    assert result is not None
    assert text[result.thinking_char_span[0] : result.thinking_char_span[1]] == "\nfirst step\n"
    assert text[result.answer_char_span[0] : result.answer_char_span[1]] == "\nParis\n"
    assert result.parser == "think_answer"


def test_think_answer_returns_none_without_markers():
    assert parse_think_answer("just some text") is None


def test_boxed_answer_extracts_inside_braces():
    text = "Reasoning text. The answer is \\boxed{Paris}."

    result = parse_boxed_answer(text)

    assert result is not None
    assert text[result.answer_char_span[0] : result.answer_char_span[1]] == "Paris"
    assert result.thinking_char_span is not None
    assert text[result.thinking_char_span[0] : result.thinking_char_span[1]].strip() == "Reasoning text."
    assert result.parser == "boxed"


def test_boxed_answer_returns_none_without_brace():
    assert parse_boxed_answer("no boxed here") is None


def test_last_paragraph_uses_final_block_as_answer():
    text = "Step 1.\nStep 2.\n\nFinal answer is Paris."

    result = parse_last_paragraph(text)

    assert result is not None
    assert text[result.answer_char_span[0] : result.answer_char_span[1]] == "Final answer is Paris."
    assert result.thinking_char_span is not None
    assert text[result.thinking_char_span[0] : result.thinking_char_span[1]].endswith("Step 2.")
    assert result.parser == "last_paragraph"


def test_last_paragraph_falls_back_to_full_text_when_single_block():
    text = "Just one block of answer."
    result = parse_last_paragraph(text)
    assert result is not None
    assert result.thinking_char_span is None
    assert text[result.answer_char_span[0] : result.answer_char_span[1]] == text


def test_run_parser_chain_prefers_think_answer():
    text = "<think>r</think><answer>Paris</answer>"
    result = run_parser_chain(text)
    assert isinstance(result, ParseResult)
    assert result.parser == "think_answer"


def test_run_parser_chain_falls_through_to_boxed_then_last_paragraph():
    boxed = run_parser_chain("Some reasoning. \\boxed{Paris}.")
    assert boxed.parser == "boxed"

    last = run_parser_chain("Step1.\n\nFinal Paris.")
    assert last.parser == "last_paragraph"
```

- [ ] **Step 5.2: Run, verify failure**

Run: `pytest tests/test_span_parsers.py -v`
Expected: 8 FAIL with `ModuleNotFoundError`

- [ ] **Step 5.3: Implement `span_parsers.py`**

Create `demo/live/span_parsers.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")


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
    thinking_span: Optional[tuple[int, int]] = None
    if think_match is not None:
        thinking_span = think_match.span(1)
    return ParseResult(
        thinking_char_span=thinking_span,
        answer_char_span=answer_match.span(1),
        parser="think_answer",
    )


def parse_boxed_answer(text: str) -> Optional[ParseResult]:
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    last = matches[-1]
    thinking_span: Optional[tuple[int, int]] = None
    if last.start() > 0:
        thinking_span = (0, last.start())
    return ParseResult(
        thinking_char_span=thinking_span,
        answer_char_span=last.span(1),
        parser="boxed",
    )


def parse_last_paragraph(text: str) -> Optional[ParseResult]:
    if not text.strip():
        return None
    blocks = re.split(r"\n\s*\n", text)
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
```

- [ ] **Step 5.4: Run, verify pass**

Run: `pytest tests/test_span_parsers.py -v`
Expected: 8 PASS

- [ ] **Step 5.5: Commit**

```bash
git add demo/live/span_parsers.py tests/test_span_parsers.py
git commit -m "feat(live): parse think/answer, boxed, and last-paragraph spans"
```

---

## Task 6: GenerationSections + `detect_sections` orchestrator

**Files:**
- Modify: `demo/live/token_overlay.py`
- Test: `tests/test_token_overlay.py`

- [ ] **Step 6.1: Append failing tests**

Add to `tests/test_token_overlay.py`:

```python
from demo.live.token_overlay import GenerationSections, detect_sections


def test_detect_sections_with_think_answer_markers():
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    text = "t10 t20 t30 t40"

    sections = detect_sections(text=text, tokenizer=tokenizer)

    assert isinstance(sections, GenerationSections)
    assert sections.generation_text == text
    assert sections.parser == "last_paragraph"
    assert sections.thinking_token_span is None
    answer_start, answer_end = sections.answer_token_span
    assert 0 <= answer_start <= answer_end


def test_detect_sections_maps_think_answer_to_token_indices():
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    text = "<think> t10 t20 </think> <answer> t30 t40 </answer>"

    sections = detect_sections(text=text, tokenizer=tokenizer)

    assert sections.parser == "think_answer"
    assert sections.thinking_token_span is not None
    t_start, t_end = sections.thinking_token_span
    a_start, a_end = sections.answer_token_span
    assert t_start <= t_end < a_start <= a_end
```

- [ ] **Step 6.2: Run, verify failure**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 2 new tests FAIL with `ImportError`

- [ ] **Step 6.3: Implement `GenerationSections` and `detect_sections`**

Append to `demo/live/token_overlay.py`:

```python
from demo.live.span_parsers import run_parser_chain


@dataclass(frozen=True)
class GenerationSections:
    generation_text: str
    thinking_char_span: tuple[int, int] | None
    answer_char_span: tuple[int, int]
    thinking_token_span: tuple[int, int] | None
    answer_token_span: tuple[int, int]
    parser: str


def detect_sections(*, text: str, tokenizer) -> GenerationSections:
    parse = run_parser_chain(text)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
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
```

- [ ] **Step 6.4: Run, verify pass**

Run: `pytest tests/test_token_overlay.py -v`
Expected: 16 PASS total

- [ ] **Step 6.5: Commit**

```bash
git add demo/live/token_overlay.py tests/test_token_overlay.py
git commit -m "feat(live): detect generation sections and map to token spans"
```

---

## Task 7: Qwen Generation Wrapper

**Files:**
- Create: `demo/live/qwen_generation.py`
- Test: `tests/test_qwen_generation.py`

- [ ] **Step 7.1: Write failing test**

Create `tests/test_qwen_generation.py`:

```python
from __future__ import annotations

import torch

from demo.live.qwen_generation import GenerationOutput, generate_with_qwen
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


def test_generate_with_qwen_returns_text_and_token_ids():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    output = generate_with_qwen(
        model=model,
        tokenizer=tokenizer,
        prompt="t10 t20",
        max_new_tokens=4,
    )

    assert isinstance(output, GenerationOutput)
    assert isinstance(output.text, str)
    assert len(output.text) > 0
    assert isinstance(output.token_ids, list)
    assert len(output.token_ids) > 0
    assert all(isinstance(tid, int) for tid in output.token_ids)


def test_generate_with_qwen_is_deterministic():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    torch.manual_seed(0)
    a = generate_with_qwen(model=model, tokenizer=tokenizer, prompt="t10 t20", max_new_tokens=4)
    torch.manual_seed(0)
    b = generate_with_qwen(model=model, tokenizer=tokenizer, prompt="t10 t20", max_new_tokens=4)

    assert a.token_ids == b.token_ids
    assert a.text == b.text
```

- [ ] **Step 7.2: Run, verify failure**

Run: `pytest tests/test_qwen_generation.py -v`
Expected: 2 FAIL with `ModuleNotFoundError`

- [ ] **Step 7.3: Implement `qwen_generation.py`**

Create `demo/live/qwen_generation.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    token_ids: list[int]


def generate_with_qwen(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> GenerationOutput:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    new_token_ids = generated[0, prompt_len:].tolist()
    text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    return GenerationOutput(text=text, token_ids=[int(t) for t in new_token_ids])
```

- [ ] **Step 7.4: Run, verify pass**

Run: `pytest tests/test_qwen_generation.py -v`
Expected: 2 PASS

- [ ] **Step 7.5: Commit**

```bash
git add demo/live/qwen_generation.py tests/test_qwen_generation.py
git commit -m "feat(live): add deterministic Qwen generation wrapper"
```

---

## Task 8: Smoke-mode Generation Stub

**Files:**
- Modify: `demo/live/qwen_generation.py`
- Test: `tests/test_qwen_generation.py`

We need a smoke generation path so the CI test (Task 12) and the smoke UI flow can exercise the full pipeline without loading Qwen.

- [ ] **Step 8.1: Append failing test**

Add to `tests/test_qwen_generation.py`:

```python
from demo.live.qwen_generation import generate_smoke_response


def test_generate_smoke_response_includes_think_and_answer():
    output = generate_smoke_response(prompt="What is the capital of France?")
    assert "<think>" in output.text and "</think>" in output.text
    assert "<answer>" in output.text and "</answer>" in output.text
    assert "Paris" in output.text
    assert output.token_ids == []
```

- [ ] **Step 8.2: Run, verify failure**

Run: `pytest tests/test_qwen_generation.py::test_generate_smoke_response_includes_think_and_answer -v`
Expected: FAIL with `ImportError`

- [ ] **Step 8.3: Implement `generate_smoke_response`**

Append to `demo/live/qwen_generation.py`:

```python
def generate_smoke_response(*, prompt: str) -> GenerationOutput:
    text = (
        "<think>\n"
        "The user is asking about the capital of France. "
        "From the context, Paris is mapped to France.\n"
        "</think>\n"
        "<answer>\nParis\n</answer>"
    )
    return GenerationOutput(text=text, token_ids=[])
```

- [ ] **Step 8.4: Run, verify pass**

Run: `pytest tests/test_qwen_generation.py -v`
Expected: 3 PASS

- [ ] **Step 8.5: Commit**

```bash
git add demo/live/qwen_generation.py tests/test_qwen_generation.py
git commit -m "feat(live): add smoke generation stub for deterministic e2e"
```

---

## Task 9: Wire Generate Button into `app.py`

**Files:**
- Modify: `demo/live/app.py`
- Test: `tests/test_live_demo.py`

- [ ] **Step 9.1: Append failing test**

Add to `tests/test_live_demo.py`:

```python
def test_generate_phase_smoke_returns_text_and_sections(tmp_path):
    module = load_live_app_module()

    text, sections, raw_rows, output_span_text, reasoning_span_text = module.run_generate_phase(
        model_name="demo/paris-smoke",
        prompt="What is the capital of France?",
        device_map="auto",
        dtype="auto",
        max_new_tokens=64,
    )

    assert "<answer>" in text and "Paris" in text
    assert sections["parser"] == "think_answer"
    assert ":" in output_span_text
    assert ":" in reasoning_span_text
    assert any("answer" in row[0] or "thinking" in row[0] for row in raw_rows)
```

- [ ] **Step 9.2: Run, verify failure**

Run: `pytest tests/test_live_demo.py::test_generate_phase_smoke_returns_text_and_sections -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'run_generate_phase'`

- [ ] **Step 9.3: Implement `run_generate_phase` in `app.py`**

Add these imports near the top of `demo/live/app.py` (after existing imports):

```python
from demo.live.qwen_generation import (
    GenerationOutput,
    generate_smoke_response,
    generate_with_qwen,
)
from demo.live.token_overlay import (
    GenerationSections,
    build_token_records,
    detect_sections,
    group_into_word_boxes,
)
```

Add this function above `build_demo`:

```python
def _format_span(span: tuple[int, int] | None) -> str:
    if span is None:
        return ""
    return f"{span[0]}:{span[1]}"


def _sections_to_dict(sections: GenerationSections) -> dict[str, object]:
    return {
        "generation_text": sections.generation_text,
        "thinking_char_span": sections.thinking_char_span,
        "answer_char_span": sections.answer_char_span,
        "thinking_token_span": sections.thinking_token_span,
        "answer_token_span": sections.answer_token_span,
        "parser": sections.parser,
    }


def _raw_token_rows(generation_text: str, tokenizer) -> list[list[object]]:
    records = build_token_records(
        text=generation_text,
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )
    return [
        [
            f"{r.section}#{r.token_index}",
            r.token_id,
            r.token_text,
            f"{r.char_start}:{r.char_end}",
            r.kind,
            "yes" if r.selectable else "no",
        ]
        for r in records
    ]


def _smoke_tokenizer():
    from tests.helpers import make_tiny_qwen2_model_and_tokenizer

    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    return tokenizer


def run_generate_phase(
    *,
    model_name: str,
    prompt: str,
    device_map: str,
    dtype: str,
    max_new_tokens: int,
    loader: Callable | None = None,
) -> tuple[str, dict[str, object], list[list[object]], str, str]:
    model_id = model_name.strip()
    prompt_text = prompt.strip()
    if not model_id:
        raise ValueError("Model is required.")
    if not prompt_text:
        raise ValueError("Prompt is required.")
    if len(prompt_text) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt must be at most {MAX_PROMPT_CHARS} characters.")

    if model_id == SMOKE_MODEL_ID:
        output = generate_smoke_response(prompt=prompt_text)
        tokenizer = _smoke_tokenizer()
    else:
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
    raw_rows = _raw_token_rows(output.text, tokenizer)
    return (
        output.text,
        _sections_to_dict(sections),
        raw_rows,
        _format_span(sections.answer_token_span),
        _format_span(sections.thinking_token_span),
    )
```

- [ ] **Step 9.4: Run, verify pass**

Run: `pytest tests/test_live_demo.py::test_generate_phase_smoke_returns_text_and_sections -v`
Expected: PASS

- [ ] **Step 9.5: Commit**

```bash
git add demo/live/app.py tests/test_live_demo.py
git commit -m "feat(live): add Generate phase that returns text, sections, raw tokens"
```

---

## Task 10: Inspector + Span UI in Gradio

**Files:**
- Modify: `demo/live/app.py`

This task wires the new generate phase into the Gradio UI. No new logic — just glue.

- [ ] **Step 10.1: Read current `build_demo` to confirm layout**

Run: `grep -n "build_demo" demo/live/app.py` and re-read lines 232-296.

- [ ] **Step 10.2: Replace `build_demo` body with the extended layout**

Edit `demo/live/app.py:232-286`. Replace the existing `build_demo` function with:

```python
def build_demo():
    import gradio as gr

    with gr.Blocks(title="FlashTrace Live Demo") as demo:
        gr.Markdown(
            "# FlashTrace Live Demo\n"
            "Generate with Qwen, inspect token spans, then trace selected answer."
        )

        with gr.Row():
            with gr.Column(scale=2):
                model_name = gr.Textbox(label="Model", value=DEFAULT_MODEL)
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=10)
                with gr.Row():
                    max_new_tokens = gr.Number(label="Max new tokens", value=128, precision=0)
                    generate_btn = gr.Button("Generate", variant="secondary")
            with gr.Column(scale=1):
                method = gr.Dropdown(
                    label="Method",
                    choices=["flashtrace", "ifr-span", "ifr-matrix"],
                    value="flashtrace",
                )
                output_span = gr.Textbox(label="Output span (auto-filled by Generate)", value="0:0")
                reasoning_span = gr.Textbox(label="Reasoning span", value="")
                hops = gr.Slider(label="Hops", minimum=1, maximum=4, step=1, value=1)
                top_k = gr.Slider(label="Top K", minimum=1, maximum=50, step=1, value=20)
                device_map = gr.Textbox(
                    label="Device map",
                    value=os.environ.get("FLASHTRACE_DEMO_DEVICE_MAP", "auto"),
                )
                dtype = gr.Dropdown(
                    label="Dtype",
                    choices=["auto", "float16", "bfloat16", "float32"],
                    value="auto",
                )
                chunk_tokens = gr.Number(label="Chunk tokens", value=128, precision=0)
                sink_chunk_tokens = gr.Number(label="Sink chunk tokens", value=32, precision=0)
                use_chat_template = gr.Checkbox(label="Use chat template", value=False)
                submit = gr.Button("Trace selected answer", variant="primary")

        target = gr.Textbox(label="Generated response (used as trace target)", value=DEFAULT_TARGET, lines=6)
        sections_state = gr.JSON(label="Detected sections")
        raw_tokens = gr.Dataframe(
            headers=["section#idx", "token_id", "token_text", "char_span", "kind", "selectable"],
            label="Raw tokens",
        )

        top_table = gr.Dataframe(headers=["Index", "Token", "Score"], label="Top input tokens")
        generation_tokens = gr.Textbox(label="Generation tokens", lines=8)
        heatmap = gr.HTML(label="Trace heatmap")
        json_file = gr.File(label="JSON trace")

        generate_btn.click(
            fn=lambda model_name, prompt, device_map, dtype, max_new_tokens: run_generate_phase(
                model_name=model_name,
                prompt=prompt,
                device_map=device_map,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
            ),
            inputs=[model_name, prompt, device_map, dtype, max_new_tokens],
            outputs=[target, sections_state, raw_tokens, output_span, reasoning_span],
        )

        submit.click(
            fn=run_trace_from_ui,
            inputs=[
                model_name,
                prompt,
                target,
                output_span,
                reasoning_span,
                method,
                hops,
                top_k,
                device_map,
                dtype,
                chunk_tokens,
                sink_chunk_tokens,
                use_chat_template,
            ],
            outputs=[top_table, generation_tokens, heatmap, json_file],
        )
    return demo
```

- [ ] **Step 10.3: Run dry-run script to verify UI builds**

Run: `FLASHTRACE_DEMO_DRY_RUN=1 python demo/live/app.py`
Expected: prints `FlashTrace live demo dry run OK` and exits 0.

- [ ] **Step 10.4: Run full live-demo test suite**

Run: `pytest tests/test_live_demo.py -v`
Expected: all existing tests PASS plus the new `test_generate_phase_smoke_returns_text_and_sections`.

- [ ] **Step 10.5: Commit**

```bash
git add demo/live/app.py
git commit -m "feat(live): add Generate button, sections inspector, raw tokens table"
```

---

## Task 11: End-to-End Smoke Test (Generate → Detect → Trace)

**Files:**
- Test: `tests/test_live_demo.py`

- [ ] **Step 11.1: Write failing e2e test**

Add to `tests/test_live_demo.py`:

```python
def test_full_pipeline_smoke_traces_paris(tmp_path):
    module = load_live_app_module()

    text, sections, _raw, output_span_text, reasoning_span_text = module.run_generate_phase(
        model_name="demo/paris-smoke",
        prompt="Context:\nParis is the capital of France.\nQuestion: What is the capital of France?",
        device_map="auto",
        dtype="auto",
        max_new_tokens=64,
    )

    assert sections["parser"] == "think_answer"
    assert "Paris" in text

    top_rows, _generation_tokens, _html, json_path = module.run_trace(
        model_name="demo/paris-smoke",
        prompt="Context:\nParis is the capital of France.\nQuestion: What is the capital of France?",
        target=text,
        output_span=output_span_text,
        reasoning_span=reasoning_span_text,
        method="flashtrace",
        hops=1,
        top_k=3,
        device_map="auto",
        dtype="auto",
        chunk_tokens=16,
        sink_chunk_tokens=4,
        use_chat_template=False,
        work_dir=tmp_path,
    )

    assert top_rows[0][1] == "Paris"
    assert top_rows[0][2] == 1.0
    assert Path(json_path).exists()
```

- [ ] **Step 11.2: Run, verify pass (smoke path is wired)**

Run: `pytest tests/test_live_demo.py::test_full_pipeline_smoke_traces_paris -v`
Expected: PASS (smoke trace already returns Paris=1.0; this exercises the new Generate phase end-to-end).

If this fails because the smoke trace code path doesn't accept the multi-line `target` from `generate_smoke_response`, fix `_run_paris_smoke_trace` in `demo/live/app.py:84-120` to ignore `target` content for ranking (it already does — just pass `target` through).

- [ ] **Step 11.3: Run the full test suite**

Run: `pytest -v`
Expected: all tests PASS.

- [ ] **Step 11.4: Commit**

```bash
git add tests/test_live_demo.py
git commit -m "test(live): cover generate→detect→trace pipeline against smoke model"
```

---

## Task 12: Update README

**Files:**
- Modify: `demo/live/README.md`

- [ ] **Step 12.1: Read current README**

Run: `cat demo/live/README.md`

- [ ] **Step 12.2: Replace with updated docs**

Use Edit/Write to replace `demo/live/README.md` content with documentation that covers:
- Existing setup (pip install, run command).
- New three-phase flow: Generate → Inspect → Trace.
- Smoke model behavior (uses `generate_smoke_response`, returns canned `<think>/<answer>`).
- How to switch to a real Qwen model (`FLASHTRACE_DEMO_MODEL=Qwen/Qwen2.5-1.5B-Instruct`).
- Span format (`START:END`, generation-token indices, inclusive).
- Parser fallback order: think/answer → boxed → last-paragraph.

Concrete content:

```markdown
# FlashTrace Live Demo

Gradio demo that traces a generated answer back to the prompt tokens that shaped it.

## Run

```bash
pip install -e ".[demo]"
python demo/live/app.py
```

Open http://127.0.0.1:7860/.

## Three-phase flow

1. **Generate** — Model produces reasoning + answer (deterministic, `do_sample=False`). Smoke model (`demo/paris-smoke`) returns a canned `<think>/<answer>` response without loading any weights.
2. **Inspect** — The generated text is parsed (`<think>/<answer>` → `\boxed{}` → last-paragraph) and the resulting answer/reasoning token spans are auto-filled into the span fields. The raw-token table shows every tokenizer token with its kind (content/whitespace/special/template/control).
3. **Trace** — Click *Trace selected answer*. The selected `output_span` and `reasoning_span` are passed to `FlashTrace.trace(...)` which produces the prompt-side attribution heatmap and JSON export.

## Span format

Spans are inclusive generation-token index pairs in `START:END` format. The Generate phase fills these in based on the parser; you can override them manually before tracing.

## Switch to a real Qwen model

```bash
FLASHTRACE_DEMO_MODEL=Qwen/Qwen2.5-1.5B-Instruct python demo/live/app.py
```

First run downloads the model. Subsequent runs use the demo model cache.

## Environment variables

- `FLASHTRACE_DEMO_MODEL` — Model id (default `demo/paris-smoke`).
- `FLASHTRACE_DEMO_OUTPUT_DIR` — JSON trace output dir (default `demo/live/out`).
- `FLASHTRACE_DEMO_DEVICE_MAP` — `device_map` for `from_pretrained` (default `auto`).
- `FLASHTRACE_DEMO_MAX_PROMPT_CHARS` — Prompt-length cap (default 4000).
```

- [ ] **Step 12.3: Commit**

```bash
git add demo/live/README.md
git commit -m "docs(live): document Generate-Inspect-Trace flow"
```

---

## Self-Review Checklist

- [x] **Spec coverage:**
  - Generate with Qwen + deterministic settings → Task 7
  - Reasoning + final-answer markers → Task 8 (smoke) + Qwen prompt is the user's prompt (no injected system prompt in v1; can be added later)
  - Real tokenizer offsets → Task 3 (`build_token_records` uses `return_offsets_mapping=True`)
  - Visible word boxes → Task 4 (`group_into_word_boxes`); UI v1 surfaces them as Dataframe rows + auto-filled spans (custom-JS click selection deferred per evaluation)
  - Special tokens preserved → Task 1 (kinds), Task 4 (own boxes)
  - Reasoning + output spans into `FlashTrace.trace` → Task 9 (`run_generate_phase` returns spans, existing `run_trace` consumes them)
  - Paris smoke ranks Paris highest after tracing → Task 11
- [x] **No placeholders:** all steps contain runnable code or exact commands.
- [x] **Type consistency:** `GenerationSections` fields used identically in Tasks 6, 9, 11; `ParseResult` only used in Task 5; `WordBox` only used in Task 4 (UI uses Dataframe rows, not WordBox directly — acceptable for v1).
- [x] **Deferred from spec (documented):**
  - Click-to-select word boxes (UI uses textbox spans + Dataframe; per evaluation discussion).
  - Cached-token-id tracing (spec marks as v2).
  - Chat-template special handling in prompt overlay (spec marks as later refinement).

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-03-qwen-token-overlay.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
