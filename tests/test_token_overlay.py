from __future__ import annotations

import pytest

from demo.live.token_overlay import classify_token_kind
from demo.live.token_overlay import char_span_to_token_span


def test_classify_whitespace_only():
    assert classify_token_kind(token_text=" ", token_id=10, special_ids=set()) == "whitespace"
    assert classify_token_kind(token_text="\n\t", token_id=11, special_ids=set()) == "whitespace"


def test_classify_special_id_marks_special():
    assert classify_token_kind(token_text="<eos>", token_id=2, special_ids={2}) == "special"
    assert classify_token_kind(token_text="abc", token_id=None, special_ids={2}) == "content"


def test_classify_chat_template_marker_is_template():
    assert classify_token_kind(token_text="<|im_start|>", token_id=151644, special_ids={151644}) == "template"
    assert classify_token_kind(token_text="<|im_end|>", token_id=151645, special_ids={151645}) == "template"


def test_classify_think_marker_is_control():
    assert classify_token_kind(token_text="<think>", token_id=200, special_ids={200}) == "control"
    assert classify_token_kind(token_text="</answer>", token_id=201, special_ids={201}) == "control"
    assert classify_token_kind(token_text="<reason>", token_id=202, special_ids={202}) == "control"
    assert classify_token_kind(token_text="</reason>", token_id=203, special_ids={203}) == "control"


def test_classify_regular_word_is_content():
    assert classify_token_kind(token_text="Paris", token_id=42, special_ids=set()) == "content"
    assert classify_token_kind(token_text="!", token_id=43, special_ids=set()) == "content"


def test_classify_empty_string_is_content():
    assert classify_token_kind(token_text="", token_id=0, special_ids=set()) == "content"


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


from tests.helpers import make_tiny_qwen2_model_and_tokenizer

from demo.live.token_overlay import TokenRecord, build_token_records, build_token_records_from_ids


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
    assert [r.display_text for r in records] == ["t10", "t20", "t30"]
    assert all(r.display_text == "t10 t20 t30"[r.char_start:r.char_end] for r in records)
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
    assert all(r.token_text == "t1" for r in eos_records)


def test_build_token_records_from_ids_uses_real_ids_and_decoded_offsets():
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    token_ids = [10, 20, tokenizer.eos_token_id]

    records, decoded = build_token_records_from_ids(
        token_ids=token_ids,
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )

    assert decoded == "t10 t20 t1"
    assert [r.token_id for r in records] == token_ids
    assert [r.token_text for r in records] == ["t10", "t20", "t1"]
    assert [(r.char_start, r.char_end) for r in records] == [(0, 3), (4, 7), (8, 10)]
    assert [r.display_text for r in records] == ["t10", "t20", "t1"]
    assert all(r.display_text == decoded[r.char_start:r.char_end] for r in records)
    assert [r.token_index for r in records] == [0, 1, 2]
    assert records[-1].kind == "special"
    assert records[-1].selectable is False


from demo.live.token_overlay import WordBox, group_into_word_boxes, TokenKind, Section


def _record(
    *,
    index: int,
    text: str,
    char_start: int,
    char_end: int,
    kind: TokenKind = "content",
    selectable: bool = True,
    section: Section = "answer",
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


def test_group_flushes_on_section_change():
    records = [
        _record(index=0, text="ab", char_start=0, char_end=2, section="thinking"),
        _record(index=1, text="cd", char_start=2, char_end=4, section="answer"),
    ]

    boxes = group_into_word_boxes(records, source_text="abcd")

    assert len(boxes) == 2
    assert [b.section for b in boxes] == ["thinking", "answer"]
    assert boxes[0].text == "ab"
    assert boxes[1].text == "cd"


def test_group_raises_when_source_text_too_short():
    records = [
        _record(index=0, text="hello", char_start=0, char_end=5),
    ]

    with pytest.raises(ValueError, match="cannot satisfy"):
        group_into_word_boxes(records, source_text="hi")


from demo.live.token_overlay import GenerationSections, detect_sections


def test_detect_sections_no_markers_falls_back_to_last_paragraph():
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


def test_detect_sections_maps_reason_answer_to_token_indices():
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    text = "</think> <reason> t10 t20 </reason> <answer> t30 t40 </answer>"

    sections = detect_sections(text=text, tokenizer=tokenizer)

    assert sections.parser == "reason_answer"
    assert sections.thinking_token_span is not None
    t_start, t_end = sections.thinking_token_span
    a_start, a_end = sections.answer_token_span
    assert t_start <= t_end < a_start <= a_end


def test_detect_sections_thinking_token_span_falls_back_to_none_when_no_overlap():
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    # The thinking section in this constructed text contains only whitespace,
    # so the parser produces a non-empty char span but no tokens overlap it.
    # Verify thinking_token_span gracefully becomes None instead of raising.

    class StubParserChain:
        def __init__(self, real_text: str):
            self.real_text = real_text

        def __call__(self, text: str):
            from demo.live.span_parsers import ParseResult

            # Force a thinking span over a whitespace-only region (no tokens overlap).
            return ParseResult(
                thinking_char_span=(3, 4),  # the whitespace between "t10" and "t20"
                answer_char_span=(0, 7),
                parser="forced",
            )

    import demo.live.token_overlay as overlay_mod
    original_chain = overlay_mod.run_parser_chain
    overlay_mod.run_parser_chain = StubParserChain("t10 t20")
    try:
        sections = detect_sections(text="t10 t20", tokenizer=tokenizer)
    finally:
        overlay_mod.run_parser_chain = original_chain

    assert sections.parser == "forced"
    assert sections.thinking_token_span is None
    assert sections.answer_token_span == (0, 1)


def test_detect_sections_routes_boxed_input_through_boxed_parser():
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    text = "t10 t20. \\boxed{t30}"

    sections = detect_sections(text=text, tokenizer=tokenizer)

    assert sections.parser == "boxed"
    a_start, a_end = sections.answer_token_span
    # The answer is "t30" — should map to a single token in the tokenized output.
    assert 0 <= a_start <= a_end
