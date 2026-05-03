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
    assert all(r.token_text == "t1" for r in eos_records)
