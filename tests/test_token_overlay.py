from __future__ import annotations

from demo.live.token_overlay import classify_token_kind


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
