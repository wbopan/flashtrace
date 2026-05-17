from __future__ import annotations

import pytest

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


def test_reason_answer_extracts_reasoning_span():
    text = "</think>\n\n<reason>Paris is the capital of France.</reason>\n<answer>Paris</answer>"

    result = run_parser_chain(text)

    assert result.thinking_char_span is not None
    assert text[result.thinking_char_span[0] : result.thinking_char_span[1]] == "Paris is the capital of France."
    assert text[result.answer_char_span[0] : result.answer_char_span[1]] == "Paris"
    assert result.parser == "reason_answer"


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


def test_last_paragraph_handles_trailing_blank_lines():
    text = "Step 1.\n\nFinal answer is Paris.\n\n"

    result = parse_last_paragraph(text)

    assert result is not None
    assert text[result.answer_char_span[0] : result.answer_char_span[1]] == "Final answer is Paris."
    assert result.thinking_char_span is not None
    assert text[result.thinking_char_span[0] : result.thinking_char_span[1]].endswith("Step 1.")


def test_boxed_answer_handles_period_immediately_before_marker():
    # No space between period and \boxed — _SENT_END_RE's `$` lookahead must work
    text = "Reasoning.\\boxed{42}"

    result = parse_boxed_answer(text)

    assert result is not None
    assert text[result.answer_char_span[0] : result.answer_char_span[1]] == "42"
    assert result.thinking_char_span is not None
    assert text[result.thinking_char_span[0] : result.thinking_char_span[1]] == "Reasoning."


def test_run_parser_chain_raises_on_empty_input():
    with pytest.raises(ValueError, match="No parser produced"):
        run_parser_chain("")

    with pytest.raises(ValueError, match="No parser produced"):
        run_parser_chain("   \n\n  ")


def test_boxed_at_start_has_no_thinking_span():
    text = "\\boxed{42}"

    result = parse_boxed_answer(text)

    assert result is not None
    assert result.thinking_char_span is None
    assert text[result.answer_char_span[0] : result.answer_char_span[1]] == "42"
