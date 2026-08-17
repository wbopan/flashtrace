from __future__ import annotations

import pytest

from evaluations.agentdojo.e2e import (
    _align_completion_token_provenance,
    _call_matches_expected,
    parse_qwen_tool_calls,
)


def test_parse_qwen_tool_calls_preserves_spans_and_arguments():
    completion = (
        "reasoning</think>\n"
        '<tool_call>\n{"name":"get_webpage","arguments":{"url":"https://x.test"}}\n'
        "</tool_call>"
    )

    calls, errors = parse_qwen_tool_calls(completion)

    assert errors == []
    assert len(calls) == 1
    assert calls[0].function == "get_webpage"
    assert calls[0].args == {"url": "https://x.test"}
    assert completion[calls[0].char_start : calls[0].char_end].startswith("<tool_call>")


def test_parse_qwen36_function_tag_tool_call():
    completion = (
        "reasoning</think>\n"
        "<tool_call>\n"
        "<function=get_webpage>\n"
        "<parameter=url>\nwww.informations.com\n</parameter>\n"
        "<parameter=limit>\n3\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    calls, errors = parse_qwen_tool_calls(completion)

    assert errors == []
    assert len(calls) == 1
    assert calls[0].function == "get_webpage"
    assert calls[0].args == {"url": "www.informations.com", "limit": 3}
    assert completion[calls[0].char_start : calls[0].char_end].endswith(
        "</tool_call>"
    )


def test_target_matching_normalizes_url_protocol_only():
    actual = {
        "function": "get_webpage",
        "args": {"url": "www.true-informations.com/"},
    }
    expected = {
        "function": "get_webpage",
        "args": {"url": "http://www.true-informations.com"},
    }

    assert _call_matches_expected(actual, expected)
    assert not _call_matches_expected(
        {"function": "get_webpage", "args": {"url": "www.informations.com"}},
        expected,
    )


def test_model_only_provenance_aligns_across_tokenizers():
    assert _align_completion_token_provenance(["model"] * 3, 5) == [
        "model"
    ] * 5


def test_mixed_provenance_requires_exact_alignment():
    with pytest.raises(ValueError, match="Mixed completion token provenance"):
        _align_completion_token_provenance(["model", "reference"], 3)
