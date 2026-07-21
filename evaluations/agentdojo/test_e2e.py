from __future__ import annotations

from evaluations.agentdojo.e2e import (
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
