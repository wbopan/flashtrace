from __future__ import annotations

from pathlib import Path

from evaluations.agentdojo.explain import _region_char_spans
from evaluations.agentdojo.pilot import PilotCase, PreparedCase


def test_region_parser_ignores_literal_xml_tag_mentions():
    prompt = (
        "Use signatures within <tools></tools> XML tags.\n"
        '<tools>\n{"name":"real_tool"}\n</tools>\n'
        "Put calls within <tool_call></tool_call> XML tags.\n"
        '<tool_call>\n{"name":"format_example"}\n</tool_call>\n'
        '<tool_call>\n{"name":"real_call"}\n</tool_call>\n'
        "<INFORMATION>attack</INFORMATION>"
    )
    injection_start = prompt.index("<INFORMATION>")
    injection_end = len(prompt)
    case = PreparedCase(
        spec=PilotCase(
            case_id="test",
            suite="slack",
            user_task="user_task_0",
            injection_task="injection_task_0",
            target_message_index=1,
            target_tool="unused",
            sha256="unused",
        ),
        source_path=Path("unused"),
        raw={
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": "real_call", "args": {}}],
                }
            ]
        },
        prompt=prompt,
        target="unused",
        injection_char_span=(injection_start, injection_end),
        injection_token_indices=(0,),
        replay_utility=True,
        replay_attack_success=True,
        replay_errors=(),
    )

    spans = _region_char_spans(case)
    tool_spans = [span for span in spans if span["label"] == "tool_schema"]
    call_spans = [span for span in spans if span["label"] == "assistant_tool_call"]
    format_spans = [
        span for span in spans if span["label"] == "tool_call_format_example"
    ]

    assert len(tool_spans) == 1
    assert prompt[tool_spans[0]["char_start"] : tool_spans[0]["char_end"]].startswith(
        "<tools>\n"
    )
    assert len(call_spans) == 1
    assert prompt[call_spans[0]["char_start"] : call_spans[0]["char_end"]].startswith(
        "<tool_call>\n"
    )
    assert len(format_spans) == 1
    assert (
        "format_example"
        in prompt[format_spans[0]["char_start"] : format_spans[0]["char_end"]]
    )
