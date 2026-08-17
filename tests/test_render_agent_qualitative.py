import pytest

from evaluations.render_agent_qualitative import select_examples


def _generation(function_trace):
    return {
        "case": {"case_id": "slack__case", "suite": "slack"},
        "turns": [
            {
                "generated_token_count_with_eos_if_emitted": 12,
            }
        ],
        "function_trace": function_trace,
        "utility": False,
        "attack_success": True,
        "target_functions": ["send_email"],
    }


def _attribution():
    return {
        "case_id": "slack__case",
        "status": "success",
        "attribution": {
            "methods": {
                "flashtrace_generated_action_via_reasoning": {
                    "localization_metrics": {
                        "recovery_at_10pct": 0.5,
                        "injection_mass_fraction": 0.25,
                    }
                }
            }
        },
    }


def test_select_examples_uses_publishable_structural_metadata():
    selected = select_examples(
        [_generation([{"function": "get_webpage"}, {"function": "send_email"}])],
        [_attribution()],
    )

    assert selected == [
        {
            "case_id": "slack__case",
            "suite": "slack",
            "utility": False,
            "attack_success": True,
            "turn_count": 1,
            "generated_token_count": 12,
            "function_trace": ["get_webpage", "send_email"],
            "target_functions": ["send_email"],
            "flashtrace_recovery_at_10pct": 0.5,
            "flashtrace_injection_mass_fraction": 0.25,
        }
    ]


def test_select_examples_rejects_tool_arguments():
    with pytest.raises(ValueError, match="forbidden keys"):
        select_examples(
            [_generation([{"function": "send_email", "args": {"body": "x"}}])],
            [_attribution()],
        )
