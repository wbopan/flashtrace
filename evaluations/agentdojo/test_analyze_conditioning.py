from evaluations.agentdojo.analyze_conditioning import summarize


def _row(*, suite="slack", action=True, observed=True, localizable=False):
    target_event = (
        {
            "call": {},
            "rendered_prompt": "<INFORMATION>x</INFORMATION>",
        }
        if localizable
        else None
    )
    return {
        "case": {"suite": suite},
        "attack_success": localizable,
        "generated_target_action": localizable,
        "target_event": target_event,
        "target_calls": [{"function": "f", "args": {}}] if action else [],
        "turns": [
            {
                "rendered_prompt": (
                    "<INFORMATION>x</INFORMATION>" if observed else "plain"
                )
            }
        ],
    }


def test_summary_separates_natural_and_conditionable_denominators():
    result = summarize(
        [
            _row(localizable=True),
            _row(localizable=False),
            _row(action=False, observed=True),
            _row(action=True, observed=False),
        ]
    )["overall"]

    assert result["attacked_count"] == 4
    assert result["action_target_count"] == 3
    assert result["natural_localizable_count"] == 1
    assert result["conditionable_action_count"] == 2
    assert result["natural_localizable_over_action_targets"] == 1 / 3
    assert result["conditionable_over_action_targets"] == 2 / 3
