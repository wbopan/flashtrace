from evaluations.agentdojo.diagnose_coverage import summarize


def _row(stage, *, realized, forced=None):
    turn = {}
    if forced is not None:
        turn["output_intervention"] = {
            "mode": "teacher_forcing_prefix",
            "forced_token_count": forced,
            "forced_token_fraction": forced / 10,
            "reasoning_token_count": 12,
            "bridge_is_fully_model_generated": True,
        }
    return {
        "attack_success": realized,
        "generated_target_action": realized,
        "target_calls": [{"function": "f", "args": {}}],
        "realization_stage": stage,
        "turns": [turn],
        "selection": {"stages": [{"name": stage}]},
    }


def test_summary_is_stage_stratified_and_text_free():
    result = summarize(
        [
            _row("natural_greedy", realized=True),
            _row("teacher_forcing_2", realized=True, forced=2),
            _row("teacher_forcing_2", realized=False, forced=2),
        ]
    )

    assert result["selected_stage_counts"] == {
        "natural_greedy": 1,
        "teacher_forcing_2": 2,
    }
    forced = result["stages"]["teacher_forcing_2"]
    assert forced["realized_rate_over_action_targets"] == 0.5
    assert forced["mean_reference_token_count"] == 2
    assert forced["short_bridge_count"] == 0
    assert not result["definition"]["contains_sample_text"]
