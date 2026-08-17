import pytest

from evaluations.agentdojo.compare_search_pilots import compare


def _row(case_id, *, realized=False, all_model=True):
    return {
        "case": {"case_id": case_id},
        "attack_success": realized,
        "generated_target_action": realized,
        "target_calls": [{"function": "f", "args": {}}],
        "turns": [
            {
                "generation_audit": {
                    "all_completion_tokens_model_generated": all_model,
                    "model_generated_token_count": 9 if all_model else 8,
                    "reference_token_count": 0 if all_model else 1,
                    "reasoning_token_count": 6,
                    "bridge_is_fully_model_generated": True,
                }
            }
        ],
        "selection": {
            "selected_attempt_index": 0,
            "attempts": [
                {
                    "generation_request_count": 1,
                    "sampled_token_count": 9,
                    "generation_seconds": 0.1,
                }
            ],
        },
    }


def test_compare_requires_and_summarizes_identical_manifest():
    result = compare(
        {
            "adaptive_best_of_n": [_row("a", realized=True)],
            "reasoning_rerank": [_row("a", realized=True)],
        }
    )
    assert result["same_case_manifest"]
    assert result["strategies"]["reasoning_rerank"]["model_generated_token_fraction"] == 1
    assert not result["contains_sample_text"]

    with pytest.raises(ValueError, match="same case ids"):
        compare(
            {
                "adaptive_best_of_n": [_row("a")],
                "reasoning_rerank": [_row("b")],
            }
        )
