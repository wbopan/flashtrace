import pytest

from evaluations.agentdojo.compare_realization_pilots import compare


def _report(method, *, cases=4, confirmed=2, fraction=0.25):
    return {
        "overall": {
            "case_count": cases,
            "mean_reference_token_count": 3.0,
            "mean_reference_token_fraction": fraction,
            "minimum_reasoning_token_count": 20,
            "short_bridge_count": 0,
            "non_model_bridge_count": 0,
        },
        "selected_stage_counts": {
            "natural_greedy": 1,
            f"{method}_4": confirmed,
        },
        "attempted_stages": {f"{method}_4": {"case_count": 3}},
    }


def test_compare_keeps_methods_separate_and_text_free():
    result = compare(
        _report("prefix_forcing", confirmed=1, fraction=0.5),
        _report("sparse_correction", confirmed=2, fraction=0.1),
    )

    assert result["case_count"] == 4
    assert result["natural_sampling"]["consistent"]
    assert result["prefix_forcing"]["confirmed_case_count"] == 1
    assert result["sparse_correction"]["confirmed_case_count"] == 2
    assert not result["contains_sample_text"]


def test_compare_rejects_different_case_counts():
    with pytest.raises(ValueError, match="different case counts"):
        compare(_report("prefix_forcing", cases=3), _report("sparse_correction"))
