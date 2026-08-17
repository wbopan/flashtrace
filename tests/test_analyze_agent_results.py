from evaluations.analyze_agent_results import (
    AIDER_METHODS,
    _aider_attribution,
    _mean_ci,
    _paired_difference,
    _repair_feedback_fraction,
    _render_tex,
)


def test_mean_ci_is_deterministic_and_contains_the_mean():
    estimate = _mean_ci([1.0, 2.0, 3.0], draws=1_000, seed=7)
    repeated = _mean_ci([1.0, 2.0, 3.0], draws=1_000, seed=7)

    assert estimate == repeated
    assert estimate["mean"] == 2.0
    assert estimate["ci95_low"] <= estimate["mean"] <= estimate["ci95_high"]


def test_paired_difference_uses_favorable_direction():
    lower = _paired_difference(
        [1.0, 2.0],
        [2.0, 4.0],
        lower_is_better=True,
        draws=1_000,
        seed=3,
    )
    higher = _paired_difference(
        [2.0, 4.0],
        [1.0, 2.0],
        lower_is_better=False,
        draws=1_000,
        seed=3,
    )

    assert lower["mean"] == higher["mean"] == 1.5
    assert lower["wins"] == higher["wins"] == 2
    assert lower["losses"] == higher["losses"] == 0


def test_aider_attribution_summarizes_repair_feedback_fraction():
    rows = []
    for method in AIDER_METHODS:
        rows.append(
            {
                "example_id": "example",
                "method": method,
                "status": "success",
                "final_passed": True,
                "repair_attempted": True,
                "result": {
                    "rise": 0.1,
                    "mas": 0.2,
                    "attribution_seconds": 0.3,
                    "source_segments": [
                        {"role": "user", "turn": 1, "fraction": 0.25},
                        {"role": "assistant", "turn": 1, "fraction": 0.35},
                        {"role": "user", "turn": 2, "fraction": 0.40},
                    ],
                },
            }
        )

    analysis = _aider_attribution(rows, draws=1_000)
    flashtrace = analysis["subsets"]["all_valid"]["methods"][
        "flashtrace_k1_full_action"
    ]

    assert _repair_feedback_fraction(rows[0]) == 0.4
    assert _repair_feedback_fraction(
        {
            "result": {
                "trace": {
                    "segments": [
                        {"role": "user", "turn": 2, "fraction": 0.4}
                    ]
                }
            }
        }
    ) == 0.4
    assert flashtrace["repair_feedback_count"] == 1
    assert flashtrace["repair_feedback_fraction"]["mean"] == 0.4


def test_render_tex_highlights_flashtrace_rows():
    estimate = {"mean": 0.1, "ci95_low": 0.0, "ci95_high": 0.2}
    method_metrics = {
        "rise": estimate,
        "mas": estimate,
        "attribution_seconds": estimate,
    }
    localization_metrics = {
        "recovery_at_5pct": estimate,
        "recovery_at_10pct": estimate,
        "recovery_at_20pct": estimate,
        "pointing_game": estimate,
        "injection_mass_fraction": estimate,
        "runtime_seconds": estimate,
    }
    analysis = {
        "aider": {
            "generation_audit": {
                "attempted": 1,
                "valid": 1,
                "format_compliance_rate": 1.0,
                "draft_pass_rate": 1.0,
                "revised_pass_rate_among_repairs": None,
                "repair_improvement_rate": None,
                "regressed": 0,
            },
            "attribution": {
                "subsets": {
                    "all_valid": {
                        "count": 1,
                        "methods": {
                            method: method_metrics
                            for method in (
                                "ifr_last_output_line",
                                "ifr_exhaustive_full_action",
                                "attnlrp_full_action",
                                "flashtrace_k1_full_action",
                            )
                        },
                    },
                    "execution_success": {
                        "count": 1,
                        "methods": {
                            method: method_metrics
                            for method in (
                                "ifr_last_output_line",
                                "ifr_exhaustive_full_action",
                                "attnlrp_full_action",
                                "flashtrace_k1_full_action",
                            )
                        },
                    },
                }
            },
        },
        "agentdojo": {
            "outcomes": {
                "suites": {
                    suite: {
                        "clean_count": 1,
                        "attacked_count": 1,
                        "clean_utility_rate": 1.0,
                        "attacked_utility_rate": 1.0,
                        "attack_success_rate": 0.0,
                        "joint_utility_and_attack_rate": 0.0,
                        "generated_target_coverage": 0.0,
                        "localizable_cases": 0,
                    }
                    for suite in ("banking", "slack", "travel", "workspace")
                },
                "overall": {
                    "clean_count": 4,
                    "attacked_count": 4,
                    "clean_utility_rate": 1.0,
                    "attacked_utility_rate": 1.0,
                    "attack_success_rate": 0.0,
                    "joint_utility_and_attack_rate": 0.0,
                    "generated_target_coverage": 0.0,
                    "localizable_cases": 0,
                },
            },
            "localization": {
                "methods": {
                    method: localization_metrics
                    for method in (
                        "random_expected",
                        "ifr_last_generated_action_token",
                        "ifr_full_generated_action",
                        "flashtrace_generated_action_via_reasoning",
                    )
                }
            },
        },
    }

    rendered = _render_tex(analysis, generator_label="Qwen3.6-27B")

    assert (
        rendered["aider_multiturn_rows.tex"].count(r"\rowcolor{cyan!10}") == 1
    )
    assert (
        rendered["agentdojo_localization_rows.tex"].count(
            r"\rowcolor{cyan!10}"
        )
        == 1
    )
