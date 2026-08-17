import json

from evaluations.agentdojo.attribute_full import (
    _eligible,
    _generation_strategy,
    _redacted,
    _summary,
    _synchronize_redacted,
)


def test_eligibility_requires_validator_and_exact_generated_target():
    base = {
        "attack_success": True,
        "generated_target_action": True,
        "target_event": {
            "call": {},
            "rendered_prompt": "<INFORMATION>injection</INFORMATION>",
        },
    }
    assert _eligible(base)
    assert not _eligible({**base, "attack_success": False})
    assert not _eligible({**base, "generated_target_action": False})
    assert not _eligible({**base, "target_event": None})
    assert not _eligible(
        {
            **base,
            "target_event": {
                "call": {},
                "rendered_prompt": "injection is absent",
            },
        }
    )


def test_model_only_eligibility_rejects_reference_writes_and_short_bridges():
    base = {
        "attack_success": True,
        "generated_target_action": True,
        "target_event": {
            "call": {},
            "rendered_prompt": "<INFORMATION>injection</INFORMATION>",
            "turn_index": 2,
            "teacher_forced": False,
            "output_intervention": None,
            "completion_token_provenance": ["model", "model"],
        },
        "turns": [
            {
                "turn_index": 2,
                "generation_audit": {
                    "all_completion_tokens_model_generated": True,
                    "bridge_is_fully_model_generated": True,
                    "reference_token_count": 0,
                    "reasoning_token_count": 12,
                },
            }
        ],
        "selection": {
            "selected_attempt_index": 0,
            "attempts": [{"all_completion_tokens_model_generated": True}],
        },
    }

    assert _eligible(base, require_model_only=True)
    forced = json.loads(json.dumps(base))
    forced["turns"][0]["generation_audit"]["reference_token_count"] = 1
    assert not _eligible(forced, require_model_only=True)
    short = json.loads(json.dumps(base))
    short["turns"][0]["generation_audit"]["reasoning_token_count"] = 2
    assert not _eligible(short, require_model_only=True)


def test_generation_strategy_distinguishes_reasoning_rerank():
    assert _generation_strategy(
        {"selection": {"reasoning_search": {"candidates": 4}}}
    ) == "reasoning_rerank"
    assert _generation_strategy({"selection": {"attempts": []}}) == (
        "adaptive_best_of_n"
    )


def test_redacted_attribution_omits_sensitive_text():
    row = {
        "case_id": "case",
        "suite": "slack",
        "realization_stage": "teacher_forcing_2",
        "status": "success",
        "error": None,
        "attribution": {
            "provenance": "conditioned_continuation",
            "teacher_forced": True,
            "prompt": "SECRET",
            "prompt_token_count": 10,
            "gold_injection": {"text": "SECRET", "token_indices": [2, 3]},
            "generated_target": {
                "text": "SECRET",
                "token_count": 4,
                "target_call_token_span_inclusive": [2, 3],
                "thinking_bridge_token_span_inclusive": [0, 1],
            },
            "methods": {
                "random_expected": {
                    "runtime_seconds": 0.0,
                    "localization_metrics": {
                        "recovery_at_5pct": 0.1,
                    },
                }
            },
        },
    }

    payload = _redacted(row)

    assert "SECRET" not in json.dumps(payload)
    assert payload["attribution"]["gold_injection_token_count"] == 2
    assert payload["attribution"]["target_provenance"] == (
        "conditioned_continuation"
    )
    assert payload["attribution"]["target_teacher_forced"]
    assert payload["realization_stage"] == "teacher_forcing_2"


def test_redacted_resume_synchronizes_missing_raw_case(tmp_path):
    raw_path = tmp_path / "attribution.jsonl"
    redacted_path = tmp_path / "attribution.redacted.jsonl"
    rows = [
        {
            "case_id": "case-1",
            "suite": "banking",
            "status": "error",
            "error": "RuntimeError: failed",
            "attribution": None,
        },
        {
            "case_id": "case-2",
            "suite": "slack",
            "status": "error",
            "error": "ValueError: failed",
            "attribution": None,
        },
    ]
    raw_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    redacted_path.write_text(
        json.dumps(_redacted(rows[0])) + "\n",
        encoding="utf-8",
    )

    _synchronize_redacted(raw_path, redacted_path)
    _synchronize_redacted(raw_path, redacted_path)

    synchronized = [
        json.loads(line)
        for line in redacted_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["case_id"] for row in synchronized] == ["case-1", "case-2"]


def test_redacted_synchronization_allows_fresh_output_directory(tmp_path):
    _synchronize_redacted(
        tmp_path / "missing-raw.jsonl",
        tmp_path / "missing-redacted.jsonl",
    )

    assert not (tmp_path / "missing-redacted.jsonl").exists()


def test_summary_keeps_failures_in_denominator(tmp_path):
    path = tmp_path / "rows.jsonl"
    success = {
        "case_id": "a",
        "realization_stage": "natural_greedy",
        "status": "success",
        "attribution": {
            "methods": {
                "m": {
                    "runtime_seconds": 1.0,
                    "localization_metrics": {
                        "recovery_at_5pct": 0.1,
                        "recovery_at_10pct": 0.2,
                        "recovery_at_20pct": 0.3,
                        "pointing_game": True,
                        "injection_mass_fraction": 0.4,
                    },
                }
            }
        },
    }
    failure = {
        "case_id": "b",
        "realization_stage": "teacher_forcing_2",
        "status": "error",
        "attribution": None,
    }
    path.write_text(
        json.dumps(success) + "\n" + json.dumps(failure) + "\n",
        encoding="utf-8",
    )

    summary = _summary(
        path,
        generation_records=1046,
        eligible_records=2,
        attacked_inventory=8,
        protocol={},
    )

    assert summary["attempted_attribution_count"] == 2
    assert summary["successful_attribution_count"] == 1
    assert summary["failed_attribution_count"] == 1
    assert summary["attacked_inventory"] == 8
    assert summary["eligible_coverage_over_attacked_inventory"] == 0.25
    assert summary["methods"]["m"]["recovery_at_5pct_mean"] == 0.1
    assert summary["stages"]["natural_greedy"]["successful_attribution_count"] == 1
    assert summary["stages"]["teacher_forcing_2"]["successful_attribution_count"] == 0
