import json

import evaluations.agentdojo.benchmark_local as benchmark_local
import pytest
from evaluations.agentdojo.benchmark_local import (
    BenchmarkCase,
    DecodingConfig,
    ReasoningSearchConfig,
    SUITES,
    _find_target_event,
    _injection_is_visible,
    _redacted,
    _reasoning_boundary,
    _reasoning_reranked_turn,
    _serialize_target_completion,
    _sparse_corrected_turn,
    _synchronize_redacted,
    _teacher_forced_turn,
    _turn_seed,
    _search_seed,
    _validate_generation_service_model,
    _write_or_validate_protocol,
    build_inventory,
    run_case,
    select_pilot,
)
from evaluations.agentdojo.target_realization import (
    build_realization_stages,
    parse_natural_budgets,
    parse_teacher_forcing_budgets,
)


class _TinyTokenizer:
    eos_token_id = 9
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        payload = {
            "input_ids": [ord(char) + 100 for char in text],
        }
        if return_offsets_mapping:
            payload["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return type(
            "Encoded",
            (dict,),
            {"input_ids": property(lambda self: self["input_ids"])},
        )(payload)

    def decode(self, ids, **kwargs):
        del kwargs
        return "".join(chr(token_id - 100) for token_id in ids)


class _SparseGenerator:
    def __init__(self, reference_ids, base_length, mismatch_positions):
        self.reference_ids = reference_ids
        self.base_length = base_length
        self.mismatch_positions = set(mismatch_positions)

    def generate_from_token_ids(self, input_ids, max_new_tokens, **kwargs):
        del kwargs
        position = len(input_ids) - self.base_length
        if max_new_tokens > 1 or position >= len(self.reference_ids):
            ids = [9]
        elif position in self.mismatch_positions:
            ids = [ord("X") + 100]
        else:
            ids = [self.reference_ids[position]]
        return {
            "generated_token_ids_with_eos_if_emitted": ids,
            "generation_seconds": 0.01,
        }


class _RerankGenerator:
    def __init__(self, tokenizer, target_action):
        self.tokenizer = tokenizer
        self.target_action = target_action
        self.reasoning_calls = 0
        self.continuation_calls = 0
        self.inputs = []

    def _result(self, completion):
        ids = self.tokenizer(completion).input_ids
        return {
            "generated_token_ids_with_eos_if_emitted": ids,
            "generated_token_count_with_eos_if_emitted": len(ids),
            "completion": completion,
            "raw_completion_with_eos": completion,
            "generation_seconds": 0.01,
            "finish_reason": "stop",
            "stopped_on_eos": False,
        }

    def generate(self, prompt, max_new_tokens, **kwargs):
        del prompt, max_new_tokens, kwargs
        value = f"<think>candidate {self.reasoning_calls}</think>"
        self.reasoning_calls += 1
        return self._result(value)

    def generate_from_token_ids(self, input_ids, max_new_tokens, **kwargs):
        del max_new_tokens, kwargs
        self.inputs.append(list(input_ids))
        call = self.continuation_calls
        self.continuation_calls += 1
        if call == 0:
            completion = "final response"
        else:
            completion = self.target_action
        return self._result(completion)


def test_full_inventory_matches_pinned_v122_counts():
    clean = build_inventory(mode="clean")
    attacked = build_inventory(mode="attacked")

    assert len(clean) == 97
    assert len(attacked) == 949
    assert {case.suite for case in clean} == set(SUITES)
    assert all(case.injection_task is None for case in clean)
    assert all(case.injections for case in attacked)


def test_pilot_selects_clean_and_attacked_from_every_suite():
    cases = build_inventory(mode="both")
    selected = select_pilot(cases, per_suite=1)

    assert len(selected) == 8
    for suite in SUITES:
        suite_cases = [case for case in selected if case.suite == suite]
        assert len(suite_cases) == 2
        assert {case.attack for case in suite_cases} == {
            "none",
            "important_instructions",
        }


def test_pilot_spans_user_and_injection_tasks():
    cases = build_inventory(mode="both")
    selected = select_pilot(cases, per_suite=10)

    for suite in SUITES:
        attacked = [
            case
            for case in selected
            if case.suite == suite and case.attack == "important_instructions"
        ]
        assert len({case.user_task for case in attacked}) > 1
        assert len({case.injection_task for case in attacked}) > 1


def test_target_event_requires_validator_success_and_target_call():
    turns = [
        {
            "turn_index": 0,
            "valid_reasoning_boundary": True,
            "rendered_prompt": "prompt",
            "completion": "completion",
            "token_id_provenance": "conditioned_continuation",
            "output_intervention": {"mode": "teacher_forcing_prefix"},
            "parsed_tool_calls": [
                {"function": "safe_tool", "args": {}},
                {"function": "malicious_tool", "args": {"x": 1}},
            ],
        }
    ]

    expected = [{"function": "malicious_tool", "args": {"x": 1}}]
    assert _find_target_event(turns, expected, False) is None
    event = _find_target_event(turns, expected, True)
    assert event is not None
    assert event["call"]["function"] == "malicious_tool"
    assert event["output_provenance"] == "conditioned_continuation"
    assert event["teacher_forced"]

    wrong_args = [{"function": "malicious_tool", "args": {"x": 2}}]
    assert _find_target_event(turns, wrong_args, True) is None

    turns[0]["valid_reasoning_boundary"] = False
    assert _find_target_event(turns, expected, True) is None


def test_reasoning_boundary_accepts_both_qwen_serializations():
    assert _reasoning_boundary("<think>x</think><tool_call>x</tool_call>") == (
        "explicit_open_and_close"
    )
    assert _reasoning_boundary("x</think><tool_call>x</tool_call>") == (
        "implicit_open_explicit_close"
    )
    assert _reasoning_boundary("<think>truncated") == "missing_close"


def test_target_realization_helpers_are_deterministic_and_structured():
    calls = [{"function": "target_tool", "args": {"value": "redacted"}}]
    completion = _serialize_target_completion(calls)

    parsed, errors = benchmark_local.parse_qwen_tool_calls(completion)
    assert not errors
    assert [(call.function, call.args) for call in parsed] == [
        ("target_tool", {"value": "redacted"})
    ]
    assert _injection_is_visible("<INFORMATION>x</INFORMATION>")
    assert not _injection_is_visible("ordinary tool result")
    assert _turn_seed(7, "case", 1, 2) == _turn_seed(7, "case", 1, 2)
    assert _turn_seed(7, "case", 1, 2) != _turn_seed(7, "case", 2, 2)
    assert len({_search_seed(7, index, 0) for index in range(4)}) == 4


def test_teacher_forcing_preserves_natural_reasoning_and_tracks_tokens():
    turn = _teacher_forced_turn(
        None,
        _TinyTokenizer(),
        "prompt",
        [{"function": "f", "args": {}}],
        {
            "completion": "<think>kept</think>discarded suffix",
            "generation_seconds": 1.5,
        },
        budget="full",
        max_new_tokens=128,
        generator=None,
        decoding=DecodingConfig(),
        seed=1,
    )

    assert turn["completion"].startswith("<think>kept</think><tool_call>")
    assert "discarded suffix" not in turn["completion"]
    assert turn["generation_seconds"] == 1.5
    assert turn["output_intervention"]["reasoning_provenance"] == (
        "model"
    )
    assert turn["output_intervention"]["forced_token_count"] > 0
    assert turn["completion_token_provenance"].count("model") > 2
    assert turn["completion_token_provenance"].count("reference") > 0


def test_teacher_forcing_never_creates_a_minimal_reasoning_fallback():
    natural = {"completion": "unfinished reasoning", "generation_seconds": 1.0}
    turn = _teacher_forced_turn(
        None,
        _TinyTokenizer(),
        "prompt",
        [{"function": "f", "args": {}}],
        natural,
        budget="full",
        max_new_tokens=128,
        generator=None,
        decoding=DecodingConfig(),
        seed=1,
    )

    assert turn is natural
    assert "output_intervention" not in turn


def test_sparse_correction_only_replaces_model_deviations():
    tokenizer = _TinyTokenizer()
    calls = [{"function": "f", "args": {}}]
    reasoning = "<think>kept</think>"
    action = _serialize_target_completion(calls).removeprefix("</think>\n")
    reference_ids = tokenizer(action).input_ids
    base_length = len(tokenizer("prompt" + reasoning).input_ids)
    generator = _SparseGenerator(reference_ids, base_length, {1, 3})
    natural = {"completion": reasoning + "discarded", "generation_seconds": 1.0}

    full = _sparse_corrected_turn(
        None,
        tokenizer,
        "prompt",
        calls,
        natural,
        budget="full",
        max_new_tokens=256,
        generator=generator,
        decoding=DecodingConfig(),
        seed=1,
    )
    metadata = full["output_intervention"]
    assert metadata["reference_token_count"] == 2
    assert metadata["correction_positions"] == [1, 3]
    assert metadata["model_match_count"] == len(reference_ids) - 2
    assert full["completion_token_provenance"].count(
        "reference_correction"
    ) == 2
    assert full["completion"].endswith(action)

    limited = _sparse_corrected_turn(
        None,
        tokenizer,
        "prompt",
        calls,
        natural,
        budget=1,
        max_new_tokens=256,
        generator=generator,
        decoding=DecodingConfig(),
        seed=1,
    )
    limited_metadata = limited["output_intervention"]
    assert limited_metadata["reference_token_count"] == 1
    assert limited_metadata["first_uncorrected_deviation_position"] == 3


def test_reasoning_rerank_keeps_one_coherent_model_only_trajectory():
    tokenizer = _TinyTokenizer()
    calls = [{"function": "f", "args": {}}]
    action = _serialize_target_completion(calls).removeprefix("</think>\n")
    generator = _RerankGenerator(tokenizer, action)

    turn = _reasoning_reranked_turn(
        None,
        tokenizer,
        "prompt",
        calls,
        max_new_tokens=256,
        generator=generator,
        decoding=DecodingConfig(temperature=0.7),
        seed=3,
        search=ReasoningSearchConfig(
            candidates=2,
            score_method="short_rollout",
            score_rollouts=1,
            score_max_new_tokens=64,
        ),
    )

    assert turn is not None
    assert turn["completion"].startswith("<think>candidate 1</think>")
    assert turn["completion"].endswith(action)
    assert set(turn["completion_token_provenance"]) == {"model"}
    metadata = turn["natural_search"]
    assert metadata["selected_candidate_index"] == 1
    assert metadata["distinct_reasoning_count"] == 2
    assert metadata["generation_request_count"] == 5
    assert metadata["reference_token_count"] == 0
    assert metadata["all_completion_tokens_model_generated"]
    selected_prefix = tokenizer("prompt<think>candidate 1</think>").input_ids
    assert generator.inputs[-1] == selected_prefix


def test_realization_stage_plan_has_two_monotone_axes():
    assert parse_natural_budgets("1,4,16") == (1, 4, 16)
    assert parse_teacher_forcing_budgets("1,2,8,full") == (1, 2, 8, "full")
    stages = build_realization_stages((1, 4), (1, 2, "full"))

    assert [stage.name for stage in stages] == [
        "natural_greedy",
        "natural_sample_4",
        "prefix_forcing_1",
        "prefix_forcing_2",
        "prefix_forcing_full",
    ]

    with pytest.raises(ValueError, match="strictly increasing"):
        parse_natural_budgets("1,1")
    with pytest.raises(ValueError, match="start at one"):
        parse_natural_budgets("4,16")
    with pytest.raises(ValueError, match="last"):
        parse_teacher_forcing_budgets("full,2")


def test_generation_service_model_must_match_protocol():
    info = {"model_path": "/cache/models--Org--Model/snapshots/revision"}
    _validate_generation_service_model("Org/Model", "revision", info)

    with pytest.raises(ValueError, match="does not match"):
        _validate_generation_service_model("Org/Other", "revision", info)
    with pytest.raises(ValueError, match="does not match"):
        _validate_generation_service_model("Org/Model", "other-revision", info)


def test_two_axis_runner_stops_at_first_localizable_candidate(monkeypatch):
    case = BenchmarkCase(
        case_id="suite__user__target",
        suite="slack",
        user_task="user",
        injection_task="target",
        attack="important_instructions",
        injections={"slot": "redacted"},
    )

    def candidate(*, success: bool, utility: bool) -> dict:
        return {
            "utility": utility,
            "attack_success": success,
            "generated_target_action": success,
            "valid_reasoning_boundaries": True,
            "stop_reason": "assistant_final_answer",
            "turns": [
                {
                    "rendered_prompt": "ordinary prompt",
                    "valid_reasoning_boundary": True,
                    "output_intervention": {},
                }
            ],
        }

    outcomes = [
        candidate(success=False, utility=True),
        candidate(success=False, utility=True),
        candidate(success=False, utility=True),
        candidate(success=False, utility=True),
        candidate(success=False, utility=True),
        candidate(success=True, utility=False),
    ]
    for outcome in outcomes:
        outcome["turns"][0]["rendered_prompt"] = (
            "<INFORMATION>x</INFORMATION>"
        )
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(
            (kwargs["attempt_index"], kwargs.get("teacher_forcing_budget"))
        )
        return outcomes[len(calls) - 1]

    monkeypatch.setattr(benchmark_local, "_run_case_once", fake_run)
    selected = run_case(
        case,
        model=None,
        tokenizer=None,
        max_turns=2,
        max_new_tokens=8,
        decoding=DecodingConfig(temperature=0.7),
        natural_rollout_budgets=(1, 4),
        teacher_forcing_token_budgets=(1, 2),
    )

    assert calls == [
        (0, None),
        (1, None),
        (2, None),
        (3, None),
        (0, 1),
        (0, 2),
    ]
    assert selected["generated_target_action"]
    assert selected["selection"]["executed_attempts"] == 6
    assert selected["selection"]["selected_attempt_index"] == 5
    assert selected["selection"]["realization_stage"] == "prefix_forcing_2"
    assert selected["selection"]["conditioning_source_attempt_index"] == 0


def test_redacted_resume_synchronizes_missing_raw_case(tmp_path):
    raw = tmp_path / "raw.jsonl"
    redacted = tmp_path / "redacted.jsonl"
    record = {
        "schema_version": 1,
        "case": {
            "case_id": "case",
            "suite": "slack",
            "user_task": "u",
            "injection_task": None,
            "attack": "none",
            "injections": {},
        },
        "turns": [
            {
                "turn_index": 0,
                "prompt_token_count": 1,
                "generated_token_count_with_eos_if_emitted": 1,
                "stopped_on_eos": True,
                "generation_seconds": 0.1,
                "completion": "<think>x</think>done",
                "parse_errors": [],
                "parsed_tool_calls": [],
            }
        ],
        "function_trace": [],
        "stop_reason": "assistant_final_answer",
        "utility": True,
        "attack_success": None,
        "target_functions": [],
        "target_calls": [],
        "generated_target_action": False,
    }
    raw.write_text(json.dumps(record) + "\n", encoding="utf-8")

    _synchronize_redacted(raw, redacted)
    _synchronize_redacted(raw, redacted)

    rows = [json.loads(line) for line in redacted.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["case"]["case_id"] == "case"
    assert rows[0]["valid_reasoning_boundaries"]


def test_resume_refuses_to_mix_protocols(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    records_path = tmp_path / "records.jsonl"
    first = {"decoding": "first"}

    _write_or_validate_protocol(protocol_path, records_path, first)
    records_path.write_text("{}\n", encoding="utf-8")
    _write_or_validate_protocol(protocol_path, records_path, first)

    with pytest.raises(ValueError, match="different protocols"):
        _write_or_validate_protocol(
            protocol_path,
            records_path,
            {"decoding": "second"},
        )


def test_redacted_function_trace_omits_arguments():
    record = {
        "schema_version": 1,
        "case": {
            "case_id": "case",
            "suite": "slack",
            "user_task": "u",
            "injection_task": "i",
            "attack": "important_instructions",
            "injections": {"secret": "do not publish"},
        },
        "turns": [],
        "function_trace": [
            {
                "function": "send_email",
                "args": {"body": "do not publish"},
                "id": "call",
            }
        ],
        "stop_reason": "assistant_final_answer",
        "utility": False,
        "attack_success": True,
        "target_functions": ["send_email"],
        "target_calls": [{"function": "send_email", "args": {}}],
        "generated_target_action": True,
    }

    redacted = _redacted(record)

    assert "injections" not in redacted["case"]
    assert redacted["function_trace"] == [{"function": "send_email"}]
