from __future__ import annotations

import json

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from exp.exp4.run_exp import _trajectory_trace_record
from exp.exp4.sample_and_filter import (
    AiderSeed,
    TrajectoryConfig,
    generate_trajectory,
    parse_bool,
    trajectory_row,
)
from exp.exp4.trajectory_utils import load_aider
from flashtrace.improved import LLMIFRAttributionBoth
from tests.helpers import _make_tiny_word_tokenizer, make_tiny_qwen2_model_and_tokenizer


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_aider_keeps_legacy_input_output_compatible(tmp_path):
    path = tmp_path / "legacy.jsonl"
    _write_jsonl(path, [{"input": "t10 t20", "output": "t30", "length": 3}])

    example = load_aider(path)[0]

    assert example.prompt == "t10 t20"
    assert example.target == "t30"
    assert example.metadata["length"] == 3
    assert example.metadata["schema_version"] == 1
    assert not example.is_multiturn


def test_load_aider_renders_multiturn_history_with_target_chat_template(tmp_path):
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    path = tmp_path / "trajectory.jsonl"
    messages = [
        {"role": "system", "content": "t10", "kind": "agent_instruction"},
        {"role": "user", "content": "t20 t30", "kind": "task"},
        {"role": "assistant", "content": "t40", "kind": "draft_edit"},
        {"role": "user", "content": "t50", "kind": "test_feedback"},
        {"role": "assistant", "content": "t60 t70", "kind": "revised_edit"},
    ]
    _write_jsonl(path, [{"schema_version": 2, "id": "ex-1", "messages": messages}])

    example = load_aider(path, tokenizer=tokenizer)[0]

    expected_prompt = tokenizer.apply_chat_template(
        [{"role": message["role"], "content": message["content"]} for message in messages[:-1]],
        tokenize=False,
        add_generation_prompt=True,
    )
    assert example.prompt == expected_prompt
    assert example.target == "t60 t70"
    assert example.is_multiturn
    assert [segment.role for segment in example.segments] == ["system", "user", "assistant", "user"]
    assert [example.prompt[start:end] for start, end in (segment.char_span for segment in example.segments)] == [
        "t10",
        "t20 t30",
        "t40",
        "t50",
    ]


def test_load_execution_grounded_polyglot_uses_frozen_prompt_and_action_span(
    tmp_path,
):
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    history = [
        {"role": "system", "content": "t10"},
        {"role": "user", "content": "t20"},
        {"role": "assistant", "content": "t30"},
        {"role": "user", "content": "t40"},
    ]
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    target = "<think>t50</think>\nt60 t70"
    action_start = target.index("t60")
    row = {
        "schema_version": 3,
        "benchmark": "aider_polyglot",
        "task": {
            "task_id": "polyglot-python-test",
            "language": "python",
            "exercise": "test",
        },
        "messages": history,
        "turns": [
            {
                "kind": "revised_edit",
                "rendered_prompt": prompt,
                "completion": target,
                "reasoning_char_span_half_open": [
                    len("<think>"),
                    target.index("</think>"),
                ],
                "action_char_span_half_open": [action_start, len(target)],
            }
        ],
        "draft_passed": False,
        "final_passed": True,
        "repair_attempted": True,
        "repair_improved": True,
        "valid_final_boundary": True,
        "valid_final_action": True,
    }
    path = tmp_path / "polyglot.jsonl"
    _write_jsonl(path, [row])

    example = load_aider(path, tokenizer=tokenizer)[0]

    assert example.prompt == prompt
    assert example.target == target
    assert example.sink_span is not None
    assert example.thinking_span is not None
    action_ids = tokenizer(
        target[action_start:],
        add_special_tokens=False,
    ).input_ids
    assert example.sink_span[1] - example.sink_span[0] + 1 == len(action_ids)
    assert example.metadata["final_passed"]


def test_load_execution_grounded_polyglot_skips_invalid_empty_turn(tmp_path):
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    history = [
        {"role": "system", "content": "t10"},
        {"role": "user", "content": "t20"},
    ]
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    valid = {
        "schema_version": 3,
        "benchmark": "aider_polyglot",
        "task": {
            "task_id": "valid",
            "language": "python",
            "exercise": "test",
        },
        "messages": history,
        "turns": [
            {
                "kind": "draft_edit",
                "rendered_prompt": prompt,
                "completion": "t30",
                "reasoning_char_span_half_open": [0, 1],
                "action_char_span_half_open": [0, 3],
            }
        ],
        "draft_passed": True,
        "final_passed": True,
        "repair_attempted": False,
        "repair_improved": False,
        "valid_final_boundary": True,
        "valid_final_action": True,
    }
    invalid = {
        **valid,
        "task": {**valid["task"], "task_id": "invalid-empty-eos"},
        "messages": [
            *history,
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "t40"},
        ],
        "valid_final_boundary": False,
        "valid_final_action": False,
    }
    path = tmp_path / "polyglot.jsonl"
    _write_jsonl(path, [invalid, valid])

    examples = load_aider(path, tokenizer=tokenizer)

    assert [example.metadata["example_id"] for example in examples] == ["valid"]


def test_load_execution_grounded_polyglot_keeps_empty_prior_assistant(tmp_path):
    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    history = [
        {"role": "system", "content": "t10"},
        {"role": "user", "content": "t20"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "t40"},
    ]
    prompt = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
    )
    row = {
        "schema_version": 3,
        "benchmark": "aider_polyglot",
        "task": {
            "task_id": "valid-empty-prior-assistant",
            "language": "python",
            "exercise": "test",
        },
        "messages": history,
        "turns": [
            {
                "kind": "revised_edit",
                "rendered_prompt": prompt,
                "completion": "t30",
                "reasoning_char_span_half_open": [0, 1],
                "action_char_span_half_open": [0, 3],
            }
        ],
        "draft_passed": False,
        "final_passed": True,
        "repair_attempted": True,
        "repair_improved": True,
        "valid_final_boundary": True,
        "valid_final_action": True,
    }
    path = tmp_path / "polyglot.jsonl"
    _write_jsonl(path, [row])

    example = load_aider(path, tokenizer=tokenizer)[0]

    assert example.metadata["example_id"] == "valid-empty-prior-assistant"
    assert example.target == "t30"
    assert example.segments[2].role == "assistant"
    assert example.segments[2].char_span[0] == example.segments[2].char_span[1]


def test_load_execution_grounded_polyglot_marks_transformed_assistant_unassigned(
    tmp_path,
):
    row = {
        "schema_version": 3,
        "benchmark": "aider_polyglot",
        "task": {
            "task_id": "valid-transformed-prior-assistant",
            "language": "python",
            "exercise": "test",
        },
        "messages": [
            {"role": "system", "content": "t10"},
            {"role": "user", "content": "t20"},
            {"role": "assistant", "content": "not-in-frozen-prompt"},
            {"role": "user", "content": "t40"},
        ],
        "turns": [
            {
                "kind": "revised_edit",
                "rendered_prompt": "prefix t10 middle t20 transformed t40 suffix",
                "completion": "t30",
                "reasoning_char_span_half_open": [0, 1],
                "action_char_span_half_open": [0, 3],
            }
        ],
        "draft_passed": False,
        "final_passed": True,
        "repair_attempted": True,
        "repair_improved": True,
        "valid_final_boundary": True,
        "valid_final_action": True,
    }
    path = tmp_path / "polyglot.jsonl"
    _write_jsonl(path, [row])

    example = load_aider(path)[0]

    assert example.segments[2].role == "assistant"
    assert example.segments[2].char_span == (-1, -1)
    assert example.prompt[slice(*example.segments[3].char_span)] == "t40"


def test_generate_trajectory_builds_aider_repair_turn_and_final_judge():
    seed = AiderSeed(
        example_id="ex-1",
        prompt="Implement t10",
        reference_output="t40",
        metadata={"length": 4},
    )
    config = TrajectoryConfig(generator_model="strong-generator", judge_model="judge", assistant_turns=2)
    responses = iter(["t20", "AssertionError: expected t40", "t40", "True"])
    purposes = []

    def requester(**kwargs):
        purposes.append(kwargs["purpose"])
        return next(responses)

    messages, judge_response = generate_trajectory(seed, config, requester)

    assert purposes == ["generation", "feedback", "generation", "judge"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert messages[2]["kind"] == "draft_edit"
    assert messages[3]["kind"] == "test_feedback"
    assert "AssertionError" in messages[3]["content"]
    assert messages[-1]["content"] == "t40"
    assert parse_bool(judge_response)
    row = trajectory_row(seed, messages, judge_response, config)
    assert row["schema_version"] == 2
    assert row["benchmark"] == "aider_multiturn"
    assert row["metadata"]["assistant_turns"] == 2
    assert "reference_output_sha256" in row["metadata"]


def test_flashtrace_attributes_rendered_multiturn_qwen3_trajectory(tmp_path):
    tokenizer = _make_tiny_word_tokenizer()
    config = Qwen3Config(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM._from_config(config, attn_implementation="eager")
    model.eval()

    path = tmp_path / "trajectory.jsonl"
    messages = [
        {"role": "system", "content": "t10", "kind": "agent_instruction"},
        {"role": "user", "content": "t20", "kind": "task"},
        {"role": "assistant", "content": "t30", "kind": "draft_edit"},
        {"role": "user", "content": "t40", "kind": "test_feedback"},
        {"role": "assistant", "content": "t50 t60", "kind": "revised_edit"},
    ]
    _write_jsonl(path, [{"schema_version": 2, "id": "qwen3-smoke", "messages": messages}])
    example = load_aider(path, tokenizer=tokenizer)[0]

    attributor = LLMIFRAttributionBoth(
        model,
        tokenizer,
        chunk_tokens=16,
        sink_chunk_tokens=4,
        show_progress=False,
    )
    attribution = attributor.calculate_ifr_multi_hop_both(
        example.prompt,
        target=example.target,
        n_hops=1,
    )
    row = attribution.get_all_token_attrs([0, 1])[1]
    record = _trajectory_trace_record(
        example_idx=0,
        example=example,
        method="ifr_multi_hop_both",
        sink="full_output",
        tokenizer=tokenizer,
        attribution_result=attribution,
        row_vector=row,
    )

    assert row.shape[1] == len(attribution.prompt_tokens) + len(attribution.generation_tokens)
    assert len(record["prompt_token_attribution"]) == len(attribution.prompt_tokens)
    assert [segment["role"] for segment in record["segments"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert record["per_hop"]
    json.dumps(record)
