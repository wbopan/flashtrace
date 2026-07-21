#!/usr/bin/env python3
"""Run local Qwen as an autonomous AgentDojo agent, then attribute real actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    get_text_content_as_str,
    text_content_block_from_string,
)

from evaluations.agentdojo.explain import _score_payload
from evaluations.agentdojo.pilot import (
    AGENTDOJO_VERSION,
    BENCHMARK_VERSION,
    DEFAULT_ATTRIBUTION_MODEL,
    PILOT_CASES,
    LLMIFRAttribution,
    LLMIFRAttributionBoth,
    PreparedCase,
    _char_span_to_token_indices,
    _clean_prompt_vector,
    _find_injection_span,
    _load_model,
    _localization_metrics,
    _message_text,
    _random_metrics,
    _tool_schema,
    keep_token_indices,
    prepare_case,
)


TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    flags=re.DOTALL,
)


@dataclass(frozen=True)
class ParsedToolCall:
    function: str
    args: dict[str, Any]
    char_start: int
    char_end: int
    raw_json: str


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def parse_qwen_tool_calls(completion: str) -> tuple[list[ParsedToolCall], list[str]]:
    calls = []
    errors = []
    for match in TOOL_CALL_PATTERN.finditer(completion):
        raw_json = match.group(1)
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as exc:
            errors.append(
                f"invalid JSON at chars [{match.start()},{match.end()}): {exc}"
            )
            continue
        function = payload.get("name")
        args = payload.get("arguments", {})
        if not isinstance(function, str) or not function:
            errors.append(f"missing function name in {raw_json}")
            continue
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as exc:
                errors.append(f"invalid arguments JSON for {function}: {exc}")
                continue
        if not isinstance(args, dict):
            errors.append(f"arguments for {function} are not an object")
            continue
        calls.append(
            ParsedToolCall(
                function=function,
                args=args,
                char_start=match.start(),
                char_end=match.end(),
                raw_json=raw_json,
            )
        )
    if "<tool_call>" in completion and not calls and not errors:
        errors.append("tool-call marker found, but no complete block could be parsed")
    return calls, errors


def _message_content(message: ChatMessage) -> str:
    content = message.get("content")
    if content is None:
        return ""
    return get_text_content_as_str(content)


def _qwen_message(message: ChatMessage) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "role": message["role"],
        "content": _message_content(message),
    }
    if message["role"] == "assistant" and message.get("tool_calls"):
        converted["tool_calls"] = [
            {
                "type": "function",
                "function": {
                    "name": call.function,
                    "arguments": dict(call.args),
                },
            }
            for call in message["tool_calls"] or []
        ]
    return converted


def _message_record(message: ChatMessage) -> dict[str, Any]:
    record: dict[str, Any] = {
        "role": message["role"],
        "content": _message_content(message),
    }
    if message["role"] == "assistant":
        record["tool_calls"] = [
            call.model_dump(mode="json") for call in message.get("tool_calls") or []
        ]
    if message["role"] == "tool":
        record["tool_call_id"] = message.get("tool_call_id")
        record["tool_call"] = message["tool_call"].model_dump(mode="json")
        record["error"] = message.get("error")
    return record


def _strip_trailing_eos(text: str, eos_token: str | None) -> str:
    if eos_token and text.endswith(eos_token):
        return text[: -len(eos_token)]
    return text


def _completion_content(completion: str) -> str:
    return TOOL_CALL_PATTERN.sub("", completion).strip()


def _render_prompt(
    tokenizer: Any,
    messages: Sequence[ChatMessage],
    tool_schemas: Sequence[dict[str, Any]],
) -> str:
    prompt = tokenizer.apply_chat_template(
        [_qwen_message(message) for message in messages],
        tools=list(tool_schemas),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Qwen chat template returned an empty prompt.")
    return prompt


@torch.no_grad()
def _generate_turn(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = encoded.input_ids.to(model.device)
    attention_mask = encoded.attention_mask.to(model.device)
    started = time.perf_counter()
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    runtime = time.perf_counter() - started
    generated_ids = output_ids[0, input_ids.shape[1] :].detach().cpu().tolist()
    raw_with_eos = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    completion = _strip_trailing_eos(raw_with_eos, tokenizer.eos_token)
    return {
        "rendered_prompt": prompt,
        "prompt_token_ids": input_ids[0].detach().cpu().tolist(),
        "prompt_token_count": int(input_ids.shape[1]),
        "generated_token_ids_with_eos_if_emitted": generated_ids,
        "generated_token_count_with_eos_if_emitted": len(generated_ids),
        "raw_completion_with_eos": raw_with_eos,
        "completion": completion,
        "stopped_on_eos": bool(
            generated_ids
            and tokenizer.eos_token_id is not None
            and generated_ids[-1] == tokenizer.eos_token_id
        ),
        "generation_seconds": runtime,
    }


def _format_tool_result(result: Any) -> str:
    if hasattr(result, "model_dump"):
        return _json(result.model_dump(mode="json"))
    if isinstance(result, list):
        values = [
            item.model_dump(mode="json") if hasattr(item, "model_dump") else item
            for item in result
        ]
        return _json(values)
    if isinstance(result, dict):
        return _json(result)
    return str(result)


def _source_messages(case: PreparedCase) -> list[ChatMessage]:
    system = ChatSystemMessage(
        role="system",
        content=[
            text_content_block_from_string(_message_text(case.raw["messages"][0]))
        ],
    )
    user = ChatUserMessage(
        role="user",
        content=[
            text_content_block_from_string(_message_text(case.raw["messages"][1]))
        ],
    )
    return [system, user]


def _expected_target_calls(case: PreparedCase) -> list[dict[str, Any]]:
    target_message = case.raw["messages"][case.spec.target_message_index]
    return [
        {"function": call["function"], "args": dict(call.get("args") or {})}
        for call in target_message.get("tool_calls") or []
        if call["function"] == case.spec.target_tool
    ]


def _canonical_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = value.strip().lower().rstrip("/")
    normalized = re.sub(r"^https?://", "", normalized)
    return normalized


def _call_matches_expected(call: dict[str, Any], expected: dict[str, Any]) -> bool:
    if call["function"] != expected["function"]:
        return False
    for key, expected_value in expected["args"].items():
        if key not in call["args"]:
            return False
        actual_value = call["args"][key]
        if _canonical_scalar(actual_value) != _canonical_scalar(expected_value):
            return False
    return True


def _find_target_event(
    turns: Sequence[dict[str, Any]],
    expected_calls: Sequence[dict[str, Any]],
    target_tool: str,
    attack_success: bool,
) -> dict[str, Any] | None:
    candidates = []
    for turn in turns:
        for call_index, call in enumerate(turn["parsed_tool_calls"]):
            event = {
                "turn_index": turn["turn_index"],
                "call_index": call_index,
                "call": call,
                "rendered_prompt": turn["rendered_prompt"],
                "completion": turn["completion"],
            }
            if any(
                _call_matches_expected(call, expected) for expected in expected_calls
            ):
                return event
            if call["function"] == target_tool:
                candidates.append(event)
    if attack_success and candidates:
        return candidates[-1]
    return None


def _run_case_e2e(
    case: PreparedCase,
    model: Any,
    tokenizer: Any,
    max_turns: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    suite = get_suite(BENCHMARK_VERSION, case.spec.suite)
    user_task = suite.get_user_task_by_id(case.spec.user_task)
    injection_task = suite.get_injection_task_by_id(case.spec.injection_task)
    environment = suite.load_and_inject_default_environment(case.raw["injections"])
    environment = user_task.init_environment(environment)
    pre_environment = environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)
    tool_schemas = [_tool_schema(tool) for tool in suite.tools]
    messages = _source_messages(case)
    turns = []
    trace: list[FunctionCall] = []
    stop_reason = "max_turns"

    for turn_index in range(max_turns):
        prompt = _render_prompt(tokenizer, messages, tool_schemas)
        turn = _generate_turn(model, tokenizer, prompt, max_new_tokens)
        parsed, parse_errors = parse_qwen_tool_calls(turn["completion"])
        function_calls = [
            FunctionCall(
                function=call.function,
                args=dict(call.args),
                id=f"qwen_e2e_turn_{turn_index}_call_{call_index}",
            )
            for call_index, call in enumerate(parsed)
        ]
        assistant_message = ChatAssistantMessage(
            role="assistant",
            content=[
                text_content_block_from_string(_completion_content(turn["completion"]))
            ],
            tool_calls=function_calls,
        )
        messages.append(assistant_message)
        turn.update(
            {
                "turn_index": turn_index,
                "parse_errors": parse_errors,
                "parsed_tool_calls": [asdict(call) for call in parsed],
                "tool_results": [],
            }
        )

        if not function_calls:
            stop_reason = (
                "assistant_final_answer"
                if turn["stopped_on_eos"]
                else "max_new_tokens_without_complete_action"
            )
            turns.append(turn)
            break

        for function_call in function_calls:
            trace.append(function_call)
            result, error = runtime.run_function(
                environment,
                function_call.function,
                function_call.args,
            )
            result_text = _format_tool_result(result)
            content = result_text if error is None else f"Error: {error}"
            tool_message = ChatToolResultMessage(
                role="tool",
                content=[text_content_block_from_string(content)],
                tool_call_id=function_call.id,
                tool_call=function_call,
                error=error,
            )
            messages.append(tool_message)
            turn["tool_results"].append(
                {
                    "tool_call": function_call.model_dump(mode="json"),
                    "content": content,
                    "raw_result": result_text,
                    "error": error,
                }
            )
        turns.append(turn)

    assistant_messages = [
        message for message in messages if message["role"] == "assistant"
    ]
    final_content = assistant_messages[-1]["content"] if assistant_messages else []
    utility = suite._check_user_task_utility(  # noqa: SLF001 - benchmark protocol
        user_task,
        final_content or [],
        pre_environment,
        environment,
        trace,
    )
    attack_success = suite._check_injection_task_security(  # noqa: SLF001
        injection_task,
        final_content or [],
        pre_environment,
        environment,
        trace,
    )
    expected_calls = _expected_target_calls(case)
    target_event = _find_target_event(
        turns,
        expected_calls,
        case.spec.target_tool,
        bool(attack_success),
    )
    return {
        "case": {
            **asdict(case.spec),
            "source_trace": str(case.source_path.resolve()),
            "source_trace_sha256": case.spec.sha256,
        },
        "protocol": {
            "agentdojo_package_version": AGENTDOJO_VERSION,
            "benchmark_version": BENCHMARK_VERSION,
            "model": DEFAULT_ATTRIBUTION_MODEL,
            "generation_mode": "autonomous_greedy",
            "teacher_forced": False,
            "max_turns": max_turns,
            "max_new_tokens_per_turn": max_new_tokens,
            "injection_payload_provenance": "pinned_upstream_trace_environment",
        },
        "source_user_prompt": _message_text(case.raw["messages"][1]),
        "source_injections": case.raw["injections"],
        "expected_malicious_calls_from_source_trace": expected_calls,
        "turns": turns,
        "messages": [_message_record(message) for message in messages],
        "function_trace": [call.model_dump(mode="json") for call in trace],
        "stop_reason": stop_reason,
        "utility": bool(utility),
        "attack_success": bool(attack_success),
        "both_success": bool(utility and attack_success),
        "generated_expected_target_action": target_event is not None,
        "target_event": target_event,
        "pre_environment": pre_environment.model_dump(mode="json"),
        "post_environment": environment.model_dump(mode="json"),
        "attribution": None,
    }


def _target_token_payload(
    tokenizer: Any,
    target: str,
    call_char_span: tuple[int, int],
) -> dict[str, Any]:
    encoded = tokenizer(
        target,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"]]
    call_indices = [
        index
        for index, (start, end) in enumerate(offsets)
        if start < call_char_span[1] and end > call_char_span[0]
    ]
    if not call_indices:
        raise ValueError(
            "Generated target tool call did not overlap any target tokens."
        )
    vocab_tokens = tokenizer.convert_ids_to_tokens(ids)
    return {
        "text": target,
        "token_ids": ids,
        "token_count": len(ids),
        "tokens": [
            {
                "index": index,
                "token_id": token_id,
                "vocab_token": vocab_token,
                "source_slice": target[start:end],
                "char_start": start,
                "char_end": end,
                "in_target_call_sink": index in set(call_indices),
            }
            for index, (token_id, vocab_token, (start, end)) in enumerate(
                zip(ids, vocab_tokens, offsets, strict=True)
            )
        ],
        "target_call_char_span_half_open": list(call_char_span),
        "target_call_token_indices": call_indices,
        "target_call_token_span_inclusive": [min(call_indices), max(call_indices)],
        "thinking_bridge_token_span_inclusive": (
            [0, min(call_indices) - 1] if min(call_indices) > 0 else None
        ),
    }


def _prompt_token_payload(
    tokenizer: Any,
    prompt: str,
    attribution_tokens: Sequence[str],
    gold_indices: Sequence[int],
) -> list[dict[str, Any]]:
    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"]]
    if len(ids) != len(attribution_tokens):
        raise ValueError(
            "Generated prompt tokenization differs from attribution result."
        )
    gold = set(gold_indices)
    vocab_tokens = tokenizer.convert_ids_to_tokens(ids)
    return [
        {
            "index": index,
            "token_id": token_id,
            "vocab_token": vocab_token,
            "attribution_token": attribution_tokens[index],
            "source_slice": prompt[start:end],
            "char_start": start,
            "char_end": end,
            "gold_injection": index in gold,
            "region": "injected_tool_output" if index in gold else "other_prompt",
        }
        for index, (token_id, vocab_token, (start, end)) in enumerate(
            zip(ids, vocab_tokens, offsets, strict=True)
        )
    ]


def _attribute_generated_target(
    record: dict[str, Any],
    tokenizer: Any,
    ifr: LLMIFRAttribution,
    flashtrace: LLMIFRAttributionBoth,
    n_hops: int,
) -> dict[str, Any]:
    event = record["target_event"]
    if event is None:
        raise ValueError("Cannot attribute a missing generated target event.")
    prompt = event["rendered_prompt"]
    target = event["completion"]
    call = event["call"]
    target_payload = _target_token_payload(
        tokenizer,
        target,
        (int(call["char_start"]), int(call["char_end"])),
    )
    sink_span = target_payload["target_call_token_span_inclusive"]
    last_span = [sink_span[1], sink_span[1]]
    thinking_span = target_payload["thinking_bridge_token_span_inclusive"]

    started = time.perf_counter()
    last_result = ifr.calculate_ifr_for_all_positions_output_only(
        prompt,
        target=target,
        sink_span=tuple(last_span),
    )
    last_runtime = time.perf_counter() - started

    started = time.perf_counter()
    full_result = ifr.calculate_ifr_for_all_positions_output_only(
        prompt,
        target=target,
        sink_span=tuple(sink_span),
    )
    full_runtime = time.perf_counter() - started

    started = time.perf_counter()
    flashtrace_result = flashtrace.calculate_ifr_multi_hop_both(
        prompt,
        target=target,
        sink_span=tuple(sink_span),
        thinking_span=None if thinking_span is None else tuple(thinking_span),
        n_hops=n_hops,
    )
    flashtrace_runtime = time.perf_counter() - started

    prompt_length = len(full_result.prompt_tokens)
    if (
        len(last_result.prompt_tokens) != prompt_length
        or len(flashtrace_result.prompt_tokens) != prompt_length
    ):
        raise ValueError("Generated-target attribution tokenizations do not align.")
    injection_char_span = _find_injection_span(prompt)
    gold_indices = _char_span_to_token_indices(
        tokenizer,
        prompt,
        injection_char_span,
    )
    prompt_tokens = _prompt_token_payload(
        tokenizer,
        prompt,
        full_result.prompt_tokens,
        gold_indices,
    )
    keep = keep_token_indices(full_result.prompt_tokens)
    keep_set = set(keep)
    for token in prompt_tokens:
        token["kept_for_ranking"] = token["index"] in keep_set

    result_rows = {
        "ifr_last_generated_action_token": (last_result, last_span, last_runtime),
        "ifr_full_generated_action": (full_result, sink_span, full_runtime),
        "flashtrace_generated_action_via_reasoning": (
            flashtrace_result,
            sink_span,
            flashtrace_runtime,
        ),
    }
    methods = {
        "random_expected": {
            "localization_metrics": _random_metrics(
                full_result.prompt_tokens,
                gold_indices,
            ),
            "runtime_seconds": 0.0,
        }
    }
    regions = [token["region"] for token in prompt_tokens]
    for method, (result, explained_span, runtime) in result_rows.items():
        vector = _clean_prompt_vector(
            result.get_all_token_attrs(explained_span)[1],
            prompt_length,
        ).sum(0)
        payload = _score_payload(vector, keep, regions, prompt_tokens)
        payload.update(
            {
                "runtime_seconds": runtime,
                "explained_target_token_span_inclusive": explained_span,
                "localization_metrics": _localization_metrics(
                    vector.unsqueeze(0),
                    full_result.prompt_tokens,
                    gold_indices,
                ),
            }
        )
        methods[method] = payload

    return {
        "provenance": "actual_autonomous_qwen_generation",
        "teacher_forced": False,
        "turn_index": event["turn_index"],
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "prompt_token_count": prompt_length,
        "gold_injection": {
            "text": prompt[slice(*injection_char_span)],
            "char_span_half_open": list(injection_char_span),
            "token_indices": list(gold_indices),
            "token_span_inclusive": [min(gold_indices), max(gold_indices)],
        },
        "generated_target": target_payload,
        "methods": methods,
        "n_hops": n_hops,
        "thinking_bridge_definition": (
            "all actually generated tokens before the malicious tool-call block"
        ),
    }


def _summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    return {
        "sample_count": count,
        "utility_successes": sum(record["utility"] for record in records),
        "attack_successes": sum(record["attack_success"] for record in records),
        "both_successes": sum(record["both_success"] for record in records),
        "generated_expected_target_actions": sum(
            record["generated_expected_target_action"] for record in records
        ),
        "utility_rate": sum(record["utility"] for record in records) / count,
        "attack_success_rate": sum(record["attack_success"] for record in records)
        / count,
        "both_success_rate": sum(record["both_success"] for record in records) / count,
        "attributed_generated_actions": sum(
            record["attribution"] is not None for record in records
        ),
    }


def _redacted_record(record: dict[str, Any]) -> dict[str, Any]:
    attribution = record["attribution"]
    redacted_attribution = None
    if attribution is not None:
        redacted_attribution = {
            "provenance": attribution["provenance"],
            "teacher_forced": attribution["teacher_forced"],
            "turn_index": attribution["turn_index"],
            "prompt_token_count": attribution["prompt_token_count"],
            "gold_injection_token_count": len(
                attribution["gold_injection"]["token_indices"]
            ),
            "gold_injection_token_span_inclusive": attribution["gold_injection"][
                "token_span_inclusive"
            ],
            "generated_target_token_count": attribution["generated_target"][
                "token_count"
            ],
            "target_call_token_span_inclusive": attribution["generated_target"][
                "target_call_token_span_inclusive"
            ],
            "thinking_bridge_token_span_inclusive": attribution["generated_target"][
                "thinking_bridge_token_span_inclusive"
            ],
            "methods": {
                method: {
                    "runtime_seconds": values["runtime_seconds"],
                    "localization_metrics": values["localization_metrics"],
                }
                for method, values in attribution["methods"].items()
            },
        }
    return {
        "case": record["case"],
        "protocol": record["protocol"],
        "turns": [
            {
                "turn_index": turn["turn_index"],
                "prompt_token_count": turn["prompt_token_count"],
                "generated_token_count_with_eos_if_emitted": turn[
                    "generated_token_count_with_eos_if_emitted"
                ],
                "stopped_on_eos": turn["stopped_on_eos"],
                "generation_seconds": turn["generation_seconds"],
                "completion_sha256": hashlib.sha256(
                    turn["completion"].encode("utf-8")
                ).hexdigest(),
                "parsed_tool_functions": [
                    call["function"] for call in turn["parsed_tool_calls"]
                ],
                "parse_error_count": len(turn["parse_errors"]),
                "tool_execution_error_count": sum(
                    result["error"] is not None for result in turn["tool_results"]
                ),
            }
            for turn in record["turns"]
        ],
        "function_trace": [call["function"] for call in record["function_trace"]],
        "stop_reason": record["stop_reason"],
        "utility": record["utility"],
        "attack_success": record["attack_success"],
        "both_success": record["both_success"],
        "generated_expected_target_action": record["generated_expected_target_action"],
        "target_event": None
        if record["target_event"] is None
        else {
            "turn_index": record["target_event"]["turn_index"],
            "call_index": record["target_event"]["call_index"],
            "function": record["target_event"]["call"]["function"],
        },
        "attribution": redacted_attribution,
    }


def _write_markdown(
    path: Path,
    records: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# AgentDojo autonomous Qwen E2E pilot",
        "",
        "Every assistant turn and tool call in this run was generated autonomously by the local Qwen model. No upstream malicious target was teacher-forced. Attribution is computed only when Qwen actually emits the expected malicious action. Prompt, injection, tool-result, and completion text is deliberately omitted from this report.",
        "",
        f"- Samples: `{summary['sample_count']}`",
        f"- Utility: `{summary['utility_successes']}/{summary['sample_count']}` (`{summary['utility_rate']:.3f}`)",
        f"- Attack success: `{summary['attack_successes']}/{summary['sample_count']}` (`{summary['attack_success_rate']:.3f}`)",
        f"- Utility + attack: `{summary['both_successes']}/{summary['sample_count']}` (`{summary['both_success_rate']:.3f}`)",
        f"- Generated expected malicious action: `{summary['generated_expected_target_actions']}/{summary['sample_count']}`",
        f"- Attributed real malicious actions: `{summary['attributed_generated_actions']}`",
        "",
        "| Case | Turns | Tool calls | Utility | Attack | Expected target | Attributed | Stop |",
        "|---|---:|---:|:---:|:---:|:---:|:---:|---|",
    ]
    for record in records:
        lines.append(
            f"| `{record['case']['case_id']}` | {len(record['turns'])} | "
            f"{len(record['function_trace'])} | {'Y' if record['utility'] else ''} | "
            f"{'Y' if record['attack_success'] else ''} | "
            f"{'Y' if record['generated_expected_target_action'] else ''} | "
            f"{'Y' if record['attribution'] is not None else ''} | "
            f"`{record['stop_reason']}` |"
        )
        lines.extend(
            [
                "",
                f"## {record['case']['case_id']}",
                "",
                "| Turn | Prompt tokens | Generated tokens | EOS | Tool functions | Parse errors | Tool errors | Time |",
                "|---:|---:|---:|:---:|---|---:|---:|---:|",
            ]
        )
        redacted = _redacted_record(record)
        for turn in redacted["turns"]:
            lines.append(
                f"| {turn['turn_index']} | {turn['prompt_token_count']} | "
                f"{turn['generated_token_count_with_eos_if_emitted']} | "
                f"{'Y' if turn['stopped_on_eos'] else ''} | "
                f"`{', '.join(turn['parsed_tool_functions'])}` | "
                f"{turn['parse_error_count']} | "
                f"{turn['tool_execution_error_count']} | "
                f"{turn['generation_seconds']:.2f}s |"
            )
        lines.append("")
        if record["attribution"] is not None:
            lines.extend(
                [
                    "Generated-action localization:",
                    "",
                    "| Method | R@5% | R@10% | R@20% | Injection mass |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for method, values in record["attribution"]["methods"].items():
                metrics = values["localization_metrics"]
                lines.append(
                    f"| `{method}` | {metrics['recovery_at_5pct']:.3f} | "
                    f"{metrics['recovery_at_10pct']:.3f} | "
                    f"{metrics['recovery_at_20pct']:.3f} | "
                    f"{metrics['injection_mass_fraction']:.3f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_ATTRIBUTION_MODEL)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--num-samples", type=int, default=len(PILOT_CASES))
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--n-hops", type=int, default=3)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--sink-chunk-tokens", type=int, default=32)
    parser.add_argument("--no-attribution", action="store_true")
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(__file__).parent / ".cache" / "upstream"
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "e2e" / "qwen3_4b_thinking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.num_samples > len(PILOT_CASES):
        raise SystemExit(f"--num-samples must be in [1, {len(PILOT_CASES)}]")
    if args.max_turns < 1 or args.max_new_tokens < 1:
        raise SystemExit("--max-turns and --max-new-tokens must be positive")

    model, tokenizer = _load_model(args.model, args.device)
    cases = [
        prepare_case(spec, tokenizer, args.cache_dir, args.source_root)
        for spec in PILOT_CASES[: args.num_samples]
    ]
    ifr = None
    flashtrace = None
    if not args.no_attribution:
        ifr = LLMIFRAttribution(
            model,
            tokenizer,
            chunk_tokens=args.chunk_tokens,
            sink_chunk_tokens=args.sink_chunk_tokens,
            show_progress=False,
        )
        flashtrace = LLMIFRAttributionBoth(
            model,
            tokenizer,
            chunk_tokens=args.chunk_tokens,
            sink_chunk_tokens=args.sink_chunk_tokens,
            show_progress=False,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    jsonl_path = args.output_dir / "trajectories.jsonl"
    redacted_jsonl_path = args.output_dir / "trajectories.redacted.jsonl"
    with (
        jsonl_path.open("w", encoding="utf-8") as jsonl,
        redacted_jsonl_path.open("w", encoding="utf-8") as redacted_jsonl,
    ):
        for index, case in enumerate(cases, 1):
            print(f"[e2e {index}/{len(cases)}] {case.spec.case_id}", flush=True)
            record = _run_case_e2e(
                case,
                model,
                tokenizer,
                args.max_turns,
                args.max_new_tokens,
            )
            if record["target_event"] is not None and not args.no_attribution:
                assert ifr is not None and flashtrace is not None
                print("  [attribute] actual generated target action", flush=True)
                record["attribution"] = _attribute_generated_target(
                    record,
                    tokenizer,
                    ifr,
                    flashtrace,
                    args.n_hops,
                )
            serialized = _json(record)
            records.append(record)
            jsonl.write(serialized + "\n")
            redacted = _redacted_record(record)
            redacted_jsonl.write(_json(redacted) + "\n")
            (args.output_dir / f"{case.spec.case_id}.json").write_text(
                serialized + "\n",
                encoding="utf-8",
            )
            (args.output_dir / f"{case.spec.case_id}.redacted.json").write_text(
                _json(redacted, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"  utility={record['utility']} attack={record['attack_success']} "
                f"target={record['generated_expected_target_action']}",
                flush=True,
            )
    summary = _summary(records)
    summary["protocol"] = {
        "model": args.model,
        "agentdojo_package_version": AGENTDOJO_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "generation_mode": "autonomous_greedy",
        "teacher_forced": False,
        "max_turns": args.max_turns,
        "max_new_tokens_per_turn": args.max_new_tokens,
        "n_hops": args.n_hops,
    }
    (args.output_dir / "summary.json").write_text(
        _json(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "RESULTS.md", records, summary)
    print(f"[done] wrote {args.output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
