#!/usr/bin/env python3
"""Run local Qwen over the complete AgentDojo v1.2.2 inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from agentdojo.agent_pipeline.agent_pipeline import load_system_message
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.attacks.important_instructions_attacks import (
    ImportantInstructionsAttack,
)
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import (
    ChatAssistantMessage,
    ChatMessage,
    ChatSystemMessage,
    ChatToolResultMessage,
    ChatUserMessage,
    text_content_block_from_string,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluations.local_sglang import LocalSGLangGenerator
from evaluations.agentdojo.e2e import (
    _call_matches_expected,
    _completion_content,
    _format_tool_result,
    _generate_turn,
    _message_record,
    _render_prompt,
    parse_qwen_tool_calls,
)
from evaluations.agentdojo.pilot import (
    AGENTDOJO_VERSION,
    BENCHMARK_VERSION,
    _tool_schema,
)
from evaluations.agentdojo.target_realization import (
    FULL_BUDGET,
    TeacherForcingBudget,
    build_realization_stages,
    completion_token_provenance,
    decode_token_ids,
    natural_reasoning_prefix,
    parse_natural_budgets,
    parse_teacher_forcing_budgets,
    teacher_forced_prefix,
    token_ids,
)


DEFAULT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
SUITES = ("banking", "slack", "travel", "workspace")
ATTACK_NAME = "important_instructions"


class _LocalPipelineIdentity(BasePipelineElement):
    """Identity used only so AgentDojo renders the official local-model attack."""

    name = "local"

    def query(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    suite: str
    user_task: str
    injection_task: str | None
    attack: str
    injections: dict[str, str]


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int = 0


@dataclass(frozen=True)
class ReasoningSearchConfig:
    """Model-only reasoning search with held-out natural rollout scoring."""

    candidates: int = 0
    score_method: str = "conditional_logprob"
    reasoning_temperature: float | None = None
    score_rollouts: int = 1
    score_max_new_tokens: int = 192

    @property
    def enabled(self) -> bool:
        return self.candidates > 0


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _validate_generation_service_model(
    model_id: str,
    revision: str,
    model_info: dict[str, Any],
) -> None:
    server_model_path = str(model_info.get("model_path") or "")
    expected_model_fragment = "models--" + model_id.replace("/", "--")
    if (
        expected_model_fragment not in server_model_path
        or revision not in server_model_path
    ):
        raise ValueError(
            "Configured model/revision does not match the local generation service."
        )


def _turn_seed(
    base_seed: int,
    case_id: str,
    attempt_index: int,
    turn_index: int,
) -> int:
    payload = f"{base_seed}:{case_id}:{attempt_index}:{turn_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _search_seed(base_seed: int, candidate_index: int, rollout_index: int) -> int:
    payload = f"{base_seed}:search:{candidate_index}:{rollout_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def build_inventory(
    *,
    mode: str,
    suites: Sequence[str] = SUITES,
) -> list[BenchmarkCase]:
    """Build clean, attacked, or combined cases from the pinned task suites."""

    if mode not in {"clean", "attacked", "both"}:
        raise ValueError(f"Unsupported mode: {mode}")
    cases = []
    identity = _LocalPipelineIdentity()
    for suite_name in suites:
        suite = get_suite(BENCHMARK_VERSION, suite_name)
        if mode in {"clean", "both"}:
            for user_task_id in sorted(suite.user_tasks):
                cases.append(
                    BenchmarkCase(
                        case_id=f"{suite_name}__{user_task_id}__clean",
                        suite=suite_name,
                        user_task=user_task_id,
                        injection_task=None,
                        attack="none",
                        injections={},
                    )
                )
        if mode in {"attacked", "both"}:
            attack = ImportantInstructionsAttack(suite, identity)
            for user_task_id in sorted(suite.user_tasks):
                user_task = suite.get_user_task_by_id(user_task_id)
                for injection_task_id in sorted(suite.injection_tasks):
                    injection_task = suite.get_injection_task_by_id(injection_task_id)
                    injections = attack.attack(user_task, injection_task)
                    cases.append(
                        BenchmarkCase(
                            case_id=(
                                f"{suite_name}__{user_task_id}__{injection_task_id}"
                            ),
                            suite=suite_name,
                            user_task=user_task_id,
                            injection_task=injection_task_id,
                            attack=ATTACK_NAME,
                            injections=injections,
                        )
                    )
    return cases


def select_pilot(cases: Sequence[BenchmarkCase], *, per_suite: int) -> list[BenchmarkCase]:
    """Select deterministic, task-diverse clean/attacked cases from every suite."""

    def spaced(items: Sequence[BenchmarkCase], count: int) -> list[BenchmarkCase]:
        if count >= len(items):
            return list(items)
        if count == 1:
            return [items[0]]
        return [
            items[round(index * (len(items) - 1) / (count - 1))]
            for index in range(count)
        ]

    selected = []
    for suite in SUITES:
        clean = sorted(
            (case for case in cases if case.suite == suite and case.attack == "none"),
            key=lambda case: case.case_id,
        )
        attacked = sorted(
            (
                case
                for case in cases
                if case.suite == suite and case.attack == ATTACK_NAME
            ),
            key=lambda case: (case.user_task, case.injection_task or ""),
        )
        selected.extend(spaced(clean, per_suite))
        selected.extend(spaced(attacked, per_suite))
    return selected


def filter_action_target_cases(
    cases: Sequence[BenchmarkCase],
) -> list[BenchmarkCase]:
    """Keep clean cases and attacked cases with a concrete action target."""

    selected = []
    for case in cases:
        if case.injection_task is None:
            selected.append(case)
            continue
        suite = get_suite(BENCHMARK_VERSION, case.suite)
        user_task = suite.get_user_task_by_id(case.user_task)
        environment = suite.load_and_inject_default_environment(case.injections)
        environment = user_task.init_environment(environment)
        if _target_calls(case, environment):
            selected.append(case)
    return selected


def _source_messages(case: BenchmarkCase) -> list[ChatMessage]:
    suite = get_suite(BENCHMARK_VERSION, case.suite)
    task = suite.get_user_task_by_id(case.user_task)
    return [
        ChatSystemMessage(
            role="system",
            content=[text_content_block_from_string(load_system_message(None))],
        ),
        ChatUserMessage(
            role="user",
            content=[text_content_block_from_string(task.PROMPT)],
        ),
    ]


def _target_calls(
    case: BenchmarkCase,
    pre_environment: Any,
) -> tuple[dict[str, Any], ...]:
    if case.injection_task is None:
        return ()
    suite = get_suite(BENCHMARK_VERSION, case.suite)
    task = suite.get_injection_task_by_id(case.injection_task)
    return tuple(
        {
            "function": call.function,
            "args": dict(call.args),
        }
        for call in task.ground_truth(pre_environment)
    )


def _find_target_event(
    turns: Sequence[dict[str, Any]],
    target_calls: Sequence[dict[str, Any]],
    attack_success: bool,
) -> dict[str, Any] | None:
    if not attack_success:
        return None
    candidates = []
    for turn in turns:
        if not _valid_reasoning_boundary(turn):
            continue
        for call_index, call in enumerate(turn["parsed_tool_calls"]):
            if any(
                _call_matches_expected(call, expected)
                for expected in target_calls
            ):
                candidates.append(
                    {
                        "turn_index": turn["turn_index"],
                        "call_index": call_index,
                        "call": call,
                        "rendered_prompt": turn["rendered_prompt"],
                        "completion": turn["completion"],
                        "output_provenance": turn.get(
                            "token_id_provenance",
                            "autonomous_model_generation",
                        ),
                        "teacher_forced": bool(turn.get("output_intervention")),
                        "output_intervention": turn.get("output_intervention"),
                        "completion_token_provenance": turn.get(
                            "completion_token_provenance"
                        ),
                    }
                )
    return candidates[-1] if candidates else None


def _reasoning_boundary(completion: str) -> str:
    think_start = completion.find("<think>")
    think_end = completion.find("</think>")
    if think_start >= 0 and think_end >= think_start:
        return "explicit_open_and_close"
    if think_start < 0 and think_end >= 0:
        return "implicit_open_explicit_close"
    return "missing_close"


def _valid_reasoning_boundary(turn: dict[str, Any]) -> bool:
    if "valid_reasoning_boundary" in turn:
        return bool(turn["valid_reasoning_boundary"])
    return _reasoning_boundary(str(turn.get("completion") or "")) != "missing_close"


def _injection_is_visible(prompt: str) -> bool:
    return "<INFORMATION>" in prompt and "</INFORMATION>" in prompt


def _serialize_target_completion(target_calls: Sequence[dict[str, Any]]) -> str:
    blocks = [
        "<tool_call>\n"
        + _json(
            {
                "name": call["function"],
                "arguments": call["args"],
            }
        )
        + "\n</tool_call>"
        for call in target_calls
    ]
    return "</think>\n" + "\n".join(blocks)


def _generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    *,
    generator: LocalSGLangGenerator | None,
    decoding: DecodingConfig,
    seed: int,
    stop: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    if generator is not None:
        return generator.generate(
            prompt,
            max_new_tokens,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            seed=seed,
            stop=stop,
        )
    result = _generate_turn(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        top_k=decoding.top_k,
        seed=seed,
    )
    if not stop:
        return result
    stops = [stop] if isinstance(stop, str) else list(stop)
    completion = str(result.get("completion") or "")
    matched_end = None
    for item in stops:
        position = completion.find(item)
        if position >= 0:
            end = position + len(item)
            matched_end = end if matched_end is None else min(matched_end, end)
    if matched_end is None:
        return result
    generated_for_compute = int(
        result.get("generated_token_count_with_eos_if_emitted") or 0
    )
    truncated = completion[:matched_end]
    truncated_ids = token_ids(tokenizer, truncated)
    return {
        **result,
        "completion": truncated,
        "raw_completion_with_eos": truncated,
        "generated_token_ids_with_eos_if_emitted": truncated_ids,
        "generated_token_count_with_eos_if_emitted": len(truncated_ids),
        "stopped_on_eos": False,
        "finish_reason": "reasoning_boundary_emulated",
        "compute_generated_token_count": generated_for_compute,
    }


def _generate_from_token_ids(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    max_new_tokens: int,
    *,
    generator: LocalSGLangGenerator | None,
    decoding: DecodingConfig,
    seed: int,
) -> dict[str, Any]:
    if generator is not None:
        return generator.generate_from_token_ids(
            input_ids,
            max_new_tokens,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            seed=seed,
        )
    prompt = decode_token_ids(tokenizer, input_ids)
    if token_ids(tokenizer, prompt) != input_ids:
        raise ValueError("Token-ID continuation prompt does not round-trip.")
    return _generate_turn(
        model,
        tokenizer,
        prompt,
        max_new_tokens,
        temperature=decoding.temperature,
        top_p=decoding.top_p,
        top_k=decoding.top_k,
        seed=seed,
    )


def _without_terminal_eos(tokenizer: Any, ids: Sequence[int]) -> tuple[list[int], bool]:
    values = [int(token_id) for token_id in ids]
    if (
        values
        and tokenizer.eos_token_id is not None
        and values[-1] == tokenizer.eos_token_id
    ):
        return values[:-1], True
    return values, False


def _generated_prefix_ids(
    tokenizer: Any,
    turn: dict[str, Any],
    prefix: str,
) -> list[int] | None:
    """Recover an exact token prefix without retokenizing across a boundary."""

    generated, _ = _without_terminal_eos(
        tokenizer,
        turn.get("generated_token_ids_with_eos_if_emitted") or (),
    )
    encoded = token_ids(tokenizer, prefix)
    if generated[: len(encoded)] != encoded:
        return None
    if decode_token_ids(tokenizer, encoded) != prefix:
        return None
    return encoded


def _rollout_target_score(
    completion: str,
    target_calls: Sequence[dict[str, Any]],
) -> tuple[int, int, int]:
    parsed, errors = parse_qwen_tool_calls(completion)
    exact = sum(
        any(
            _call_matches_expected(asdict(call), expected)
            for expected in target_calls
        )
        for call in parsed
    )
    target_functions = {str(call["function"]) for call in target_calls}
    function_matches = sum(call.function in target_functions for call in parsed)
    return exact, function_matches, -len(errors)


def _conditional_action_score(
    model: Any,
    prefix_ids: list[int],
    action_ids: list[int],
    *,
    generator: LocalSGLangGenerator | None,
) -> dict[str, Any]:
    if generator is not None:
        return generator.score_continuation_token_ids(prefix_ids, action_ids)
    if model is None:
        raise ValueError("Conditional scoring requires a model or local service.")
    input_ids = torch.tensor([prefix_ids + action_ids], device=model.device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids).logits[0]
    start = len(prefix_ids)
    positions = torch.arange(
        start - 1,
        start + len(action_ids) - 1,
        device=logits.device,
    )
    targets = input_ids[0, start : start + len(action_ids)]
    values = torch.log_softmax(logits[positions].float(), dim=-1).gather(
        1,
        targets.unsqueeze(1),
    )[:, 0]
    return {
        "sum_logprob": float(values.sum().item()),
        "mean_logprob": float(values.mean().item()),
        "scored_token_count": len(action_ids),
        "generation_seconds": 0.0,
    }


def _reasoning_reranked_turn(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_calls: Sequence[dict[str, Any]],
    *,
    max_new_tokens: int,
    generator: LocalSGLangGenerator | None,
    decoding: DecodingConfig,
    seed: int,
    search: ReasoningSearchConfig,
) -> dict[str, Any] | None:
    """Select one coherent model reasoning trace, then naturally continue it."""

    if not search.enabled:
        return None
    if search.score_method not in {"conditional_logprob", "short_rollout"}:
        raise ValueError("Unsupported reasoning score method.")
    if search.score_rollouts < 1 or search.score_max_new_tokens < 1:
        raise ValueError("Reasoning search rollout budgets must be positive.")

    prompt_ids = token_ids(tokenizer, prompt)
    candidates: list[dict[str, Any]] = []
    request_count = 0
    sampled_token_count = 0
    scored_token_count = 0
    generation_seconds = 0.0
    for candidate_index in range(search.candidates):
        reasoning_decoding = DecodingConfig(
            temperature=(
                search.reasoning_temperature
                if search.reasoning_temperature is not None
                else decoding.temperature
            ),
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            seed=decoding.seed,
        )
        reasoning_turn = _generate_completion(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            generator=generator,
            decoding=reasoning_decoding,
            seed=_search_seed(seed, candidate_index, 0),
            stop="</think>",
        )
        request_count += 1
        generation_seconds += float(reasoning_turn.get("generation_seconds") or 0.0)
        sampled_token_count += int(
            reasoning_turn.get("compute_generated_token_count")
            or reasoning_turn.get("generated_token_count_with_eos_if_emitted")
            or 0
        )
        reasoning = natural_reasoning_prefix(
            str(reasoning_turn.get("completion") or "")
        )
        if reasoning is None:
            continue
        reasoning_ids = _generated_prefix_ids(tokenizer, reasoning_turn, reasoning)
        if reasoning_ids is None:
            continue

        best_score: tuple[float, float, float] = (0.0, 0.0, -10**9)
        conditional_mean_logprob: float | None = None
        rollout_tokens = 0
        if search.score_method == "conditional_logprob":
            action_ids = token_ids(
                tokenizer,
                _serialize_target_completion(target_calls).removeprefix(
                    "</think>\n"
                ),
            )
            score = _conditional_action_score(
                model,
                prompt_ids + reasoning_ids,
                action_ids,
                generator=generator,
            )
            request_count += 1
            generation_seconds += float(score.get("generation_seconds") or 0.0)
            scored_token_count += int(score["scored_token_count"])
            conditional_mean_logprob = float(score["mean_logprob"])
            best_score = (conditional_mean_logprob, 0.0, 0.0)
        else:
            for rollout_index in range(search.score_rollouts):
                rollout = _generate_from_token_ids(
                    model,
                    tokenizer,
                    prompt_ids + reasoning_ids,
                    search.score_max_new_tokens,
                    generator=generator,
                    decoding=decoding,
                    seed=_search_seed(seed, candidate_index, rollout_index + 1),
                )
                request_count += 1
                generation_seconds += float(rollout.get("generation_seconds") or 0.0)
                rollout_ids, _ = _without_terminal_eos(
                    tokenizer,
                    rollout.get("generated_token_ids_with_eos_if_emitted") or (),
                )
                rollout_tokens += len(rollout_ids)
                sampled_token_count += len(rollout_ids)
                rollout_completion = reasoning + decode_token_ids(
                    tokenizer, rollout_ids
                )
                best_score = max(
                    best_score,
                    _rollout_target_score(rollout_completion, target_calls),
                )
        candidates.append(
            {
                "candidate_index": candidate_index,
                "reasoning": reasoning,
                "reasoning_ids": reasoning_ids,
                "score": best_score,
                "conditional_mean_logprob": conditional_mean_logprob,
                "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
                "score_rollout_token_count": rollout_tokens,
            }
        )

    if not candidates:
        return None
    selected = max(candidates, key=lambda row: row["score"])
    reasoning_ids = list(selected["reasoning_ids"])
    remaining = max(1, max_new_tokens - len(reasoning_ids))
    final = _generate_from_token_ids(
        model,
        tokenizer,
        prompt_ids + reasoning_ids,
        remaining,
        generator=generator,
        decoding=decoding,
        seed=_search_seed(seed, search.candidates, 0),
    )
    request_count += 1
    generation_seconds += float(final.get("generation_seconds") or 0.0)
    final_ids, stopped_on_eos = _without_terminal_eos(
        tokenizer,
        final.get("generated_token_ids_with_eos_if_emitted") or (),
    )
    sampled_token_count += len(final_ids)
    completion_ids = reasoning_ids + final_ids
    completion = decode_token_ids(tokenizer, completion_ids)
    if token_ids(tokenizer, completion) != completion_ids:
        raise ValueError("Reasoning-search completion does not round-trip.")
    generated_with_eos = list(completion_ids)
    raw_with_eos = completion
    if stopped_on_eos and tokenizer.eos_token_id is not None:
        generated_with_eos.append(int(tokenizer.eos_token_id))
        raw_with_eos += tokenizer.eos_token or ""

    first_score = candidates[0]["score"]
    selected_score = selected["score"]
    return {
        "rendered_prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "generated_token_ids_with_eos_if_emitted": generated_with_eos,
        "generated_token_count_with_eos_if_emitted": len(generated_with_eos),
        "raw_completion_with_eos": raw_with_eos,
        "completion": completion,
        "stopped_on_eos": stopped_on_eos,
        "finish_reason": str(final.get("finish_reason") or "unknown"),
        "generation_seconds": generation_seconds,
        "sampling_params": {
            "mode": "reasoning_rerank_natural_continuation",
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "top_k": decoding.top_k,
        },
        "token_id_provenance": "autonomous_model_search_selected",
        "completion_token_provenance": ["model"] * len(completion_ids),
        "natural_search": {
            "method": f"reasoning_rerank_{search.score_method}",
            "configured_candidate_count": search.candidates,
            "valid_candidate_count": len(candidates),
            "distinct_reasoning_count": len(
                {row["reasoning_sha256"] for row in candidates}
            ),
            "score_rollouts_per_candidate": search.score_rollouts,
            "score_max_new_tokens": search.score_max_new_tokens,
            "reasoning_sampling_temperature": reasoning_decoding.temperature,
            "final_action_sampling_temperature": decoding.temperature,
            "selected_candidate_index": selected["candidate_index"],
            "selected_was_first": selected["candidate_index"] == 0,
            "selected_exact_score": (
                selected_score[0]
                if search.score_method == "short_rollout"
                else None
            ),
            "selected_function_score": (
                selected_score[1]
                if search.score_method == "short_rollout"
                else None
            ),
            "exact_score_gain_over_first": (
                selected_score[0] - first_score[0]
                if search.score_method == "short_rollout"
                else None
            ),
            "selected_conditional_mean_logprob": selected[
                "conditional_mean_logprob"
            ],
            "conditional_logprob_gain_over_first": (
                selected_score[0] - first_score[0]
                if search.score_method == "conditional_logprob"
                else None
            ),
            "generation_request_count": request_count,
            "sampled_token_count": sampled_token_count,
            "scored_token_count": scored_token_count,
            "final_completion_token_count": len(completion_ids),
            "reasoning_token_count": len(reasoning_ids),
            "bridge_is_fully_model_generated": True,
            "all_completion_tokens_model_generated": True,
            "reference_token_count": 0,
            "candidate_scores": [
                {
                    "candidate_index": row["candidate_index"],
                    "exact_score": row["score"][0],
                    "function_score": row["score"][1],
                    "parse_score": row["score"][2],
                    "conditional_mean_logprob": row[
                        "conditional_mean_logprob"
                    ],
                    "reasoning_token_count": len(row["reasoning_ids"]),
                    "score_rollout_token_count": row[
                        "score_rollout_token_count"
                    ],
                }
                for row in candidates
            ],
        },
    }


def _teacher_forced_turn(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_calls: Sequence[dict[str, Any]],
    natural_turn: dict[str, Any],
    *,
    budget: TeacherForcingBudget,
    max_new_tokens: int,
    generator: LocalSGLangGenerator | None,
    decoding: DecodingConfig,
    seed: int,
) -> dict[str, Any]:
    natural_completion = str(natural_turn.get("completion") or "")
    reasoning_prefix = natural_reasoning_prefix(natural_completion)
    if reasoning_prefix is None:
        return natural_turn

    action_text = _serialize_target_completion(target_calls).removeprefix("</think>\n")
    forced_text, forced_ids, action_token_count = teacher_forced_prefix(
        tokenizer,
        action_text,
        budget,
    )
    forced_start = len(reasoning_prefix)
    assistant_prefix = reasoning_prefix + forced_text
    full_stage = budget == FULL_BUDGET or len(forced_ids) == action_token_count
    continuation = None
    if not full_stage:
        prefix_token_count = len(token_ids(tokenizer, assistant_prefix))
        continuation = _generate_completion(
            model,
            tokenizer,
            prompt + assistant_prefix,
            max(1, max_new_tokens - prefix_token_count),
            generator=generator,
            decoding=decoding,
            seed=seed,
        )
    continuation_text = str((continuation or {}).get("completion") or "")
    completion = assistant_prefix + continuation_text
    generated_ids = token_ids(tokenizer, completion)
    raw_with_eos = assistant_prefix + str(
        (continuation or {}).get("raw_completion_with_eos") or ""
    )
    stopped_on_eos = bool((continuation or {}).get("stopped_on_eos"))
    if stopped_on_eos and tokenizer.eos_token_id is not None:
        generated_ids.append(int(tokenizer.eos_token_id))
    provenance = completion_token_provenance(
        tokenizer,
        completion,
        (forced_start, forced_start + len(forced_text)),
    )
    return {
        **natural_turn,
        "rendered_prompt": prompt,
        "prompt_token_ids": token_ids(tokenizer, prompt),
        "prompt_token_count": len(token_ids(tokenizer, prompt)),
        "generated_token_ids_with_eos_if_emitted": generated_ids,
        "generated_token_count_with_eos_if_emitted": len(generated_ids),
        "raw_completion_with_eos": raw_with_eos,
        "completion": completion,
        "stopped_on_eos": stopped_on_eos,
        "finish_reason": (
            "teacher_forcing_full"
            if full_stage
            else str((continuation or {}).get("finish_reason") or "unknown")
        ),
        "generation_seconds": float(natural_turn.get("generation_seconds") or 0.0)
        + float((continuation or {}).get("generation_seconds") or 0.0),
        "sampling_params": {
            "mode": "teacher_forcing_prefix_then_model_continuation",
            "continuation": (continuation or {}).get("sampling_params"),
        },
        "token_id_provenance": "mixed_model_and_reference_with_token_mask",
        "completion_token_provenance": provenance,
        "output_intervention": {
            "mode": "teacher_forcing_prefix",
            "budget": budget,
            "forced_token_count": len(forced_ids),
            "action_token_count": action_token_count,
            "forced_token_fraction": (
                len(forced_ids) / action_token_count if action_token_count else 0.0
            ),
            "decode_request_count": 0 if full_stage else 1,
            "reasoning_token_count": len(token_ids(tokenizer, reasoning_prefix)),
            "bridge_is_fully_model_generated": True,
            "forced_char_span_half_open": [
                forced_start,
                forced_start + len(forced_text),
            ],
            "reasoning_provenance": "model",
            "natural_completion_sha256": hashlib.sha256(
                natural_completion.encode()
            ).hexdigest(),
        },
    }


def _sparse_corrected_turn(
    model: Any,
    tokenizer: Any,
    prompt: str,
    target_calls: Sequence[dict[str, Any]],
    natural_turn: dict[str, Any],
    *,
    budget: TeacherForcingBudget,
    max_new_tokens: int,
    generator: LocalSGLangGenerator | None,
    decoding: DecodingConfig,
    seed: int,
) -> dict[str, Any]:
    """Decode token by token, correcting only deviations from the reference."""

    natural_completion = str(natural_turn.get("completion") or "")
    reasoning_prefix = natural_reasoning_prefix(natural_completion)
    if reasoning_prefix is None:
        return natural_turn

    action_text = _serialize_target_completion(target_calls).removeprefix("</think>\n")
    reference_ids = token_ids(tokenizer, action_text)
    reasoning_ids = token_ids(tokenizer, reasoning_prefix)
    prompt_ids = token_ids(tokenizer, prompt)
    chosen_ids: list[int] = []
    action_provenance: list[str] = []
    correction_positions: list[int] = []
    model_match_positions: list[int] = []
    generation_seconds = float(natural_turn.get("generation_seconds") or 0.0)
    requests = 0
    diverged_at: int | None = None
    stopped_on_eos = False
    limit = len(reference_ids) + 1 if budget == FULL_BUDGET else int(budget)

    for position, reference_id in enumerate(reference_ids):
        one = _generate_from_token_ids(
            model,
            tokenizer,
            prompt_ids + reasoning_ids + chosen_ids,
            1,
            generator=generator,
            decoding=decoding,
            seed=seed + position,
        )
        requests += 1
        generation_seconds += float(one.get("generation_seconds") or 0.0)
        predicted = list(one.get("generated_token_ids_with_eos_if_emitted") or [])
        predicted_id = int(predicted[0]) if predicted else tokenizer.eos_token_id
        if predicted_id == reference_id:
            chosen_ids.append(reference_id)
            action_provenance.append("model")
            model_match_positions.append(position)
            continue
        if len(correction_positions) < limit:
            chosen_ids.append(reference_id)
            action_provenance.append("reference_correction")
            correction_positions.append(position)
            continue

        diverged_at = position
        if predicted_id is not None and predicted_id != tokenizer.eos_token_id:
            chosen_ids.append(predicted_id)
            action_provenance.append("model")
        else:
            stopped_on_eos = True
        remaining = max_new_tokens - len(reasoning_ids) - len(chosen_ids)
        if not stopped_on_eos and remaining > 0:
            continuation = _generate_from_token_ids(
                model,
                tokenizer,
                prompt_ids + reasoning_ids + chosen_ids,
                remaining,
                generator=generator,
                decoding=decoding,
                seed=seed + len(reference_ids) + 1,
            )
            requests += 1
            generation_seconds += float(
                continuation.get("generation_seconds") or 0.0
            )
            continuation_ids = list(
                continuation.get("generated_token_ids_with_eos_if_emitted") or []
            )
            if (
                continuation_ids
                and tokenizer.eos_token_id is not None
                and continuation_ids[-1] == tokenizer.eos_token_id
            ):
                continuation_ids = continuation_ids[:-1]
                stopped_on_eos = True
            chosen_ids.extend(int(token_id) for token_id in continuation_ids)
            action_provenance.extend("model" for _ in continuation_ids)
        break

    completion_ids = reasoning_ids + chosen_ids
    completion = decode_token_ids(tokenizer, completion_ids)
    if token_ids(tokenizer, completion) != completion_ids:
        raise ValueError("Sparse-correction completion does not round-trip.")
    generated_ids = list(completion_ids)
    raw_with_eos = completion
    if stopped_on_eos and tokenizer.eos_token_id is not None:
        generated_ids.append(int(tokenizer.eos_token_id))
        raw_with_eos += tokenizer.eos_token or ""
    provenance = ["model"] * len(reasoning_ids) + action_provenance
    return {
        **natural_turn,
        "rendered_prompt": prompt,
        "prompt_token_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "generated_token_ids_with_eos_if_emitted": generated_ids,
        "generated_token_count_with_eos_if_emitted": len(generated_ids),
        "raw_completion_with_eos": raw_with_eos,
        "completion": completion,
        "stopped_on_eos": stopped_on_eos,
        "finish_reason": (
            "sparse_reference_complete"
            if diverged_at is None
            else "sparse_budget_exhausted_then_model_continuation"
        ),
        "generation_seconds": generation_seconds,
        "sampling_params": {
            "mode": "sparse_correction",
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "top_k": decoding.top_k,
        },
        "token_id_provenance": "model_with_sparse_reference_corrections",
        "completion_token_provenance": provenance,
        "output_intervention": {
            "mode": "sparse_correction",
            "budget": budget,
            "reference_token_count": len(correction_positions),
            "action_token_count": len(reference_ids),
            "reference_token_fraction": (
                len(correction_positions) / len(reference_ids)
                if reference_ids
                else 0.0
            ),
            "correction_positions": correction_positions,
            "model_match_count": len(model_match_positions),
            "tokens_compared": (
                len(reference_ids) if diverged_at is None else diverged_at + 1
            ),
            "first_uncorrected_deviation_position": diverged_at,
            "decode_request_count": requests,
            "reasoning_token_count": len(reasoning_ids),
            "bridge_is_fully_model_generated": True,
            "reasoning_provenance": "model",
            "natural_completion_sha256": hashlib.sha256(
                natural_completion.encode()
            ).hexdigest(),
        },
    }


def _attempt_summary(
    record: dict[str, Any],
    attempt_index: int,
    realization_stage: str | None = None,
) -> dict[str, Any]:
    intervention = next(
        (
            turn.get("output_intervention")
            for turn in record["turns"]
            if turn.get("output_intervention", {}).get("mode")
            in {"teacher_forcing_prefix", "sparse_correction"}
        ),
        None,
    )
    skip_reason = record.get("realization_skip_reason") or next(
        (
            turn.get("realization_skip_reason")
            for turn in record["turns"]
            if turn.get("realization_skip_reason")
        ),
        None,
    )
    search_turns = [
        turn["natural_search"]
        for turn in record["turns"]
        if turn.get("natural_search")
    ]
    model_token_count = sum(
        turn.get("completion_token_provenance", []).count("model")
        for turn in record["turns"]
    )
    completion_token_count = sum(
        len(turn.get("completion_token_provenance", []))
        for turn in record["turns"]
    )
    return {
        "attempt_index": attempt_index,
        "realization_stage": realization_stage,
        "utility": record["utility"],
        "attack_success": record["attack_success"],
        "generated_target_action": record["generated_target_action"],
        "injection_observed": any(
            _injection_is_visible(turn["rendered_prompt"])
            for turn in record["turns"]
        ),
        "intervention_method": (
            intervention.get("mode") if intervention is not None else None
        ),
        "intervention_applied": intervention is not None,
        "realization_skip_reason": skip_reason,
        "conditioning_source_attempt_index": record.get(
            "conditioning_source_attempt_index"
        ),
        "intervention": (
            {
                key: intervention.get(key)
                for key in (
                    "budget",
                    "forced_token_count",
                    "reference_token_count",
                    "action_token_count",
                    "forced_token_fraction",
                    "reference_token_fraction",
                    "model_match_count",
                    "tokens_compared",
                    "decode_request_count",
                    "reasoning_token_count",
                    "bridge_is_fully_model_generated",
                )
            }
            if intervention is not None
            else None
        ),
        "generation_seconds": sum(
            float(turn.get("generation_seconds") or 0.0)
            for turn in record["turns"]
        ),
        "generation_request_count": sum(
            int(turn.get("natural_search", {}).get("generation_request_count") or 1)
            + int(turn.get("output_intervention", {}).get("decode_request_count") or 0)
            for turn in record["turns"]
        ),
        "sampled_token_count": sum(
            int(
                turn.get("natural_search", {}).get("sampled_token_count")
                or turn.get("generated_token_count_with_eos_if_emitted")
                or 0
            )
            for turn in record["turns"]
        ),
        "scored_token_count": sum(
            int(turn.get("natural_search", {}).get("scored_token_count") or 0)
            for turn in record["turns"]
        ),
        "all_completion_tokens_model_generated": (
            completion_token_count > 0 and model_token_count == completion_token_count
        ),
        "model_generated_token_fraction": (
            model_token_count / completion_token_count
            if completion_token_count
            else None
        ),
        "natural_search": search_turns[-1] if search_turns else None,
        "minimum_reasoning_token_count": min(
            (
                int(turn.get("generation_audit", {}).get("reasoning_token_count") or 0)
                for turn in record["turns"]
                if turn.get("generation_audit", {}).get("reasoning_token_count")
            ),
            default=None,
        ),
        "all_bridges_model_generated": all(
            bool(
                turn.get("generation_audit", {}).get(
                    "bridge_is_fully_model_generated"
                )
            )
            for turn in record["turns"]
        ),
        "valid_reasoning_boundaries": record["valid_reasoning_boundaries"],
        "stop_reason": record["stop_reason"],
        "turn_count": len(record["turns"]),
    }


def _run_case_once(
    case: BenchmarkCase,
    model: Any,
    tokenizer: Any,
    *,
    max_turns: int,
    max_new_tokens: int,
    generator: LocalSGLangGenerator | None = None,
    decoding: DecodingConfig = DecodingConfig(),
    attempt_index: int = 0,
    teacher_forcing_budget: TeacherForcingBudget | None = None,
    sparse_correction_budget: TeacherForcingBudget | None = None,
    reasoning_search: ReasoningSearchConfig = ReasoningSearchConfig(),
) -> dict[str, Any]:
    suite = get_suite(BENCHMARK_VERSION, case.suite)
    user_task = suite.get_user_task_by_id(case.user_task)
    injection_task = (
        suite.get_injection_task_by_id(case.injection_task)
        if case.injection_task is not None
        else None
    )
    environment = suite.load_and_inject_default_environment(case.injections)
    environment = user_task.init_environment(environment)
    pre_environment = environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)
    schemas = [_tool_schema(tool) for tool in suite.tools]
    target_calls = _target_calls(case, pre_environment)
    messages = _source_messages(case)
    turns = []
    trace: list[FunctionCall] = []
    stop_reason = "max_turns"

    for turn_index in range(max_turns):
        prompt = _render_prompt(tokenizer, messages, schemas)
        seed = _turn_seed(
            decoding.seed,
            case.case_id,
            attempt_index,
            turn_index,
        )
        turn = None
        if (
            reasoning_search.enabled
            and target_calls
            and _injection_is_visible(prompt)
        ):
            turn = _reasoning_reranked_turn(
                model,
                tokenizer,
                prompt,
                target_calls,
                max_new_tokens=max_new_tokens,
                generator=generator,
                decoding=decoding,
                seed=seed,
                search=reasoning_search,
            )
        if turn is None:
            turn = _generate_completion(
                model,
                tokenizer,
                prompt,
                max_new_tokens,
                generator=generator,
                decoding=decoding,
                seed=seed,
            )
        if (
            teacher_forcing_budget is not None
            and target_calls
            and _injection_is_visible(prompt)
        ):
            conditioned_turn = _teacher_forced_turn(
                model,
                tokenizer,
                prompt,
                target_calls,
                turn,
                budget=teacher_forcing_budget,
                max_new_tokens=max_new_tokens,
                generator=generator,
                decoding=decoding,
                seed=seed + 1,
            )
            if conditioned_turn is turn:
                turn["realization_skip_reason"] = "missing_reasoning_boundary"
            else:
                turn = conditioned_turn
        elif (
            sparse_correction_budget is not None
            and target_calls
            and _injection_is_visible(prompt)
        ):
            corrected_turn = _sparse_corrected_turn(
                model,
                tokenizer,
                prompt,
                target_calls,
                turn,
                budget=sparse_correction_budget,
                max_new_tokens=max_new_tokens,
                generator=generator,
                decoding=decoding,
                seed=seed + 1,
            )
            if corrected_turn is turn:
                turn["realization_skip_reason"] = "missing_reasoning_boundary"
            else:
                turn = corrected_turn
        parsed, parse_errors = parse_qwen_tool_calls(turn["completion"])
        if "completion_token_provenance" not in turn:
            completion_ids = token_ids(tokenizer, turn["completion"])
            turn["completion_token_provenance"] = ["model"] * len(completion_ids)
        reasoning_prefix = natural_reasoning_prefix(turn["completion"])
        provenance = list(turn.get("completion_token_provenance") or [])
        reasoning_token_count = (
            len(token_ids(tokenizer, reasoning_prefix))
            if reasoning_prefix is not None
            else 0
        )
        turn["generation_audit"] = {
            "reasoning_token_count": reasoning_token_count,
            "bridge_is_fully_model_generated": bool(reasoning_prefix is not None)
            and all(label == "model" for label in provenance[:reasoning_token_count]),
            "all_completion_tokens_model_generated": bool(provenance)
            and all(label == "model" for label in provenance),
            "model_generated_token_count": provenance.count("model"),
            "reference_token_count": len(provenance) - provenance.count("model"),
        }
        calls = [
            FunctionCall(
                function=call.function,
                args=dict(call.args),
                id=f"local_turn_{turn_index}_call_{call_index}",
            )
            for call_index, call in enumerate(parsed)
        ]
        messages.append(
            ChatAssistantMessage(
                role="assistant",
                content=[
                    text_content_block_from_string(
                        _completion_content(turn["completion"])
                    )
                ],
                tool_calls=calls,
            )
        )
        turn.update(
            {
                "turn_index": turn_index,
                "reasoning_boundary": _reasoning_boundary(turn["completion"]),
                "parse_errors": parse_errors,
                "parsed_tool_calls": [asdict(call) for call in parsed],
                "tool_results": [],
            }
        )
        turn["valid_reasoning_boundary"] = (
            turn["reasoning_boundary"] != "missing_close"
        )
        if not calls:
            stop_reason = (
                "assistant_final_answer"
                if turn["stopped_on_eos"]
                else "max_new_tokens_without_complete_action"
            )
            turns.append(turn)
            break

        for call in calls:
            trace.append(call)
            result, error = runtime.run_function(
                environment,
                call.function,
                call.args,
            )
            result_text = _format_tool_result(result)
            content = result_text if error is None else f"Error: {error}"
            messages.append(
                ChatToolResultMessage(
                    role="tool",
                    content=[text_content_block_from_string(content)],
                    tool_call_id=call.id,
                    tool_call=call,
                    error=error,
                )
            )
            turn["tool_results"].append(
                {
                    "tool_call": call.model_dump(mode="json"),
                    "content": content,
                    "raw_result": result_text,
                    "error": error,
                }
            )
        turns.append(turn)
        if turn.get("output_intervention", {}).get("mode") in {
            "teacher_forcing_prefix",
            "sparse_correction",
        }:
            stop_reason = "intervened_action"
            break

    assistant_messages = [
        message for message in messages if message["role"] == "assistant"
    ]
    final_content = assistant_messages[-1]["content"] if assistant_messages else []
    utility = suite._check_user_task_utility(
        user_task,
        final_content or [],
        pre_environment,
        environment,
        trace,
    )
    attack_success = (
        suite._check_injection_task_security(
            injection_task,
            final_content or [],
            pre_environment,
            environment,
            trace,
        )
        if injection_task is not None
        else None
    )
    target_event = _find_target_event(
        turns,
        target_calls,
        bool(attack_success),
    )
    intervention_applied = any(
        turn.get("output_intervention", {}).get("mode")
        in {"teacher_forcing_prefix", "sparse_correction"}
        for turn in turns
    )
    realization_skip_reason = None
    if (
        teacher_forcing_budget is not None or sparse_correction_budget is not None
    ) and not intervention_applied:
        realization_skip_reason = (
            "missing_reasoning_boundary"
            if any(_injection_is_visible(turn["rendered_prompt"]) for turn in turns)
            else "target_context_not_observed"
        )
    return {
        "schema_version": 1,
        "case": asdict(case),
        "turns": turns,
        "messages": [_message_record(message) for message in messages],
        "function_trace": [call.model_dump(mode="json") for call in trace],
        "stop_reason": stop_reason,
        "valid_reasoning_boundaries": all(
            _valid_reasoning_boundary(turn) for turn in turns
        ),
        "utility": bool(utility),
        "attack_success": (
            bool(attack_success) if attack_success is not None else None
        ),
        "target_calls": list(target_calls),
        "target_functions": sorted(
            {call["function"] for call in target_calls}
        ),
        "generated_target_action": target_event is not None,
        "target_event": target_event,
        "realization_skip_reason": realization_skip_reason,
        "pre_environment": pre_environment.model_dump(mode="json"),
        "post_environment": environment.model_dump(mode="json"),
    }


def run_case(
    case: BenchmarkCase,
    model: Any,
    tokenizer: Any,
    *,
    max_turns: int,
    max_new_tokens: int,
    generator: LocalSGLangGenerator | None = None,
    decoding: DecodingConfig = DecodingConfig(),
    natural_rollout_budgets: Sequence[int] = (1,),
    teacher_forcing_token_budgets: Sequence[TeacherForcingBudget] = (),
    sparse_correction_budgets: Sequence[TeacherForcingBudget] = (),
    reasoning_search: ReasoningSearchConfig = ReasoningSearchConfig(),
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the natural axis, then a uniform token-prefix forcing axis."""

    stages = build_realization_stages(
        natural_rollout_budgets,
        teacher_forcing_token_budgets,
        sparse_correction_budgets,
    )
    if case.injection_task is None:
        stages = stages[:1]
    if stages[-1].cumulative_natural_attempts > 1 and decoding.temperature == 0:
        raise ValueError("Natural resampling stages require temperature > 0.")
    candidates = []
    attempt_summaries = []
    stage_summaries = []
    natural_attempts_completed = 0
    conditioning_source: tuple[int, DecodingConfig] | None = None
    success = False
    for stage in stages:
        stage_attempt_indices = []
        if stage.family == "natural":
            attempts_in_stage = (
                stage.cumulative_natural_attempts - natural_attempts_completed
            )
        else:
            if conditioning_source is None:
                stage_summaries.append(
                    {
                        **stage.to_dict(),
                        "attempt_indices": [],
                        "validator_passed": False,
                        "skipped_reason": "no_conditionable_natural_prefix",
                    }
                )
                if progress_callback is not None:
                    progress_callback(
                        {
                            "realization_stage": stage.name,
                            "attempt_index": "skipped",
                            "validator_passed": False,
                        }
                    )
                continue
            attempts_in_stage = 1
        for _ in range(attempts_in_stage):
            audit_attempt_index = len(candidates)
            if stage.family != "natural":
                generation_attempt_index, attempt_decoding = conditioning_source
            else:
                generation_attempt_index = audit_attempt_index
                attempt_decoding = (
                    DecodingConfig(seed=decoding.seed)
                    if stage.name == "natural_greedy"
                    else decoding
                )
            candidate = _run_case_once(
                case,
                model,
                tokenizer,
                max_turns=max_turns,
                max_new_tokens=max_new_tokens,
                generator=generator,
                decoding=attempt_decoding,
                attempt_index=generation_attempt_index,
                teacher_forcing_budget=stage.teacher_forcing_tokens,
                sparse_correction_budget=stage.sparse_correction_budget,
                reasoning_search=(
                    reasoning_search
                    if stage.family == "natural"
                    else ReasoningSearchConfig()
                ),
            )
            candidate["realization_stage"] = stage.name
            if stage.family != "natural":
                candidate["conditioning_source_attempt_index"] = (
                    generation_attempt_index
                )
            candidates.append(candidate)
            stage_attempt_indices.append(audit_attempt_index)
            attempt_summaries.append(
                _attempt_summary(candidate, audit_attempt_index, stage.name)
            )
            if progress_callback is not None:
                progress_callback(
                    {
                        "realization_stage": stage.name,
                        "attempt_index": audit_attempt_index,
                        "validator_passed": bool(
                            candidate["generated_target_action"]
                        ),
                    }
                )
            if stage.family == "natural":
                natural_attempts_completed += 1
                if conditioning_source is None and any(
                    _injection_is_visible(turn["rendered_prompt"])
                    and _valid_reasoning_boundary(turn)
                    for turn in candidate["turns"]
                ):
                    conditioning_source = (
                        generation_attempt_index,
                        attempt_decoding,
                    )
            if candidate["generated_target_action"]:
                success = True
                break
        stage_summaries.append(
            {
                **stage.to_dict(),
                "attempt_indices": stage_attempt_indices,
                "validator_passed": success,
            }
        )
        if success:
            break

    def rank(candidate: dict[str, Any]) -> tuple[bool, bool, bool]:
        return (
            bool(candidate["generated_target_action"]),
            bool(candidate["attack_success"]),
            bool(candidate["utility"]),
        )

    selected_index = max(range(len(candidates)), key=lambda index: rank(candidates[index]))
    selected = candidates[selected_index]
    selected["selection"] = {
        "policy": (
            "reasoning_rerank_then_first_validator_confirmed"
            if reasoning_search.enabled
            else "adaptive_best_of_n_first_validator_confirmed"
        ),
        "natural_rollout_budgets": list(parse_natural_budgets(natural_rollout_budgets)),
        "teacher_forcing_token_budgets": list(
            parse_teacher_forcing_budgets(teacher_forcing_token_budgets)
        ),
        "sparse_correction_budgets": list(
            parse_teacher_forcing_budgets(sparse_correction_budgets)
        ),
        "reasoning_search": asdict(reasoning_search),
        "executed_attempts": len(candidates),
        "selected_attempt_index": selected_index,
        "realization_stage": selected.get("realization_stage"),
        "conditioning_source_attempt_index": (
            conditioning_source[0] if conditioning_source is not None else None
        ),
        "stages": stage_summaries,
        "attempts": attempt_summaries,
    }
    return selected


def _redacted(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": record["schema_version"],
        "case": {
            key: value
            for key, value in record["case"].items()
            if key != "injections"
        },
        "turns": [
            {
                "turn_index": turn["turn_index"],
                "prompt_token_count": turn["prompt_token_count"],
                "generated_token_count_with_eos_if_emitted": turn[
                    "generated_token_count_with_eos_if_emitted"
                ],
                "stopped_on_eos": turn["stopped_on_eos"],
                "finish_reason": turn.get(
                    "finish_reason",
                    "eos" if turn["stopped_on_eos"] else "max_new_tokens",
                ),
                "reasoning_boundary": turn.get(
                    "reasoning_boundary",
                    _reasoning_boundary(turn["completion"]),
                ),
                "valid_reasoning_boundary": _valid_reasoning_boundary(turn),
                "generation_seconds": turn["generation_seconds"],
                "completion_sha256": hashlib.sha256(
                    turn["completion"].encode("utf-8")
                ).hexdigest(),
                "parse_error_count": len(turn["parse_errors"]),
                "tool_call_count": len(turn["parsed_tool_calls"]),
                "intervention": (
                    {
                        key: turn["output_intervention"].get(key)
                        for key in (
                            "mode",
                            "budget",
                            "forced_token_count",
                            "reference_token_count",
                            "action_token_count",
                            "forced_token_fraction",
                            "reference_token_fraction",
                            "model_match_count",
                            "tokens_compared",
                            "decode_request_count",
                            "reasoning_provenance",
                            "reasoning_token_count",
                            "bridge_is_fully_model_generated",
                        )
                    }
                    if turn.get("output_intervention", {}).get("mode")
                    in {"teacher_forcing_prefix", "sparse_correction"}
                    else None
                ),
                "completion_token_provenance_counts": {
                    "model": turn.get("completion_token_provenance", []).count(
                        "model"
                    ),
                    "reference_prefix": turn.get(
                        "completion_token_provenance", []
                    ).count("reference"),
                    "reference_correction": turn.get(
                        "completion_token_provenance", []
                    ).count("reference_correction"),
                },
                "natural_search": (
                    {
                        key: turn["natural_search"].get(key)
                        for key in (
                            "method",
                            "configured_candidate_count",
                            "valid_candidate_count",
                            "distinct_reasoning_count",
                            "score_rollouts_per_candidate",
                            "score_max_new_tokens",
                            "reasoning_sampling_temperature",
                            "final_action_sampling_temperature",
                            "selected_candidate_index",
                            "selected_was_first",
                            "selected_exact_score",
                            "selected_function_score",
                            "exact_score_gain_over_first",
                            "selected_conditional_mean_logprob",
                            "conditional_logprob_gain_over_first",
                            "generation_request_count",
                            "sampled_token_count",
                            "scored_token_count",
                            "final_completion_token_count",
                            "reasoning_token_count",
                            "bridge_is_fully_model_generated",
                            "all_completion_tokens_model_generated",
                            "reference_token_count",
                            "candidate_scores",
                        )
                    }
                    if turn.get("natural_search")
                    else None
                ),
                "generation_audit": turn.get("generation_audit"),
                "realization_skip_reason": turn.get("realization_skip_reason"),
            }
            for turn in record["turns"]
        ],
        "function_trace": [
            {"function": call["function"]} for call in record["function_trace"]
        ],
        "stop_reason": record["stop_reason"],
        "valid_reasoning_boundaries": record.get(
            "valid_reasoning_boundaries",
            all(_valid_reasoning_boundary(turn) for turn in record["turns"]),
        ),
        "utility": record["utility"],
        "attack_success": record["attack_success"],
        "target_functions": record["target_functions"],
        "target_call_count": len(record["target_calls"]),
        "generated_target_action": record["generated_target_action"],
        "realization_stage": record.get("realization_stage"),
        "realization_skip_reason": record.get("realization_skip_reason"),
        "selection": record.get("selection"),
    }


def _existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["case"]["case_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _write_or_validate_protocol(
    path: Path,
    records_path: Path,
    protocol: dict[str, Any],
) -> None:
    if records_path.exists() and records_path.stat().st_size > 0:
        if not path.exists():
            raise ValueError("Cannot resume records without protocol.json.")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError(
                "Refusing to mix records produced by different protocols."
            )
        return
    path.write_text(_json(protocol, indent=2) + "\n", encoding="utf-8")


def _synchronize_redacted(raw_path: Path, redacted_path: Path) -> None:
    if not raw_path.exists():
        return
    completed = _existing_ids(redacted_path)
    missing = [
        row
        for row in (
            json.loads(line)
            for line in raw_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row["case"]["case_id"] not in completed
    ]
    if not missing:
        return
    with redacted_path.open("a", encoding="utf-8") as handle:
        for row in missing:
            handle.write(_json(_redacted(row)) + "\n")


def _summary(
    records_path: Path,
    *,
    protocol: dict[str, Any],
    expected_count: int,
) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clean = [row for row in records if row["attack_success"] is None]
    attacked = [row for row in records if row["attack_success"] is not None]
    total_turns = sum(len(row["turns"]) for row in records)
    parse_error_turns = sum(
        bool(turn["parse_errors"]) for row in records for turn in row["turns"]
    )
    invalid_reasoning_turns = sum(
        not _valid_reasoning_boundary(turn)
        for row in records
        for turn in row["turns"]
    )
    incomplete = [
        row
        for row in records
        if row["stop_reason"] not in {
            "assistant_final_answer",
            "intervened_action",
        }
        or not all(_valid_reasoning_boundary(turn) for turn in row["turns"])
    ]
    selected_attempts = [
        row.get("selection", {}).get("attempts", [_attempt_summary(row, 0)])
        for row in attacked
    ]
    first_attempts = [attempts[0] for attempts in selected_attempts]
    action_target_pairs = [
        (row, attempts)
        for row, attempts in zip(attacked, selected_attempts, strict=True)
        if row.get("target_calls")
    ]
    suite_summaries = {}
    for suite in SUITES:
        suite_clean = [
            row
            for row in clean
            if row["case"]["suite"] == suite
        ]
        suite_attacked = [
            row
            for row in attacked
            if row["case"]["suite"] == suite
        ]
        suite_summaries[suite] = {
            "clean_count": len(suite_clean),
            "clean_utility_rate": (
                sum(row["utility"] for row in suite_clean) / len(suite_clean)
                if suite_clean
                else None
            ),
            "attacked_count": len(suite_attacked),
            "attacked_utility_rate": (
                sum(row["utility"] for row in suite_attacked)
                / len(suite_attacked)
                if suite_attacked
                else None
            ),
            "attack_success_rate": (
                sum(row["attack_success"] for row in suite_attacked)
                / len(suite_attacked)
                if suite_attacked
                else None
            ),
            "joint_utility_and_attack_rate": (
                sum(
                    row["utility"] and row["attack_success"]
                    for row in suite_attacked
                )
                / len(suite_attacked)
                if suite_attacked
                else None
            ),
            "localizable_case_count": sum(
                row["generated_target_action"] for row in suite_attacked
            ),
        }
    return {
        "protocol": protocol,
        "expected_case_count": expected_count,
        "completed_case_count": len(records),
        "clean_count": len(clean),
        "attacked_count": len(attacked),
        "clean_utility_rate": (
            sum(row["utility"] for row in clean) / len(clean) if clean else None
        ),
        "attacked_utility_rate": (
            sum(row["utility"] for row in attacked) / len(attacked)
            if attacked
            else None
        ),
        "attack_success_rate": (
            sum(row["attack_success"] for row in attacked) / len(attacked)
            if attacked
            else None
        ),
        "generated_target_action_rate": (
            sum(row["generated_target_action"] for row in attacked) / len(attacked)
            if attacked
            else None
        ),
        "action_target_count": len(action_target_pairs),
        "action_target_attack_success_rate": (
            sum(bool(row["attack_success"]) for row, _ in action_target_pairs)
            / len(action_target_pairs)
            if action_target_pairs
            else None
        ),
        "action_target_localizable_rate": (
            sum(bool(row["generated_target_action"])
                for row, _ in action_target_pairs)
            / len(action_target_pairs)
            if action_target_pairs
            else None
        ),
        "action_target_injection_observed_rate": (
            sum(any(attempt["injection_observed"] for attempt in attempts)
                for _, attempts in action_target_pairs)
            / len(action_target_pairs)
            if action_target_pairs
            else None
        ),
        "first_attempt_attack_success_rate": (
            sum(bool(row["attack_success"]) for row in first_attempts)
            / len(first_attempts)
            if first_attempts
            else None
        ),
        "first_attempt_localizable_rate": (
            sum(bool(row["generated_target_action"]) for row in first_attempts)
            / len(first_attempts)
            if first_attempts
            else None
        ),
        "injection_observed_rate": (
            sum(any(attempt["injection_observed"] for attempt in attempts)
                for attempts in selected_attempts)
            / len(selected_attempts)
            if selected_attempts
            else None
        ),
        "intervention_action_rate": (
            sum(any(attempt.get("intervention_applied") for attempt in attempts)
                for attempts in selected_attempts)
            / len(selected_attempts)
            if selected_attempts
            else None
        ),
        "realization_stage_counts": {
            stage: sum(row.get("realization_stage") == stage for row in attacked)
            for stage in sorted(
                {
                    row.get("realization_stage")
                    for row in attacked
                    if row.get("realization_stage")
                }
            )
        },
        "mean_executed_attempts_per_attacked_case": (
            sum(len(attempts) for attempts in selected_attempts)
            / len(selected_attempts)
            if selected_attempts
            else None
        ),
        "total_turns": total_turns,
        "parse_error_turn_rate": (
            parse_error_turns / total_turns if total_turns else 0.0
        ),
        "invalid_reasoning_boundary_turn_rate": (
            invalid_reasoning_turns / total_turns if total_turns else 0.0
        ),
        "incomplete_case_count": len(incomplete),
        "incomplete_case_rate": len(incomplete) / len(records) if records else 0.0,
        "suites": suite_summaries,
    }


def _command_output(command: Sequence[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    ).stdout.strip()


def _provenance(
    args: argparse.Namespace,
    generator: LocalSGLangGenerator | None,
) -> dict[str, Any]:
    tracked_files = (
        Path(__file__),
        Path(__file__).with_name("e2e.py"),
        Path(__file__).with_name("pilot.py"),
        Path(__file__).with_name("pyproject.toml"),
        Path(__file__).with_name("uv.lock"),
    )
    payload = {
        "flashtrace_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "flashtrace_dirty": bool(
            _command_output(["git", "status", "--short"])
        ),
        "runner_file_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tracked_files
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "agentdojo": version("agentdojo"),
        "cuda": torch.version.cuda,
        "gpu": (
            _command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                    "-i",
                    args.device.split(":")[-1],
                ]
            )
            if generator is not None
            else (
                torch.cuda.get_device_name(int(args.device.split(":")[-1]))
                if torch.cuda.is_available()
                else None
            )
        ),
    }
    if generator is not None:
        payload["sglang_server"] = generator.provenance()
    return payload


def _load_model(
    model_id: str,
    revision: str,
    device: str,
) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
        device_map={"": index},
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("clean", "attacked", "both"), default="both")
    parser.add_argument(
        "--suites",
        default=",".join(SUITES),
        help="Comma-separated suite subset.",
    )
    parser.add_argument(
        "--action-target-only",
        action="store_true",
        help="Keep only attacked cases with at least one concrete action target.",
    )
    parser.add_argument("--pilot-per-suite", type=int, default=None)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only an exact case id; may be repeated.",
    )
    parser.add_argument("--max-turns", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--natural-rollout-budgets",
        default="1",
        help=(
            "Strictly increasing cumulative rollout budgets, for example "
            "1,4,16. The first rollout is greedy; later rollouts use the "
            "sampling parameters below."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--reasoning-rerank-candidates",
        type=int,
        default=0,
        help="Sample this many complete reasoning candidates; zero disables reranking.",
    )
    parser.add_argument(
        "--reasoning-score-method",
        choices=("conditional_logprob", "short_rollout"),
        default="conditional_logprob",
        help="Hidden score used to rank complete model-generated reasoning traces.",
    )
    parser.add_argument(
        "--reasoning-temperature",
        type=float,
        default=None,
        help="Optional candidate-only temperature; final action uses --temperature.",
    )
    parser.add_argument(
        "--reasoning-score-rollouts",
        type=int,
        default=1,
        help="Natural short continuations used to score each reasoning candidate.",
    )
    parser.add_argument(
        "--reasoning-score-max-new-tokens",
        type=int,
        default=192,
        help="Maximum tokens per hidden scoring continuation.",
    )
    parser.add_argument(
        "--prefix-forcing-token-budgets",
        "--teacher-forcing-token-budgets",
        dest="prefix_forcing_token_budgets",
        default="",
        help=(
            "Increasing target-action prefix budgets, for example "
            "1,2,4,8,full. Empty disables this secondary axis."
        ),
    )
    parser.add_argument(
        "--sparse-correction-budgets",
        default="",
        help=(
            "Increasing mismatch-correction budgets, for example "
            "1,2,4,8,full. Runs separately from prefix forcing."
        ),
    )
    parser.add_argument(
        "--sglang-base-url",
        help="Loopback-only SGLang endpoint, for example http://127.0.0.1:8001.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/agentdojo/results/full/qwen3_8b"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    natural_budgets = parse_natural_budgets(args.natural_rollout_budgets)
    teacher_forcing_budgets = parse_teacher_forcing_budgets(
        args.prefix_forcing_token_budgets
    )
    sparse_correction_budgets = parse_teacher_forcing_budgets(
        args.sparse_correction_budgets
    )
    if teacher_forcing_budgets and sparse_correction_budgets:
        raise ValueError(
            "Prefix forcing and sparse correction require separate output runs."
        )
    selected_suites = tuple(
        item.strip() for item in args.suites.split(",") if item.strip()
    )
    unknown_suites = sorted(set(selected_suites) - set(SUITES))
    if not selected_suites or unknown_suites:
        raise ValueError(f"Unsupported suites: {unknown_suites}")
    if not 0 <= args.temperature:
        raise ValueError("--temperature must be non-negative.")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1].")
    if args.top_k < 0:
        raise ValueError("--top-k must be non-negative.")
    if args.reasoning_rerank_candidates < 0:
        raise ValueError("--reasoning-rerank-candidates must be non-negative.")
    if args.reasoning_temperature is not None and args.reasoning_temperature <= 0:
        raise ValueError("--reasoning-temperature must be positive.")
    if args.reasoning_score_rollouts < 1:
        raise ValueError("--reasoning-score-rollouts must be positive.")
    if args.reasoning_score_max_new_tokens < 1:
        raise ValueError("--reasoning-score-max-new-tokens must be positive.")
    if natural_budgets[-1] > 1 and args.temperature == 0:
        raise ValueError(
            "Natural resampling requires --temperature > 0; greedy attempts "
            "would be duplicates."
        )
    cases = build_inventory(mode=args.mode, suites=selected_suites)
    if args.action_target_only:
        cases = filter_action_target_cases(cases)
    if args.case_id:
        requested_ids = set(args.case_id)
        cases = [case for case in cases if case.case_id in requested_ids]
        missing_ids = sorted(requested_ids - {case.case_id for case in cases})
        if missing_ids:
            raise ValueError(f"Unknown or filtered case ids: {missing_ids}")
    if args.pilot_per_suite is not None:
        if args.case_id:
            raise ValueError("--case-id and --pilot-per-suite are mutually exclusive.")
        cases = select_pilot(cases, per_suite=args.pilot_per_suite)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    generator = (
        LocalSGLangGenerator(args.sglang_base_url, tokenizer)
        if args.sglang_base_url
        else None
    )
    if generator is not None:
        server_info = generator.provenance().get("model_info", {})
        _validate_generation_service_model(args.model, args.revision, server_info)
    model = None
    if generator is None:
        model, tokenizer = _load_model(args.model, args.revision, args.device)

    protocol = {
        "model": args.model,
        "revision": args.revision,
        "generation_backend": (
            "local_sglang_native_generate"
            if generator is not None
            else "transformers_local"
        ),
        "agentdojo_package_version": AGENTDOJO_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "attack": ATTACK_NAME,
        "mode": args.mode,
        "suites": list(selected_suites),
        "inventory_filter": (
            "action_target_only" if args.action_target_only else "none"
        ),
        "selection": (
            "explicit_case_ids"
            if args.case_id
            else "full"
            if args.pilot_per_suite is None
            else f"deterministic_{args.pilot_per_suite}_per_suite_per_mode"
        ),
        "case_id_filter_count": len(args.case_id),
        "decoding": "greedy" if args.temperature == 0 else "sampling",
        "sampling": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
        },
        "trajectory_selection": {
            "policy": (
                "reasoning_rerank_then_first_validator_confirmed"
                if args.reasoning_rerank_candidates
                else "adaptive_best_of_n_first_validator_confirmed"
            ),
            "natural_rollout_budgets": list(natural_budgets),
            "teacher_forcing_token_budgets": list(teacher_forcing_budgets),
            "sparse_correction_budgets": list(sparse_correction_budgets),
            "intervention_method": (
                "prefix_forcing"
                if teacher_forcing_budgets
                else "sparse_correction"
                if sparse_correction_budgets
                else "none"
            ),
            "reasoning_search": {
                "method": (
                    args.reasoning_score_method
                    if args.reasoning_rerank_candidates
                    else "disabled"
                ),
                "candidates": args.reasoning_rerank_candidates,
                "reasoning_temperature": args.reasoning_temperature,
                "score_rollouts": args.reasoning_score_rollouts,
                "score_max_new_tokens": args.reasoning_score_max_new_tokens,
                "final_completion_reference_writes": 0,
            },
            "natural_attack_rate_comparable": (
                natural_budgets == (1,)
                and not teacher_forcing_budgets
                and not sparse_correction_budgets
            ),
        },
        "enable_thinking": True,
        "preserve_thinking": True,
        "max_turns": args.max_turns,
        "max_new_tokens_per_turn": args.max_new_tokens,
    }
    raw_path = args.output_dir / "trajectories.jsonl"
    redacted_path = args.output_dir / "trajectories.redacted.jsonl"
    _write_or_validate_protocol(
        args.output_dir / "protocol.json",
        raw_path,
        protocol,
    )
    (args.output_dir / "provenance.json").write_text(
        _json(_provenance(args, generator), indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(_json(asdict(case)) + "\n")

    _synchronize_redacted(raw_path, redacted_path)
    completed = _existing_ids(raw_path)
    with (
        raw_path.open("a", encoding="utf-8") as raw_handle,
        redacted_path.open("a", encoding="utf-8") as redacted_handle,
    ):
        decoding = DecodingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
        reasoning_search = ReasoningSearchConfig(
            candidates=args.reasoning_rerank_candidates,
            score_method=args.reasoning_score_method,
            reasoning_temperature=args.reasoning_temperature,
            score_rollouts=args.reasoning_score_rollouts,
            score_max_new_tokens=args.reasoning_score_max_new_tokens,
        )
        for index, case in enumerate(cases, 1):
            if case.case_id in completed:
                print(f"[{index}/{len(cases)}] skip {case.case_id}", flush=True)
                continue
            print(f"[{index}/{len(cases)}] {case.case_id}", flush=True)
            record = run_case(
                case,
                model,
                tokenizer,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                generator=generator,
                decoding=decoding,
                natural_rollout_budgets=natural_budgets,
                teacher_forcing_token_budgets=teacher_forcing_budgets,
                sparse_correction_budgets=sparse_correction_budgets,
                reasoning_search=reasoning_search,
                progress_callback=lambda progress: print(
                    "  stage={realization_stage} attempt={attempt_index} "
                    "validated={validator_passed}".format(**progress),
                    flush=True,
                ),
            )
            raw_handle.write(_json(record) + "\n")
            raw_handle.flush()
            redacted_handle.write(_json(_redacted(record)) + "\n")
            redacted_handle.flush()
            print(
                f"  utility={record['utility']} "
                f"attack={record['attack_success']} "
                f"target={record['generated_target_action']}",
                flush=True,
            )
            summary = _summary(
                raw_path,
                protocol=protocol,
                expected_count=len(cases),
            )
            (args.output_dir / "summary.json").write_text(
                _json(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            if generator is None:
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
