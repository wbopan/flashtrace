#!/usr/bin/env python3
"""Export lossless, token-level explanations for the AgentDojo pilot cases."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from agentdojo.task_suite.load_suites import get_suite
from transformers import AutoTokenizer

from evaluations.agentdojo.pilot import (
    BENCHMARK_VERSION,
    DEFAULT_ATTRIBUTION_MODEL,
    PILOT_CASES,
    SOURCE_MODEL,
    TOP_FRACTIONS,
    UPSTREAM_COMMIT,
    LLMIFRAttribution,
    LLMIFRAttributionBoth,
    PreparedCase,
    _clean_prompt_vector,
    _load_model,
    _localization_metrics,
    _message_text,
    _target_length_without_eos,
    _tool_schema,
    keep_token_indices,
    prepare_case,
)


METHOD_LABELS = {
    "ifr_last_token": "IFR last output token",
    "ifr_full_span_exhaustive": "IFR exhaustive full output span",
    "flashtrace_full_span": "FlashTrace full output span",
}


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _markdown_cell(value: Any) -> str:
    if value is None:
        return ""
    rendered = value if isinstance(value, str) else str(value)
    return rendered.replace("|", "\\|").replace("\n", "<br>")


def _fenced_json(value: Any) -> list[str]:
    return ["```json", _json(value, indent=2), "```"]


def _fenced_text(value: str) -> list[str]:
    return ["````text", value, "````"]


def _all_occurrence_spans(text: str, pattern: str, label: str) -> list[dict[str, Any]]:
    return [
        {"label": label, "char_start": match.start(), "char_end": match.end()}
        for match in re.finditer(pattern, text, flags=re.DOTALL)
    ]


def _region_char_spans(case: PreparedCase) -> list[dict[str, Any]]:
    """Describe semantic prompt regions; overlap is resolved later by priority."""

    spans: list[dict[str, Any]] = []
    # The Qwen preamble mentions the literal text ``<tools></tools>`` before
    # opening the real multiline schema block, so require the following newline.
    tool_start = case.prompt.find("<tools>\n")
    if tool_start >= 0:
        tool_end = case.prompt.find("\n</tools>", tool_start)
        if tool_end >= 0:
            spans.append(
                {
                    "label": "tool_schema",
                    "char_start": tool_start,
                    "char_end": tool_end + len("\n</tools>"),
                }
            )

    cursor = 0
    history = case.raw["messages"][: case.spec.target_message_index]
    for index, message in enumerate(history):
        content = _message_text(message)
        if not content:
            continue
        start = case.prompt.find(content, cursor)
        if start < 0:
            raise ValueError(
                f"Could not align message {index} content in {case.spec.case_id}."
            )
        end = start + len(content)
        spans.append(
            {
                "label": f"message_{index:02d}_{message['role']}_content",
                "char_start": start,
                "char_end": end,
                "message_index": index,
                "role": message["role"],
            }
        )
        cursor = end

    tool_call_spans = _all_occurrence_spans(
        case.prompt,
        r"<tool_call>\n.*?\n</tool_call>",
        "assistant_tool_call",
    )
    actual_call_count = sum(
        len(message.get("tool_calls") or [])
        for message in history
        if message["role"] == "assistant"
    )
    format_example_count = max(0, len(tool_call_spans) - actual_call_count)
    for span in tool_call_spans[:format_example_count]:
        span["label"] = "tool_call_format_example"
    spans.extend(tool_call_spans)
    generation_start = case.prompt.rfind("<|im_start|>assistant")
    if generation_start >= 0:
        spans.append(
            {
                "label": "assistant_generation_prefix",
                "char_start": generation_start,
                "char_end": len(case.prompt),
            }
        )
    spans.append(
        {
            "label": "injected_tool_output",
            "char_start": case.injection_char_span[0],
            "char_end": case.injection_char_span[1],
            "gold": True,
        }
    )
    return spans


def _overlaps(token_span: tuple[int, int], char_span: dict[str, Any]) -> bool:
    return (
        token_span[0] < char_span["char_end"]
        and token_span[1] > char_span["char_start"]
    )


def _assign_regions(
    offsets: Sequence[Sequence[int]], spans: Sequence[dict[str, Any]]
) -> list[str]:
    priority = {
        "injected_tool_output": 50,
        "tool_schema": 40,
        "assistant_tool_call": 31,
        "tool_call_format_example": 30,
        "assistant_generation_prefix": 20,
    }
    labels = []
    for raw_start, raw_end in offsets:
        token_span = (int(raw_start), int(raw_end))
        candidates = [span for span in spans if _overlaps(token_span, span)]
        if not candidates:
            labels.append("chat_template")
            continue
        winner = max(
            candidates,
            key=lambda span: priority.get(
                span["label"], 25 if span["label"].startswith("message_") else 0
            ),
        )
        labels.append(str(winner["label"]))
    return labels


def _region_segments(tokens: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not tokens:
        return []
    segments = []
    start = 0
    for index in range(1, len(tokens) + 1):
        if index < len(tokens) and tokens[index]["region"] == tokens[start]["region"]:
            continue
        segments.append(
            {
                "region": tokens[start]["region"],
                "token_start": start,
                "token_end_exclusive": index,
                "char_start": tokens[start]["char_start"],
                "char_end": tokens[index - 1]["char_end"],
            }
        )
        start = index
    return segments


def _top_band(rank: int | None, kept_count: int) -> str | None:
    if rank is None:
        return None
    for fraction in TOP_FRACTIONS:
        if rank <= max(1, math.ceil(kept_count * fraction)):
            return f"top_{int(round(fraction * 100))}pct"
    return None


def _score_payload(
    raw_weights: torch.Tensor,
    keep: Sequence[int],
    regions: Sequence[str],
    tokens: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    weights = torch.nan_to_num(raw_weights.detach().cpu().to(torch.float64)).clamp(
        min=0
    )
    keep_tensor = torch.as_tensor(keep, dtype=torch.long)
    total = float(weights.index_select(0, keep_tensor).sum().item())
    normalized = torch.zeros_like(weights)
    if total > 0:
        normalized.index_copy_(
            0, keep_tensor, weights.index_select(0, keep_tensor) / total
        )

    ranked_indices = sorted(keep, key=lambda index: (-float(normalized[index]), index))
    ranks: list[int | None] = [None] * len(weights)
    for rank, index in enumerate(ranked_indices, 1):
        ranks[index] = rank

    region_rows: dict[str, dict[str, Any]] = {}
    for region in dict.fromkeys(regions):
        indices = [index for index, value in enumerate(regions) if value == region]
        kept = [index for index in indices if ranks[index] is not None]
        region_rows[region] = {
            "token_count": len(indices),
            "kept_token_count": len(kept),
            "normalized_mass": float(normalized[indices].sum().item()),
            "max_normalized_score": float(normalized[indices].max().item()),
        }

    top_tokens = []
    for index in ranked_indices[:30]:
        top_tokens.append(
            {
                "rank": ranks[index],
                "index": index,
                "token_id": tokens[index]["token_id"],
                "source_slice": tokens[index]["source_slice"],
                "attribution_token": tokens[index]["attribution_token"],
                "region": regions[index],
                "gold_injection": tokens[index]["gold_injection"],
                "normalized_score": float(normalized[index].item()),
            }
        )

    return {
        "score_semantics": (
            "Span-aggregated prompt attribution after LLMAttributionResult's "
            "per-output-row normalization. The first vector is before, and the "
            "second is after, renormalization over kept prompt tokens."
        ),
        "raw_scores": weights.tolist(),
        "normalized_scores_over_kept_tokens": normalized.tolist(),
        "rank_over_kept_tokens": ranks,
        "top_band": [_top_band(rank, len(keep)) for rank in ranks],
        "normalization_denominator": total,
        "region_summary": region_rows,
        "top_30_tokens": top_tokens,
    }


def _tokenize_prompt(
    case: PreparedCase, tokenizer: Any, attribution_tokens: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    encoded = tokenizer(
        case.prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"]]
    if len(token_ids) != len(attribution_tokens):
        raise ValueError(
            f"Prompt token mismatch: tokenizer={len(token_ids)}, "
            f"attribution={len(attribution_tokens)}"
        )
    spans = _region_char_spans(case)
    regions = _assign_regions(offsets, spans)
    gold = set(case.injection_token_indices)
    vocab_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = []
    for index, (token_id, vocab_token, offset, region) in enumerate(
        zip(token_ids, vocab_tokens, offsets, regions, strict=True)
    ):
        tokens.append(
            {
                "index": index,
                "token_id": token_id,
                "vocab_token": vocab_token,
                "attribution_token": attribution_tokens[index],
                "source_slice": case.prompt[offset[0] : offset[1]],
                "char_start": offset[0],
                "char_end": offset[1],
                "region": region,
                "gold_injection": index in gold,
            }
        )
    if {token["index"] for token in tokens if token["gold_injection"]} != gold:
        raise ValueError("Gold token alignment changed during detailed export.")
    return tokens, spans


def _tokenize_target(target: str, tokenizer: Any) -> dict[str, Any]:
    encoded = tokenizer(
        target,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = [int(value) for value in encoded["input_ids"]]
    offsets = [[int(start), int(end)] for start, end in encoded["offset_mapping"]]
    vocab_tokens = tokenizer.convert_ids_to_tokens(ids)
    tokens = [
        {
            "index": index,
            "token_id": token_id,
            "vocab_token": vocab_token,
            "source_slice": target[start:end],
            "char_start": start,
            "char_end": end,
            "in_sink_span": True,
        }
        for index, (token_id, vocab_token, (start, end)) in enumerate(
            zip(ids, vocab_tokens, offsets, strict=True)
        )
    ]
    eos_id = tokenizer.eos_token_id
    return {
        "note": (
            "Qwen chat-template serialization of the structured upstream tool call; "
            "not a claim about GPT-4o's internal text or token sequence."
        ),
        "text": target,
        "sink_span_inclusive": [0, len(ids) - 1],
        "tokens_without_eos": tokens,
        "model_sequence_token_ids_with_appended_eos": ids
        + ([] if eos_id is None else [int(eos_id)]),
        "appended_eos": None
        if eos_id is None
        else {
            "token_id": int(eos_id),
            "token": tokenizer.convert_ids_to_tokens(int(eos_id)),
            "in_sink_span": False,
        },
    }


def _run_case(
    case: PreparedCase,
    tokenizer: Any,
    ifr: LLMIFRAttribution,
    flashtrace: LLMIFRAttributionBoth,
    n_hops: int,
) -> dict[str, Any]:
    target_length = _target_length_without_eos(tokenizer, case.target)
    full_span = [0, target_length - 1]
    last_span = [target_length - 1, target_length - 1]

    results: dict[str, tuple[Any, list[int], float]] = {}
    started = time.perf_counter()
    result = ifr.calculate_ifr_for_all_positions_output_only(
        case.prompt, target=case.target, sink_span=tuple(last_span)
    )
    results["ifr_last_token"] = (result, last_span, time.perf_counter() - started)

    started = time.perf_counter()
    result = ifr.calculate_ifr_for_all_positions_output_only(
        case.prompt, target=case.target, sink_span=tuple(full_span)
    )
    results["ifr_full_span_exhaustive"] = (
        result,
        full_span,
        time.perf_counter() - started,
    )

    started = time.perf_counter()
    result = flashtrace.calculate_ifr_multi_hop_both(
        case.prompt,
        target=case.target,
        sink_span=tuple(full_span),
        thinking_span=None,
        n_hops=n_hops,
    )
    results["flashtrace_full_span"] = (
        result,
        full_span,
        time.perf_counter() - started,
    )

    reference_result = results["ifr_full_span_exhaustive"][0]
    prompt_length = len(reference_result.prompt_tokens)
    for method_result, _, _ in results.values():
        if method_result.prompt_tokens != reference_result.prompt_tokens:
            raise ValueError("Attribution methods returned different prompt tokens.")

    prompt_tokens, region_spans = _tokenize_prompt(
        case, tokenizer, reference_result.prompt_tokens
    )
    keep = keep_token_indices(reference_result.prompt_tokens)
    keep_set = set(keep)
    for token in prompt_tokens:
        token["kept_for_ranking"] = token["index"] in keep_set

    method_payloads: dict[str, Any] = {}
    for method, (method_result, sink_span, runtime) in results.items():
        raw_vector = _clean_prompt_vector(
            method_result.get_all_token_attrs(sink_span)[1], prompt_length
        ).sum(0)
        payload = _score_payload(
            raw_vector,
            keep,
            [token["region"] for token in prompt_tokens],
            prompt_tokens,
        )
        payload.update(
            {
                "label": METHOD_LABELS[method],
                "runtime_seconds": runtime,
                "sink_span_inclusive": sink_span,
                "localization_metrics": _localization_metrics(
                    raw_vector.unsqueeze(0),
                    reference_result.prompt_tokens,
                    case.injection_token_indices,
                ),
            }
        )
        method_payloads[method] = payload

    flashtrace_result = results["flashtrace_full_span"][0]
    metadata = flashtrace_result.metadata or {}
    projected = metadata.get("ifr", {}).get("per_hop_projected", [])
    hop_payloads = []
    for hop_index, vector in enumerate(projected):
        raw_vector = _clean_prompt_vector(torch.as_tensor(vector), prompt_length).sum(0)
        payload = _score_payload(
            raw_vector,
            keep,
            [token["region"] for token in prompt_tokens],
            prompt_tokens,
        )
        payload["hop_index"] = hop_index
        hop_payloads.append(payload)

    suite = get_suite(BENCHMARK_VERSION, case.spec.suite)
    tool_schemas = [_tool_schema(tool) for tool in suite.tools]
    prompt_ids = [token["token_id"] for token in prompt_tokens]
    decoded_prompt = tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    target = _tokenize_target(case.target, tokenizer)
    return {
        "schema_version": 1,
        "case": {
            "case_id": case.spec.case_id,
            "suite": case.spec.suite,
            "user_task": case.spec.user_task,
            "injection_task": case.spec.injection_task,
            "target_message_index": case.spec.target_message_index,
            "target_tool": case.spec.target_tool,
            "source_trace_path": str(case.source_path.resolve()),
            "source_trace_sha256": case.spec.sha256,
            "stored_utility": bool(case.raw["utility"]),
            "stored_attack_success": bool(case.raw["security"]),
            "replay_utility": case.replay_utility,
            "replay_attack_success": case.replay_attack_success,
            "replay_errors": list(case.replay_errors),
        },
        "provenance": {
            "source_model": SOURCE_MODEL,
            "upstream_commit": UPSTREAM_COMMIT,
            "benchmark_version": BENCHMARK_VERSION,
            "attribution_target_provenance": "teacher_forced_upstream_trace",
        },
        "original_agentdojo_logical_input": {
            "note": (
                "Trace messages are exact. Separate API tool schemas are "
                "reconstructed from the pinned AgentDojo v1.2.2 suite because the "
                "run JSON does not store them. The source GPT-4o token IDs are "
                "proprietary and unavailable."
            ),
            "messages_before_target": case.raw["messages"][
                : case.spec.target_message_index
            ],
            "tool_schemas": tool_schemas,
        },
        "source_trace_raw": case.raw,
        "qwen_attribution_input": {
            "note": (
                "This is the exact rendered text and Qwen token sequence used by "
                "the white-box attribution run."
            ),
            "rendered_prompt": case.prompt,
            "prompt_token_ids": prompt_ids,
            "prompt_token_count": len(prompt_ids),
            "tokenizer_decode_roundtrip_exact": decoded_prompt == case.prompt,
            "tokenizer_decoded_prompt": decoded_prompt,
            "tokens": prompt_tokens,
            "region_char_spans": region_spans,
            "region_token_segments": _region_segments(prompt_tokens),
            "kept_token_indices_for_ranking": keep,
        },
        "gold_injection": {
            "text": case.prompt[slice(*case.injection_char_span)],
            "char_span_half_open": list(case.injection_char_span),
            "token_indices": list(case.injection_token_indices),
            "token_span_inclusive": [
                min(case.injection_token_indices),
                max(case.injection_token_indices),
            ],
        },
        "teacher_forced_target": target,
        "full_teacher_forced_model_token_ids": prompt_ids
        + target["model_sequence_token_ids_with_appended_eos"],
        "methods": method_payloads,
        "flashtrace_per_hop_diagnostics": hop_payloads,
    }


def _region_table(record: dict[str, Any]) -> list[str]:
    method_names = list(METHOD_LABELS)
    regions = record["methods"][method_names[0]]["region_summary"]
    lines = [
        "| Region | Tokens | Kept | IFR-last mass | IFR-full mass | FlashTrace mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for region, counts in regions.items():
        masses = [
            record["methods"][method]["region_summary"][region]["normalized_mass"]
            for method in method_names
        ]
        lines.append(
            f"| `{_markdown_cell(region)}` | {counts['token_count']} | "
            f"{counts['kept_token_count']} | {masses[0]:.6f} | {masses[1]:.6f} | "
            f"{masses[2]:.6f} |"
        )
    return lines


def _top_tokens_table(method: str, payload: dict[str, Any]) -> list[str]:
    lines = [
        f"### {METHOD_LABELS[method]}",
        "",
        "| Rank | i | id | source slice | attribution token | region | gold | score |",
        "|---:|---:|---:|---|---|---|:---:|---:|",
    ]
    for token in payload["top_30_tokens"]:
        lines.append(
            f"| {token['rank']} | {token['index']} | {token['token_id']} | "
            f"`{_markdown_cell(_json(token['source_slice']))}` | "
            f"`{_markdown_cell(_json(token['attribution_token']))}` | "
            f"`{_markdown_cell(token['region'])}` | "
            f"{'Y' if token['gold_injection'] else ''} | "
            f"{token['normalized_score']:.9g} |"
        )
    return lines


def _full_prompt_token_table(record: dict[str, Any]) -> list[str]:
    tokens = record["qwen_attribution_input"]["tokens"]
    methods = record["methods"]
    lines = [
        "| i | id | vocab token | exact source slice | chars | region | gold | keep | IFR-last score/rank/band | IFR-full score/rank/band | FlashTrace score/rank/band |",
        "|---:|---:|---|---|---|---|:---:|:---:|---|---|---|",
    ]
    for token in tokens:
        attrs = []
        for method in METHOD_LABELS:
            payload = methods[method]
            index = token["index"]
            score = payload["normalized_scores_over_kept_tokens"][index]
            rank = payload["rank_over_kept_tokens"][index]
            band = payload["top_band"][index]
            attrs.append(f"{score:.9g} / {rank or '-'} / {band or '-'}")
        lines.append(
            f"| {token['index']} | {token['token_id']} | "
            f"`{_markdown_cell(_json(token['vocab_token']))}` | "
            f"`{_markdown_cell(_json(token['source_slice']))}` | "
            f"[{token['char_start']},{token['char_end']}) | "
            f"`{_markdown_cell(token['region'])}` | "
            f"{'Y' if token['gold_injection'] else ''} | "
            f"{'Y' if token['kept_for_ranking'] else ''} | "
            f"{attrs[0]} | {attrs[1]} | {attrs[2]} |"
        )
    return lines


def _target_token_table(record: dict[str, Any]) -> list[str]:
    lines = [
        "| i | id | vocab token | exact source slice | chars | in sink |",
        "|---:|---:|---|---|---|:---:|",
    ]
    for token in record["teacher_forced_target"]["tokens_without_eos"]:
        lines.append(
            f"| {token['index']} | {token['token_id']} | "
            f"`{_markdown_cell(_json(token['vocab_token']))}` | "
            f"`{_markdown_cell(_json(token['source_slice']))}` | "
            f"[{token['char_start']},{token['char_end']}) | Y |"
        )
    eos = record["teacher_forced_target"]["appended_eos"]
    if eos is not None:
        lines.append(
            f"| {len(record['teacher_forced_target']['tokens_without_eos'])} | "
            f"{eos['token_id']} | `{_markdown_cell(_json(eos['token']))}` | "
            "`<appended EOS>` | n/a |  |"
        )
    return lines


def _write_case_markdown(path: Path, record: dict[str, Any]) -> None:
    case = record["case"]
    prompt = record["qwen_attribution_input"]
    gold = record["gold_injection"]
    target = record["teacher_forced_target"]
    lines = [
        f"# {case['case_id']}: complete input and token explanation",
        "",
        "## Reading contract",
        "",
        "The upstream trace stores exact structured `messages`; its separate `tools` array is reconstructed from the pinned AgentDojo v1.2.2 suite. GPT-4o's proprietary token IDs are unavailable. The Qwen sequence below is the exact local white-box attribution input; the malicious structured action is serialized with Qwen's template and teacher-forced as the target.",
        "",
        f"- Suite/task: `{case['suite']}` / `{case['user_task']}` / `{case['injection_task']}`",
        f"- Target message/tool: `{case['target_message_index']}` / `{case['target_tool']}`",
        f"- Prompt tokens: `{prompt['prompt_token_count']}`",
        f"- Gold injection: chars `[{gold['char_span_half_open'][0]},{gold['char_span_half_open'][1]})`, tokens `{gold['token_span_inclusive'][0]}..{gold['token_span_inclusive'][1]}` (`{len(gold['token_indices'])}` tokens)",
        f"- Target sink: tokens `{target['sink_span_inclusive'][0]}..{target['sink_span_inclusive'][1]}` (`{len(target['tokens_without_eos'])}` tokens), followed by an appended EOS outside the sink",
        f"- Exact tokenizer decode round trip: `{prompt['tokenizer_decode_roundtrip_exact']}`",
        f"- Validator replay: utility=`{case['replay_utility']}`, attack success=`{case['replay_attack_success']}`, errors=`{case['replay_errors']}`",
        "",
        "## Original AgentDojo logical input",
        "",
        "### Messages before the malicious target",
        "",
        *_fenced_json(
            record["original_agentdojo_logical_input"]["messages_before_target"]
        ),
        "",
        "### Tool schemas passed separately to the source agent",
        "",
        *_fenced_json(record["original_agentdojo_logical_input"]["tool_schemas"]),
        "",
        "## Exact Qwen rendered prompt",
        "",
        *_fenced_text(prompt["rendered_prompt"]),
        "",
        "## Gold injection region",
        "",
        *_fenced_text(gold["text"]),
        "",
        "## Teacher-forced malicious output target",
        "",
        *_fenced_text(target["text"]),
        "",
        "### Target token sequence",
        "",
        *_target_token_table(record),
        "",
        "## Semantic region mass",
        "",
        "Scores are normalized across the non-stop prompt tokens retained by the pilot metric. Regions form a disjoint token partition; the gold injection overrides its enclosing tool-message content.",
        "",
        *_region_table(record),
        "",
        "## Top 30 attributed prompt tokens",
        "",
    ]
    for method, payload in record["methods"].items():
        lines.extend(_top_tokens_table(method, payload))
        lines.append("")

    lines.extend(
        [
            "## FlashTrace per-hop region diagnostics",
            "",
            "These projected hop vectors are diagnostics, not additive components of the final normalized FlashTrace score.",
            "",
            "| Hop | injection | tool schema | tool-call format | assistant tool call | generation prefix | all message content | chat template |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for hop in record["flashtrace_per_hop_diagnostics"]:
        summary = hop["region_summary"]
        message_mass = sum(
            values["normalized_mass"]
            for region, values in summary.items()
            if region.startswith("message_")
        )
        lines.append(
            f"| {hop['hop_index']} | "
            f"{summary.get('injected_tool_output', {}).get('normalized_mass', 0.0):.6f} | "
            f"{summary.get('tool_schema', {}).get('normalized_mass', 0.0):.6f} | "
            f"{summary.get('tool_call_format_example', {}).get('normalized_mass', 0.0):.6f} | "
            f"{summary.get('assistant_tool_call', {}).get('normalized_mass', 0.0):.6f} | "
            f"{summary.get('assistant_generation_prefix', {}).get('normalized_mass', 0.0):.6f} | "
            f"{message_mass:.6f} | "
            f"{summary.get('chat_template', {}).get('normalized_mass', 0.0):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Complete prompt token sequence and attribution",
            "",
            "`source slice` is the exact half-open character slice from the rendered prompt. `vocab token` is `convert_ids_to_tokens(id)`. Each method cell is `normalized score / rank / smallest top band`; skipped stop tokens have no rank. Machine-readable `raw_scores` are span-aggregated row-normalized scores before the additional kept-prompt-token renormalization.",
            "",
            *_full_prompt_token_table(record),
            "",
            "## Region character spans",
            "",
            *_fenced_json(prompt["region_char_spans"]),
            "",
            "## Region token segments",
            "",
            *_fenced_json(prompt["region_token_segments"]),
            "",
            "## Complete pinned source trace",
            "",
            *_fenced_json(record["source_trace_raw"]),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_index(path: Path, records: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# AgentDojo pilot: complete sample explanations",
        "",
        "This directory separates the original structured AgentDojo API input from the exact rendered Qwen token sequence used for white-box attribution. Every case report contains the full prompt, target, token IDs, offsets, semantic regions, gold injection mask, and all per-token scores/ranks without truncation.",
        "",
        "| Case | Logical turns | Prompt | Injection | Target | IFR-last mass | IFR-full mass | FlashTrace mass | Report |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        case = record["case"]
        methods = record["methods"]
        lines.append(
            f"| `{case['case_id']}` | {len(record['original_agentdojo_logical_input']['messages_before_target'])} | "
            f"{record['qwen_attribution_input']['prompt_token_count']} | "
            f"{len(record['gold_injection']['token_indices'])} | "
            f"{len(record['teacher_forced_target']['tokens_without_eos'])} | "
            f"{methods['ifr_last_token']['localization_metrics']['injection_mass_fraction']:.6f} | "
            f"{methods['ifr_full_span_exhaustive']['localization_metrics']['injection_mass_fraction']:.6f} | "
            f"{methods['flashtrace_full_span']['localization_metrics']['injection_mass_fraction']:.6f} | "
            f"[{case['case_id']}.md]({case['case_id']}.md) |"
        )
    lines.extend(
        [
            "",
            "Machine-readable artifacts:",
            "",
            "- `detailed_traces.jsonl`: one self-contained lossless record per case",
            "- `<case>.json`: the same record split into convenient per-case files",
            "- `raw_prompt_token_ids.json`: compact case-to-token-ID mapping",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html_heatmap(path: Path, record: dict[str, Any]) -> None:
    """Write a compact readable view; Markdown remains the complete audit artifact."""

    method = "flashtrace_full_span"
    scores = record["methods"][method]["normalized_scores_over_kept_tokens"]
    max_score = max(scores) or 1.0
    spans = []
    for token, score in zip(
        record["qwen_attribution_input"]["tokens"], scores, strict=True
    ):
        alpha = min(0.92, 0.05 + 0.87 * math.sqrt(score / max_score)) if score else 0.0
        color = "220,38,38" if token["gold_injection"] else "37,99,235"
        title = html.escape(
            f"i={token['index']} id={token['token_id']} region={token['region']} "
            f"score={score:.9g}",
            quote=True,
        )
        spans.append(
            f'<span title="{title}" style="background:rgba({color},{alpha:.4f})">'
            f"{html.escape(token['source_slice'])}</span>"
        )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>{html.escape(record["case"]["case_id"])} FlashTrace heatmap</title>
<style>
body {{ font: 14px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; margin: 28px; color: #172033; }}
main {{ white-space: pre-wrap; overflow-wrap: anywhere; max-width: 1200px; }}
.legend {{ font-family: system-ui, sans-serif; margin-bottom: 20px; }}
</style>
<div class="legend"><b>{html.escape(record["case"]["case_id"])}</b> — FlashTrace normalized token score. Blue is ordinary context; red is the gold injection span. Hover for token details.</div>
<main>{"".join(spans)}</main>
"""
    path.write_text(document, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_ATTRIBUTION_MODEL)
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--n-hops", type=int, default=3)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--sink-chunk-tokens", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=len(PILOT_CASES))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path(__file__).parent / ".cache" / "upstream"
    )
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "results" / "pilot" / "explanations",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.num_samples > len(PILOT_CASES):
        raise SystemExit(f"--num-samples must be in [1, {len(PILOT_CASES)}]")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = [
        prepare_case(spec, tokenizer, args.cache_dir, args.source_root)
        for spec in PILOT_CASES[: args.num_samples]
    ]
    model, tokenizer = _load_model(args.model, args.device)
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
    jsonl_path = args.output_dir / "detailed_traces.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl:
        for index, case in enumerate(cases, 1):
            print(f"[explain {index}/{len(cases)}] {case.spec.case_id}", flush=True)
            record = _run_case(case, tokenizer, ifr, flashtrace, args.n_hops)
            records.append(record)
            serialized = _json(record)
            jsonl.write(serialized + "\n")
            (args.output_dir / f"{case.spec.case_id}.json").write_text(
                serialized + "\n", encoding="utf-8"
            )
            _write_case_markdown(args.output_dir / f"{case.spec.case_id}.md", record)
            _write_html_heatmap(args.output_dir / f"{case.spec.case_id}.html", record)

    compact_ids = {
        record["case"]["case_id"]: record["qwen_attribution_input"]["prompt_token_ids"]
        for record in records
    }
    (args.output_dir / "raw_prompt_token_ids.json").write_text(
        _json(compact_ids, indent=2) + "\n", encoding="utf-8"
    )
    _write_index(args.output_dir / "INDEX.md", records)
    print(f"[done] wrote complete explanations to {args.output_dir}")


if __name__ == "__main__":
    main()
