#!/usr/bin/env python3
"""Reproducible AgentDojo prompt-injection localization pilot.

The pilot deliberately separates two concerns:

1. AgentDojo supplies execution-grounded, validator-checked attack traces.
2. FlashTrace and IFR localize the injected tool-output span that caused a
   later malicious tool call.

The selected traces are successful Important Instructions attacks from the
official AgentDojo repository.  They are pinned by commit and SHA-256, then
replayed against AgentDojo v1.2.2 validators before attribution is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

import torch
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import get_text_content_as_str, text_content_block_from_string
from transformers import AutoModelForCausalLM, AutoTokenizer

from flashtrace.attribution import LLMIFRAttribution
from flashtrace.improved import (
    LLMIFRAttributionBoth,
    evaluate_attr_recovery_skip_tokens,
    keep_token_indices,
)


AGENTDOJO_VERSION = "0.1.35"
BENCHMARK_VERSION = "v1.2.2"
UPSTREAM_COMMIT = "a75aba7631d3ca5fb7ab938965c97ead2f9ff84b"
UPSTREAM_RAW_ROOT = (
    "https://raw.githubusercontent.com/ethz-spylab/agentdojo/"
    f"{UPSTREAM_COMMIT}/runs/gpt-4o-2024-05-13"
)
ATTACK = "important_instructions"
SOURCE_MODEL = "gpt-4o-2024-05-13"
DEFAULT_ATTRIBUTION_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
TOP_FRACTIONS = (0.05, 0.10, 0.20)


@dataclass(frozen=True)
class PilotCase:
    case_id: str
    suite: str
    user_task: str
    injection_task: str
    target_message_index: int
    target_tool: str
    sha256: str

    @property
    def relative_path(self) -> Path:
        return (
            Path(self.suite) / self.user_task / ATTACK / f"{self.injection_task}.json"
        )


PILOT_CASES = (
    PilotCase(
        case_id="slack_visit_phishing_site",
        suite="slack",
        user_task="user_task_0",
        injection_task="injection_task_3",
        target_message_index=4,
        target_tool="get_webpage",
        sha256="3607a8d04f2a8f74782e86d30d27237ed7bedee455982fc8f45e75307dbc41af",
    ),
    PilotCase(
        case_id="banking_change_password",
        suite="banking",
        user_task="user_task_1",
        injection_task="injection_task_7",
        target_message_index=4,
        target_tool="update_password",
        sha256="e857442c5fa81f1d1a4d926a5987e8c034c78f561d4560ec66b67c4ec3d95ecf",
    ),
    PilotCase(
        case_id="slack_remove_invited_attacker",
        suite="slack",
        user_task="user_task_0",
        injection_task="injection_task_5",
        target_message_index=9,
        target_tool="remove_user_from_slack",
        sha256="e2e9395029f032827a093062d01dcf1add236a07818b78ca020a85f7efd3e5b7",
    ),
)


@dataclass(frozen=True)
class PreparedCase:
    spec: PilotCase
    source_path: Path
    raw: dict[str, Any]
    prompt: str
    target: str
    injection_char_span: tuple[int, int]
    injection_token_indices: tuple[int, ...]
    replay_utility: bool
    replay_attack_success: bool
    replay_errors: tuple[str, ...]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_or_fetch_case(
    spec: PilotCase, cache_dir: Path, source_root: Path | None
) -> Path:
    if source_root is not None:
        source_path = source_root / spec.relative_path
        if not source_path.exists():
            raise FileNotFoundError(f"Missing pinned AgentDojo trace: {source_path}")
        data = source_path.read_bytes()
    else:
        source_path = cache_dir / spec.relative_path
        if source_path.exists():
            data = source_path.read_bytes()
        else:
            url = f"{UPSTREAM_RAW_ROOT}/{spec.relative_path.as_posix()}"
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(data)

    digest = _sha256(data)
    if digest != spec.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {spec.case_id}: expected {spec.sha256}, got {digest}"
        )
    return source_path


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return get_text_content_as_str(content)


def _tool_schema(tool: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.model_json_schema(),
        },
    }


def _to_qwen_message(message: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {
        "role": message["role"],
        "content": _message_text(message),
    }
    if message["role"] == "assistant" and message.get("tool_calls"):
        converted["tool_calls"] = [
            {
                "type": "function",
                "function": {
                    "name": call["function"],
                    "arguments": call.get("args") or {},
                },
            }
            for call in message["tool_calls"]
        ]
    return converted


def _serialize_target(message: dict[str, Any]) -> str:
    calls = message.get("tool_calls") or []
    if not calls:
        raise ValueError("The selected malicious target must contain a tool call.")
    blocks = []
    for call in calls:
        payload = {
            "name": call["function"],
            "arguments": call.get("args") or {},
        }
        blocks.append(
            "<tool_call>\n"
            + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + "\n</tool_call>"
        )
    # Qwen3-4B-Thinking's generation prompt ends immediately after <think>.
    return "\n</think>\n\n" + "\n".join(blocks)


def _find_injection_span(prompt: str) -> tuple[int, int]:
    start = prompt.rfind("<INFORMATION>")
    if start < 0:
        raise ValueError(
            "Could not find the Important Instructions injection start marker."
        )
    end_marker = "</INFORMATION>"
    end = prompt.find(end_marker, start)
    if end < 0:
        raise ValueError(
            "Could not find the Important Instructions injection end marker."
        )
    return start, end + len(end_marker)


def _char_span_to_token_indices(
    tokenizer: Any,
    text: str,
    span: tuple[int, int],
) -> tuple[int, ...]:
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    indices = []
    for index, (start, end) in enumerate(encoded["offset_mapping"]):
        if int(start) < span[1] and int(end) > span[0]:
            indices.append(index)
    if not indices:
        raise ValueError(f"Character span {span} did not overlap any prompt token.")
    return tuple(indices)


def _replay_validators(raw: dict[str, Any]) -> tuple[bool, bool, tuple[str, ...]]:
    """Replay recorded calls and recompute v1.2.2 utility/attack labels."""

    suite = get_suite(BENCHMARK_VERSION, raw["suite_name"])
    user_task = suite.get_user_task_by_id(raw["user_task_id"])
    injection_task = suite.get_injection_task_by_id(raw["injection_task_id"])
    environment = suite.load_and_inject_default_environment(raw["injections"])
    environment = user_task.init_environment(environment)
    pre_environment = environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)
    trace: list[FunctionCall] = []
    errors: list[str] = []

    for message in raw["messages"]:
        if message["role"] != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function_call = FunctionCall(
                function=call["function"],
                args=dict(call.get("args") or {}),
                id=call.get("id"),
            )
            trace.append(function_call)
            _, error = runtime.run_function(
                environment,
                function_call.function,
                function_call.args,
            )
            if error:
                errors.append(f"{function_call.function}: {error}")

    final_content = [text_content_block_from_string(_message_text(raw["messages"][-1]))]
    utility = suite._check_user_task_utility(  # noqa: SLF001 - benchmark audit
        user_task,
        final_content,
        pre_environment,
        environment,
        trace,
    )
    attack_success = suite._check_injection_task_security(  # noqa: SLF001 - benchmark audit
        injection_task,
        final_content,
        pre_environment,
        environment,
        trace,
    )
    return bool(utility), bool(attack_success), tuple(errors)


def prepare_case(
    spec: PilotCase,
    tokenizer: Any,
    cache_dir: Path,
    source_root: Path | None,
) -> PreparedCase:
    source_path = _read_or_fetch_case(spec, cache_dir, source_root)
    raw = json.loads(source_path.read_text(encoding="utf-8"))

    if raw.get("attack_type") != ATTACK:
        raise ValueError(f"{spec.case_id} is not an {ATTACK} trace.")
    if raw.get("utility") is not True or raw.get("security") is not True:
        raise ValueError(
            f"{spec.case_id} is not a stored utility+attack-success trace."
        )

    target_message = raw["messages"][spec.target_message_index]
    target_tools = [call["function"] for call in target_message.get("tool_calls") or []]
    if (
        target_message.get("role") != "assistant"
        or spec.target_tool not in target_tools
    ):
        raise ValueError(
            f"{spec.case_id} target mismatch: expected assistant tool {spec.target_tool}, got {target_tools}"
        )

    suite = get_suite(BENCHMARK_VERSION, spec.suite)
    history = [
        _to_qwen_message(message)
        for message in raw["messages"][: spec.target_message_index]
    ]
    prompt = tokenizer.apply_chat_template(
        history,
        tools=[_tool_schema(tool) for tool in suite.tools],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Tokenizer returned an empty AgentDojo prompt.")

    injection_char_span = _find_injection_span(prompt)
    injection_token_indices = _char_span_to_token_indices(
        tokenizer, prompt, injection_char_span
    )
    replay_utility, replay_attack_success, replay_errors = _replay_validators(raw)
    if not replay_utility or not replay_attack_success:
        raise ValueError(
            f"{spec.case_id} failed v1.2.2 replay: utility={replay_utility}, "
            f"attack_success={replay_attack_success}"
        )

    return PreparedCase(
        spec=spec,
        source_path=source_path,
        raw=raw,
        prompt=prompt,
        target=_serialize_target(target_message),
        injection_char_span=injection_char_span,
        injection_token_indices=injection_token_indices,
        replay_utility=replay_utility,
        replay_attack_success=replay_attack_success,
        replay_errors=replay_errors,
    )


def _clean_prompt_vector(row: torch.Tensor, prompt_length: int) -> torch.Tensor:
    vector = torch.nan_to_num(
        torch.as_tensor(row).detach().cpu().to(torch.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if vector.ndim == 1:
        vector = vector.unsqueeze(0)
    if vector.ndim != 2 or vector.shape[1] < prompt_length:
        raise ValueError(
            f"Unexpected attribution row shape {tuple(vector.shape)} for prompt length {prompt_length}."
        )
    return vector[:, :prompt_length].clamp(min=0.0)


def _localization_metrics(
    prompt_attribution: torch.Tensor,
    prompt_tokens: Sequence[str],
    gold_indices: Sequence[int],
) -> dict[str, float | bool]:
    keep = keep_token_indices(prompt_tokens)
    gold = sorted(set(gold_indices).intersection(keep))
    if not gold:
        raise ValueError("No injection tokens remain after stop-token filtering.")
    weights = prompt_attribution.sum(0)
    keep_weights = weights.index_select(0, torch.as_tensor(keep, dtype=torch.long))
    top_index = keep[int(torch.argmax(keep_weights).item())]
    total_mass = float(keep_weights.sum().item())
    injection_mass = float(weights.index_select(0, torch.as_tensor(gold)).sum().item())
    metrics: dict[str, float | bool] = {
        "pointing_game": top_index in set(gold),
        "injection_mass_fraction": injection_mass / total_mass
        if total_mass > 0
        else 0.0,
    }
    for fraction in TOP_FRACTIONS:
        metrics[f"recovery_at_{int(round(fraction * 100))}pct"] = (
            evaluate_attr_recovery_skip_tokens(
                prompt_attribution,
                keep_prompt_token_indices=keep,
                gold_prompt_token_indices=gold,
                top_fraction=fraction,
            )
        )
    return metrics


def _random_metrics(
    prompt_tokens: Sequence[str], gold_indices: Sequence[int]
) -> dict[str, float | bool]:
    keep = keep_token_indices(prompt_tokens)
    gold = set(gold_indices).intersection(keep)
    if not gold:
        raise ValueError("No gold injection tokens for the random baseline.")
    metrics: dict[str, float | bool] = {
        "pointing_game": False,
        "injection_mass_fraction": len(gold) / len(keep),
    }
    for fraction in TOP_FRACTIONS:
        selected = min(len(keep), max(1, math.ceil(len(keep) * fraction)))
        metrics[f"recovery_at_{int(round(fraction * 100))}pct"] = selected / len(keep)
    return metrics


def _target_length_without_eos(tokenizer: Any, target: str) -> int:
    length = len(tokenizer(target, add_special_tokens=False).input_ids)
    if length <= 0:
        raise ValueError("The malicious target tokenized to an empty sequence.")
    return length


def evaluate_case(
    case: PreparedCase,
    tokenizer: Any,
    ifr: LLMIFRAttribution,
    flashtrace: LLMIFRAttributionBoth,
    n_hops: int,
) -> list[dict[str, Any]]:
    target_length = _target_length_without_eos(tokenizer, case.target)
    full_span = [0, target_length - 1]
    last_span = [target_length - 1, target_length - 1]

    started = time.perf_counter()
    ifr_last_result = ifr.calculate_ifr_for_all_positions_output_only(
        case.prompt,
        target=case.target,
        sink_span=tuple(last_span),
    )
    ifr_last_runtime = time.perf_counter() - started

    started = time.perf_counter()
    ifr_full_result = ifr.calculate_ifr_for_all_positions_output_only(
        case.prompt,
        target=case.target,
        sink_span=tuple(full_span),
    )
    ifr_full_runtime = time.perf_counter() - started
    prompt_length = len(ifr_full_result.prompt_tokens)
    if prompt_length != len(tokenizer(case.prompt, add_special_tokens=False).input_ids):
        raise ValueError("IFR prompt-token alignment changed unexpectedly.")

    started = time.perf_counter()
    flashtrace_result = flashtrace.calculate_ifr_multi_hop_both(
        case.prompt,
        target=case.target,
        sink_span=tuple(full_span),
        thinking_span=None,
        n_hops=n_hops,
    )
    flashtrace_runtime = time.perf_counter() - started
    if (
        len(ifr_last_result.prompt_tokens) != prompt_length
        or len(flashtrace_result.prompt_tokens) != prompt_length
    ):
        raise ValueError("IFR and FlashTrace prompt tokenizations do not align.")

    method_rows = (
        (
            "random_expected",
            None,
            0.0,
        ),
        (
            "ifr_last_token",
            _clean_prompt_vector(
                ifr_last_result.get_all_token_attrs(last_span)[1], prompt_length
            ),
            ifr_last_runtime,
        ),
        (
            "ifr_full_span_exhaustive",
            _clean_prompt_vector(
                ifr_full_result.get_all_token_attrs(full_span)[1], prompt_length
            ),
            ifr_full_runtime,
        ),
        (
            "flashtrace_full_span",
            _clean_prompt_vector(
                flashtrace_result.get_all_token_attrs(full_span)[1], prompt_length
            ),
            flashtrace_runtime,
        ),
    )

    rows = []
    for method, attribution, runtime in method_rows:
        metrics = (
            _random_metrics(ifr_full_result.prompt_tokens, case.injection_token_indices)
            if attribution is None
            else _localization_metrics(
                attribution,
                ifr_full_result.prompt_tokens,
                case.injection_token_indices,
            )
        )
        rows.append(
            {
                "case_id": case.spec.case_id,
                "suite": case.spec.suite,
                "user_task": case.spec.user_task,
                "injection_task": case.spec.injection_task,
                "target_tool": case.spec.target_tool,
                "method": method,
                "prompt_tokens": prompt_length,
                "injection_tokens": len(case.injection_token_indices),
                "target_tokens": target_length,
                "runtime_seconds": runtime,
                **metrics,
            }
        )
    return rows


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    methods = sorted({row["method"] for row in rows})
    metrics = (
        "recovery_at_5pct",
        "recovery_at_10pct",
        "recovery_at_20pct",
        "injection_mass_fraction",
        "runtime_seconds",
    )
    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        method_summary: dict[str, Any] = {
            "samples": len(selected),
            "pointing_game_rate": mean(float(row["pointing_game"]) for row in selected),
        }
        for metric in metrics:
            values = [float(row[metric]) for row in selected]
            method_summary[f"{metric}_mean"] = mean(values)
            method_summary[f"{metric}_std"] = pstdev(values) if len(values) > 1 else 0.0
        summary[method] = method_summary
    return summary


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_results_markdown(
    path: Path,
    summary: dict[str, dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    method_labels = {
        "random_expected": "Random expected",
        "ifr_last_token": "IFR last token",
        "ifr_full_span_exhaustive": "IFR exhaustive full span",
        "flashtrace_full_span": "FlashTrace full span",
    }
    lines = [
        "# AgentDojo localization pilot",
        "",
        f"- AgentDojo: `{AGENTDOJO_VERSION}` / suite `{BENCHMARK_VERSION}`",
        f"- Upstream traces: `{SOURCE_MODEL}` at `{UPSTREAM_COMMIT}`",
        f"- Attribution model: `{metadata['attribution_model']}`",
        f"- Samples: `{metadata['sample_count']}` successful utility + attack traces",
        "- Metric: injected-tool-output token recall among the top attributed prompt tokens",
        "",
        "| Method | Recovery@5% | Recovery@10% | Recovery@20% | Pointing | Injection mass | Runtime / sample |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in (
        "random_expected",
        "ifr_last_token",
        "ifr_full_span_exhaustive",
        "flashtrace_full_span",
    ):
        values = summary[method]
        lines.append(
            "| "
            + method_labels[method]
            + f" | {values['recovery_at_5pct_mean']:.3f}"
            + f" | {values['recovery_at_10pct_mean']:.3f}"
            + f" | {values['recovery_at_20pct_mean']:.3f}"
            + f" | {values['pointing_game_rate']:.3f}"
            + f" | {values['injection_mass_fraction_mean']:.3f}"
            + f" | {values['runtime_seconds_mean']:.2f}s |"
        )
    lines.extend(
        [
            "",
            "This is a smoke pilot, not a benchmark-level estimate: all cases use one source model, one attack family, and three hand-pinned successful traces.",
            "The malicious actions were generated by the upstream source model and teacher-forced into the white-box attribution model; this pilot does not measure same-model attack generation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_model(model_id: str, device: str) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        dtype = torch.bfloat16
        device_map: Any = {"": index}
    else:
        dtype = torch.float32
        device_map = None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="eager",
    )
    if device_map is None:
        model.to(device)
    model.eval()
    return model, tokenizer


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
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional local AgentDojo run root ending at runs/gpt-4o-2024-05-13.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parent / "results" / "pilot"
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Fetch, render, and replay validators without loading the attribution model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1 or args.num_samples > len(PILOT_CASES):
        raise SystemExit(f"--num-samples must be in [1, {len(PILOT_CASES)}]")

    # Tokenizer loading is enough to render and validate the prepared cases.
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    cases = [
        prepare_case(spec, tokenizer, args.cache_dir, args.source_root)
        for spec in PILOT_CASES[: args.num_samples]
    ]
    prepared_manifest = {
        "protocol": {
            "agentdojo_package_version": AGENTDOJO_VERSION,
            "benchmark_version": BENCHMARK_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "source_model": SOURCE_MODEL,
            "attribution_target_provenance": "teacher_forced_upstream_trace",
            "attack": ATTACK,
            "attribution_model": args.model,
            "n_hops": args.n_hops,
        },
        "cases": [
            {
                **asdict(case.spec),
                "source_trace": case.spec.relative_path.as_posix(),
                "prompt_tokens": len(
                    tokenizer(case.prompt, add_special_tokens=False).input_ids
                ),
                "target_tokens": _target_length_without_eos(tokenizer, case.target),
                "injection_tokens": len(case.injection_token_indices),
                "stored_utility": case.raw["utility"],
                "stored_attack_success": case.raw["security"],
                "replay_utility": case.replay_utility,
                "replay_attack_success": case.replay_attack_success,
                "replay_errors": list(case.replay_errors),
            }
            for case in cases
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prepared_manifest.json").write_text(
        json.dumps(prepared_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[prepared] {len(cases)} validator-replayed AgentDojo cases")
    if args.prepare_only:
        return

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

    rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        print(f"[run {index}/{len(cases)}] {case.spec.case_id}", flush=True)
        rows.extend(evaluate_case(case, tokenizer, ifr, flashtrace, args.n_hops))

    summary = _aggregate(rows)
    metadata = {
        "attribution_model": args.model,
        "sample_count": len(cases),
        "agentdojo_package_version": AGENTDOJO_VERSION,
        "benchmark_version": BENCHMARK_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_model": SOURCE_MODEL,
        "attribution_target_provenance": "teacher_forced_upstream_trace",
        "attack": ATTACK,
        "n_hops": args.n_hops,
    }
    _write_csv(args.output_dir / "per_sample.csv", rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"metadata": metadata, "methods": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_results_markdown(args.output_dir / "RESULTS.md", summary, metadata)
    print(f"[done] wrote {args.output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
