#!/usr/bin/env python3
"""Render deterministic, structurally redacted AgentDojo examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


SUITES = ("banking", "slack", "travel", "workspace")
FORBIDDEN_KEYS = {
    "args",
    "completion",
    "content",
    "injections",
    "messages",
    "prompt",
    "raw_result",
    "reasoning",
    "rendered_prompt",
    "tool_results",
}
FLASHTRACE_METHOD = "flashtrace_generated_action_via_reasoning"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _assert_publishable(value: Any) -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_KEYS.intersection(value)
        if forbidden:
            raise ValueError(
                f"Redacted input contains forbidden keys: {sorted(forbidden)}"
            )
        for child in value.values():
            _assert_publishable(child)
    elif isinstance(value, list):
        for child in value:
            _assert_publishable(child)


def _tex(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def select_examples(
    generation_rows: Iterable[dict[str, Any]],
    attribution_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    generation = list(generation_rows)
    for row in generation:
        _assert_publishable(row)
    attribution = list(attribution_rows)
    for row in attribution:
        _assert_publishable(row)
    successful = {
        row["case_id"]: row
        for row in attribution
        if row.get("status") == "success"
    }
    selected = []
    seen_suites: set[str] = set()
    for row in generation:
        case_id = row["case"]["case_id"]
        suite = row["case"]["suite"]
        if suite in seen_suites or case_id not in successful:
            continue
        attribution = successful[case_id]["attribution"]
        flashtrace = attribution["methods"][FLASHTRACE_METHOD][
            "localization_metrics"
        ]
        selected.append(
            {
                "case_id": case_id,
                "suite": suite,
                "utility": row["utility"],
                "attack_success": row["attack_success"],
                "turn_count": len(row["turns"]),
                "generated_token_count": sum(
                    turn["generated_token_count_with_eos_if_emitted"]
                    for turn in row["turns"]
                ),
                "function_trace": [
                    call["function"] for call in row["function_trace"]
                ],
                "target_functions": row["target_functions"],
                "flashtrace_recovery_at_10pct": flashtrace[
                    "recovery_at_10pct"
                ],
                "flashtrace_injection_mass_fraction": flashtrace[
                    "injection_mass_fraction"
                ],
            }
        )
        seen_suites.add(suite)
        if len(seen_suites) == len(SUITES):
            break
    return selected


def render_tex(rows: Iterable[dict[str, Any]]) -> str:
    output = []
    for row in rows:
        trace = r"$\rightarrow$".join(
            _tex(function) for function in row["function_trace"]
        )
        targets = ", ".join(_tex(value) for value in row["target_functions"])
        output.append(
            f"{_tex(row['suite'].title())} & {_tex(row['case_id'])} & "
            f"{row['turn_count']} & {row['generated_token_count']} & "
            f"{trace or '--'} & {targets or '--'} & "
            f"{row['flashtrace_recovery_at_10pct']:.3f} & "
            f"{row['flashtrace_injection_mass_fraction']:.3f} \\\\"
        )
    return "\n".join(output) + ("\n" if output else "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-redacted", type=Path, required=True)
    parser.add_argument("--attribution-redacted", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-tex", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = select_examples(
        _jsonl(args.generation_redacted),
        _jsonl(args.attribution_redacted),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.output_tex.write_text(render_tex(selected), encoding="utf-8")


if __name__ == "__main__":
    main()
