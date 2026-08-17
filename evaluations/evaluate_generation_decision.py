#!/usr/bin/env python3
"""Apply the frozen external-verifier gates to the complete 8B pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(
    protocol: dict[str, Any],
    aider_rows: list[dict[str, Any]],
    agentdojo_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    configured = protocol["decision_pilot"]
    aider_config = configured["aider_polyglot"]
    dojo_config = configured["agentdojo"]
    if len(aider_rows) != int(aider_config["cases"]):
        raise ValueError(
            f"Aider pilot incomplete: {len(aider_rows)}/{aider_config['cases']}"
        )
    if len(agentdojo_rows) != int(dojo_config["cases"]):
        raise ValueError(
            "AgentDojo pilot incomplete: "
            f"{len(agentdojo_rows)}/{dojo_config['cases']}"
        )

    clean = [row for row in agentdojo_rows if row["attack_success"] is None]
    attacked = [
        row for row in agentdojo_rows if row["attack_success"] is not None
    ]
    turns = [turn for row in agentdojo_rows for turn in row["turns"]]

    def valid_reasoning_boundary(turn: dict[str, Any]) -> bool:
        if "valid_reasoning_boundary" in turn:
            return bool(turn["valid_reasoning_boundary"])
        return "</think>" in str(turn.get("completion") or "")

    incomplete = [
        row
        for row in agentdojo_rows
        if row["stop_reason"] != "assistant_final_answer"
        or any(not valid_reasoning_boundary(turn) for turn in row["turns"])
    ]
    aider_turns = [turn for row in aider_rows for turn in row["turns"]]
    infrastructure_errors = [
        turn["verification"]["infrastructure_error"]
        for turn in aider_turns
        if turn["verification"].get("infrastructure_error")
    ]
    observed = {
        "agentdojo_clean_utility_rate": (
            sum(row["utility"] for row in clean) / len(clean)
        ),
        "agentdojo_attacked_utility_rate": (
            sum(row["utility"] for row in attacked) / len(attacked)
        ),
        "agentdojo_parse_error_turn_rate": (
            sum(bool(turn["parse_errors"]) for turn in turns) / len(turns)
        ),
        "agentdojo_incomplete_case_rate": len(incomplete) / len(agentdojo_rows),
        "aider_final_execution_pass_rate": (
            sum(row["final_passed"] for row in aider_rows) / len(aider_rows)
        ),
        "aider_format_compliance_rate": (
            sum(turn["parsed_edit"]["format_compliant"] for turn in aider_turns)
            / len(aider_turns)
        ),
        "aider_infrastructure_error_count": len(infrastructure_errors),
        "aider_infrastructure_error_categories": sorted(set(infrastructure_errors)),
    }
    checks = {
        "agentdojo_clean_utility": (
            observed["agentdojo_clean_utility_rate"]
            >= dojo_config["minimum_clean_utility_rate"]
        ),
        "agentdojo_attacked_utility": (
            observed["agentdojo_attacked_utility_rate"]
            >= dojo_config["minimum_attacked_utility_rate"]
        ),
        "agentdojo_parse_errors": (
            observed["agentdojo_parse_error_turn_rate"]
            <= dojo_config["maximum_parse_error_turn_rate"]
        ),
        "agentdojo_incomplete_cases": (
            observed["agentdojo_incomplete_case_rate"]
            <= dojo_config["maximum_incomplete_case_rate"]
        ),
        "aider_final_execution": (
            observed["aider_final_execution_pass_rate"]
            >= aider_config["minimum_final_execution_pass_rate"]
        ),
        "aider_format_compliance": (
            observed["aider_format_compliance_rate"]
            >= aider_config["minimum_format_compliance_rate"]
        ),
        "aider_harness": not infrastructure_errors,
    }
    candidate_passed = all(checks.values())
    selected = (
        protocol["generation_candidate"]
        if candidate_passed
        else protocol["generation_fallback"]
    )
    return {
        "schema_version": 1,
        "decision": (
            "retain_candidate_8b"
            if candidate_passed
            else "global_fallback_to_qwen36_27b"
        ),
        "candidate_passed_all_frozen_gates": candidate_passed,
        "checks": checks,
        "observed": observed,
        "selected_formal_generator": selected,
        "attribution_model": protocol["attribution_model"],
        "judge": configured["judge"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("evaluations/agent_protocol.json"),
    )
    parser.add_argument("--aider-trajectories", type=Path, required=True)
    parser.add_argument("--agentdojo-trajectories", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluations/generation_decision.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    decision = evaluate(
        protocol,
        _rows(args.aider_trajectories),
        _rows(args.agentdojo_trajectories),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
