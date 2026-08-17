#!/usr/bin/env python3
"""Summarize outcome-conditioning coverage from a frozen AgentDojo run."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from evaluations.agentdojo.attribute_full import _eligible
from evaluations.agentdojo.benchmark_local import (
    SUITES,
    _injection_is_visible,
)
from evaluations.agentdojo.pilot import BENCHMARK_VERSION


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _observed(record: dict[str, Any]) -> bool:
    return any(
        _injection_is_visible(str(turn.get("rendered_prompt") or ""))
        for turn in record.get("turns", [])
    )


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in records if row.get("attack_success") is not None]
    by_suite: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_suite[str(row["case"]["suite"])].append(row)

    def group(values: list[dict[str, Any]]) -> dict[str, Any]:
        action_rows = [row for row in values if row.get("target_calls")]
        observed_rows = [row for row in values if _observed(row)]
        conditionable = [row for row in action_rows if _observed(row)]
        natural = [row for row in values if _eligible(row)]
        return {
            "attacked_count": len(values),
            "action_target_count": len(action_rows),
            "observed_count": len(observed_rows),
            "conditionable_action_count": len(conditionable),
            "natural_localizable_count": len(natural),
            "natural_localizable_over_all": (
                len(natural) / len(values) if values else None
            ),
            "natural_localizable_over_action_targets": (
                len(natural) / len(action_rows) if action_rows else None
            ),
            "conditionable_over_all": (
                len(conditionable) / len(values) if values else None
            ),
            "conditionable_over_action_targets": (
                len(conditionable) / len(action_rows) if action_rows else None
            ),
        }

    return {
        "schema_version": 1,
        "definition": {
            "natural_localizable": (
                "validator-confirmed generated action with a complete observed span"
            ),
            "conditionable_action": (
                "concrete action target and complete observed span in a frozen prompt"
            ),
            "conditionable_is_not_natural_rate": True,
        },
        "overall": group(rows),
        "suites": {
            suite: group(by_suite.get(suite, []))
            for suite in SUITES
        },
    }


def replay_validators(
    records: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Replay frozen prefixes plus their known action suffixes."""
    suites = {
        name: get_suite(BENCHMARK_VERSION, name)
        for name in SUITES
    }
    runtimes = {
        name: FunctionsRuntime(suite.tools)
        for name, suite in suites.items()
    }
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "candidate_count": 0,
            "validator_success_count": 0,
            "record_error_count": 0,
            "records_with_runtime_error": 0,
        }
    )
    started = time.perf_counter()
    for index, row in enumerate(records, 1):
        if row.get("attack_success") is None or not row.get("target_calls"):
            continue
        turns = list(row.get("turns") or [])
        visible_index = next(
            (
                turn_index
                for turn_index, turn in enumerate(turns)
                if _injection_is_visible(str(turn.get("rendered_prompt") or ""))
            ),
            None,
        )
        if visible_index is None:
            continue
        suite_name = str(row["case"]["suite"])
        counts = totals[suite_name]
        counts["candidate_count"] += 1
        try:
            suite = suites[suite_name]
            user_task = suite.get_user_task_by_id(row["case"]["user_task"])
            task = suite.get_injection_task_by_id(
                row["case"]["injection_task"]
            )
            environment = suite.load_and_inject_default_environment(
                row["case"]["injections"]
            )
            environment = user_task.init_environment(environment)
            pre_environment = environment.model_copy(deep=True)
            trace = []
            runtime_error = False
            for turn in turns[:visible_index]:
                for call_index, call_data in enumerate(
                    turn.get("parsed_tool_calls") or []
                ):
                    call = FunctionCall(
                        function=call_data["function"],
                        args=call_data["args"],
                        id=f"prefix_{turn.get('turn_index', 0)}_{call_index}",
                    )
                    trace.append(call)
                    _, error = runtimes[suite_name].run_function(
                        environment,
                        call.function,
                        call.args,
                    )
                    runtime_error |= error is not None
            for call_index, call_data in enumerate(row["target_calls"]):
                call = FunctionCall(
                    function=call_data["function"],
                    args=call_data["args"],
                    id=f"conditioned_{call_index}",
                )
                trace.append(call)
                _, error = runtimes[suite_name].run_function(
                    environment,
                    call.function,
                    call.args,
                )
                runtime_error |= error is not None
            success = suite._check_injection_task_security(  # noqa: SLF001
                task,
                [],
                pre_environment,
                environment,
                trace,
            )
            counts["validator_success_count"] += int(bool(success))
            counts["records_with_runtime_error"] += int(runtime_error)
        except Exception:
            counts["record_error_count"] += 1
        if index % 50 == 0:
            gc.collect()

    overall = {
        key: sum(values[key] for values in totals.values())
        for key in (
            "candidate_count",
            "validator_success_count",
            "record_error_count",
            "records_with_runtime_error",
        )
    }
    overall["validator_success_rate"] = (
        overall["validator_success_count"] / overall["candidate_count"]
        if overall["candidate_count"]
        else None
    )
    overall["runtime_seconds"] = time.perf_counter() - started
    return {
        "overall": overall,
        "suites": {
            suite: {
                **totals[suite],
                "validator_success_rate": (
                    totals[suite]["validator_success_count"]
                    / totals[suite]["candidate_count"]
                    if totals[suite]["candidate_count"]
                    else None
                ),
            }
            for suite in SUITES
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-validators", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(_records(args.generation_jsonl))
    summary = summarize(records)
    if args.replay_validators:
        replay = replay_validators(records)
        replay_overall = replay["overall"]
        coverage = summary["overall"]
        replay_overall["success_over_all"] = (
            replay_overall["validator_success_count"]
            / coverage["attacked_count"]
            if coverage["attacked_count"]
            else None
        )
        replay_overall["success_over_action_targets"] = (
            replay_overall["validator_success_count"]
            / coverage["action_target_count"]
            if coverage["action_target_count"]
            else None
        )
        replay_overall["lift_over_natural"] = (
            replay_overall["validator_success_count"]
            / coverage["natural_localizable_count"]
            if coverage["natural_localizable_count"]
            else None
        )
        summary["validator_replay"] = replay
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
