#!/usr/bin/env python3
"""Compare aggregate natural-search and diagnostic-control pilots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluations.agentdojo.diagnose_coverage import summarize


def _load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strategy_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    overall = summarize(records)["overall"]
    return {
        key: overall.get(key)
        for key in (
            "case_count",
            "action_target_count",
            "realized_count",
            "realized_rate_over_action_targets",
            "all_model_generated_case_count",
            "model_generated_token_fraction",
            "mean_generation_requests_per_case",
            "mean_sampled_tokens_per_case",
            "mean_scored_tokens_per_case",
            "mean_generation_seconds_per_case",
            "mean_requests_per_successful_case",
            "mean_sampled_tokens_per_successful_case",
            "minimum_model_reasoning_token_count",
            "missing_reasoning_boundary_count",
            "non_model_natural_bridge_count",
            "mean_executed_attempts",
            "trajectory_nonfirst_selection_rate",
            "mean_selected_attempt_index",
            "reasoning_search_case_count",
            "reasoning_rerank_nonfirst_selection_rate",
            "mean_rerank_exact_score_gain",
            "mean_rerank_conditional_logprob_gain",
            "mean_distinct_reasoning_fraction",
            "candidate_temperature_shift_case_rate",
            "mean_reference_token_count",
            "mean_reference_token_fraction",
        )
    }


def compare(inputs: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    if not inputs:
        raise ValueError("At least one pilot is required.")
    id_sets = {
        name: {str(row["case"]["case_id"]) for row in records}
        for name, records in inputs.items()
    }
    first_ids = next(iter(id_sets.values()))
    if any(ids != first_ids for ids in id_sets.values()):
        raise ValueError("Pilot manifests do not contain the same case ids.")
    return {
        "schema_version": 1,
        "contains_sample_text": False,
        "same_case_manifest": True,
        "case_count": len(first_ids),
        "strategies": {
            name: _strategy_summary(records)
            for name, records in inputs.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-of", type=Path, required=True)
    parser.add_argument("--reasoning-rerank", type=Path, required=True)
    parser.add_argument("--prefix-control", type=Path)
    parser.add_argument("--sparse-control", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = {
        "adaptive_best_of_n": args.best_of,
        "reasoning_rerank": args.reasoning_rerank,
    }
    if args.prefix_control:
        paths["prefix_control"] = args.prefix_control
    if args.sparse_control:
        paths["sparse_control"] = args.sparse_control
    payload = compare({name: _load(path) for name, path in paths.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
