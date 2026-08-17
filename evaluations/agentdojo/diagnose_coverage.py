#!/usr/bin/env python3
"""Aggregate realization-stage coverage without emitting sample contents."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def _records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _stage(record: dict[str, Any]) -> str:
    return str(
        record.get("realization_stage")
        or record.get("selection", {}).get("realization_stage")
        or "unrecorded"
    )


def _interventions(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(turn["output_intervention"])
        for turn in record.get("turns", [])
        if turn.get("output_intervention", {}).get("mode")
        in {"teacher_forcing_prefix", "sparse_correction"}
    ]


def _attempts(record: dict[str, Any]) -> list[dict[str, Any]]:
    return list(record.get("selection", {}).get("attempts") or [])


def _generation_audits(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(turn["generation_audit"])
        for turn in record.get("turns", [])
        if turn.get("generation_audit")
    ]


def _natural_searches(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(turn["natural_search"])
        for turn in record.get("turns", [])
        if turn.get("natural_search")
    ]


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return only aggregate counts and provenance diagnostics."""

    rows = [row for row in records if row.get("attack_success") is not None]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_stage(row)].append(row)

    def aggregate(values: list[dict[str, Any]]) -> dict[str, Any]:
        action_rows = [row for row in values if row.get("target_calls")]
        realized = [row for row in values if row.get("generated_target_action")]
        interventions = [item for row in values for item in _interventions(row)]
        reference_counts = [
            int(
                item.get("reference_token_count")
                if item.get("reference_token_count") is not None
                else item.get("forced_token_count")
                or 0
            )
            for item in interventions
        ]
        reference_fractions = [
            float(
                item.get("reference_token_fraction")
                if item.get("reference_token_fraction") is not None
                else item.get("forced_token_fraction")
                or 0.0
            )
            for item in interventions
        ]
        reasoning_counts = [
            int(item.get("reasoning_token_count") or 0) for item in interventions
        ]
        audits = [item for row in values for item in _generation_audits(row)]
        searches = [item for row in values for item in _natural_searches(row)]
        per_case_requests = [
            sum(int(item.get("generation_request_count") or 0) for item in _attempts(row))
            for row in values
        ]
        per_case_tokens = [
            sum(int(item.get("sampled_token_count") or 0) for item in _attempts(row))
            for row in values
        ]
        per_case_scored_tokens = [
            sum(int(item.get("scored_token_count") or 0) for item in _attempts(row))
            for row in values
        ]
        per_case_seconds = [
            sum(float(item.get("generation_seconds") or 0.0) for item in _attempts(row))
            for row in values
        ]
        successful_indices = [
            index for index, row in enumerate(values) if row.get("generated_target_action")
        ]
        audited_tokens = sum(
            int(item.get("model_generated_token_count") or 0)
            + int(item.get("reference_token_count") or 0)
            for item in audits
        )
        model_tokens = sum(
            int(item.get("model_generated_token_count") or 0) for item in audits
        )
        selected_attempt_indices = [
            int(row.get("selection", {}).get("selected_attempt_index") or 0)
            for row in values
        ]
        return {
            "case_count": len(values),
            "action_target_count": len(action_rows),
            "realized_count": len(realized),
            "realized_rate_over_action_targets": (
                len(realized) / len(action_rows) if action_rows else None
            ),
            "intervention_case_count": sum(bool(_interventions(row)) for row in values),
            "intervention_method_counts": {
                method: sum(item.get("mode") == method for item in interventions)
                for method in sorted(
                    {str(item.get("mode")) for item in interventions}
                )
            },
            "mean_reference_token_count": (
                sum(reference_counts) / len(reference_counts)
                if reference_counts
                else None
            ),
            "mean_reference_token_fraction": (
                sum(reference_fractions) / len(reference_fractions)
                if reference_fractions
                else None
            ),
            "mean_decode_request_count": (
                sum(int(item.get("decode_request_count") or 1) for item in interventions)
                / len(interventions)
                if interventions
                else None
            ),
            "minimum_reasoning_token_count": (
                min(reasoning_counts) if reasoning_counts else None
            ),
            "short_bridge_count": sum(count <= 2 for count in reasoning_counts),
            "non_model_bridge_count": sum(
                not bool(item.get("bridge_is_fully_model_generated"))
                for item in interventions
            ),
            "all_model_generated_case_count": sum(
                bool(_generation_audits(row))
                and all(
                    bool(item.get("all_completion_tokens_model_generated"))
                    for item in _generation_audits(row)
                )
                for row in values
            ),
            "model_generated_token_fraction": (
                model_tokens / audited_tokens if audited_tokens else None
            ),
            "minimum_model_reasoning_token_count": (
                min(
                    int(item.get("reasoning_token_count") or 0)
                    for item in audits
                    if item.get("reasoning_token_count")
                )
                if any(item.get("reasoning_token_count") for item in audits)
                else None
            ),
            "missing_reasoning_boundary_count": sum(
                not int(item.get("reasoning_token_count") or 0)
                for item in audits
            ),
            "non_model_natural_bridge_count": sum(
                bool(item.get("reasoning_token_count"))
                and not bool(item.get("bridge_is_fully_model_generated"))
                for item in audits
            ),
            "mean_generation_requests_per_case": (
                sum(per_case_requests) / len(per_case_requests)
                if per_case_requests
                else None
            ),
            "mean_sampled_tokens_per_case": (
                sum(per_case_tokens) / len(per_case_tokens)
                if per_case_tokens
                else None
            ),
            "mean_scored_tokens_per_case": (
                sum(per_case_scored_tokens) / len(per_case_scored_tokens)
                if per_case_scored_tokens
                else None
            ),
            "mean_generation_seconds_per_case": (
                sum(per_case_seconds) / len(per_case_seconds)
                if per_case_seconds
                else None
            ),
            "mean_requests_per_successful_case": (
                sum(per_case_requests[index] for index in successful_indices)
                / len(successful_indices)
                if successful_indices
                else None
            ),
            "mean_sampled_tokens_per_successful_case": (
                sum(per_case_tokens[index] for index in successful_indices)
                / len(successful_indices)
                if successful_indices
                else None
            ),
            "reasoning_search_case_count": sum(
                bool(_natural_searches(row)) for row in values
            ),
            "reasoning_rerank_nonfirst_selection_rate": (
                sum(not bool(item.get("selected_was_first")) for item in searches)
                / len(searches)
                if searches
                else None
            ),
            "mean_rerank_exact_score_gain": (
                sum(float(item.get("exact_score_gain_over_first") or 0) for item in searches)
                / len(searches)
                if searches
                else None
            ),
            "mean_rerank_conditional_logprob_gain": (
                sum(
                    float(item.get("conditional_logprob_gain_over_first") or 0)
                    for item in searches
                )
                / len(searches)
                if searches
                else None
            ),
            "mean_distinct_reasoning_fraction": (
                sum(
                    int(item.get("distinct_reasoning_count") or 0)
                    / max(1, int(item.get("valid_candidate_count") or 0))
                    for item in searches
                )
                / len(searches)
                if searches
                else None
            ),
            "candidate_temperature_shift_case_rate": (
                sum(
                    item.get("reasoning_sampling_temperature")
                    != item.get("final_action_sampling_temperature")
                    for item in searches
                )
                / len(searches)
                if searches
                else None
            ),
            "mean_executed_attempts": (
                sum(len(_attempts(row)) for row in values) / len(values)
                if values
                else None
            ),
            "trajectory_nonfirst_selection_rate": (
                sum(index > 0 for index in selected_attempt_indices) / len(values)
                if values
                else None
            ),
            "mean_selected_attempt_index": (
                sum(selected_attempt_indices) / len(values) if values else None
            ),
        }

    stage_summary = {
        stage: aggregate(values) for stage, values in sorted(groups.items())
    }
    attempted_stage_counts: dict[str, int] = defaultdict(int)
    attempted_stage_passes: dict[str, int] = defaultdict(int)
    attempted_stage_skips: dict[str, int] = defaultdict(int)
    attempted_stage_interventions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attempted_stage_seconds: dict[str, list[float]] = defaultdict(list)
    attempted_stage_requests: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        for stage in row.get("selection", {}).get("stages", []):
            stage_name = str(stage.get("name") or "unrecorded")
            attempted_stage_counts[stage_name] += 1
            attempted_stage_skips[stage_name] += int(
                bool(stage.get("skipped_reason"))
            )
        for attempt in row.get("selection", {}).get("attempts", []):
            stage = str(attempt.get("realization_stage") or "unrecorded")
            attempted_stage_passes[stage] += int(
                bool(attempt.get("generated_target_action"))
            )
            attempted_stage_skips[stage] += int(
                bool(attempt.get("realization_skip_reason"))
            )
            attempted_stage_seconds[stage].append(
                float(attempt.get("generation_seconds") or 0.0)
            )
            attempted_stage_requests[stage].append(
                int(attempt.get("generation_request_count") or 0)
            )
            if attempt.get("intervention"):
                attempted_stage_interventions[stage].append(
                    dict(attempt["intervention"])
                )
    return {
        "schema_version": 1,
        "definition": {
            "realized": "official-validator-confirmed target action",
            "short_bridge": "recorded reasoning token count at most two",
            "contains_sample_text": False,
        },
        "overall": aggregate(rows),
        "selected_stage_counts": {
            stage: len(values) for stage, values in sorted(groups.items())
        },
        "attempted_stage_counts": dict(sorted(attempted_stage_counts.items())),
        "attempted_stages": {
            stage: {
                "case_count": count,
                "validator_pass_count": attempted_stage_passes[stage],
                "skip_count": attempted_stage_skips[stage],
                "mean_generation_seconds": (
                    sum(attempted_stage_seconds[stage])
                    / len(attempted_stage_seconds[stage])
                    if attempted_stage_seconds[stage]
                    else None
                ),
                "mean_generation_request_count": (
                    sum(attempted_stage_requests[stage])
                    / len(attempted_stage_requests[stage])
                    if attempted_stage_requests[stage]
                    else None
                ),
                "mean_reference_token_count": (
                    sum(
                        int(
                            item.get("reference_token_count")
                            if item.get("reference_token_count") is not None
                            else item.get("forced_token_count")
                            or 0
                        )
                        for item in attempted_stage_interventions[stage]
                    )
                    / len(attempted_stage_interventions[stage])
                    if attempted_stage_interventions[stage]
                    else None
                ),
                "mean_reference_token_fraction": (
                    sum(
                        float(
                            item.get("reference_token_fraction")
                            if item.get("reference_token_fraction") is not None
                            else item.get("forced_token_fraction")
                            or 0.0
                        )
                        for item in attempted_stage_interventions[stage]
                    )
                    / len(attempted_stage_interventions[stage])
                    if attempted_stage_interventions[stage]
                    else None
                ),
                "minimum_reasoning_token_count": (
                    min(
                        int(item.get("reasoning_token_count") or 0)
                        for item in attempted_stage_interventions[stage]
                    )
                    if attempted_stage_interventions[stage]
                    else None
                ),
                "short_bridge_count": sum(
                    int(item.get("reasoning_token_count") or 0) <= 2
                    for item in attempted_stage_interventions[stage]
                ),
                "non_model_bridge_count": sum(
                    not bool(item.get("bridge_is_fully_model_generated"))
                    for item in attempted_stage_interventions[stage]
                ),
            }
            for stage, count in sorted(attempted_stage_counts.items())
        },
        "stages": stage_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize(_records(args.generation_jsonl))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
