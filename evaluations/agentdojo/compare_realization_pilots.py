#!/usr/bin/env python3
"""Compare aggregate prefix-forcing and sparse-correction pilot reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _method_summary(report: dict[str, Any], prefix: str) -> dict[str, Any]:
    selected = {
        stage: count
        for stage, count in report.get("selected_stage_counts", {}).items()
        if stage.startswith(prefix)
    }
    stages = {
        stage: values
        for stage, values in report.get("attempted_stages", {}).items()
        if stage.startswith(prefix)
    }
    return {
        "confirmed_case_count": sum(selected.values()),
        "selected_stage_counts": selected,
        "attempted_stages": stages,
        "mean_reference_token_count_at_confirmation": report["overall"].get(
            "mean_reference_token_count"
        ),
        "mean_reference_token_fraction_at_confirmation": report["overall"].get(
            "mean_reference_token_fraction"
        ),
        "minimum_reasoning_token_count_at_confirmation": report["overall"].get(
            "minimum_reasoning_token_count"
        ),
        "short_bridge_count": report["overall"].get("short_bridge_count"),
        "non_model_bridge_count": report["overall"].get(
            "non_model_bridge_count"
        ),
    }


def compare(prefix: dict[str, Any], sparse: dict[str, Any]) -> dict[str, Any]:
    prefix_cases = int(prefix["overall"]["case_count"])
    sparse_cases = int(sparse["overall"]["case_count"])
    if prefix_cases != sparse_cases:
        raise ValueError("Pilot reports have different case counts.")
    prefix_natural = {
        stage: count
        for stage, count in prefix.get("selected_stage_counts", {}).items()
        if stage.startswith("natural_")
    }
    sparse_natural = {
        stage: count
        for stage, count in sparse.get("selected_stage_counts", {}).items()
        if stage.startswith("natural_")
    }
    return {
        "schema_version": 1,
        "case_count": prefix_cases,
        "contains_sample_text": False,
        "natural_sampling": {
            "prefix_run_selected_stage_counts": prefix_natural,
            "sparse_run_selected_stage_counts": sparse_natural,
            "consistent": prefix_natural == sparse_natural,
        },
        "prefix_forcing": _method_summary(prefix, "prefix_forcing_"),
        "sparse_correction": _method_summary(sparse, "sparse_correction_"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-report", type=Path, required=True)
    parser.add_argument("--sparse-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = compare(
        json.loads(args.prefix_report.read_text(encoding="utf-8")),
        json.loads(args.sparse_report.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
