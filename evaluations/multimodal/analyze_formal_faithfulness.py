"""Paired bootstrap analysis for formal frozen-response faithfulness runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .strict_generation import read_jsonl


METRIC_DIRECTIONS = {
    "deletion_auc": "lower",
    "insertion_auc": "higher",
    "visual_mas": "lower",
}


def _interval(
    values: Sequence[float], rng: np.random.Generator, draws: int
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    indices = rng.integers(0, array.size, size=(draws, array.size))
    bootstrap = array[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": float(array.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _favorable_difference(metric: str, method: float, baseline: float) -> float:
    return method - baseline if METRIC_DIRECTIONS[metric] == "higher" else baseline - method


def _analyze_subset(
    sample_ids: Sequence[str],
    methods: Sequence[str],
    records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    rng: np.random.Generator,
    draws: int,
    positive_only: bool,
) -> dict[str, Any]:
    def value(method: str, sample_id: str, metric: str) -> float:
        faithfulness = records[method][sample_id]["faithfulness"]
        source = faithfulness.get("positive_only_ordering") if positive_only else faithfulness
        if source is None:
            raise ValueError("positive-only ordering curves are absent")
        return float(source[metric])

    estimates = {
        method: {
            metric: _interval(
                [value(method, sample_id, metric) for sample_id in sample_ids],
                rng,
                draws,
            )
            for metric in METRIC_DIRECTIONS
        }
        for method in methods
    }
    paired: dict[str, Any] = {}
    if "flashtrace" in methods:
        for baseline in methods:
            if baseline == "flashtrace":
                continue
            paired[baseline] = {}
            for metric in METRIC_DIRECTIONS:
                differences = [
                    _favorable_difference(
                        metric,
                        value("flashtrace", sample_id, metric),
                        value(baseline, sample_id, metric),
                    )
                    for sample_id in sample_ids
                ]
                interval = _interval(differences, rng, draws)
                array = np.asarray(differences)
                paired[baseline][metric] = {
                    **interval,
                    "wins": int(np.sum(array > 1e-12)),
                    "ties": int(np.sum(np.abs(array) <= 1e-12)),
                    "losses": int(np.sum(array < -1e-12)),
                }
    return {
        "samples": len(sample_ids),
        "sample_ids": list(sample_ids),
        "estimates": estimates,
        "flashtrace_favorable_difference": paired,
    }


def _thinking_buckets(
    sample_ids: Sequence[str], models: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    ranked = sorted(
        sample_ids,
        key=lambda sample_id: (
            int(models[sample_id]["generation_metadata"]["thinking_tokens"]),
            sample_id,
        ),
    )
    names = ("short", "medium", "long")
    return {
        sample_id: names[min(2, index * 3 // len(ranked))]
        for index, sample_id in enumerate(ranked)
    }


def analyze(
    faithfulness_dir: Path,
    *,
    generation_evaluation: Path | None = None,
    model_output: Path | None = None,
    draws: int = 50_000,
    seed: int = 17,
) -> dict[str, Any]:
    summary = json.loads((faithfulness_dir / "summary.json").read_text())
    methods = list(summary["methods"])
    common_ids = list(summary["common_sample_ids"])
    records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in read_jsonl(faithfulness_dir / "faithfulness_records.jsonl"):
        if record.get("status") == "ok" and record["sample_id"] in common_ids:
            records[record["method"]][record["sample_id"]] = record
    missing = {
        method: sorted(set(common_ids) - set(records[method]))
        for method in methods
        if set(common_ids) - set(records[method])
    }
    if missing:
        raise ValueError(f"faithfulness analysis is not paired: {missing}")

    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "schema_version": 1,
        "faithfulness_dir": str(faithfulness_dir),
        "generation_evaluation": (
            str(generation_evaluation)
            if generation_evaluation is not None
            else None
        ),
        "model_output": str(model_output) if model_output is not None else None,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "metric_directions": METRIC_DIRECTIONS,
        "overall": _analyze_subset(
            common_ids, methods, records, rng=rng, draws=draws, positive_only=False
        ),
    }
    has_positive = all(
        "positive_only_ordering" in records[method][sample_id]["faithfulness"]
        for method in methods
        for sample_id in common_ids
    )
    result["positive_only_available"] = has_positive
    if has_positive:
        result["positive_only_ordering"] = _analyze_subset(
            common_ids, methods, records, rng=rng, draws=draws, positive_only=True
        )

    if generation_evaluation is not None:
        evaluations = {
            record["sample_id"]: record
            for record in read_jsonl(generation_evaluation)
        }
        fully_correct = [
            sample_id
            for sample_id in common_ids
            if (evaluations.get(sample_id, {}).get("semantic_correctness") or {}).get(
                "label"
            )
            == "fully"
        ]
        result["fully_correct_subset"] = (
            _analyze_subset(
                fully_correct,
                methods,
                records,
                rng=rng,
                draws=draws,
                positive_only=False,
            )
            if fully_correct
            else {"samples": 0, "status": "no_fully_correct_labels"}
        )

    if model_output is not None and {"ifr-span", "flashtrace"}.issubset(methods):
        models = {record["sample_id"]: record for record in read_jsonl(model_output)}
        buckets = _thinking_buckets(common_ids, models)
        result["recursion_by_thinking_bucket"] = {
            bucket: _analyze_subset(
                [sample_id for sample_id in common_ids if buckets[sample_id] == bucket],
                ("ifr-span", "flashtrace"),
                records,
                rng=rng,
                draws=draws,
                positive_only=False,
            )
            for bucket in ("short", "medium", "long")
        }
    return result


def _markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Formal frozen-response faithfulness analysis",
        "",
        f"Common samples: {analysis['overall']['samples']}; "
        f"bootstrap draws: {analysis['bootstrap_draws']}.",
        "",
        "| method | deletion AUC | insertion AUC | Visual-MAS |",
        "|---|---:|---:|---:|",
    ]
    for method, estimates in analysis["overall"]["estimates"].items():
        lines.append(
            f"| {method} | {estimates['deletion_auc']['mean']:.4f} | "
            f"{estimates['insertion_auc']['mean']:.4f} | "
            f"{estimates['visual_mas']['mean']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faithfulness-dir", type=Path, required=True)
    parser.add_argument("--generation-evaluation", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    analysis = analyze(
        args.faithfulness_dir,
        generation_evaluation=args.generation_evaluation,
        model_output=args.model_output,
        draws=args.draws,
        seed=args.seed,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(
        _markdown(analysis) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "json": str(args.output_json),
                "markdown": str(args.output_markdown),
                "common_samples": analysis["overall"]["samples"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
