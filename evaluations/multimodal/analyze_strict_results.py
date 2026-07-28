"""Create paired bootstrap analyses for strict multimodal attribution runs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .strict_generation import read_jsonl


METRICS = (
    "pointing_game",
    "energy_in_mask",
    "evidence_rank_auc",
    "top_evidence_iou",
    "recovery_at_1pct",
    "recovery_at_5pct",
    "recovery_at_10pct",
    "recovery_at_20pct",
)


def _interval(values: np.ndarray, rng: np.random.Generator, draws: int) -> dict[str, float]:
    n = values.size
    bootstrap = values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def analyze(
    manifest: Path,
    attribution_dir: Path,
    *,
    draws: int = 10_000,
    seed: int = 17,
    metric_prefix: str = "",
) -> dict[str, Any]:
    summary = json.loads((attribution_dir / "summary.json").read_text())
    methods = summary["requested_methods"]
    common_ids = summary["common_sample_ids"]
    datasets = {record["sample_id"]: record for record in read_jsonl(manifest)}
    records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in read_jsonl(attribution_dir / "attribution_records.jsonl"):
        if record.get("status") == "ok" and record["sample_id"] in common_ids:
            records[record["method"]][record["sample_id"]] = record

    missing = {
        method: sorted(set(common_ids) - set(records[method]))
        for method in methods
        if set(common_ids) - set(records[method])
    }
    if missing:
        raise ValueError(f"paired analysis has missing method/sample records: {missing}")

    rng = np.random.default_rng(seed)

    def metric_value(method: str, sample_id: str, metric: str) -> float:
        return float(
            records[method][sample_id]["localization"][f"{metric_prefix}{metric}"]
        )

    estimates: dict[str, dict[str, dict[str, float]]] = {}
    ranks: dict[str, list[str]] = {}
    for metric in METRICS:
        estimates[metric] = {}
        for method in methods:
            values = np.asarray(
                [metric_value(method, sample_id, metric) for sample_id in common_ids],
                dtype=np.float64,
            )
            estimates[metric][method] = _interval(values, rng, draws)
        ranks[metric] = sorted(
            methods, key=lambda method: estimates[metric][method]["mean"], reverse=True
        )

    paired_flashtrace: dict[str, dict[str, dict[str, float | int]]] = {}
    if "flashtrace" in methods:
        for metric in METRICS:
            paired_flashtrace[metric] = {}
            flashtrace = np.asarray(
                [
                    metric_value("flashtrace", sample_id, metric)
                    for sample_id in common_ids
                ]
            )
            for method in methods:
                if method == "flashtrace":
                    continue
                baseline = np.asarray(
                    [
                        metric_value(method, sample_id, metric)
                        for sample_id in common_ids
                    ]
                )
                differences = flashtrace - baseline
                interval = _interval(differences, rng, draws)
                paired_flashtrace[metric][method] = {
                    **interval,
                    "wins": int(np.sum(differences > 1e-12)),
                    "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                    "losses": int(np.sum(differences < -1e-12)),
                }

    group_by_sample: dict[str, str] = {}
    for sample_id in common_ids:
        metadata = datasets[sample_id]["evaluation"].get("metadata", {})
        group_by_sample[sample_id] = str(
            metadata.get("reasoning_family", metadata.get("stratum", "all"))
        )
    per_group: dict[str, dict[str, dict[str, float]]] = {}
    per_group_paired: dict[str, Any] = {}
    for group in sorted(set(group_by_sample.values())):
        ids = [sample_id for sample_id in common_ids if group_by_sample[sample_id] == group]
        per_group[group] = {}
        for method in methods:
            per_group[group][method] = {
                metric: float(
                    np.mean(
                        [metric_value(method, sample_id, metric) for sample_id in ids]
                    )
                )
                for metric in METRICS
            }
            per_group[group][method]["samples"] = len(ids)
        group_estimates: dict[str, Any] = {}
        group_flashtrace: dict[str, Any] = {}
        for metric in METRICS:
            group_estimates[metric] = {
                method: _interval(
                    np.asarray(
                        [metric_value(method, sample_id, metric) for sample_id in ids],
                        dtype=np.float64,
                    ),
                    rng,
                    draws,
                )
                for method in methods
            }
            if "flashtrace" not in methods:
                continue
            flashtrace = np.asarray(
                [
                    metric_value("flashtrace", sample_id, metric)
                    for sample_id in ids
                ],
                dtype=np.float64,
            )
            group_flashtrace[metric] = {}
            for method in methods:
                if method == "flashtrace":
                    continue
                baseline = np.asarray(
                    [metric_value(method, sample_id, metric) for sample_id in ids],
                    dtype=np.float64,
                )
                differences = flashtrace - baseline
                group_flashtrace[metric][method] = {
                    **_interval(differences, rng, draws),
                    "wins": int(np.sum(differences > 1e-12)),
                    "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                    "losses": int(np.sum(differences < -1e-12)),
                }
        per_group_paired[group] = {
            "samples": len(ids),
            "sample_ids": ids,
            "estimates": group_estimates,
            "flashtrace_minus_baseline": group_flashtrace,
        }

    return {
        "schema_version": 1,
        "dataset_manifest": str(manifest),
        "attribution_dir": str(attribution_dir),
        "common_samples": len(common_ids),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "metric_prefix": metric_prefix,
        "methods": methods,
        "estimates": estimates,
        "ranks": ranks,
        "flashtrace_minus_baseline": paired_flashtrace,
        "per_group": per_group,
        "per_group_paired": per_group_paired,
    }


def _markdown(analysis: dict[str, Any]) -> str:
    methods = analysis["methods"]
    lines = [
        "# Strict paired attribution analysis",
        "",
        f"Common paired samples: {analysis['common_samples']}; "
        f"bootstrap draws: {analysis['bootstrap_draws']}.",
        "",
        "All intervals are paired-sample nonparametric 95% bootstrap intervals.",
        "",
    ]
    for metric in METRICS:
        lines.extend(
            [
                f"## {metric}",
                "",
                "| rank | method | mean | median | 95% CI |",
                "|---:|---|---:|---:|---:|",
            ]
        )
        for rank, method in enumerate(analysis["ranks"][metric], start=1):
            estimate = analysis["estimates"][metric][method]
            lines.append(
                f"| {rank} | {method} | {estimate['mean']:.4f} | "
                f"{estimate['median']:.4f} | "
                f"[{estimate['ci95_low']:.4f}, {estimate['ci95_high']:.4f}] |"
            )
        lines.append("")

    if analysis["flashtrace_minus_baseline"]:
        lines.extend(
            [
                "## FlashTrace paired differences",
                "",
                "Positive values favor FlashTrace. W/T/L is counted per paired sample.",
                "",
            ]
        )
        for metric in METRICS:
            lines.extend(
                [
                    f"### {metric}",
                    "",
                    "| baseline | mean delta | 95% CI | W/T/L |",
                    "|---|---:|---:|---:|",
                ]
            )
            for method in methods:
                if method == "flashtrace":
                    continue
                delta = analysis["flashtrace_minus_baseline"][metric][method]
                lines.append(
                    f"| {method} | {delta['mean']:+.4f} | "
                    f"[{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] | "
                    f"{delta['wins']}/{delta['ties']}/{delta['losses']} |"
                )
            lines.append("")
    if analysis.get("per_group_paired"):
        lines.extend(
            [
                "## Stratum-level primary endpoints",
                "",
                "Intervals and W/T/L use the same paired bootstrap protocol as the "
                "overall analysis.",
                "",
                "| stratum | n | method | Energy [95% CI] | R@5 [95% CI] |",
                "|---|---:|---|---:|---:|",
            ]
        )
        for group, group_analysis in analysis["per_group_paired"].items():
            for method in methods:
                energy = group_analysis["estimates"]["energy_in_mask"][method]
                recovery = group_analysis["estimates"]["recovery_at_5pct"][method]
                lines.append(
                    f"| {group} | {group_analysis['samples']} | {method} | "
                    f"{energy['mean']:.4f} "
                    f"[{energy['ci95_low']:.4f}, {energy['ci95_high']:.4f}] | "
                    f"{recovery['mean']:.4f} "
                    f"[{recovery['ci95_low']:.4f}, {recovery['ci95_high']:.4f}] |"
                )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--metric-prefix",
        default="",
        help="Localization-key prefix, e.g. sensitivity_union. for CLEVR union GT.",
    )
    args = parser.parse_args()
    analysis = analyze(
        args.manifest,
        args.attribution_dir,
        draws=args.draws,
        seed=args.seed,
        metric_prefix=args.metric_prefix,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(analysis) + "\n")
    print(json.dumps({"json": str(args.output_json), "markdown": str(args.output_markdown)}, indent=2))


if __name__ == "__main__":
    main()
