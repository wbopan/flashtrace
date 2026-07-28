"""Analyze recursion, geometry, and sign sensitivity in formal visual runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .strict_attribution import _evidence_masks, localization_metrics
from .strict_generation import read_jsonl


RECURSION_METHODS = ("ifr-span", "flashtrace", "flashtrace-all-gen")
LOCALIZATION_METRICS = (
    "energy_in_mask",
    "evidence_rank_auc",
    "recovery_at_5pct",
    "recovery_at_20pct",
)


def _normalized_positive(grid: Any) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    positive = np.clip(values, 0.0, None)
    total = float(positive.sum())
    return positive / total if total > 0.0 else np.zeros_like(positive)


def _cosine(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0.0 else 0.0


def _geometry(grid: Any) -> dict[str, float]:
    mass = _normalized_positive(grid)
    rows, columns = mass.shape
    border = np.zeros_like(mass, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    row_coordinates = (
        np.linspace(0.0, 1.0, rows) if rows > 1 else np.asarray([0.5])
    )
    column_coordinates = (
        np.linspace(0.0, 1.0, columns) if columns > 1 else np.asarray([0.5])
    )
    row_centroid = float((mass * row_coordinates[:, None]).sum())
    column_centroid = float((mass * column_coordinates[None, :]).sum())
    return {
        "border_mass_ratio": float(mass[border].sum()),
        "top_row_mass_ratio": float(mass[0].sum()),
        "left_column_mass_ratio": float(mass[:, 0].sum()),
        "heatmap_centroid_row": row_centroid,
        "heatmap_centroid_column": column_centroid,
        "heatmap_centroid_distance_to_center": math.hypot(
            row_centroid - 0.5, column_centroid - 0.5
        ),
        "negative_cell_fraction": float(
            np.mean(np.asarray(grid, dtype=np.float64) < 0.0)
        ),
    }


def _ground_truth_centroid(dataset: Mapping[str, Any]) -> dict[str, float] | None:
    try:
        masks = _evidence_masks(dataset)
    except ValueError:
        return None
    if not masks:
        return None
    primary_name = (
        "primary_unique_firstnonempty"
        if "primary_unique_firstnonempty" in masks
        else "primary"
        if "primary" in masks
        else "primary_bbox"
        if "primary_bbox" in masks
        else next(iter(masks))
    )
    mask = np.asarray(masks[primary_name], dtype=bool)
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        return None
    row = float(coordinates[:, 0].mean() / max(1, mask.shape[0] - 1))
    column = float(coordinates[:, 1].mean() / max(1, mask.shape[1] - 1))
    return {
        "mask": primary_name,
        "row": row,
        "column": column,
        "distance_to_center": math.hypot(row - 0.5, column - 0.5),
    }


def _visual_direct_recursive_mass(record: Mapping[str, Any]) -> dict[str, float]:
    trace = record["method_metadata"]["trace_metadata"]
    projected = trace["ifr"]["observation_projected"]
    visual_indices = np.asarray(
        trace["multimodal"]["visual_token_indices_prompt"], dtype=np.int64
    )
    direct = np.asarray(projected["base"], dtype=np.float64)[visual_indices]
    recursive_rows = projected.get("per_hop") or []
    recursive = (
        np.asarray(recursive_rows, dtype=np.float64)[:, visual_indices].sum(axis=0)
        if recursive_rows
        else np.zeros_like(direct)
    )
    direct_positive = float(np.clip(direct, 0.0, None).sum())
    recursive_positive = float(np.clip(recursive, 0.0, None).sum())
    positive_total = direct_positive + recursive_positive
    direct_absolute = float(np.abs(direct).sum())
    recursive_absolute = float(np.abs(recursive).sum())
    absolute_total = direct_absolute + recursive_absolute
    return {
        "direct_positive_mass": direct_positive,
        "recursive_positive_mass": recursive_positive,
        "recursive_positive_fraction": (
            recursive_positive / positive_total if positive_total > 0.0 else 0.0
        ),
        "direct_absolute_mass": direct_absolute,
        "recursive_absolute_mass": recursive_absolute,
        "recursive_absolute_fraction": (
            recursive_absolute / absolute_total if absolute_total > 0.0 else 0.0
        ),
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
    labels = ("short", "medium", "long")
    return {
        sample_id: labels[min(2, index * 3 // len(ranked))]
        for index, sample_id in enumerate(ranked)
    }


def analyze(
    manifest: Path,
    model_output: Path,
    attribution_dir: Path,
    *,
    draws: int = 50_000,
    seed: int = 17,
) -> dict[str, Any]:
    datasets = {record["sample_id"]: record for record in read_jsonl(manifest)}
    models = {record["sample_id"]: record for record in read_jsonl(model_output)}
    summary = json.loads((attribution_dir / "summary.json").read_text())
    common_ids = list(summary["common_sample_ids"])
    absent_methods = set(RECURSION_METHODS) - set(summary["requested_methods"])
    if absent_methods:
        raise ValueError(
            f"formal diagnostics require recursion methods: {sorted(absent_methods)}"
        )
    by_method: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in read_jsonl(attribution_dir / "attribution_records.jsonl"):
        if record.get("status") == "ok" and record["sample_id"] in common_ids:
            by_method[record["method"]][record["sample_id"]] = record

    missing = {
        method: sorted(set(common_ids) - set(by_method[method]))
        for method in RECURSION_METHODS
        if method in summary["requested_methods"]
        and set(common_ids) - set(by_method[method])
    }
    if missing:
        raise ValueError(f"missing paired recursion records: {missing}")
    buckets = _thinking_buckets(common_ids, models)
    rng = np.random.default_rng(seed)
    sample_rows = []
    for sample_id in common_ids:
        exact = by_method["flashtrace"][sample_id]
        all_gen = by_method["flashtrace-all-gen"][sample_id]
        row: dict[str, Any] = {
            "sample_id": sample_id,
            "thinking_tokens": int(
                models[sample_id]["generation_metadata"]["thinking_tokens"]
            ),
            "thinking_bucket": buckets[sample_id],
            "exact_all_gen_cosine": _cosine(
                exact["visual_grid"], all_gen["visual_grid"]
            ),
            "flashtrace_mass": _visual_direct_recursive_mass(exact),
            "ground_truth_centroid": _ground_truth_centroid(datasets[sample_id]),
            "methods": {
                method: _geometry(by_method[method][sample_id]["visual_grid"])
                for method in summary["requested_methods"]
            },
        }
        if exact.get("localization") is not None:
            row["localization_deltas"] = {}
            for baseline in ("ifr-span", "flashtrace-all-gen"):
                row["localization_deltas"][baseline] = {
                    metric: float(exact["localization"][metric])
                    - float(by_method[baseline][sample_id]["localization"][metric])
                    for metric in LOCALIZATION_METRICS
                }
            row["positive_only_localization_sensitivity"] = {}
            for method in summary["requested_methods"]:
                source = by_method[method][sample_id]
                clipped = np.clip(
                    np.asarray(source["visual_grid"], dtype=np.float64), 0.0, None
                )
                recomputed = localization_metrics(clipped, datasets[sample_id])
                row["positive_only_localization_sensitivity"][method] = {
                    metric: float(recomputed[metric])
                    - float(source["localization"][metric])
                    for metric in LOCALIZATION_METRICS
                }
        sample_rows.append(row)

    recursion_by_bucket: dict[str, Any] = {}
    if sample_rows and "localization_deltas" in sample_rows[0]:
        for bucket in ("short", "medium", "long"):
            rows = [row for row in sample_rows if row["thinking_bucket"] == bucket]
            recursion_by_bucket[bucket] = {
                baseline: {
                    metric: _interval(
                        [
                            row["localization_deltas"][baseline][metric]
                            for row in rows
                        ],
                        rng,
                        draws,
                    )
                    for metric in LOCALIZATION_METRICS
                }
                for baseline in ("ifr-span", "flashtrace-all-gen")
            }

    geometry_summary = {
        method: {
            metric: _interval(
                [row["methods"][method][metric] for row in sample_rows], rng, draws
            )
            for metric in (
                "border_mass_ratio",
                "top_row_mass_ratio",
                "left_column_mass_ratio",
                "heatmap_centroid_distance_to_center",
                "negative_cell_fraction",
            )
        }
        for method in summary["requested_methods"]
    }
    gt_distances: dict[str, list[float]] = defaultdict(list)
    for row in sample_rows:
        centroid = row["ground_truth_centroid"]
        if centroid is not None:
            metadata = datasets[row["sample_id"]]["evaluation"]["metadata"]
            group = str(metadata.get("stratum", metadata.get("reasoning_family", "all")))
            gt_distances[group].append(float(centroid["distance_to_center"]))

    return {
        "schema_version": 1,
        "manifest": str(manifest),
        "model_output": str(model_output),
        "attribution_dir": str(attribution_dir),
        "common_samples": len(common_ids),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "exact_all_gen_cosine": _interval(
            [row["exact_all_gen_cosine"] for row in sample_rows], rng, draws
        ),
        "recursive_positive_fraction": _interval(
            [
                row["flashtrace_mass"]["recursive_positive_fraction"]
                for row in sample_rows
            ],
            rng,
            draws,
        ),
        "direct_positive_fraction": _interval(
            [
                1.0
                - row["flashtrace_mass"]["recursive_positive_fraction"]
                for row in sample_rows
            ],
            rng,
            draws,
        ),
        "recursive_absolute_fraction": _interval(
            [
                row["flashtrace_mass"]["recursive_absolute_fraction"]
                for row in sample_rows
            ],
            rng,
            draws,
        ),
        "direct_absolute_fraction": _interval(
            [
                1.0
                - row["flashtrace_mass"]["recursive_absolute_fraction"]
                for row in sample_rows
            ],
            rng,
            draws,
        ),
        "recursion_by_thinking_bucket": recursion_by_bucket,
        "geometry": geometry_summary,
        "ground_truth_centroid_distance": {
            group: _interval(values, rng, draws)
            for group, values in sorted(gt_distances.items())
        },
        "samples": sample_rows,
    }


def _markdown(analysis: Mapping[str, Any]) -> str:
    cosine = analysis["exact_all_gen_cosine"]
    recursive = analysis["recursive_positive_fraction"]
    lines = [
        "# Formal visual diagnostics",
        "",
        f"Common paired samples: {analysis['common_samples']}; "
        f"bootstrap draws: {analysis['bootstrap_draws']}.",
        "",
        f"- Exact vs all-generation cosine: {cosine['mean']:.4f} "
        f"[{cosine['ci95_low']:.4f}, {cosine['ci95_high']:.4f}]",
        f"- Recursive positive visual-mass fraction: {recursive['mean']:.4f} "
        f"[{recursive['ci95_low']:.4f}, {recursive['ci95_high']:.4f}]",
        "",
        "## Geometry",
        "",
        "| method | border mass | top-row mass | centroid distance | negative cells |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in analysis["geometry"].items():
        lines.append(
            f"| {method} | {metrics['border_mass_ratio']['mean']:.4f} | "
            f"{metrics['top_row_mass_ratio']['mean']:.4f} | "
            f"{metrics['heatmap_centroid_distance_to_center']['mean']:.4f} | "
            f"{metrics['negative_cell_fraction']['mean']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    analysis = analyze(
        args.manifest,
        args.model_output,
        args.attribution_dir,
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
                "common_samples": analysis["common_samples"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
