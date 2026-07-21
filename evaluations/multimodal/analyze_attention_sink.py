"""Cross-fitted diagnostics and post-hoc corrections for visual position sinks.

The corrections in this module never inspect evidence masks while estimating a
sink.  For each held-out sample, the shared positional prior is estimated from
the other samples only.  This makes the experiment a diagnostic of reusable
position bias rather than a ground-truth-guided cleanup.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .strict_attribution import _evidence_masks, _metric_bundle, localization_metrics
from .strict_generation import read_jsonl, write_jsonl


METRICS = (
    "pointing_game",
    "energy_in_mask",
    "evidence_rank_auc",
    "top_evidence_iou",
    "recovery_at_5pct",
    "recovery_at_20pct",
)
UNION_METRICS = (
    "sensitivity_union.energy_in_mask",
    "sensitivity_union.evidence_rank_auc",
    "sensitivity_union.top_evidence_iou",
    "sensitivity_union.recovery_at_20pct",
)

BASELINE_METHODS = ("visual-ig", "attnlrp")
FLASH_METHOD = "flashtrace"


def normalized_positive(grid: Any) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all():
        raise ValueError(f"Expected a finite non-empty 2-D grid, got {values.shape}")
    positive = np.clip(values, 0.0, None)
    total = float(positive.sum())
    if total <= 0:
        return np.zeros_like(positive)
    return positive / total


def leave_one_out_priors(grids: list[np.ndarray]) -> list[np.ndarray]:
    if len(grids) < 2:
        raise ValueError("Leave-one-out correction requires at least two grids")
    normalized = np.stack([normalized_positive(grid) for grid in grids])
    total = normalized.sum(axis=0)
    return [(total - grid) / (len(grids) - 1) for grid in normalized]


def mask_top_fraction(grid: Any, prior: Any, fraction: float) -> np.ndarray:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    values = np.asarray(grid, dtype=np.float64).copy()
    position_prior = np.asarray(prior, dtype=np.float64)
    if values.shape != position_prior.shape:
        raise ValueError("grid and prior must have the same shape")
    count = max(1, math.ceil(values.size * fraction))
    masked = np.argpartition(position_prior.reshape(-1), -count)[-count:]
    values.reshape(-1)[masked] = 0.0
    return values


def residualize_position_prior(grid: Any, prior: Any) -> np.ndarray:
    values = normalized_positive(grid)
    position_prior = np.asarray(prior, dtype=np.float64)
    if values.shape != position_prior.shape:
        raise ValueError("grid and prior must have the same shape")
    residual = np.clip(values - position_prior, 0.0, None)
    residual[residual <= 1e-15] = 0.0
    return residual


def _fixed_mask(grid: Any, *, first_cell: bool = False, top_row: bool = False) -> np.ndarray:
    values = np.asarray(grid, dtype=np.float64).copy()
    if first_cell:
        values[0, 0] = 0.0
    if top_row:
        values[0, :] = 0.0
    return values


def _geometry(grid: Any) -> dict[str, float]:
    mass = normalized_positive(grid)
    edge = np.zeros_like(mass, dtype=bool)
    edge[[0, -1], :] = True
    edge[:, [0, -1]] = True
    top_left_rows = max(1, math.ceil(mass.shape[0] * 0.25))
    top_left_columns = max(1, math.ceil(mass.shape[1] * 0.25))
    maximum = np.unravel_index(int(np.argmax(mass)), mass.shape)
    return {
        "first_cell_mass": float(mass[0, 0]),
        "top_row_mass": float(mass[0].sum()),
        "left_column_mass": float(mass[:, 0].sum()),
        "outer_edge_mass": float(mass[edge].sum()),
        "top_left_quarter_mass": float(
            mass[:top_left_rows, :top_left_columns].sum()
        ),
        "argmax_first_cell": float(maximum == (0, 0)),
        "argmax_top_row": float(maximum[0] == 0),
        "argmax_outer_edge": float(edge[maximum]),
    }


def _bootstrap_delta(
    corrected: np.ndarray,
    source: np.ndarray,
    *,
    rng: np.random.Generator,
    draws: int,
) -> dict[str, float | int]:
    differences = corrected - source
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean": float(differences.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "wins": int(np.sum(differences > 1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
    }


def _random_mask_null(
    grids: list[np.ndarray],
    datasets: list[dict[str, Any]],
    *,
    fraction: float,
    observed: dict[str, float],
    rng: np.random.Generator,
    draws: int,
) -> dict[str, dict[str, float]]:
    """Compare a position-prior mask with equally sized random masks."""

    count = max(1, math.ceil(grids[0].size * fraction))
    null = {metric: np.empty(draws, dtype=np.float64) for metric in METRICS}
    primary_masks = []
    for dataset in datasets:
        masks = _evidence_masks(dataset)
        primary_name = (
            "primary_unique_firstnonempty"
            if "primary_unique_firstnonempty" in masks
            else "primary_bbox"
        )
        primary_masks.append(masks[primary_name])
    for draw in range(draws):
        per_metric = {metric: [] for metric in METRICS}
        for grid, mask in zip(grids, primary_masks, strict=True):
            corrected = np.asarray(grid, dtype=np.float64).copy()
            masked = rng.choice(corrected.size, size=count, replace=False)
            corrected.reshape(-1)[masked] = 0.0
            localization = _metric_bundle(corrected, mask)
            for metric in METRICS:
                per_metric[metric].append(float(localization[metric]))
        for metric in METRICS:
            null[metric][draw] = float(np.mean(per_metric[metric]))
    return {
        metric: {
            "null_mean": float(values.mean()),
            "null_ci95_low": float(np.quantile(values, 0.025)),
            "null_ci95_high": float(np.quantile(values, 0.975)),
            "observed": float(observed[metric]),
            "observed_percentile": float(np.mean(values <= observed[metric])),
        }
        for metric, values in null.items()
    }


def analyze(
    manifest: Path,
    attribution_dir: Path,
    output_dir: Path,
    *,
    prior_fraction: float = 0.05,
    draws: int = 10_000,
    null_draws: int = 500,
    seed: int = 17,
) -> dict[str, Any]:
    datasets = {record["sample_id"]: record for record in read_jsonl(manifest)}
    source_records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in read_jsonl(attribution_dir / "attribution_records.jsonl"):
        if record.get("status") == "ok":
            source_records[record["method"]][record["sample_id"]] = record

    required = (FLASH_METHOD, *BASELINE_METHODS)
    common_ids = sorted(
        set.intersection(*(set(source_records[method]) for method in required))
    )
    if len(common_ids) < 2:
        raise ValueError(f"Need at least two paired samples, got {len(common_ids)}")
    if set(common_ids) - set(datasets):
        raise ValueError("Attribution records contain samples absent from the manifest")

    flash_grids = [
        np.asarray(source_records[FLASH_METHOD][sample_id]["visual_grid"])
        for sample_id in common_ids
    ]
    flash_priors = leave_one_out_priors(flash_grids)

    variants: dict[str, list[np.ndarray]] = {
        "visual-ig": [
            np.asarray(source_records["visual-ig"][sample_id]["visual_grid"])
            for sample_id in common_ids
        ],
        "attnlrp": [
            np.asarray(source_records["attnlrp"][sample_id]["visual_grid"])
            for sample_id in common_ids
        ],
        "flashtrace": flash_grids,
        "flashtrace-no-first-cell": [
            _fixed_mask(grid, first_cell=True) for grid in flash_grids
        ],
        "flashtrace-no-top-row": [
            _fixed_mask(grid, top_row=True) for grid in flash_grids
        ],
        "flashtrace-loo-prior-mask-5pct": [
            mask_top_fraction(grid, prior, prior_fraction)
            for grid, prior in zip(flash_grids, flash_priors, strict=True)
        ],
        "flashtrace-loo-prior-residual": [
            residualize_position_prior(grid, prior)
            for grid, prior in zip(flash_grids, flash_priors, strict=True)
        ],
    }
    variants["visual-ig-no-top-row"] = [
        _fixed_mask(grid, top_row=True) for grid in variants["visual-ig"]
    ]
    variants["attnlrp-no-top-row"] = [
        _fixed_mask(grid, top_row=True) for grid in variants["attnlrp"]
    ]
    variants["visual-ig-flash-prior-mask-5pct"] = [
        mask_top_fraction(grid, prior, prior_fraction)
        for grid, prior in zip(variants["visual-ig"], flash_priors, strict=True)
    ]
    variants["attnlrp-flash-prior-mask-5pct"] = [
        mask_top_fraction(grid, prior, prior_fraction)
        for grid, prior in zip(variants["attnlrp"], flash_priors, strict=True)
    ]

    result_records: list[dict[str, Any]] = []
    metric_values: dict[str, dict[str, list[float]]] = {}
    union_metric_values: dict[str, dict[str, list[float]]] = {}
    geometry: dict[str, dict[str, float]] = {}
    for method, grids in variants.items():
        metric_values[method] = {metric: [] for metric in METRICS}
        union_metric_values[method] = {metric: [] for metric in UNION_METRICS}
        geometry_rows = [_geometry(grid) for grid in grids]
        geometry[method] = {
            key: float(np.mean([row[key] for row in geometry_rows]))
            for key in geometry_rows[0]
        }
        for sample_id, grid in zip(common_ids, grids, strict=True):
            localization = localization_metrics(grid, datasets[sample_id])
            for metric in METRICS:
                metric_values[method][metric].append(float(localization[metric]))
            for metric in UNION_METRICS:
                union_metric_values[method][metric].append(float(localization[metric]))
            result_records.append(
                {
                    "schema_version": 2,
                    "status": "ok",
                    "benchmark": "clevr_xai_complex",
                    "sample_id": sample_id,
                    "method": method,
                    "target_span": "output_only",
                    "bridge_span": "thinking" if "flashtrace" in method else None,
                    "visual_grid": np.asarray(grid).tolist(),
                    "visual_grid_shape": list(np.asarray(grid).shape),
                    "localization": localization,
                    "overlay_path": "",
                    "method_metadata": {
                        "diagnostic_posthoc": method not in BASELINE_METHODS
                        and method != FLASH_METHOD,
                        "uses_ground_truth_to_estimate_sink": False,
                        "leave_one_out_prior": "loo" in method,
                        "prior_fraction": prior_fraction if "mask" in method else None,
                    },
                    "seconds": 0.0,
                    "peak_vram_gb": 0.0,
                    "incremental_peak_vram_gb": 0.0,
                }
            )

    # Guard against an accidental protocol drift while reconstructing metrics.
    for sample_id, metrics in zip(
        common_ids,
        [
            localization_metrics(grid, datasets[sample_id])
            for sample_id, grid in zip(common_ids, flash_grids, strict=True)
        ],
        strict=True,
    ):
        saved = source_records[FLASH_METHOD][sample_id]["localization"]
        for metric in METRICS:
            if not np.isclose(metrics[metric], saved[metric], atol=1e-10):
                raise AssertionError(
                    f"Recomputed {sample_id}/{metric} differs from saved result"
                )

    estimates = {
        method: {
            metric: float(np.mean(values))
            for metric, values in per_metric.items()
        }
        for method, per_metric in metric_values.items()
    }
    union_estimates = {
        method: {
            metric: float(np.mean(values))
            for metric, values in per_metric.items()
        }
        for method, per_metric in union_metric_values.items()
    }
    random_mask_null = _random_mask_null(
        flash_grids,
        [datasets[sample_id] for sample_id in common_ids],
        fraction=prior_fraction,
        observed=estimates["flashtrace-loo-prior-mask-5pct"],
        rng=np.random.default_rng(seed + 1),
        draws=null_draws,
    )
    rng = np.random.default_rng(seed)
    source_for_variant = {
        "flashtrace-no-first-cell": "flashtrace",
        "flashtrace-no-top-row": "flashtrace",
        "flashtrace-loo-prior-mask-5pct": "flashtrace",
        "flashtrace-loo-prior-residual": "flashtrace",
        "visual-ig-no-top-row": "visual-ig",
        "attnlrp-no-top-row": "attnlrp",
        "visual-ig-flash-prior-mask-5pct": "visual-ig",
        "attnlrp-flash-prior-mask-5pct": "attnlrp",
    }
    paired_deltas: dict[str, dict[str, dict[str, float | int]]] = {}
    for corrected, source in source_for_variant.items():
        paired_deltas[corrected] = {}
        for metric in METRICS:
            paired_deltas[corrected][metric] = _bootstrap_delta(
                np.asarray(metric_values[corrected][metric]),
                np.asarray(metric_values[source][metric]),
                rng=rng,
                draws=draws,
            )

    prior_mean = np.stack(
        [normalized_positive(grid) for grid in flash_grids]
    ).mean(axis=0)
    top_count = max(1, math.ceil(prior_mean.size * prior_fraction))
    top_indices = np.argpartition(prior_mean.reshape(-1), -top_count)[-top_count:]
    top_positions = sorted(
        (
            {
                "row": int(index // prior_mean.shape[1]),
                "column": int(index % prior_mean.shape[1]),
                "mean_mass": float(prior_mean.reshape(-1)[index]),
            }
            for index in top_indices
        ),
        key=lambda item: item["mean_mass"],
        reverse=True,
    )

    analysis = {
        "schema_version": 1,
        "samples": len(common_ids),
        "sample_ids": common_ids,
        "protocol": {
            "ground_truth_used_to_estimate_sink": False,
            "position_prior": "leave-one-out mean of per-sample normalized positive maps",
            "prior_mask_fraction": prior_fraction,
            "metric_resampling": "existing bilinear protocol for apples-to-apples comparison",
            "bootstrap_draws": draws,
            "random_mask_null_draws": null_draws,
            "bootstrap_seed": seed,
        },
        "top_shared_flashtrace_positions": top_positions,
        "geometry": geometry,
        "estimates": estimates,
        "union_estimates": union_estimates,
        "random_mask_null": random_mask_null,
        "paired_deltas": paired_deltas,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(result_records, output_dir / "attribution_records.jsonl")
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "requested_methods": list(variants),
                "common_sample_ids": common_ids,
                "methods": {
                    method: {"common_samples": len(common_ids)} for method in variants
                },
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
    (output_dir / "analysis.md").write_text(_markdown(analysis) + "\n")
    return analysis


def _markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# CLEVR-XAI FlashTrace attention-sink diagnostic",
        "",
        f"Paired samples: {analysis['samples']}.",
        "",
        "Sink priors are estimated without evidence masks. Each sample is corrected "
        "using the other samples only. Localization metrics retain the existing "
        "bilinear evaluation protocol so deltas are directly comparable to the "
        "saved strict pilot.",
        "",
        "## Position geometry",
        "",
        "| method | first cell | top row | left column | outer edge | top-left quarter | argmax top row |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in analysis["geometry"].items():
        lines.append(
            f"| {method} | {values['first_cell_mass']:.4f} | "
            f"{values['top_row_mass']:.4f} | {values['left_column_mass']:.4f} | "
            f"{values['outer_edge_mass']:.4f} | "
            f"{values['top_left_quarter_mass']:.4f} | "
            f"{values['argmax_top_row']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Localization",
            "",
            "| method | Pointing | Energy | Rank-AUC | Top-IoU | R@5 | R@20 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method, values in analysis["estimates"].items():
        lines.append(
            f"| {method} | {values['pointing_game']:.4f} | "
            f"{values['energy_in_mask']:.4f} | "
            f"{values['evidence_rank_auc']:.4f} | "
            f"{values['top_evidence_iou']:.4f} | "
            f"{values['recovery_at_5pct']:.4f} | "
            f"{values['recovery_at_20pct']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Random-mask control",
            "",
            "The observed cross-fitted 5% position-prior mask is compared with "
            "equally sized random masks. Percentile is the fraction of random-mask "
            "runs at or below the observed score.",
            "",
            "| metric | observed | random mean | random 95% interval | percentile |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, values in analysis["random_mask_null"].items():
        lines.append(
            f"| {metric} | {values['observed']:.4f} | "
            f"{values['null_mean']:.4f} | "
            f"[{values['null_ci95_low']:.4f}, {values['null_ci95_high']:.4f}] | "
            f"{values['observed_percentile']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Union-evidence localization",
            "",
            "| method | Energy | Rank-AUC | Top-IoU | R@20 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, values in analysis["union_estimates"].items():
        lines.append(
            f"| {method} | "
            f"{values['sensitivity_union.energy_in_mask']:.4f} | "
            f"{values['sensitivity_union.evidence_rank_auc']:.4f} | "
            f"{values['sensitivity_union.top_evidence_iou']:.4f} | "
            f"{values['sensitivity_union.recovery_at_20pct']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Paired correction deltas",
            "",
            "Positive values favor the corrected map. Intervals are paired "
            "nonparametric bootstrap 95% CIs.",
            "",
            "| correction | metric | delta | 95% CI | W/T/L |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for correction, per_metric in analysis["paired_deltas"].items():
        for metric, values in per_metric.items():
            lines.append(
                f"| {correction} | {metric} | {values['mean']:+.4f} | "
                f"[{values['ci95_low']:+.4f}, {values['ci95_high']:+.4f}] | "
                f"{values['wins']}/{values['ties']}/{values['losses']} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prior-fraction", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--null-draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    analysis = analyze(
        args.manifest,
        args.attribution_dir,
        args.output_dir,
        prior_fraction=args.prior_fraction,
        draws=args.draws,
        null_draws=args.null_draws,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "samples": analysis["samples"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
