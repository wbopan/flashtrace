"""Freeze A6 diagnostic evidence from the existing CLEVR/VISTAQA pilots.

This analysis performs no new model inference and never merges pilot samples
with the formal Wiki-VISA or VizWiz-LF estimates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METRICS = (
    "energy_in_mask",
    "evidence_rank_auc",
    "recovery_at_5pct",
    "recovery_at_20pct",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze(root: Path) -> dict[str, Any]:
    final = root / "evaluations/multimodal/results/strict/final"
    native = root / "evaluations/multimodal/results/strict/native_pilot"
    clevr_dir = final / "clevr_xai_complex_strict_n20_methods_v2"
    primary = _read(clevr_dir / "analysis_primary.json")
    union = _read(clevr_dir / "analysis_union.json")
    faithfulness = _read(
        final / "clevr_xai_complex_strict_n20_faithfulness_v2/summary.json"
    )
    vista = _read(native / "vistaqa_n10_attribution/summary.json")

    if primary["common_samples"] != 20 or union["common_samples"] != 20:
        raise ValueError("CLEVR mask sensitivity must use the strict paired n=20 set")
    if primary["methods"] != union["methods"]:
        raise ValueError("CLEVR primary/union analyses have different method panels")

    mask_sensitivity: dict[str, Any] = {}
    for method in primary["methods"]:
        mask_sensitivity[method] = {}
        for metric in METRICS:
            primary_estimate = primary["estimates"][metric][method]
            union_estimate = union["estimates"][metric][method]
            mask_sensitivity[method][metric] = {
                "unique_first_nonempty": primary_estimate,
                "union": union_estimate,
                "union_minus_unique_mean": (
                    float(union_estimate["mean"]) - float(primary_estimate["mean"])
                ),
            }

    return {
        "schema_version": 1,
        "analysis_id": "A6",
        "new_gpu_inference": False,
        "aggregation_policy": (
            "diagnostic pilots remain separate from formal Wiki-VISA/VizWiz-LF"
        ),
        "clevr_xai": {
            "sample_count": 20,
            "bootstrap_draws": primary["bootstrap_draws"],
            "mask_conventions": {
                "primary": "unique_first_nonempty",
                "sensitivity": "union",
            },
            "mask_sensitivity": mask_sensitivity,
            "faithfulness_budget": {
                "regions": faithfulness["target_regions"],
                "steps": faithfulness["steps"],
                "common_samples": faithfulness["common_samples"],
            },
            "faithfulness": {
                method: {
                    metric: faithfulness["methods"][method][metric]
                    for metric in ("deletion_auc", "insertion_auc", "visual_mas")
                }
                for method in faithfulness["methods"]
            },
        },
        "vistaqa": {
            "manifest_sample_count": 10,
            "common_success_samples": vista["common_samples"],
            "methods": {
                method: {
                    metric: values[metric]
                    for metric in METRICS
                }
                for method, values in vista["methods"].items()
            },
            "interpretation": (
                "failure-analysis pilot only; the paired common intersection is "
                "too small for a formal benchmark claim"
            ),
        },
    }


def _markdown(result: dict[str, Any]) -> str:
    clevr = result["clevr_xai"]
    vista = result["vistaqa"]
    lines = [
        "# A6: retained CLEVR-XAI and VISTAQA diagnostics",
        "",
        "These are frozen protocol-validation diagnostics. They are not pooled with "
        "the formal Wiki-VISA or VizWiz-LF samples, and this analysis runs no new "
        "GPU inference.",
        "",
        "## CLEVR-XAI dual-mask sensitivity",
        "",
        f"Paired samples: {clevr['sample_count']}; bootstrap draws: "
        f"{clevr['bootstrap_draws']}. The primary mask is Unique "
        "First-nonempty; Union is a sensitivity convention.",
        "",
        "| method | Energy unique | Energy union | R@5 unique | R@5 union |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, metrics in clevr["mask_sensitivity"].items():
        energy = metrics["energy_in_mask"]
        recovery = metrics["recovery_at_5pct"]
        lines.append(
            f"| {method} | {energy['unique_first_nonempty']['mean']:.4f} | "
            f"{energy['union']['mean']:.4f} | "
            f"{recovery['unique_first_nonempty']['mean']:.4f} | "
            f"{recovery['union']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The large convention-dependent shifts, especially for Center and "
            "Visual IG, show why CLEVR does not serve as the formal localization "
            "benchmark. Its centered synthetic objects can reward a spatial prior.",
            "",
            "## CLEVR-XAI center-prior faithfulness counterexample",
            "",
            f"Budget: {clevr['faithfulness_budget']['regions']} regions and "
            f"{clevr['faithfulness_budget']['steps']} steps on "
            f"{clevr['faithfulness_budget']['common_samples']} paired samples.",
            "",
            "| method | deletion AUC ↓ | insertion AUC ↑ | Visual-MAS ↓ |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in ("center", "visual-ig", "attnlrp", "flashtrace"):
        values = clevr["faithfulness"][method]
        lines.append(
            f"| {method} | {values['deletion_auc']:.4f} | "
            f"{values['insertion_auc']:.4f} | {values['visual_mas']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Center is strongest on all three retained faithfulness metrics in "
            "this synthetic diagnostic. We therefore treat centered-subject "
            "priors as an explicit baseline, not as evidence of causal grounding.",
            "",
            "## VISTAQA failure-analysis pilot",
            "",
            f"The native manifest contains {vista['manifest_sample_count']} samples, "
            f"but only {vista['common_success_samples']} lie in the common successful "
            "method intersection. These values are descriptive only.",
            "",
            "| method | Energy | Rank AUC | R@5 | R@20 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method, values in vista["methods"].items():
        lines.append(
            f"| {method} | {values['energy_in_mask']:.4f} | "
            f"{values['evidence_rank_auc']:.4f} | "
            f"{values['recovery_at_5pct']:.4f} | "
            f"{values['recovery_at_20pct']:.4f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(_markdown(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.output_json),
                "markdown": str(args.output_markdown),
                "new_gpu_inference": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
