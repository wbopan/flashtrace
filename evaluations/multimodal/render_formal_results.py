"""Render formal visual results and paper table rows from audited artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


METHODS = (
    "random",
    "center",
    "visual-loo",
    "visual-ig",
    "attnlrp",
    "flashtrace",
    "ifr-span",
    "flashtrace-all-gen",
)
PRIMARY_FAITHFULNESS_METHODS = ("visual-ig", "attnlrp", "flashtrace")
METHOD_LABELS = {
    "random": "Random",
    "center": "Center prior",
    "visual-loo": "Visual LOO",
    "visual-ig": "Visual IG",
    "attnlrp": "AttnLRP",
    "flashtrace": "FlashTrace (exact, K=1)",
    "ifr-span": "IFR-span (K=0)",
    "flashtrace-all-gen": "FlashTrace all-generation",
}
LATEX_LABELS = {
    "random": "Random",
    "center": "Center prior",
    "visual-loo": "Visual LOO",
    "visual-ig": "Visual IG",
    "attnlrp": "AttnLRP",
    "flashtrace": r"\textbf{\flashtrace{} (exact, $K{=}1$)}",
    "ifr-span": r"\textit{IFR-span ($K{=}0$)}",
    "flashtrace-all-gen": r"\textit{\flashtrace{} all-generation}",
}


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required formal artifact is absent: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _estimate_cell(estimate: Mapping[str, Any]) -> str:
    return (
        f"{float(estimate['mean']):.4f} "
        f"[{float(estimate['ci95_low']):.4f}, "
        f"{float(estimate['ci95_high']):.4f}]"
    )


def _latex_cell(estimate: Mapping[str, Any]) -> str:
    return (
        r"\shortstack{"
        f"{float(estimate['mean']):.3f}"
        r"\\{\scriptsize["
        f"{float(estimate['ci95_low']):.3f},"
        f"{float(estimate['ci95_high']):.3f}"
        r"]}}"
    )


def _funnel_markdown(name: str, funnel: Mapping[str, Any]) -> list[str]:
    lines = [
        f"### {name}",
        "",
        "| gate stage | passed | eliminated at stage | not evaluated at stage |",
        "|---|---:|---:|---:|",
    ]
    for stage in funnel["stages"]:
        lines.append(
            f"| {stage['stage']} | {stage['passed']} | "
            f"{stage['eliminated_at_stage']} | "
            f"{stage.get('not_evaluated_at_stage', 0)} |"
        )
    lines.extend(
        [
            "",
            "| gate marginal | passed | failed | not evaluated |",
            "|---|---:|---:|---:|",
        ]
    )
    for gate, counts in funnel["gate_marginal_counts"].items():
        lines.append(
            f"| {gate} | {counts['passed']} | {counts['failed']} | "
            f"{counts['not_evaluated']} |"
        )
    lines.append("")
    return lines


def _localization_markdown(analysis: Mapping[str, Any]) -> list[str]:
    metric_labels = (
        ("energy_in_mask", "Energy"),
        ("evidence_rank_auc", "Rank AUC"),
        ("recovery_at_5pct", "R@5"),
        ("recovery_at_20pct", "R@20"),
    )
    lines = [
        "## E3: Wiki-VISA localization",
        "",
        f"Common paired samples: {analysis['common_samples']}; paired bootstrap "
        f"draws: {analysis['bootstrap_draws']}. Energy and R@5 are primary.",
        "",
        "| method | Energy | Rank AUC | R@5 | R@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        cells = [
            _estimate_cell(analysis["estimates"][metric][method])
            for metric, _ in metric_labels
        ]
        lines.append(f"| {METHOD_LABELS[method]} | " + " | ".join(cells) + " |")
    lines.extend(["", "### Primary paired differences: FlashTrace minus baseline", ""])
    for metric, label in metric_labels:
        if metric not in {"energy_in_mask", "recovery_at_5pct"}:
            continue
        lines.extend(
            [
                f"#### {label}",
                "",
                "| baseline | favorable delta [95% CI] | W/T/L |",
                "|---|---:|---:|",
            ]
        )
        for method in METHODS:
            if method == "flashtrace":
                continue
            delta = analysis["flashtrace_minus_baseline"][metric][method]
            lines.append(
                f"| {METHOD_LABELS[method]} | {_estimate_cell(delta)} | "
                f"{delta['wins']}/{delta['ties']}/{delta['losses']} |"
            )
        lines.append("")

    lines.extend(
        [
            "### Wiki strata",
            "",
            "| stratum | n | method | Energy | R@5 |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for group, group_analysis in analysis["per_group_paired"].items():
        for method in METHODS:
            lines.append(
                f"| {group} | {group_analysis['samples']} | "
                f"{METHOD_LABELS[method]} | "
                f"{_estimate_cell(group_analysis['estimates']['energy_in_mask'][method])} | "
                f"{_estimate_cell(group_analysis['estimates']['recovery_at_5pct'][method])} |"
            )
    lines.extend(
        [
            "",
            "### Supplemental localization endpoints",
            "",
            "These endpoints are computed from the same whole-patch, tie-aware "
            "maps and paired n=120 intersection; they are not primary endpoints.",
            "",
            "| method | Pointing Game | Top-area IoU | R@1 | R@10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{_estimate_cell(analysis['estimates']['pointing_game'][method])} | "
            f"{_estimate_cell(analysis['estimates']['top_evidence_iou'][method])} | "
            f"{_estimate_cell(analysis['estimates']['recovery_at_1pct'][method])} | "
            f"{_estimate_cell(analysis['estimates']['recovery_at_10pct'][method])} |"
        )
    lines.append("")
    return lines


def _faithfulness_markdown(
    name: str, analysis: Mapping[str, Any], *, primary: bool
) -> list[str]:
    overall = analysis["overall"]
    lines = [
        f"## {'E4' if primary else 'E5'}: {name} frozen-response faithfulness",
        "",
        f"Common paired samples: {overall['samples']}; paired bootstrap draws: "
        f"{analysis['bootstrap_draws']}. Deletion AUC is the primary endpoint.",
        "",
        "| method | Deletion AUC ↓ | Insertion AUC ↑ | Visual-MAS ↓ |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        estimates = overall["estimates"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{_estimate_cell(estimates['deletion_auc'])} | "
            f"{_estimate_cell(estimates['insertion_auc'])} | "
            f"{_estimate_cell(estimates['visual_mas'])} |"
        )
    if primary:
        lines.extend(
            [
                "",
                "Visual LOO is retained in the complete eight-method appendix "
                "as a cost-insensitive perturbation diagnostic. The practical "
                "main comparison and interpretation use Visual IG, AttnLRP, "
                "and FlashTrace; Center remains an explicit spatial-prior "
                "check.",
            ]
        )
    lines.extend(["", "### FlashTrace favorable deletion-AUC differences", ""])
    lines.extend(
        [
            "| baseline | favorable delta [95% CI] | W/T/L |",
            "|---|---:|---:|",
        ]
    )
    for method in METHODS:
        if method == "flashtrace":
            continue
        delta = overall["flashtrace_favorable_difference"][method]["deletion_auc"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {_estimate_cell(delta)} | "
            f"{delta['wins']}/{delta['ties']}/{delta['losses']} |"
        )
    if analysis.get("positive_only_available"):
        lines.extend(
            [
                "",
                "### Signed-order vs positive-only sensitivity",
                "",
                "| method | signed deletion AUC | positive-only deletion AUC | shift |",
                "|---|---:|---:|---:|",
            ]
        )
        positive = analysis["positive_only_ordering"]["estimates"]
        for method in METHODS:
            signed_mean = float(
                overall["estimates"][method]["deletion_auc"]["mean"]
            )
            positive_mean = float(positive[method]["deletion_auc"]["mean"])
            lines.append(
                f"| {METHOD_LABELS[method]} | {signed_mean:.4f} | "
                f"{positive_mean:.4f} | {positive_mean - signed_mean:+.4f} |"
            )
    lines.append("")
    return lines


def _diagnostics_markdown(
    wiki: Mapping[str, Any], viz: Mapping[str, Any]
) -> list[str]:
    lines = [
        "## A1–A4: recursion and geometry diagnostics",
        "",
        "| dataset | n | exact/all-generation cosine | direct positive mass | recursive positive mass | recursive absolute mass |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, analysis in (("Wiki-VISA", wiki), ("VizWiz-LF", viz)):
        lines.append(
            f"| {name} | {analysis['common_samples']} | "
            f"{_estimate_cell(analysis['exact_all_gen_cosine'])} | "
            f"{_estimate_cell(analysis['direct_positive_fraction'])} | "
            f"{_estimate_cell(analysis['recursive_positive_fraction'])} | "
            f"{_estimate_cell(analysis['recursive_absolute_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "### Native-evidence centroid distance to image center",
            "",
            "| Wiki-VISA stratum | GT centroid distance [95% CI] |",
            "|---|---:|",
        ]
    )
    for stratum, estimate in wiki["ground_truth_centroid_distance"].items():
        lines.append(f"| {stratum} | {_estimate_cell(estimate)} |")
    lines.extend(
        [
            "",
            "### Heatmap geometry and sign diagnostics",
            "",
            "| dataset | method | border mass | top-row mass | heatmap centroid distance | negative cells |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name, analysis in (("Wiki-VISA", wiki), ("VizWiz-LF", viz)):
        for method in METHODS:
            geometry = analysis["geometry"][method]
            lines.append(
                f"| {name} | {METHOD_LABELS[method]} | "
                f"{_estimate_cell(geometry['border_mass_ratio'])} | "
                f"{_estimate_cell(geometry['top_row_mass_ratio'])} | "
                f"{_estimate_cell(geometry['heatmap_centroid_distance_to_center'])} | "
                f"{_estimate_cell(geometry['negative_cell_fraction'])} |"
            )
    lines.extend(
        [
            "",
            "VizWiz-LF has no native evidence mask, so a ground-truth evidence "
            "centroid is not defined; its A3 report is restricted to heatmap "
            "centroids and border/top-row mass. Wiki-VISA additionally reports "
            "native-box centroid distance by stratum.",
            "",
        ]
    )
    return lines


def _recursion_bucket_markdown(
    wiki_diagnostics: Mapping[str, Any],
    viz_diagnostics: Mapping[str, Any],
    wiki_faith: Mapping[str, Any],
    viz_faith: Mapping[str, Any],
) -> list[str]:
    lines = [
        "### One-hop recursion gain by THINKING length",
        "",
        "Localization deltas are K=1 minus K=0; deletion is oriented so "
        "positive values favor K=1.",
        "",
        "| dataset | bucket | Δ Energy | Δ R@5 | favorable Δ deletion AUC |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, diagnostics, faithfulness in (
        ("Wiki-VISA", wiki_diagnostics, wiki_faith),
        ("VizWiz-LF", viz_diagnostics, viz_faith),
    ):
        for bucket in ("short", "medium", "long"):
            if dataset == "Wiki-VISA":
                localization = diagnostics["recursion_by_thinking_bucket"][
                    bucket
                ]["ifr-span"]
                energy = _estimate_cell(localization["energy_in_mask"])
                recovery = _estimate_cell(localization["recovery_at_5pct"])
            else:
                energy = recovery = "--"
            deletion = faithfulness["recursion_by_thinking_bucket"][bucket][
                "flashtrace_favorable_difference"
            ]["ifr-span"]["deletion_auc"]
            lines.append(
                f"| {dataset} | {bucket} | {energy} | {recovery} | "
                f"{_estimate_cell(deletion)} |"
            )
    lines.append("")
    return lines


def _fully_correct_markdown(
    analysis: Mapping[str, Any], semantic: Mapping[str, Any]
) -> list[str]:
    subset = analysis["fully_correct_subset"]
    lines = [
        "## A8: VizWiz semantic correctness sensitivity",
        "",
        f"Labels: {semantic['label_counts']}. Independent human audit: "
        f"{semantic['audit_reviewed']}/{len(semantic['audit_sample_ids'])}. "
        f"Fully-correct subset size: {subset['samples']}.",
        "",
        "| method | Deletion AUC ↓ | Insertion AUC ↑ | Visual-MAS ↓ |",
        "|---|---:|---:|---:|",
    ]
    for method in METHODS:
        estimates = subset["estimates"][method]
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{_estimate_cell(estimates['deletion_auc'])} | "
            f"{_estimate_cell(estimates['insertion_auc'])} | "
            f"{_estimate_cell(estimates['visual_mas'])} |"
        )
    lines.extend(
        [
            "",
            "### Fully-correct FlashTrace favorable deletion-AUC differences",
            "",
            "| baseline | favorable delta [95% CI] | W/T/L |",
            "|---|---:|---:|",
        ]
    )
    for method in METHODS:
        if method == "flashtrace":
            continue
        delta = subset["flashtrace_favorable_difference"][method]["deletion_auc"]
        lines.append(
            f"| {METHOD_LABELS[method]} | {_estimate_cell(delta)} | "
            f"{delta['wins']}/{delta['ties']}/{delta['losses']} |"
        )
    lines.append("")
    return lines


def _timing_markdown(
    wiki_attribution: Mapping[str, Any],
    viz_attribution: Mapping[str, Any],
    wiki_faithfulness: Mapping[str, Any],
    viz_faithfulness: Mapping[str, Any],
) -> list[str]:
    lines = [
        "## Observed visual compute",
        "",
        "Times are per successful sample-method on the formal common "
        "intersection. Attribution VRAM is incremental peak allocation; "
        "faithfulness time covers the matched 64-region/10-step perturbations.",
        "",
        "| dataset | method | attribution seconds | incremental peak VRAM GiB | faithfulness seconds |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset, attribution, faithfulness in (
        ("Wiki-VISA", wiki_attribution, wiki_faithfulness),
        ("VizWiz-LF", viz_attribution, viz_faithfulness),
    ):
        for method in METHODS:
            attr = attribution["methods"][method]
            faith = faithfulness["methods"][method]
            lines.append(
                f"| {dataset} | {METHOD_LABELS[method]} | "
                f"{float(attr['mean_seconds']):.3f} | "
                f"{float(attr['mean_peak_vram_gb']):.3f} | "
                f"{float(faith['mean_seconds']):.3f} |"
            )
    lines.append("")
    return lines


def _shape_counts_cell(value: Mapping[str, Any]) -> str:
    return ", ".join(
        f"{shape} (n={int(count)})" for shape, count in value.items()
    ) or "--"


def _resolution_markdown(
    wiki_attribution: Mapping[str, Any],
    viz_attribution: Mapping[str, Any],
    wiki_faithfulness: Mapping[str, Any],
    viz_faithfulness: Mapping[str, Any],
) -> list[str]:
    lines = [
        "## Spatial resolution disclosure",
        "",
        "Native attribution grids are method outputs before nearest-patch "
        "resampling. Faithfulness layouts are shared by every method for a "
        "given image and contain approximately 64 perturbation regions.",
        "",
        "| dataset | method | native attribution grid shapes | faithfulness layouts |",
        "|---|---|---|---|",
    ]
    for dataset, attribution, faithfulness in (
        ("Wiki-VISA", wiki_attribution, wiki_faithfulness),
        ("VizWiz-LF", viz_attribution, viz_faithfulness),
    ):
        for method in METHODS:
            lines.append(
                f"| {dataset} | {METHOD_LABELS[method]} | "
                f"{_shape_counts_cell(attribution['methods'][method].get('native_grid_shapes') or {})} | "
                f"{_shape_counts_cell(faithfulness['methods'][method].get('region_layouts') or {})} |"
            )
    lines.extend(
        [
            "",
            "IFR-span, Visual IG, AttnLRP, FlashTrace, and "
            "FlashTrace all-generation share the same native model-token grid "
            "within each sample. Random/Center use a 32x32 synthetic grid and "
            "Visual LOO uses a coarse 4x4 perturbation grid. Nearest-neighbor "
            "resampling does not create sub-patch attribution detail.",
            "",
        ]
    )
    return lines


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def _visual_discussion_tex(
    localization: Mapping[str, Any], viz_faith: Mapping[str, Any]
) -> str:
    localization_leaders = {}
    for metric in (
        "energy_in_mask",
        "evidence_rank_auc",
        "recovery_at_5pct",
        "recovery_at_20pct",
    ):
        localization_leaders[metric] = max(
            METHODS,
            key=lambda method: float(
                localization["estimates"][metric][method]["mean"]
            ),
        )
    faithfulness_leaders = {
        "deletion_auc": min(
            PRIMARY_FAITHFULNESS_METHODS,
            key=lambda method: float(
                viz_faith["overall"]["estimates"][method]["deletion_auc"]["mean"]
            ),
        ),
        "insertion_auc": max(
            PRIMARY_FAITHFULNESS_METHODS,
            key=lambda method: float(
                viz_faith["overall"]["estimates"][method]["insertion_auc"]["mean"]
            ),
        ),
        "visual_mas": min(
            PRIMARY_FAITHFULNESS_METHODS,
            key=lambda method: float(
                viz_faith["overall"]["estimates"][method]["visual_mas"]["mean"]
            ),
        ),
    }

    def evidence(metric: str, baseline: str) -> str:
        delta = localization["flashtrace_minus_baseline"][metric][baseline]
        low = float(delta["ci95_low"])
        high = float(delta["ci95_high"])
        if low > 0:
            return "favored"
        if high < 0:
            return "was lower than"
        return "was directionally indistinguishable from"

    def faithfulness_evidence(metric: str, baseline: str) -> str:
        delta = viz_faith["overall"]["flashtrace_favorable_difference"][
            baseline
        ][metric]
        low = float(delta["ci95_low"])
        high = float(delta["ci95_high"])
        if low > 0:
            return "favored \\flashtrace{}"
        if high < 0:
            return f"favored {LATEX_LABELS[baseline]}"
        return "did not resolve a difference"

    lines = [
        r"\noindent\textbf{Visual-results interpretation.}",
        "The formal Wiki-VISA panel separates concentration from coverage: "
        f"{LATEX_LABELS[localization_leaders['energy_in_mask']]} had the largest "
        "mean Energy, while "
        f"{LATEX_LABELS[localization_leaders['evidence_rank_auc']]} and "
        f"{LATEX_LABELS[localization_leaders['recovery_at_5pct']]} led mean "
        "Rank AUC and R@5, respectively. Under the paired 95\\% intervals, "
        f"\\flashtrace{{}} {evidence('energy_in_mask', 'ifr-span')} IFR-span "
        f"but {evidence('energy_in_mask', 'attnlrp')} AttnLRP for Energy; "
        f"it {evidence('recovery_at_5pct', 'ifr-span')} IFR-span and "
        f"{evidence('recovery_at_5pct', 'attnlrp')} AttnLRP for R@5.",
        "Within the practical VizWiz-LF learned-method panel, the mean leaders "
        "were "
        f"{LATEX_LABELS[faithfulness_leaders['deletion_auc']]} for deletion AUC, "
        f"{LATEX_LABELS[faithfulness_leaders['insertion_auc']]} for insertion AUC, "
        f"and {LATEX_LABELS[faithfulness_leaders['visual_mas']]} for Visual-MAS. "
        "The paired intervals against AttnLRP "
        f"{faithfulness_evidence('deletion_auc', 'attnlrp')} on deletion, "
        f"{faithfulness_evidence('insertion_auc', 'attnlrp')} on insertion, and "
        f"{faithfulness_evidence('visual_mas', 'attnlrp')} on Visual-MAS. "
        "Against Center prior they "
        f"{faithfulness_evidence('deletion_auc', 'center')} on deletion, "
        f"{faithfulness_evidence('insertion_auc', 'center')} on insertion, and "
        f"{faithfulness_evidence('visual_mas', 'center')} on Visual-MAS. "
        "Visual LOO remains in the full appendix as a cost-insensitive "
        "perturbation diagnostic rather than the practical main comparison. "
        "We therefore do not claim an across-metric winner; the main conclusions "
        "are endpoint-specific and are interpreted alongside the center, border, "
        "mask-convention, sign, and fully-correct sensitivity analyses.",
        "",
    ]
    return "\n".join(lines)


def _appendix_tex(
    *,
    wiki_funnel: Mapping[str, Any],
    viz_funnel: Mapping[str, Any],
    localization: Mapping[str, Any],
    wiki_diagnostics: Mapping[str, Any],
    viz_diagnostics: Mapping[str, Any],
    viz_faith: Mapping[str, Any],
    wiki_faith: Mapping[str, Any],
    wiki_attribution_summary: Mapping[str, Any],
    viz_attribution_summary: Mapping[str, Any],
    wiki_faithfulness_summary: Mapping[str, Any],
    viz_faithfulness_summary: Mapping[str, Any],
    wiki_manual_audit: Mapping[str, Any],
    viz_manual_audit: Mapping[str, Any],
) -> str:
    lines = [
        "% Generated by evaluations.multimodal.render_formal_results.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{\textbf{Formal visual gate funnels.} Every filter precedes attribution; prior pilot samples are excluded before freezing IDs.}",
        r"\label{tab:visual_gate_funnels}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\rowcolor{gray!7}",
        r"\textbf{Dataset} & \textbf{Stage} & \textbf{Passed} & \textbf{Eliminated at stage} & \textbf{Not evaluated} \\",
        r"\midrule",
    ]
    for dataset, funnel in (
        ("Wiki-VISA", wiki_funnel),
        ("VizWiz-LF", viz_funnel),
    ):
        for stage in funnel["stages"]:
            lines.append(
                f"{dataset} & {_latex_escape(str(stage['stage']))} & "
                f"{stage['passed']} & {stage['eliminated_at_stage']} & "
                f"{stage.get('not_evaluated_at_stage', 0)} \\\\"
            )
        if dataset == "Wiki-VISA":
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Supplemental Wiki-VISA localization endpoints.} Pointing Game, top-area IoU, R@1, and R@10 use the same whole-patch tie-aware protocol and paired $n=120$ intersection as the preregistered endpoints, but are reported only as supplemental diagnostics.}",
            r"\label{tab:visual_wiki_localization_supplemental}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Method} & \textbf{Pointing Game} $\uparrow$ & \textbf{Top-area IoU} $\uparrow$ & \textbf{R@1} $\uparrow$ & \textbf{R@10} $\uparrow$ \\",
            r"\midrule",
        ]
    )
    for method in METHODS:
        lines.append(
            f"{LATEX_LABELS[method]} & "
            f"{_latex_cell(localization['estimates']['pointing_game'][method])} & "
            f"{_latex_cell(localization['estimates']['top_evidence_iou'][method])} & "
            f"{_latex_cell(localization['estimates']['recovery_at_1pct'][method])} & "
            f"{_latex_cell(localization['estimates']['recovery_at_10pct'][method])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Native attribution and common perturbation resolutions.} Shapes are reported as rows$\times$columns with the number of frozen samples using each shape. Learned visual-token methods share a native grid within each image; Visual LOO is the registered coarse $4{\times}4$ perturbation reference.}",
            r"\label{tab:visual_spatial_resolution}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llll}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Dataset} & \textbf{Method} & \textbf{Native attribution grids} & \textbf{Faithfulness layouts} \\",
            r"\midrule",
        ]
    )
    for dataset_index, (
        dataset,
        attribution,
        faithfulness,
    ) in enumerate(
        (
            (
                "Wiki-VISA",
                wiki_attribution_summary,
                wiki_faithfulness_summary,
            ),
            (
                "VizWiz-LF",
                viz_attribution_summary,
                viz_faithfulness_summary,
            ),
        )
    ):
        for method in METHODS:
            native = _latex_escape(
                _shape_counts_cell(
                    attribution["methods"][method].get(
                        "native_grid_shapes"
                    )
                    or {}
                )
            )
            perturbation = _latex_escape(
                _shape_counts_cell(
                    faithfulness["methods"][method].get("region_layouts")
                    or {}
                )
            )
            lines.append(
                f"{dataset} & {LATEX_LABELS[method]} & "
                f"{native} & {perturbation} \\\\"
            )
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Wiki-VISA localization by frozen stratum.} Values are paired means with 95\% bootstrap intervals; Energy and R@5 are the primary endpoints.}",
            r"\label{tab:visual_wiki_strata}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llcc}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Stratum} & \textbf{Method} & \textbf{Energy} $\uparrow$ & \textbf{R@5} $\uparrow$ \\",
            r"\midrule",
        ]
    )
    groups = list(localization["per_group_paired"])
    for group_index, group in enumerate(groups):
        analysis = localization["per_group_paired"][group]
        for method in METHODS:
            lines.append(
                f"{_latex_escape(group)} & {LATEX_LABELS[method]} & "
                f"{_latex_cell(analysis['estimates']['energy_in_mask'][method])} & "
                f"{_latex_cell(analysis['estimates']['recovery_at_5pct'][method])} \\\\"
            )
        if group_index + 1 < len(groups):
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Complete frozen-response faithfulness panel and sign sensitivity.} Signed-score ordering is the registered analysis; positive-only deletion ordering is a sensitivity check.}",
            r"\label{tab:visual_faithfulness_full}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Dataset} & \textbf{Method} & \textbf{Deletion signed} $\downarrow$ & \textbf{Deletion positive-only} $\downarrow$ & \textbf{Insertion} $\uparrow$ & \textbf{Visual-MAS} $\downarrow$ \\",
            r"\midrule",
        ]
    )
    for dataset_index, (dataset, analysis) in enumerate(
        (("VizWiz-LF", viz_faith), ("Wiki-VISA", wiki_faith))
    ):
        signed = analysis["overall"]["estimates"]
        positive = analysis["positive_only_ordering"]["estimates"]
        for method in METHODS:
            lines.append(
                f"{dataset} & {LATEX_LABELS[method]} & "
                f"{_latex_cell(signed[method]['deletion_auc'])} & "
                f"{_latex_cell(positive[method]['deletion_auc'])} & "
                f"{_latex_cell(signed[method]['insertion_auc'])} & "
                f"{_latex_cell(signed[method]['visual_mas'])} \\\\"
            )
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{VizWiz-LF fully-correct sensitivity.} The subset is defined by the adjudicated semantic labels and is not used to select the frozen faithfulness panel.}",
            r"\label{tab:visual_vizwiz_fully_correct}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Method} & \textbf{Deletion AUC} $\downarrow$ & \textbf{Insertion AUC} $\uparrow$ & \textbf{Visual-MAS} $\downarrow$ \\",
            r"\midrule",
        ]
    )
    fully_correct = viz_faith["fully_correct_subset"]["estimates"]
    for method in METHODS:
        lines.append(
            f"{LATEX_LABELS[method]} & "
            f"{_latex_cell(fully_correct[method]['deletion_auc'])} & "
            f"{_latex_cell(fully_correct[method]['insertion_auc'])} & "
            f"{_latex_cell(fully_correct[method]['visual_mas'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Single-hop recursion and geometry diagnostics.} Cosine compares exact-reasoning and all-generation maps; recursive mass is the positive visual mass contributed by the one-hop term.}",
            r"\label{tab:visual_recursion_geometry}",
            r"\resizebox{\linewidth}{!}{%",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Dataset} & \textbf{Exact/all-gen cosine} & \textbf{Direct positive mass} & \textbf{Recursive positive mass} & \textbf{Recursive absolute mass} & \textbf{FlashTrace border mass} \\",
            r"\midrule",
        ]
    )
    for dataset, analysis in (
        ("Wiki-VISA", wiki_diagnostics),
        ("VizWiz-LF", viz_diagnostics),
    ):
        lines.append(
            f"{dataset} & {_latex_cell(analysis['exact_all_gen_cosine'])} & "
            f"{_latex_cell(analysis['direct_positive_fraction'])} & "
            f"{_latex_cell(analysis['recursive_positive_fraction'])} & "
            f"{_latex_cell(analysis['recursive_absolute_fraction'])} & "
            f"{_latex_cell(analysis['geometry']['flashtrace']['border_mass_ratio'])} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Formal visual geometry and sign diagnostics.} Geometry uses positive visual mass; negative-cell fraction is reported from each signed attribution map before positive-mass normalization.}",
            r"\label{tab:visual_geometry_bias}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llcccc}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Dataset} & \textbf{Method} & \textbf{Border mass} & \textbf{Top-row mass} & \textbf{Centroid distance} & \textbf{Negative cells} \\",
            r"\midrule",
        ]
    )
    for dataset_index, (dataset, analysis) in enumerate(
        (("Wiki-VISA", wiki_diagnostics), ("VizWiz-LF", viz_diagnostics))
    ):
        for method in METHODS:
            geometry = analysis["geometry"][method]
            lines.append(
                f"{dataset} & {LATEX_LABELS[method]} & "
                f"{_latex_cell(geometry['border_mass_ratio'])} & "
                f"{_latex_cell(geometry['top_row_mass_ratio'])} & "
                f"{_latex_cell(geometry['heatmap_centroid_distance_to_center'])} & "
                f"{_latex_cell(geometry['negative_cell_fraction'])} \\\\"
            )
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Wiki-VISA native-evidence centroid bias by stratum.} Distance is measured from the evidence centroid to image center.}",
            r"\label{tab:visual_gt_centroid_bias}",
            r"\begin{tabular}{lc}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Stratum} & \textbf{Centroid distance} \\",
            r"\midrule",
        ]
    )
    for stratum, estimate in wiki_diagnostics[
        "ground_truth_centroid_distance"
    ].items():
        lines.append(f"{_latex_escape(stratum)} & {_latex_cell(estimate)} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Effect of one-hop recursion by reasoning length.} Localization deltas are $K{=}1-K{=}0$; deletion delta is oriented so positive favors $K{=}1$. VizWiz-LF has no localization ground truth.}",
            r"\label{tab:visual_recursion_buckets}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llccc}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Dataset} & \textbf{THINKING bucket} & \textbf{$\Delta$ Energy} & \textbf{$\Delta$ R@5} & \textbf{Favorable $\Delta$ deletion AUC} \\",
            r"\midrule",
        ]
    )
    for dataset_index, (
        dataset,
        diagnostics,
        faithfulness,
    ) in enumerate(
        (
            ("Wiki-VISA", wiki_diagnostics, wiki_faith),
            ("VizWiz-LF", viz_diagnostics, viz_faith),
        )
    ):
        for bucket in ("short", "medium", "long"):
            if dataset == "Wiki-VISA":
                localization_bucket = diagnostics["recursion_by_thinking_bucket"][
                    bucket
                ]["ifr-span"]
                energy = _latex_cell(localization_bucket["energy_in_mask"])
                recovery = _latex_cell(localization_bucket["recovery_at_5pct"])
            else:
                energy = recovery = "--"
            deletion = faithfulness["recursion_by_thinking_bucket"][bucket][
                "flashtrace_favorable_difference"
            ]["ifr-span"]["deletion_auc"]
            lines.append(
                f"{dataset} & {bucket} & {energy} & {recovery} & "
                f"{_latex_cell(deletion)} \\\\"
            )
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\caption{\textbf{Observed formal visual compute.} Values are means per successful sample-method. VRAM is incremental peak allocation during attribution; faithfulness uses the common 64-region/10-step budget.}",
            r"\label{tab:visual_observed_compute}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"\rowcolor{gray!7}",
            r"\textbf{Dataset} & \textbf{Method} & \textbf{Attribution (s)} & \textbf{Peak VRAM (GiB)} & \textbf{Faithfulness (s)} \\",
            r"\midrule",
        ]
    )
    for dataset_index, (
        dataset,
        attribution,
        faithfulness,
    ) in enumerate(
        (
            (
                "Wiki-VISA",
                wiki_attribution_summary,
                wiki_faithfulness_summary,
            ),
            (
                "VizWiz-LF",
                viz_attribution_summary,
                viz_faithfulness_summary,
            ),
        )
    ):
        for method in METHODS:
            attr = attribution["methods"][method]
            faith = faithfulness["methods"][method]
            lines.append(
                f"{dataset} & {LATEX_LABELS[method]} & "
                f"{float(attr['mean_seconds']):.2f} & "
                f"{float(attr['mean_peak_vram_gb']):.2f} & "
                f"{float(faith['mean_seconds']):.2f} \\\\"
            )
        if dataset_index == 0:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
            r"\paragraph{Independent protocol audit.}",
            "A deterministic 10\\% caveat-only review inspected "
            f"{wiki_manual_audit['reviewed_count']}/120 Wiki-VISA and "
            f"{viz_manual_audit['reviewed_count']}/100 VizWiz-LF rows for "
            "image dependence and THINKING quality. Wiki image-dependence "
            f"labels were {_latex_escape(str(wiki_manual_audit['image_dependence_counts']))}; "
            "VizWiz labels were "
            f"{_latex_escape(str(viz_manual_audit['image_dependence_counts']))}. "
            "No review changed the frozen sample set.",
            "",
        ]
    )
    return "\n".join(lines)


def render(formal_dir: Path) -> tuple[str, str, str, str, str]:
    wiki_funnel = _read(formal_dir / "wiki_visa_funnel.json")
    viz_funnel = _read(formal_dir / "vizwiz_lf_funnel.json")
    localization = _read(formal_dir / "wiki_visa_n120_methods/analysis.json")
    wiki_diagnostics = _read(
        formal_dir / "wiki_visa_n120_methods/diagnostics.json"
    )
    viz_diagnostics = _read(
        formal_dir / "vizwiz_lf_n100_methods/diagnostics.json"
    )
    viz_faith = _read(formal_dir / "vizwiz_lf_n100_faithfulness/analysis.json")
    wiki_faith = _read(formal_dir / "wiki_visa_n120_faithfulness/analysis.json")
    wiki_attribution_summary = _read(
        formal_dir / "wiki_visa_n120_methods/summary.json"
    )
    viz_attribution_summary = _read(
        formal_dir / "vizwiz_lf_n100_methods/summary.json"
    )
    viz_faithfulness_summary = _read(
        formal_dir / "vizwiz_lf_n100_faithfulness/summary.json"
    )
    wiki_faithfulness_summary = _read(
        formal_dir / "wiki_visa_n120_faithfulness/summary.json"
    )
    semantic = _read(formal_dir / "vizwiz_lf_n100.semantic_summary.json")
    wiki_preview_reuse = _read(
        formal_dir / "wiki_visa_n120.preview_reuse.json"
    )
    viz_preview_reuse = _read(
        formal_dir / "vizwiz_lf_n100.preview_reuse.json"
    )
    wiki_manual_audit = _read(
        formal_dir / "wiki_visa_n120.protocol_audit_summary.json"
    )
    viz_manual_audit = _read(
        formal_dir / "vizwiz_lf_n100.protocol_audit_summary.json"
    )
    if semantic.get("complete") is not True:
        raise ValueError("VizWiz semantic judgment/audit is incomplete")
    if (
        wiki_manual_audit.get("complete") is not True
        or wiki_manual_audit.get("reviewed_count") != 12
        or viz_manual_audit.get("complete") is not True
        or viz_manual_audit.get("reviewed_count") != 10
    ):
        raise ValueError("independent 10% protocol audits are incomplete")
    if localization["common_samples"] != 120:
        raise ValueError("Wiki localization common intersection is not n=120")
    if viz_faith["overall"]["samples"] != 100:
        raise ValueError("VizWiz faithfulness common intersection is not n=100")
    if wiki_faith["overall"]["samples"] != 120:
        raise ValueError("Wiki faithfulness common intersection is not n=120")
    if "fully_correct_subset" not in viz_faith:
        raise ValueError("VizWiz A8 fully-correct subset analysis is absent")
    if (
        wiki_preview_reuse.get("identity_mismatched_sample_ids")
        or viz_preview_reuse.get("identity_mismatched_sample_ids")
    ):
        raise ValueError("preview checkpoint reuse contains a response mismatch")

    lines = [
        "# Formal visual evaluation v2 results",
        "",
        "Protocol frozen 2026-07-24. Formal samples exclude the strict and native "
        "pilots; pilot estimates are never pooled with these results.",
        "",
        "The earlier breadth-first n=20 previews overlap the formal fixed-seed "
        f"freeze by {wiki_preview_reuse['identity_matched_samples']} Wiki-VISA "
        f"and {viz_preview_reuse['identity_matched_samples']} VizWiz-LF samples. "
        "For those rows only, deterministic GPU records were reused after exact "
        "image/question, frozen-response, token-ID, and model-revision identity "
        "checks; every formal estimate and bootstrap denominator is recomputed "
        "from the complete n=120/n=100 frozen sets.",
        "",
        "## E1/E2 gate funnels",
        "",
    ]
    lines.extend(_funnel_markdown("Wiki-VISA", wiki_funnel))
    lines.extend(_funnel_markdown("VizWiz-LF", viz_funnel))
    lines.extend(_localization_markdown(localization))
    lines.extend(_faithfulness_markdown("VizWiz-LF", viz_faith, primary=True))
    lines.extend(_faithfulness_markdown("Wiki-VISA", wiki_faith, primary=False))
    lines.extend(_diagnostics_markdown(wiki_diagnostics, viz_diagnostics))
    lines.extend(
        _recursion_bucket_markdown(
            wiki_diagnostics,
            viz_diagnostics,
            wiki_faith,
            viz_faith,
        )
    )
    lines.extend(
        _resolution_markdown(
            wiki_attribution_summary,
            viz_attribution_summary,
            wiki_faithfulness_summary,
            viz_faithfulness_summary,
        )
    )
    lines.extend(
        _timing_markdown(
            wiki_attribution_summary,
            viz_attribution_summary,
            wiki_faithfulness_summary,
            viz_faithfulness_summary,
        )
    )
    lines.extend(_fully_correct_markdown(viz_faith, semantic))
    lines.extend(
        [
            "## Independent frozen-sample protocol audits",
            "",
            f"Wiki-VISA ({wiki_manual_audit['reviewed_count']}/120): "
            f"image dependence {wiki_manual_audit['image_dependence_counts']}; "
            f"THINKING quality {wiki_manual_audit['thinking_quality_counts']}.",
            "",
            f"VizWiz-LF ({viz_manual_audit['reviewed_count']}/100): "
            f"image dependence {viz_manual_audit['image_dependence_counts']}; "
            f"THINKING quality {viz_manual_audit['thinking_quality_counts']}.",
            "",
            "These reviews are caveat-only and did not change frozen IDs.",
            "",
            "## Scope and limitations",
            "",
            "- Wiki-VISA boxes mark supporting HTML elements, not exhaustive "
            "word-level evidence.",
            "- VizWiz-LF evaluates faithfulness to prompted long-form model "
            "responses; answer correctness is a sensitivity label, not a gate.",
            "- Strict stability and image-dependence gates improve internal "
            "validity while reducing representativeness; the complete funnels "
            "above expose that selection.",
            "- Center prior remains in every method panel. Claims are limited to "
            "one frozen VLM and one recursive hop.",
            "- CLEVR-XAI n=20 and VISTAQA n=10 remain separate diagnostics; see "
            "`A6_LEGACY_DIAGNOSTICS.md`.",
            "",
        ]
    )

    localization_rows = []
    localization_metrics = (
        "energy_in_mask",
        "evidence_rank_auc",
        "recovery_at_5pct",
        "recovery_at_20pct",
    )
    for method in METHODS:
        prefix = r"\rowcolor{cyan!10} " if method == "flashtrace" else ""
        cells = [
            _latex_cell(localization["estimates"][metric][method])
            for metric in localization_metrics
        ]
        localization_rows.append(
            prefix + LATEX_LABELS[method] + " & " + " & ".join(cells) + r" \\"
        )
        if method == "flashtrace":
            localization_rows.append(r"\midrule")

    faith_rows = []
    for method in ("visual-ig", "attnlrp", "flashtrace"):
        prefix = r"\rowcolor{cyan!10} " if method == "flashtrace" else ""
        estimates = viz_faith["overall"]["estimates"][method]
        cells = [
            _latex_cell(estimates[metric])
            for metric in ("deletion_auc", "insertion_auc", "visual_mas")
        ]
        faith_rows.append(
            prefix + LATEX_LABELS[method] + " & " + " & ".join(cells) + r" \\"
        )
    appendix = _appendix_tex(
        wiki_funnel=wiki_funnel,
        viz_funnel=viz_funnel,
        localization=localization,
        wiki_diagnostics=wiki_diagnostics,
        viz_diagnostics=viz_diagnostics,
        viz_faith=viz_faith,
        wiki_faith=wiki_faith,
        wiki_attribution_summary=wiki_attribution_summary,
        viz_attribution_summary=viz_attribution_summary,
        wiki_faithfulness_summary=wiki_faithfulness_summary,
        viz_faithfulness_summary=viz_faithfulness_summary,
        wiki_manual_audit=wiki_manual_audit,
        viz_manual_audit=viz_manual_audit,
    )
    discussion = _visual_discussion_tex(localization, viz_faith)
    return (
        "\n".join(lines),
        "\n".join(localization_rows) + "\n",
        "\n".join(faith_rows) + "\n",
        appendix + "\n",
        discussion,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--localization-tex", type=Path, required=True)
    parser.add_argument("--faithfulness-tex", type=Path, required=True)
    parser.add_argument("--appendix-tex", type=Path, required=True)
    parser.add_argument("--discussion-tex", type=Path, required=True)
    args = parser.parse_args()
    (
        markdown,
        localization_tex,
        faithfulness_tex,
        appendix_tex,
        discussion_tex,
    ) = render(
        args.formal_dir
    )
    for path, text in (
        (args.output_markdown, markdown),
        (args.localization_tex, localization_tex),
        (args.faithfulness_tex, faithfulness_tex),
        (args.appendix_tex, appendix_tex),
        (args.discussion_tex, discussion_tex),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "markdown": str(args.output_markdown),
                "localization_tex": str(args.localization_tex),
                "faithfulness_tex": str(args.faithfulness_tex),
                "appendix_tex": str(args.appendix_tex),
                "discussion_tex": str(args.discussion_tex),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
