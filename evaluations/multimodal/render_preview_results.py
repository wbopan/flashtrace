"""Render the isolated, pilot-disjoint n=20 formal-pipeline preview."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl


METHODS = (
    "random",
    "center",
    "visual-loo",
    "ifr-span",
    "visual-ig",
    "attnlrp",
    "flashtrace",
    "flashtrace-all-gen",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _estimate(value: dict[str, Any], digits: int = 4) -> str:
    return (
        f"{float(value['mean']):.{digits}f} "
        f"[{float(value['ci95_low']):.{digits}f}, "
        f"{float(value['ci95_high']):.{digits}f}]"
    )


def _delta(value: dict[str, Any], digits: int = 4) -> str:
    return (
        f"{float(value['mean']):+.{digits}f} "
        f"[{float(value['ci95_low']):+.{digits}f}, "
        f"{float(value['ci95_high']):+.{digits}f}] "
        f"({value['wins']}/{value['ties']}/{value['losses']})"
    )


def _assert_complete(
    attribution: dict[str, Any], faithfulness: dict[str, Any]
) -> None:
    if int(attribution["common_samples"]) != 20:
        raise ValueError("attribution preview is not a complete paired n=20 matrix")
    if int(faithfulness["common_samples"]) != 20:
        raise ValueError("faithfulness preview is not a complete paired n=20 matrix")
    if tuple(attribution["methods"]) != METHODS:
        raise ValueError("attribution method panel does not match the frozen eight methods")
    if tuple(faithfulness["methods"]) != METHODS:
        raise ValueError("faithfulness method panel does not match the frozen eight methods")
    for method in METHODS:
        if int(attribution["methods"][method]["common_samples"]) != 20:
            raise ValueError(f"incomplete attribution method: {method}")
        faith = faithfulness["methods"][method]
        if int(faith["common_samples"]) != 20:
            raise ValueError(f"incomplete faithfulness method: {method}")
        if faith["degenerate_deletion_curves"] or faith["degenerate_insertion_curves"]:
            raise ValueError(f"degenerate faithfulness curve: {method}")


def _shape_distributions(
    records: list[dict[str, Any]], *, faithfulness: bool
) -> dict[str, Counter[str]]:
    output = {method: Counter() for method in METHODS}
    for record in records:
        if record.get("status") != "ok" or record.get("method") not in output:
            continue
        shape = (
            (record.get("faithfulness") or {}).get("region_layout")
            if faithfulness
            else record.get("visual_grid_shape")
        )
        if isinstance(shape, list) and len(shape) == 2:
            output[str(record["method"])][f"{shape[0]}x{shape[1]}"] += 1
    return output


def _shape_cell(counts: Counter[str]) -> str:
    return ", ".join(
        f"{shape} (n={count})" for shape, count in sorted(counts.items())
    )


def render(preview_dir: Path) -> str:
    wiki_manifest = read_jsonl(preview_dir / "wiki_visa_n20.dataset.jsonl")
    viz_manifest = read_jsonl(preview_dir / "vizwiz_lf_n20.dataset.jsonl")
    wiki_funnel = _read_json(preview_dir / "wiki_visa_n20_funnel.json")
    viz_funnel = _read_json(preview_dir / "vizwiz_lf_n20_funnel.json")
    wiki_attr_summary = _read_json(
        preview_dir / "wiki_visa_n20_methods" / "summary.json"
    )
    viz_attr_summary = _read_json(
        preview_dir / "vizwiz_lf_n20_methods" / "summary.json"
    )
    wiki_attr = _read_json(preview_dir / "wiki_visa_n20_methods" / "analysis.json")
    wiki_diag = _read_json(
        preview_dir / "wiki_visa_n20_methods" / "diagnostics.json"
    )
    viz_diag = _read_json(
        preview_dir / "vizwiz_lf_n20_methods" / "diagnostics.json"
    )
    wiki_faith_summary = _read_json(
        preview_dir / "wiki_visa_n20_faithfulness" / "summary.json"
    )
    viz_faith_summary = _read_json(
        preview_dir / "vizwiz_lf_n20_faithfulness" / "summary.json"
    )
    wiki_faith = _read_json(
        preview_dir / "wiki_visa_n20_faithfulness" / "analysis.json"
    )
    viz_faith = _read_json(
        preview_dir / "vizwiz_lf_n20_faithfulness" / "analysis.json"
    )
    semantic = _read_json(preview_dir / "vizwiz_lf_n20.semantic_summary.json")
    wiki_native_shapes = _shape_distributions(
        read_jsonl(
            preview_dir
            / "wiki_visa_n20_methods"
            / "attribution_records.jsonl"
        ),
        faithfulness=False,
    )
    viz_native_shapes = _shape_distributions(
        read_jsonl(
            preview_dir
            / "vizwiz_lf_n20_methods"
            / "attribution_records.jsonl"
        ),
        faithfulness=False,
    )
    wiki_faith_shapes = _shape_distributions(
        read_jsonl(
            preview_dir
            / "wiki_visa_n20_faithfulness"
            / "faithfulness_records.jsonl"
        ),
        faithfulness=True,
    )
    viz_faith_shapes = _shape_distributions(
        read_jsonl(
            preview_dir
            / "vizwiz_lf_n20_faithfulness"
            / "faithfulness_records.jsonl"
        ),
        faithfulness=True,
    )

    _assert_complete(wiki_attr_summary, wiki_faith_summary)
    _assert_complete(viz_attr_summary, viz_faith_summary)
    if len(wiki_manifest) != 20 or len(viz_manifest) != 20:
        raise ValueError("frozen preview manifests must each contain 20 samples")
    if wiki_attr["common_samples"] != 20:
        raise ValueError("Wiki localization bootstrap is not paired n=20")
    if viz_faith.get("fully_correct_subset", {}).get("samples", 0) == 0:
        raise ValueError("VizWiz fully-correct sensitivity analysis is missing")

    strata = Counter(record["evaluation"]["metadata"]["stratum"] for record in wiki_manifest)
    lines = [
        "# Formal-pipeline preview: Wiki-VISA n=20 / VizWiz-LF n=20",
        "",
        "> **Preview only — not the frozen formal result.** These samples are disjoint "
        "from prior pilots and use the formal model, gates, eight-method panel, "
        "64-region/10-step faithfulness budget, and 50,000-draw paired bootstrap. "
        "The formal targets remain Wiki-VISA n=120 and VizWiz-LF n=100.",
        "",
        "## Completion and freeze",
        "",
        "| dataset | candidates | strict eligible | frozen | attribution | faithfulness | localization GT |",
        "|---|---:|---:|---:|---:|---:|---|",
        f"| Wiki-VISA | {wiki_funnel['candidate_count']} | "
        f"{wiki_funnel['strict_eligible_count']} | 20 | 160/160 | 160/160 | yes |",
        f"| VizWiz-LF | {viz_funnel['candidate_count']} | "
        f"{viz_funnel['strict_eligible_count']} | 20 | 160/160 | 160/160 | no, by design |",
        "",
        f"Wiki strata are balanced as frozen: first-page passage "
        f"{strata['first_page_passage']}, later-page passage "
        f"{strata['later_page_passage']}, non-passage {strata['non_passage']}. "
        "Both faithfulness panels have zero degenerate deletion or insertion curves.",
        "",
        "VizWiz semantic grading covers all 20 samples: "
        f"{semantic['label_counts']['fully']} fully correct, "
        f"{semantic['label_counts']['partial']} partial, and "
        f"{semantic['label_counts']['wrong']} wrong. The deterministic 10% human "
        f"audit is still unsigned ({semantic['audit_reviewed']}/"
        f"{len(semantic['audit_sample_ids'])}); therefore the fully-correct "
        "sensitivity result below is LLM-judged and provisional.",
        "",
        "## Wiki-VISA localization",
        "",
        "| method | Energy | Rank AUC | R@5 | R@20 |",
        "|---|---:|---:|---:|---:|",
    ]
    localization_index = lines.index("## Wiki-VISA localization")
    resolution_lines = [
        "## Spatial resolution disclosure",
        "",
        "| dataset | method | native attribution grids | faithfulness layouts |",
        "|---|---|---|---|",
    ]
    for dataset, native, faith in (
        ("Wiki-VISA", wiki_native_shapes, wiki_faith_shapes),
        ("VizWiz-LF", viz_native_shapes, viz_faith_shapes),
    ):
        for method in METHODS:
            resolution_lines.append(
                f"| {dataset} | {method} | {_shape_cell(native[method])} | "
                f"{_shape_cell(faith[method])} |"
            )
    resolution_lines.extend(
        [
            "",
            "IFR-span, Visual IG, AttnLRP, FlashTrace, and "
            "FlashTrace all-generation share the same native visual-token "
            "grid within each image. Random/Center use 32x32 synthetic grids; "
            "Visual LOO is a coarse 4x4 perturbation reference. Every method "
            "uses the same approximately 64-region faithfulness layout for a "
            "given image, via nearest-patch resampling.",
            "",
        ]
    )
    lines[localization_index:localization_index] = resolution_lines
    for method in METHODS:
        lines.append(
            f"| {method} | "
            f"{_estimate(wiki_attr['estimates']['energy_in_mask'][method])} | "
            f"{_estimate(wiki_attr['estimates']['evidence_rank_auc'][method])} | "
            f"{_estimate(wiki_attr['estimates']['recovery_at_5pct'][method])} | "
            f"{_estimate(wiki_attr['estimates']['recovery_at_20pct'][method])} |"
        )

    lines.extend(
        [
            "",
            "FlashTrace has the highest Rank AUC (0.8240) and R@5 (0.3315); "
            "AttnLRP has the highest Energy (0.3020); Visual LOO and FlashTrace "
            "are effectively tied at R@20 (0.6593 vs. 0.6573). With n=20, these "
            "are preview estimates rather than final rankings.",
            "",
            "### Paired FlashTrace differences on primary endpoints",
            "",
            "Positive values favor exact-span FlashTrace; parentheses are W/T/L.",
            "",
            "| baseline | Energy Δ [95% CI] | Rank AUC Δ [95% CI] | R@5 Δ [95% CI] |",
            "|---|---:|---:|---:|",
        ]
    )
    for baseline in ("visual-loo", "ifr-span", "visual-ig", "attnlrp", "flashtrace-all-gen"):
        differences = wiki_attr["flashtrace_minus_baseline"]
        lines.append(
            f"| {baseline} | {_delta(differences['energy_in_mask'][baseline])} | "
            f"{_delta(differences['evidence_rank_auc'][baseline])} | "
            f"{_delta(differences['recovery_at_5pct'][baseline])} |"
        )

    lines.extend(
        [
            "",
            "## Frozen-response visual faithfulness",
            "",
            "Deletion AUC and Visual-MAS are lower-is-better; insertion AUC is "
            "higher-is-better. Values are means with paired-sample bootstrap CIs.",
        ]
    )
    for dataset, analysis in (("Wiki-VISA", wiki_faith), ("VizWiz-LF", viz_faith)):
        lines.extend(
            [
                "",
                f"### {dataset}",
                "",
                "| method | deletion AUC | insertion AUC | Visual-MAS |",
                "|---|---:|---:|---:|",
            ]
        )
        for method in METHODS:
            estimates = analysis["overall"]["estimates"][method]
            lines.append(
                f"| {method} | {_estimate(estimates['deletion_auc'])} | "
                f"{_estimate(estimates['insertion_auc'])} | "
                f"{_estimate(estimates['visual_mas'])} |"
            )

    fully = viz_faith["fully_correct_subset"]
    lines.extend(
        [
            "",
            f"### VizWiz-LF fully-correct sensitivity (LLM-judged n={fully['samples']})",
            "",
            "| method | deletion AUC | insertion AUC | Visual-MAS |",
            "|---|---:|---:|---:|",
        ]
    )
    for method in METHODS:
        estimates = fully["estimates"][method]
        lines.append(
            f"| {method} | {_estimate(estimates['deletion_auc'])} | "
            f"{_estimate(estimates['insertion_auc'])} | "
            f"{_estimate(estimates['visual_mas'])} |"
        )

    lines.extend(
        [
            "",
            "On Wiki-VISA, Visual LOO is strongest on all three faithfulness "
            "summaries, while FlashTrace is consistently stronger than IFR-span "
            "and exact-span is modestly stronger than all-generation. On VizWiz-LF "
            "there is no single overall winner: Visual LOO has the lowest deletion "
            "AUC, center has the highest insertion AUC, and all-generation has the "
            "lowest overall Visual-MAS. In the provisional fully-correct subset, "
            "Visual LOO narrowly leads deletion and insertion, while FlashTrace has "
            "the lowest Visual-MAS. The wide n=20/n=13 intervals argue against a "
            "final winner claim.",
            "",
            "## Recursive mechanism and map diagnostics",
            "",
            "| dataset | exact/all-gen cosine | recursive positive mass |",
            "|---|---:|---:|",
            f"| Wiki-VISA | {_estimate(wiki_diag['exact_all_gen_cosine'])} | "
            f"{_estimate(wiki_diag['recursive_positive_fraction'])} |",
            f"| VizWiz-LF | {_estimate(viz_diag['exact_all_gen_cosine'])} | "
            f"{_estimate(viz_diag['recursive_positive_fraction'])} |",
            "",
            "Exact-span and all-generation maps are highly aligned, but the paired "
            "Wiki localization analysis still favors exact-span FlashTrace over "
            "all-generation on Energy, Rank AUC, and R@5. Recursive visual mass is "
            "substantial (about 45% on Wiki and 34% on VizWiz). Signed baselines "
            "also show material negative-cell fractions, so their positive-only "
            "ordering sensitivity remains a required appendix result.",
            "",
            "## Attribution resource profile",
            "",
            "| dataset | method | seconds/sample | peak VRAM GiB |",
            "|---|---|---:|---:|",
        ]
    )
    for dataset, summary in (
        ("Wiki-VISA", wiki_attr_summary),
        ("VizWiz-LF", viz_attr_summary),
    ):
        for method in METHODS[2:]:
            values = summary["methods"][method]
            lines.append(
                f"| {dataset} | {method} | {values['mean_seconds']:.3f} | "
                f"{values['mean_peak_vram_gb']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Pipeline findings before the full run",
            "",
            "- VizWiz gate refresh rejected 9/40 explicit unanswerable responses "
            "that the earlier refusal pattern missed.",
            "",
            "- Two long VizWiz AttnLRP cases originally exhausted device memory. "
            "Gradient checkpointing and target-row-only LM-head projection fixed "
            "them without changing the frozen target-logit objective; the final "
            "preview matrices are complete.",
            "",
            "- No CLEVR-XAI rerun is part of this preview. A6 remains the existing "
            "offline legacy diagnostic appendix only.",
            "",
            "## Interpretation boundary",
            "",
            "This preview validates the execution path and exposes likely effect "
            "directions. It does not replace the formal n=120/n=100 freeze, does "
            "not justify paper-level significance claims, and does not satisfy the "
            "final independent human-audit requirement. The next execution step is "
            "to resume the full Wiki-VISA and VizWiz-LF queues with the corrected "
            "gates and memory-safe AttnLRP implementation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render(args.preview_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {"output": str(args.output), "bytes": len(rendered.encode("utf-8"))},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
