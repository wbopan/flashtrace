#!/usr/bin/env python3
"""Audit agent experiments, bootstrap paired metrics, and render paper rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from evaluations.agentdojo.attribute_full import _eligible as _dojo_localizable


BOOTSTRAP_DRAWS = 50_000
SEED = 20260728
AIDER_METHODS = (
    "ifr_last_output_line",
    "ifr_exhaustive_full_action",
    "attnlrp_full_action",
    "flashtrace_k1_full_action",
)
AGENTDOJO_METHODS = (
    "random_expected",
    "ifr_last_generated_action_token",
    "ifr_full_generated_action",
    "flashtrace_generated_action_via_reasoning",
)
METHOD_LABELS = {
    "ifr_last_output_line": "IFR",
    "ifr_exhaustive_full_action": "IFR exhaustive",
    "attnlrp_full_action": "AttnLRP",
    "flashtrace_k1_full_action": r"\textbf{\flashtrace{}}",
    "random_expected": "Random",
    "ifr_last_generated_action_token": "IFR",
    "ifr_full_generated_action": "IFR",
    "flashtrace_generated_action_via_reasoning": r"\textbf{\flashtrace{}}",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _input_provenance(path: Path, *, records: int) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "records": records,
    }


def _mean_ci(
    values: Iterable[float],
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> dict[str, float | int]:
    samples = np.asarray(list(values), dtype=np.float64)
    if samples.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(draws, dtype=np.float64)
    chunk = 5_000
    for start in range(0, draws, chunk):
        stop = min(start + chunk, draws)
        indices = rng.integers(0, samples.size, size=(stop - start, samples.size))
        bootstrap_means[start:stop] = samples[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return {
        "count": int(samples.size),
        "mean": float(samples.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _paired_difference(
    candidate: Iterable[float],
    baseline: Iterable[float],
    *,
    lower_is_better: bool,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = SEED,
) -> dict[str, float | int]:
    candidate_array = np.asarray(list(candidate), dtype=np.float64)
    baseline_array = np.asarray(list(baseline), dtype=np.float64)
    if candidate_array.shape != baseline_array.shape or candidate_array.size == 0:
        raise ValueError("Paired arrays must have the same non-zero shape.")
    favorable = (
        baseline_array - candidate_array
        if lower_is_better
        else candidate_array - baseline_array
    )
    estimate = _mean_ci(favorable, draws=draws, seed=seed)
    tolerance = 1e-12
    estimate.update(
        {
            "wins": int(np.sum(favorable > tolerance)),
            "ties": int(np.sum(np.abs(favorable) <= tolerance)),
            "losses": int(np.sum(favorable < -tolerance)),
        }
    )
    return estimate


def _aider_generation_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    turns = [turn for row in rows for turn in row["turns"]]
    repairs = [row for row in rows if row["repair_attempted"]]
    valid = [row for row in rows if row["valid_final_action"]]
    regressions = [
        row for row in rows if row["draft_passed"] and not row["final_passed"]
    ]
    return {
        "attempted": len(rows),
        "valid": len(valid),
        "valid_rate": len(valid) / len(rows),
        "format_compliance_rate": (
            sum(turn["parsed_edit"]["format_compliant"] for turn in turns)
            / len(turns)
        ),
        "draft_pass_rate": sum(row["draft_passed"] for row in rows) / len(rows),
        "final_pass_rate": sum(row["final_passed"] for row in rows) / len(rows),
        "repair_attempts": len(repairs),
        "revised_pass_rate_among_repairs": (
            sum(row["final_passed"] for row in repairs) / len(repairs)
            if repairs
            else None
        ),
        "repair_improvement_rate": (
            sum(row["repair_improved"] for row in repairs) / len(repairs)
            if repairs
            else None
        ),
        "regressed": len(regressions),
        "self_judge_used": False,
    }


def _repair_feedback_fraction(row: Mapping[str, Any]) -> float | None:
    result = row["result"]
    segments = result.get("source_segments")
    if segments is None:
        segments = result.get("trace", {}).get("segments", [])
    feedback = [
        float(segment["fraction"])
        for segment in segments
        if segment.get("role") == "user" and int(segment.get("turn", -1)) == 2
    ]
    return sum(feedback) if feedback else None


def _aider_attribution(
    rows: list[dict[str, Any]],
    *,
    draws: int,
) -> dict[str, Any]:
    successful: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row["status"] == "success":
            successful[str(row["example_id"])][row["method"]] = row
    common_ids = sorted(
        example_id
        for example_id, methods in successful.items()
        if all(method in methods for method in AIDER_METHODS)
    )
    subsets = {
        "all_valid": common_ids,
        "execution_success": [
            example_id
            for example_id in common_ids
            if successful[example_id][AIDER_METHODS[0]]["final_passed"]
        ],
    }
    output: dict[str, Any] = {
        "bootstrap_draws": draws,
        "common_success_count": len(common_ids),
        "subsets": {},
    }
    for subset_name, example_ids in subsets.items():
        subset: dict[str, Any] = {
            "count": len(example_ids),
            "methods": {},
            "flashtrace_favorable_difference": {},
        }
        if not example_ids:
            output["subsets"][subset_name] = subset
            continue
        for method_index, method in enumerate(AIDER_METHODS):
            method_rows = [successful[example_id][method] for example_id in example_ids]
            method_metrics = {
                metric: _mean_ci(
                    (row["result"][metric] for row in method_rows),
                    draws=draws,
                    seed=SEED + method_index,
                )
                for metric in ("rise", "mas", "attribution_seconds")
            }
            feedback_fractions = [
                fraction
                for row in method_rows
                if row.get("repair_attempted")
                if (fraction := _repair_feedback_fraction(row)) is not None
            ]
            method_metrics["repair_feedback_count"] = len(feedback_fractions)
            method_metrics["repair_feedback_fraction"] = (
                _mean_ci(
                    feedback_fractions,
                    draws=draws,
                    seed=SEED + method_index,
                )
                if feedback_fractions
                else None
            )
            subset["methods"][method] = method_metrics
        flashtrace = subset["methods"]["flashtrace_k1_full_action"]
        for baseline_index, baseline in enumerate(AIDER_METHODS[:-1]):
            subset["flashtrace_favorable_difference"][baseline] = {}
            for metric in ("rise", "mas"):
                subset["flashtrace_favorable_difference"][baseline][metric] = (
                    _paired_difference(
                        (
                            successful[example_id]["flashtrace_k1_full_action"][
                                "result"
                            ][metric]
                            for example_id in example_ids
                        ),
                        (
                            successful[example_id][baseline]["result"][metric]
                            for example_id in example_ids
                        ),
                        lower_is_better=True,
                        draws=draws,
                        seed=SEED + baseline_index,
                    )
                )
        del flashtrace
        output["subsets"][subset_name] = subset
    output["attempted_by_method"] = {
        method: sum(row["method"] == method for row in rows)
        for method in AIDER_METHODS
    }
    output["failed_by_method"] = {
        method: sum(
            row["method"] == method and row["status"] != "success" for row in rows
        )
        for method in AIDER_METHODS
    }
    return output


def _agentdojo_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        clean = [row for row in subset if row["attack_success"] is None]
        attacked = [row for row in subset if row["attack_success"] is not None]
        localizable = sum(_dojo_localizable(row) for row in attacked)
        return {
            "clean_count": len(clean),
            "attacked_count": len(attacked),
            "clean_utility_rate": sum(row["utility"] for row in clean) / len(clean),
            "attacked_utility_rate": (
                sum(row["utility"] for row in attacked) / len(attacked)
            ),
            "attack_success_rate": (
                sum(row["attack_success"] for row in attacked) / len(attacked)
            ),
            "joint_utility_and_attack_rate": (
                sum(row["utility"] and row["attack_success"] for row in attacked)
                / len(attacked)
            ),
            "generated_target_coverage": localizable / len(attacked),
            "localizable_cases": localizable,
        }

    suites = {
        suite: summarize([row for row in rows if row["case"]["suite"] == suite])
        for suite in ("banking", "slack", "travel", "workspace")
    }
    return {"overall": summarize(rows), "suites": suites}


def _agentdojo_localization(
    rows: list[dict[str, Any]],
    *,
    eligible_count: int,
    attacked_inventory: int,
    draws: int,
) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] == "success"]
    output: dict[str, Any] = {
        "eligible_count": eligible_count,
        "attacked_inventory": attacked_inventory,
        "eligible_coverage_over_attacked_inventory": (
            eligible_count / attacked_inventory
        ),
        "attempted": len(rows),
        "common_success_count": len(successful),
        "failed": len(rows) - len(successful),
        "bootstrap_draws": draws,
        "methods": {},
        "flashtrace_favorable_difference": {},
    }
    if not successful:
        return output
    metrics = (
        "recovery_at_5pct",
        "recovery_at_10pct",
        "recovery_at_20pct",
        "pointing_game",
        "injection_mass_fraction",
    )
    for method_index, method in enumerate(AGENTDOJO_METHODS):
        output["methods"][method] = {
            metric: _mean_ci(
                (
                    row["attribution"]["methods"][method]["localization_metrics"][
                        metric
                    ]
                    for row in successful
                ),
                draws=draws,
                seed=SEED + method_index,
            )
            for metric in metrics
        }
        output["methods"][method]["runtime_seconds"] = _mean_ci(
            (
                row["attribution"]["methods"][method]["runtime_seconds"]
                for row in successful
            ),
            draws=draws,
            seed=SEED + method_index,
        )
    flashtrace = "flashtrace_generated_action_via_reasoning"
    for baseline_index, baseline in enumerate(AGENTDOJO_METHODS[:-1]):
        output["flashtrace_favorable_difference"][baseline] = {}
        for metric in metrics:
            output["flashtrace_favorable_difference"][baseline][metric] = (
                _paired_difference(
                    (
                        row["attribution"]["methods"][flashtrace][
                            "localization_metrics"
                        ][metric]
                        for row in successful
                    ),
                    (
                        row["attribution"]["methods"][baseline][
                            "localization_metrics"
                        ][metric]
                        for row in successful
                    ),
                    lower_is_better=False,
                    draws=draws,
                    seed=SEED + baseline_index,
                )
            )
    return output


def _cell(estimate: Mapping[str, Any], digits: int = 3) -> str:
    return (
        r"\shortstack{"
        f"{float(estimate['mean']):.{digits}f}"
        r"\\{\scriptsize["
        f"{float(estimate['ci95_low']):.{digits}f},"
        f"{float(estimate['ci95_high']):.{digits}f}"
        r"]}}"
    )


def _optional_cell(
    estimate: Mapping[str, Any] | None,
    digits: int = 3,
) -> str:
    return "--" if estimate is None else _cell(estimate, digits)


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.1f}\\%"


def _render_tex(
    analysis: Mapping[str, Any],
    *,
    generator_label: str,
) -> dict[str, str]:
    audit = analysis["aider"]["generation_audit"]
    audit_row = (
        f"{generator_label} & {audit['attempted']} & {audit['valid']} & "
        f"{_pct(audit['format_compliance_rate'])} & "
        f"{_pct(audit['draft_pass_rate'])} & "
        f"{_pct(audit['revised_pass_rate_among_repairs'])} & "
        f"{_pct(audit['repair_improvement_rate'])} & "
        f"{audit['regressed']} & -- \\\\\n"
    )
    aider = analysis["aider"]["attribution"]
    all_valid = aider["subsets"]["all_valid"]
    execution = aider["subsets"]["execution_success"]
    scopes = {
        "ifr_last_output_line": "Last output line",
        "ifr_exhaustive_full_action": "Full final edit",
        "attnlrp_full_action": "Full final edit",
        "flashtrace_k1_full_action": "Final edit via repair turns",
    }
    aider_rows = []
    for method in AIDER_METHODS:
        all_method = all_valid.get("methods", {}).get(method, {})
        execution_method = execution.get("methods", {}).get(method, {})
        row_prefix = (
            r"\rowcolor{cyan!10}" + "\n"
            if method == "flashtrace_k1_full_action"
            else ""
        )
        aider_rows.append(
            f"{row_prefix}{METHOD_LABELS[method]} & {scopes[method]} & "
            f"{all_valid['count']} & "
            f"{_optional_cell(all_method.get('rise'))} & "
            f"{_optional_cell(execution_method.get('rise'))} & "
            f"{_optional_cell(all_method.get('mas'))} & "
            f"{_optional_cell(execution_method.get('mas'))} & "
            f"{_optional_cell(all_method.get('attribution_seconds'), 2)} \\\\"
        )
    outcomes = analysis["agentdojo"]["outcomes"]
    outcome_rows = []
    for suite in ("banking", "slack", "travel", "workspace"):
        values = outcomes["suites"][suite]
        outcome_rows.append(
            f"{generator_label} & {suite.title()} & "
            f"{values['clean_count']}/{values['attacked_count']} & "
            f"{_pct(values['clean_utility_rate'])} & "
            f"{_pct(values['attacked_utility_rate'])} & "
            f"{_pct(values['attack_success_rate'])} & "
            f"{_pct(values['joint_utility_and_attack_rate'])} & "
            f"{_pct(values['generated_target_coverage'])} & "
            f"{values['localizable_cases']} \\\\"
        )
    values = outcomes["overall"]
    outcome_rows.append(
        f"{generator_label} & Overall & "
        f"{values['clean_count']}/{values['attacked_count']} & "
        f"{_pct(values['clean_utility_rate'])} & "
        f"{_pct(values['attacked_utility_rate'])} & "
        f"{_pct(values['attack_success_rate'])} & "
        f"{_pct(values['joint_utility_and_attack_rate'])} & "
        f"{_pct(values['generated_target_coverage'])} & "
        f"{values['localizable_cases']} \\\\"
    )
    localization = analysis["agentdojo"]["localization"]
    target_labels = {
        "random_expected": "---",
        "ifr_last_generated_action_token": "Last action token",
        "ifr_full_generated_action": "Full generated action",
        "flashtrace_generated_action_via_reasoning": (
            "Full action via generated reasoning"
        ),
    }
    localization_rows = []
    for method in AGENTDOJO_METHODS:
        values = localization["methods"].get(method, {})
        row_prefix = (
            r"\rowcolor{cyan!10}" + "\n"
            if method == "flashtrace_generated_action_via_reasoning"
            else ""
        )
        localization_rows.append(
            f"{row_prefix}{METHOD_LABELS[method]} & {target_labels[method]} & "
            f"{_optional_cell(values.get('recovery_at_5pct'))} & "
            f"{_optional_cell(values.get('recovery_at_10pct'))} & "
            f"{_optional_cell(values.get('recovery_at_20pct'))} & "
            f"{_optional_cell(values.get('pointing_game'))} & "
            f"{_optional_cell(values.get('injection_mass_fraction'))} & "
            f"{_optional_cell(values.get('runtime_seconds'), 2)} \\\\"
        )
    return {
        "aider_audit_rows.tex": audit_row,
        "aider_multiturn_rows.tex": "\n".join(aider_rows) + "\n",
        "agentdojo_outcome_rows.tex": "\n".join(outcome_rows) + "\n",
        "agentdojo_localization_rows.tex": "\n".join(localization_rows) + "\n",
    }


def _render_markdown(analysis: Mapping[str, Any]) -> str:
    audit = analysis["aider"]["generation_audit"]
    aider = analysis["aider"]["attribution"]
    outcomes = analysis["agentdojo"]["outcomes"]["overall"]
    localization = analysis["agentdojo"]["localization"]
    aider_methods = aider["subsets"]["all_valid"]["methods"]
    dojo_methods = localization["methods"]
    method_labels = {
        "ifr_last_output_line": "IFR last",
        "ifr_exhaustive_full_action": "IFR exhaustive",
        "attnlrp_full_action": "AttnLRP",
        "flashtrace_k1_full_action": "FlashTrace",
    }
    aider_lines = [
        (
            f"- {method_labels[method]}: "
            f"RISE {values['rise']['mean']:.4f}, "
            f"MAS {values['mas']['mean']:.4f}, "
            f"time {values['attribution_seconds']['mean']:.2f}s"
        )
        for method, values in aider_methods.items()
    ]
    dojo_labels = {
        "random_expected": "Random",
        "ifr_last_generated_action_token": "IFR last",
        "ifr_full_generated_action": "IFR full",
        "flashtrace_generated_action_via_reasoning": "FlashTrace",
    }
    dojo_lines = [
        (
            f"- {dojo_labels[method]}: "
            f"R@5 {values['recovery_at_5pct']['mean']:.4f}, "
            f"R@20 {values['recovery_at_20pct']['mean']:.4f}, "
            f"time {values['runtime_seconds']['mean']:.2f}s"
        )
        for method, values in dojo_methods.items()
    ]
    return "\n".join(
        [
            "# Final agent experiment audit",
            "",
            "## Aider",
            "",
            f"- Attempted/valid: {audit['attempted']}/{audit['valid']}",
            f"- Draft/final pass rate: {audit['draft_pass_rate']:.4f} / "
            f"{audit['final_pass_rate']:.4f}",
            f"- Format compliance: {audit['format_compliance_rate']:.4f}",
            f"- Paired attribution intersection: "
            f"{aider['common_success_count']}",
            *aider_lines,
            "",
            "## AgentDojo",
            "",
            f"- Clean/attacked inventory: {outcomes['clean_count']}/"
            f"{outcomes['attacked_count']}",
            f"- Clean/attacked utility: {outcomes['clean_utility_rate']:.4f} / "
            f"{outcomes['attacked_utility_rate']:.4f}",
            f"- Attack success: {outcomes['attack_success_rate']:.4f}",
            f"- Localizable: {localization['eligible_count']}/"
            f"{localization['attacked_inventory']} "
            f"({localization['eligible_coverage_over_attacked_inventory']:.4f})",
            f"- Successful attribution: {localization['common_success_count']}/"
            f"{localization['attempted']}",
            *dojo_lines,
            "",
            f"All confidence intervals use {analysis['bootstrap_draws']:,} "
            "paired sample-bootstrap draws.",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aider-generation", type=Path, required=True)
    parser.add_argument("--aider-attribution", type=Path, required=True)
    parser.add_argument("--agentdojo-generation", type=Path, required=True)
    parser.add_argument(
        "--agentdojo-localization-generation",
        type=Path,
        help=(
            "Optional generation pool used to define localization eligibility. "
            "Defaults to --agentdojo-generation."
        ),
    )
    parser.add_argument("--agentdojo-attribution", type=Path, required=True)
    parser.add_argument("--generator-label", required=True)
    parser.add_argument(
        "--agentdojo-attacked-inventory",
        type=int,
        default=949,
    )
    parser.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluations/results/agent_final"),
    )
    parser.add_argument(
        "--paper-generated-dir",
        type=Path,
        default=Path("paper/generated"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_draws < 1:
        raise ValueError("--bootstrap-draws must be positive.")
    if args.agentdojo_attacked_inventory < 1:
        raise ValueError("--agentdojo-attacked-inventory must be positive.")
    aider_generation = _jsonl(args.aider_generation)
    aider_attribution = _jsonl(args.aider_attribution)
    agentdojo_generation = _jsonl(args.agentdojo_generation)
    agentdojo_localization_generation = (
        _jsonl(args.agentdojo_localization_generation)
        if args.agentdojo_localization_generation is not None
        else agentdojo_generation
    )
    agentdojo_attribution = _jsonl(args.agentdojo_attribution)
    eligible_count = sum(
        _dojo_localizable(row)
        for row in agentdojo_localization_generation
    )
    analysis = {
        "schema_version": 1,
        "bootstrap_draws": args.bootstrap_draws,
        "generator_label": args.generator_label,
        "inputs": {
            "aider_generation": _input_provenance(
                args.aider_generation,
                records=len(aider_generation),
            ),
            "aider_attribution": _input_provenance(
                args.aider_attribution,
                records=len(aider_attribution),
            ),
            "agentdojo_generation": _input_provenance(
                args.agentdojo_generation,
                records=len(agentdojo_generation),
            ),
            "agentdojo_attribution": _input_provenance(
                args.agentdojo_attribution,
                records=len(agentdojo_attribution),
            ),
        },
        "aider": {
            "generation_audit": _aider_generation_audit(aider_generation),
            "attribution": _aider_attribution(
                aider_attribution,
                draws=args.bootstrap_draws,
            ),
        },
        "agentdojo": {
            "outcomes": _agentdojo_outcomes(agentdojo_generation),
            "localization": _agentdojo_localization(
                agentdojo_attribution,
                eligible_count=eligible_count,
                attacked_inventory=args.agentdojo_attacked_inventory,
                draws=args.bootstrap_draws,
            ),
            "localization_generation_pool": {
                "path": str(
                    (
                        args.agentdojo_localization_generation
                        or args.agentdojo_generation
                    ).resolve()
                ),
                "record_count": len(agentdojo_localization_generation),
            },
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "REPORT.md").write_text(
        _render_markdown(analysis),
        encoding="utf-8",
    )
    args.paper_generated_dir.mkdir(parents=True, exist_ok=True)
    for name, content in _render_tex(
        analysis,
        generator_label=args.generator_label,
    ).items():
        (args.paper_generated_dir / name).write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
