"""Audit completeness and protocol compliance of the formal visual evaluation.

The command is intentionally read-only.  It distinguishes incomplete artifacts
from protocol violations, writes a machine-readable report, and exits non-zero
until every E1--E5/A1--A8 deliverable required by protocol v2 is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .freeze_formal_input_hashes import canonical_record_sha256
from .freeze_formal_response_hashes import token_ids_sha256
from .formal_manual_audit import (
    IMAGE_DEPENDENCE_LABELS,
    THINKING_QUALITY_LABELS,
    prepare_audit as prepare_protocol_audit,
)
from .render_formal_results import (
    LATEX_LABELS,
    METHODS as RENDER_METHODS,
    _latex_cell,
    _visual_discussion_tex,
    render as render_formal_results,
)
from .select_strict_subset import (
    gate_funnel,
    select as replay_strict_selection,
)
from .strict_generation import normalized_output, read_jsonl, validate_model_record
from .strict_visual_faithfulness import CURVE_NORMALIZATION_POLICY
from .vizwiz_semantic_judgments import audit_sample_ids
from .vizwiz_semantic_judgments import (
    prepare_human_review as prepare_semantic_human_review,
    prepare_tasks as prepare_semantic_tasks,
)


EXPECTED_REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
EXPECTED_METHODS = {
    "random",
    "center",
    "visual-loo",
    "visual-ig",
    "attnlrp",
    "flashtrace",
    "ifr-span",
    "flashtrace-all-gen",
}
LEARNED_GRID_METHODS = {
    "visual-ig",
    "attnlrp",
    "flashtrace",
    "ifr-span",
    "flashtrace-all-gen",
}
EXPECTED_STRATA = {
    "first_page_passage": 40,
    "later_page_passage": 40,
    "non_passage": 40,
}
EXPECTED_LOCALIZATION_METRICS = {
    "energy_in_mask",
    "evidence_rank_auc",
    "pointing_game",
    "recovery_at_1pct",
    "recovery_at_5pct",
    "recovery_at_10pct",
    "recovery_at_20pct",
    "top_evidence_iou",
}


def _ids(records: Iterable[Mapping[str, Any]]) -> list[str]:
    return [str(record["sample_id"]) for record in records]


def _duplicates(values: Iterable[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _recorded_path_matches(value: Any, *, root: Path, expected: Path) -> bool:
    if not str(value or "").strip():
        return False
    recorded = Path(str(value))
    if not recorded.is_absolute():
        recorded = root / recorded
    return recorded.resolve() == expected.resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _funnel_conservation_issues(payload: Mapping[str, Any]) -> list[str]:
    """Return population-accounting violations in a serialized gate funnel."""

    issues: list[str] = []
    stages = payload.get("stages")
    if not isinstance(stages, list) or not stages:
        return ["stages must be a non-empty list"]
    if not all(isinstance(stage, Mapping) for stage in stages):
        return ["every stage must be an object"]

    def count(value: Any, label: str) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            issues.append(f"{label} must be a non-negative integer")
            return None
        return value

    candidate_count = count(payload.get("candidate_count"), "candidate_count")
    model_count = count(payload.get("model_record_count"), "model_record_count")
    evaluation_count = count(
        payload.get("evaluation_record_count"), "evaluation_record_count"
    )
    strict_count = count(
        payload.get("strict_eligible_count"), "strict_eligible_count"
    )
    frozen_count = count(payload.get("frozen_sample_count"), "frozen_sample_count")
    excluded_count = count(
        payload.get("excluded_prior_pilot_count", 0),
        "excluded_prior_pilot_count",
    )

    names = [str(stage.get("stage") or "") for stage in stages]
    if any(not name for name in names):
        issues.append("every stage must have a non-empty name")
    if len(set(names)) != len(names):
        issues.append("stage names must be unique")

    stage_counts: list[tuple[int | None, int | None]] = []
    for index, stage in enumerate(stages):
        stage_counts.append(
            (
                count(stage.get("passed"), f"stages[{index}].passed"),
                count(
                    stage.get("eliminated_at_stage"),
                    f"stages[{index}].eliminated_at_stage",
                ),
            )
        )

    first_passed, first_eliminated = stage_counts[0]
    if names[0] != "candidate_manifest":
        issues.append("the first stage must be candidate_manifest")
    if candidate_count is not None and first_passed != candidate_count:
        issues.append("candidate_manifest passed must equal candidate_count")
    if first_eliminated != 0:
        issues.append("candidate_manifest must eliminate zero records")

    for index in range(1, len(stages)):
        previous_passed = stage_counts[index - 1][0]
        passed, eliminated = stage_counts[index]
        if (
            previous_passed is not None
            and passed is not None
            and eliminated is not None
            and previous_passed != passed + eliminated
        ):
            issues.append(
                f"population is not conserved between {names[index - 1]} "
                f"and {names[index]}"
            )

    stage_by_name = {
        name: stage_counts[index] for index, name in enumerate(names) if name
    }
    prior_counts = stage_by_name.get("prior_pilot_sample_exclusion")
    if prior_counts is None:
        issues.append("prior_pilot_sample_exclusion stage is absent")
    elif excluded_count is not None and prior_counts[1] != excluded_count:
        issues.append(
            "prior-pilot stage elimination must equal excluded_prior_pilot_count"
        )
    strict_stage = stage_by_name.get("final_strict_eligible")
    if strict_stage is None:
        issues.append("final_strict_eligible stage is absent")
    elif strict_count is not None and strict_stage[0] != strict_count:
        issues.append(
            "final_strict_eligible passed must equal strict_eligible_count"
        )

    if names[-1] != "unique_image_and_fixed_seed_freeze":
        issues.append(
            "the last stage must be unique_image_and_fixed_seed_freeze"
        )
    if frozen_count is not None and stage_counts[-1][0] != frozen_count:
        issues.append("last-stage passed must equal frozen_sample_count")
    if (
        candidate_count is not None
        and evaluation_count is not None
        and evaluation_count != candidate_count
    ):
        issues.append("evaluation_record_count must equal candidate_count")
    if (
        candidate_count is not None
        and model_count is not None
        and model_count > candidate_count
    ):
        issues.append("model_record_count cannot exceed candidate_count")
    return issues


def _frozen_protocol_metadata_issues(
    frozen: Mapping[str, Any],
) -> list[str]:
    """Validate the deterministic protocol metadata stored with frozen IDs."""

    issues: list[str] = []
    if frozen.get("schema_version") != 1:
        issues.append("schema_version must be 1")
    if frozen.get("frozen_on") != "2026-07-24":
        issues.append("frozen_on must match the preregistered freeze date")
    if frozen.get("selection_seed") != 17:
        issues.append("selection_seed must be 17")
    datasets = frozen.get("datasets")
    if not isinstance(datasets, Mapping):
        return issues + ["datasets must be an object"]
    if set(datasets) != {"wiki_visa", "vizwiz_lf"}:
        issues.append("datasets must contain exactly wiki_visa and vizwiz_lf")

    expected_bundles = {
        "wiki_visa": {
            "count": 120,
            "balance_key": "stratum",
            "selection_mode": "balanced_fixed_seed",
        },
        "vizwiz_lf": {
            "count": 100,
            "balance_key": None,
            "selection_mode": "unstratified_fixed_seed",
        },
    }
    for benchmark, expected in expected_bundles.items():
        bundle = datasets.get(benchmark)
        if not isinstance(bundle, Mapping):
            issues.append(f"{benchmark} bundle is absent")
            continue
        for field, value in expected.items():
            if bundle.get(field) != value:
                issues.append(f"{benchmark}.{field} must equal {value!r}")
        samples = bundle.get("samples")
        if not isinstance(samples, list) or not all(
            isinstance(sample, Mapping) for sample in samples
        ):
            issues.append(f"{benchmark}.samples must be a list of objects")
            continue
        expected_count = int(expected["count"])
        sample_ids = [str(sample.get("sample_id") or "") for sample in samples]
        if len(samples) != expected_count:
            issues.append(f"{benchmark}.samples must contain {expected_count} rows")
        if any(not sample_id for sample_id in sample_ids):
            issues.append(f"{benchmark}.samples contains an empty sample_id")
        if len(set(sample_ids)) != len(sample_ids):
            issues.append(f"{benchmark}.samples contains duplicate sample IDs")
        if sample_ids != sorted(sample_ids):
            issues.append(f"{benchmark}.samples must be sorted by sample_id")

        lengths = [sample.get("output_tokens") for sample in samples]
        if not all(
            isinstance(length, int)
            and not isinstance(length, bool)
            and length > 0
            for length in lengths
        ):
            issues.append(f"{benchmark}.output_tokens must be positive integers")
            continue
        sorted_lengths = sorted(int(length) for length in lengths)
        lower = sorted_lengths[(len(sorted_lengths) - 1) // 3]
        upper = sorted_lengths[(2 * (len(sorted_lengths) - 1)) // 3]
        if bundle.get("output_token_tercile_cutpoints") != [lower, upper]:
            issues.append(
                f"{benchmark}.output_token_tercile_cutpoints cannot be reproduced"
            )

        if benchmark == "wiki_visa":
            balance_counts = Counter(
                str(sample.get("balance_group")) for sample in samples
            )
            if balance_counts != Counter(EXPECTED_STRATA):
                issues.append("wiki_visa balance groups must be exactly 40/40/40")
        else:
            if any(sample.get("balance_group") is not None for sample in samples):
                issues.append("vizwiz_lf balance_group must be null")
            if any(
                not str(sample.get("question_type") or "").strip()
                for sample in samples
            ):
                issues.append("vizwiz_lf question_type must be populated")
            expected_labels = [
                (
                    "short"
                    if int(length) <= lower
                    else "medium"
                    if int(length) <= upper
                    else "long"
                )
                for length in lengths
            ]
            actual_labels = [
                sample.get("output_length_tercile") for sample in samples
            ]
            if actual_labels != expected_labels:
                issues.append(
                    "vizwiz_lf output_length_tercile cannot be reproduced"
                )
    return issues


def _frozen_record_protocol_issues(
    datasets: list[Mapping[str, Any]],
    models: list[Mapping[str, Any]],
    evaluations: list[Mapping[str, Any]],
    *,
    benchmark: str,
) -> list[str]:
    """Revalidate frozen dataset/model/evaluation joins and all hard gates."""

    issues: list[str] = []
    collections = {
        "dataset": datasets,
        "model": models,
        "evaluation": evaluations,
    }
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    for label, records in collections.items():
        sample_ids = [str(record.get("sample_id") or "") for record in records]
        if any(not sample_id for sample_id in sample_ids):
            issues.append(f"{label} records contain an empty sample_id")
        if len(set(sample_ids)) != len(sample_ids):
            issues.append(f"{label} records contain duplicate sample IDs")
        indexed[label] = {
            str(record.get("sample_id")): record for record in records
        }
    id_sets = {label: set(records) for label, records in indexed.items()}
    if len({frozenset(values) for values in id_sets.values()}) != 1:
        issues.append("dataset/model/evaluation sample IDs do not match")
        return issues

    expected_gates = {
        "generation_stable",
        "positive_blur_logprob_drop",
        "generated_teacher_forced_ids_match",
        "thinking_closed",
        "generation_ablation_changes_output",
    }
    if benchmark == "wiki_visa":
        expected_gates.add("whole_output_correct")
        expected_profile = "concise"
        max_new_tokens = 1024
    elif benchmark == "vizwiz_lf":
        expected_gates.update(
            {
                "output_non_refusal",
                "thinking_within_token_limit",
                "output_meets_min_tokens",
            }
        )
        expected_profile = "long_form"
        max_new_tokens = 2048
    else:
        return [f"unsupported formal benchmark: {benchmark}"]

    invalid: dict[str, list[str]] = {}

    def reject(sample_id: str, reason: str) -> None:
        invalid.setdefault(sample_id, []).append(reason)

    def finite_number(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )

    for sample_id in sorted(id_sets["dataset"]):
        dataset = indexed["dataset"][sample_id]
        model = indexed["model"][sample_id]
        evaluation = indexed["evaluation"][sample_id]
        dataset_input = dataset.get("input")
        if not isinstance(dataset_input, Mapping) or set(dataset_input) != {
            "I_IMAGE",
            "I_QUESTION",
        }:
            reject(sample_id, "dataset input fields are not exactly image+question")
        elif (
            model.get("I_IMAGE") != dataset_input.get("I_IMAGE")
            or model.get("I_QUESTION") != dataset_input.get("I_QUESTION")
        ):
            reject(sample_id, "model input does not match dataset input")

        dataset_evaluation = dataset.get("evaluation")
        if not isinstance(dataset_evaluation, Mapping) or evaluation.get(
            "REFERENCE_OUTPUT"
        ) != dataset_evaluation.get("REFERENCE_OUTPUT"):
            reject(sample_id, "evaluation reference does not match dataset reference")

        try:
            validate_model_record(model)
        except (KeyError, TypeError, ValueError) as error:
            reject(sample_id, f"invalid model record: {error}")
        generation_metadata = model.get("generation_metadata")
        if not isinstance(generation_metadata, Mapping):
            reject(sample_id, "generation_metadata is absent")
            generation_metadata = {}
        generated_ids = generation_metadata.get("original_generated_token_ids")
        teacher_forced_ids = generation_metadata.get("teacher_forced_token_ids")
        if (
            not isinstance(generated_ids, list)
            or not generated_ids
            or generated_ids != teacher_forced_ids
        ):
            reject(sample_id, "generated and teacher-forced token IDs differ")
        generated_tokens = generation_metadata.get(
            "original_generated_tokens_without_eos"
        )
        if (
            not isinstance(generated_tokens, int)
            or isinstance(generated_tokens, bool)
            or generated_tokens <= 0
            or generated_tokens > max_new_tokens
        ):
            reject(sample_id, "generated token count violates the frozen budget")

        model_metadata = model.get("model")
        generation = (
            model_metadata.get("generation")
            if isinstance(model_metadata, Mapping)
            else None
        )
        if (
            not isinstance(model_metadata, Mapping)
            or model_metadata.get("repo_id")
            != "Qwen/Qwen3-VL-8B-Thinking"
            or model_metadata.get("requested_revision") != EXPECTED_REVISION
            or model_metadata.get("resolved_revision") != EXPECTED_REVISION
            or not isinstance(generation, Mapping)
            or generation.get("do_sample") is not False
            or generation.get("max_new_tokens") != max_new_tokens
            or generation.get("prompt_profile") != expected_profile
            or generation_metadata.get("prompt_profile") != expected_profile
        ):
            reject(sample_id, "model generation configuration is not frozen")

        gates = evaluation.get("gates")
        if (
            not isinstance(gates, Mapping)
            or set(gates) != expected_gates
            or not all(gates.get(gate) is True for gate in expected_gates)
        ):
            reject(sample_id, "hard-gate set is not exact and all true")
        if (
            evaluation.get("strict_eligible") is not True
            or evaluation.get("pre_ablation_eligible") is not True
            or evaluation.get("generation_stable") is not True
            or evaluation.get("generated_teacher_forced_ids_match") is not True
            or evaluation.get("image_dependent") is not True
            or evaluation.get("image_dependent_by_generation_ablation") is not True
            or evaluation.get("stability_repeats") != 2
        ):
            reject(sample_id, "derived gate fields are inconsistent")

        original_score = evaluation.get("original_output_mean_logprob")
        blurred_score = evaluation.get("blurred_output_mean_logprob")
        delta = evaluation.get("image_dependence_delta")
        if not (
            finite_number(original_score)
            and finite_number(blurred_score)
            and finite_number(delta)
            and float(original_score) > float(blurred_score)
            and math.isclose(
                float(delta),
                float(original_score) - float(blurred_score),
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
        ):
            reject(sample_id, "blur log-probability drop is invalid")

        ablations = evaluation.get("ablation_outputs")
        if not isinstance(ablations, Mapping) or set(ablations) != {
            "global_blur",
            "uniform_gray",
        }:
            reject(sample_id, "ablation outputs are incomplete")
        elif not any(
            isinstance(ablations.get(name), Mapping)
            and ablations[name].get("status") in {"ok", "parse_error"}
            and ablations[name].get("same_as_original_output") is False
            for name in ("global_blur", "uniform_gray")
        ):
            reject(sample_id, "neither image ablation changes the frozen output")

        if benchmark == "wiki_visa":
            if (
                evaluation.get("correctness_gate_required") is not True
                or evaluation.get("output_correct") is not True
                or evaluation.get("reference_exact_match") is not True
            ):
                reject(sample_id, "Wiki correctness hard gate is not satisfied")
        else:
            output_tokens = generation_metadata.get("output_tokens")
            if (
                evaluation.get("correctness_gate_required") is not False
                or evaluation.get("output_correct") is not None
                or not isinstance(output_tokens, int)
                or isinstance(output_tokens, bool)
                or output_tokens < 16
            ):
                reject(sample_id, "VizWiz correctness/minimum-output policy is invalid")

    if invalid:
        issues.append(
            f"{len(invalid)} frozen rows violate record/gate protocol: "
            + ", ".join(
                f"{sample_id} ({'; '.join(reasons)})"
                for sample_id, reasons in list(invalid.items())[:10]
            )
        )
    return issues


def _ablation_provenance_issues(
    models: list[Mapping[str, Any]],
    evaluations: list[Mapping[str, Any]],
    *,
    paths: Iterable[Path],
    benchmark: str,
) -> list[str]:
    """Recompute frozen ablation comparisons from the separated raw records."""

    issues: list[str] = []
    model_by_id = {str(record["sample_id"]): record for record in models}
    evaluation_by_id = {
        str(record["sample_id"]): record for record in evaluations
    }
    source_rows = [
        record
        for path in paths
        if path.is_file()
        for record in read_jsonl(path)
        if record.get("status") == "complete"
    ]
    source_ids = [str(record.get("sample_id") or "") for record in source_rows]
    duplicates = _duplicates(source_ids)
    if duplicates:
        issues.append(f"duplicate complete ablation rows: {duplicates[:10]}")
    source_by_id = {
        str(record.get("sample_id")): record for record in source_rows
    }
    missing = sorted(set(model_by_id) - set(source_by_id))
    if missing:
        issues.append(f"frozen samples missing raw ablation records: {missing[:10]}")

    max_new_tokens = 1024 if benchmark == "wiki_visa" else 2048
    invalid: dict[str, list[str]] = {}

    def reject(sample_id: str, reason: str) -> None:
        invalid.setdefault(sample_id, []).append(reason)

    for sample_id in sorted(set(model_by_id) & set(source_by_id)):
        model = model_by_id[sample_id]
        evaluation = evaluation_by_id.get(sample_id) or {}
        source = source_by_id[sample_id]
        source_model = source.get("model")
        if (
            source.get("benchmark") != benchmark
            or source.get("I_QUESTION") != model.get("I_QUESTION")
            or not isinstance(source_model, Mapping)
            or source_model.get("repo_id") != "Qwen/Qwen3-VL-8B-Thinking"
            or source_model.get("revision") != EXPECTED_REVISION
            or source_model.get("do_sample") is not False
            or source_model.get("max_new_tokens") != max_new_tokens
        ):
            reject(sample_id, "raw ablation generation configuration is invalid")
        ablations = source.get("ablations")
        if not isinstance(ablations, Mapping) or set(ablations) != {
            "global_blur",
            "uniform_gray",
        }:
            reject(sample_id, "raw ablation pair is incomplete")
            continue

        expected_comparisons: dict[str, dict[str, Any]] = {}
        for name in ("global_blur", "uniform_gray"):
            generated = ablations[name]
            if not isinstance(generated, Mapping):
                reject(sample_id, f"{name} raw ablation is not an object")
                continue
            status = generated.get("status")
            raw_response = generated.get("raw_response")
            token_ids = generated.get("generated_token_ids")
            output = generated.get("OUTPUT")
            if (
                status not in {"ok", "parse_error"}
                or not str(raw_response or "").strip()
                or not isinstance(token_ids, list)
                or not token_ids
                or (status == "ok" and not str(output or "").strip())
                or (status == "parse_error" and output is not None)
            ):
                reject(sample_id, f"{name} raw ablation record is invalid")
                continue
            normalized_ablation = (
                normalized_output(str(output)) if output is not None else None
            )
            expected_comparisons[name] = {
                "status": status,
                "normalized_output": normalized_ablation,
                "same_as_original_output": (
                    output is not None
                    and normalized_ablation
                    == normalized_output(str(model.get("OUTPUT") or ""))
                ),
            }
        if (
            len(expected_comparisons) == 2
            and evaluation.get("ablation_outputs") != expected_comparisons
        ):
            reject(
                sample_id,
                "evaluation ablation comparison cannot be reproduced",
            )

    if invalid:
        issues.append(
            f"{len(invalid)} frozen ablation rows violate provenance: "
            + ", ".join(
                f"{sample_id} ({'; '.join(reasons)})"
                for sample_id, reasons in list(invalid.items())[:10]
            )
        )
    return issues


def _protocol_v2_issues(protocol: Mapping[str, Any]) -> list[str]:
    """Validate every frozen top-level decision that defines protocol v2."""

    issues: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            issues.append(message)

    require(protocol.get("schema_version") == 2, "schema_version must be 2")
    require(
        protocol.get("frozen_on") == "2026-07-24",
        "frozen_on must match the plan date",
    )
    require(
        protocol.get("status") == "main_experiment_scope_frozen",
        "protocol status must be frozen",
    )
    require(
        "must not be changed" in str(protocol.get("amendment_policy") or ""),
        "amendment policy must forbid outcome-responsive changes",
    )

    model = protocol.get("model")
    require(isinstance(model, Mapping), "model configuration is absent")
    if isinstance(model, Mapping):
        require(
            model.get("repo_id") == "Qwen/Qwen3-VL-8B-Thinking"
            and model.get("revision") == EXPECTED_REVISION,
            "model identity/revision changed",
        )
        require(
            model.get("wiki_visa_max_pixels") == 2_007_040,
            "Wiki max_pixels changed",
        )
        require(
            model.get("generation_budgets")
            == {
                "wiki_visa_max_new_tokens": 1024,
                "vizwiz_lf_max_new_tokens": 2048,
            },
            "generation budgets changed",
        )
        require(
            model.get("input_fields") == ["I_IMAGE", "I_QUESTION"]
            and model.get("generated_fields") == ["THINKING", "OUTPUT"]
            and model.get("attribution_sink") == "complete_OUTPUT_SPAN"
            and model.get("teacher_force_same_frozen_response_across_methods")
            is True,
            "model input/output or attribution-sink policy changed",
        )

    separation = protocol.get("artifact_separation")
    require(
        isinstance(separation, Mapping)
        and separation.get("join_key") == "sample_id"
        and set(separation.get("files") or [])
        == {
            "dataset.jsonl",
            "model.jsonl",
            "generation_eval.jsonl",
            "ablation.model.jsonl",
            "attribution_records.jsonl",
        }
        and separation.get("dataset_rationales_or_programs_enter_prompt")
        is False,
        "artifact separation or prompt-leakage policy changed",
    )
    exclusion = protocol.get("formal_sample_exclusion")
    require(
        isinstance(exclusion, Mapping)
        and exclusion.get("reported_in_gate_funnel") is True
        and "excluded before formal subset freezing"
        in str(exclusion.get("policy") or ""),
        "pilot exclusion policy changed",
    )

    benchmarks = {
        str(bundle.get("id")): bundle
        for bundle in protocol.get("benchmarks") or []
        if isinstance(bundle, Mapping)
    }
    require(
        set(benchmarks)
        == {"wiki_visa", "vizwiz_lf", "vistaqa", "clevr_xai_complex"},
        "benchmark registry changed",
    )
    wiki = benchmarks.get("wiki_visa") or {}
    require(
        wiki.get("tier") == "primary"
        and wiki.get("role") == "localization"
        and wiki.get("candidate_count_initial") == 240
        and wiki.get("final_count") == 120
        and wiki.get("strata") == EXPECTED_STRATA
        and wiki.get("ground_truth") == "native_supporting_html_element_boxes"
        and set(wiki.get("hard_gates") or [])
        == {
            "whole_output_correct",
            "two_identical_greedy_generations",
            "generated_teacher_forced_token_identity",
            "positive_output_logprob_drop_under_global_blur",
            "blur_or_gray_generation_changes_whole_output",
            "thinking_closes_within_generation_budget",
        }
        and set(wiki.get("primary_endpoints") or [])
        == {"energy_in_evidence", "recovery_at_5pct"},
        "Wiki role, sample policy, gates, or primary endpoints changed",
    )
    require(
        "never relax a gate"
        in str(wiki.get("candidate_extension_policy") or ""),
        "Wiki low-yield extension policy changed",
    )
    viz = benchmarks.get("vizwiz_lf") or {}
    correctness = viz.get("correctness_policy") or {}
    require(
        viz.get("tier") == "primary"
        and viz.get("role") == "frozen_response_faithfulness"
        and viz.get("candidate_count") == 200
        and viz.get("final_count") == 100
        and viz.get("selection")
        == "fixed_seed_sample_from_full_strict_eligible_pool"
        and set(viz.get("recorded_strata") or [])
        == {"output_length_tercile", "question_type"}
        and set(viz.get("hard_gates") or [])
        == {
            "output_is_not_refusal_or_unanswerable",
            "two_identical_greedy_generations",
            "generated_teacher_forced_token_identity",
            "positive_output_logprob_drop_under_global_blur",
            "blur_or_gray_generation_changes_whole_output",
            "thinking_closes_within_2048_tokens",
            "output_has_at_least_16_tokens",
        }
        and viz.get("primary_endpoints") == ["deletion_auc"]
        and isinstance(correctness, Mapping)
        and correctness.get("hard_gate") is False
        and set(correctness.get("labels") or [])
        == {"fully", "partial", "wrong"}
        and correctness.get("fully_correct_subset_sensitivity") is True,
        "VizWiz role, sample policy, gates, or primary endpoint changed",
    )
    require(
        (benchmarks.get("vistaqa") or {}).get("new_gpu_run") is False
        and (benchmarks.get("vistaqa") or {}).get("sample_count") == 10
        and (benchmarks.get("clevr_xai_complex") or {}).get("new_gpu_run")
        is False
        and (benchmarks.get("clevr_xai_complex") or {}).get("sample_count")
        == 20,
        "legacy diagnostic scope changed",
    )

    methods = {
        str(method.get("id")): method
        for method in protocol.get("methods") or []
        if isinstance(method, Mapping)
    }
    require(set(methods) == EXPECTED_METHODS, "frozen method registry changed")
    require(
        (methods.get("flashtrace") or {}).get("role") == "primary"
        and (methods.get("flashtrace") or {}).get("recursive_hops") == 1
        and (methods.get("ifr-span") or {}).get("role") == "k0_ablation"
        and (methods.get("flashtrace-all-gen") or {}).get("role")
        == "bridge_ablation",
        "FlashTrace/ablation method roles changed",
    )
    spatial = protocol.get("shared_spatial_protocol")
    require(
        spatial
        == {
            "complete_visual_patches": True,
            "cutoff_ties": "expected_credit_under_uniform_tie_break",
            "metric_interpolation": "none",
            "display_interpolation": "nearest_neighbor_only",
        },
        "whole-patch spatial protocol changed",
    )
    faithfulness = protocol.get("faithfulness_protocol")
    require(
        isinstance(faithfulness, Mapping)
        and faithfulness.get("frozen_tokens") == "complete_THINKING_plus_OUTPUT"
        and faithfulness.get("scored_tokens") == "OUTPUT_SPAN_only"
        and faithfulness.get("region_budget") == 64
        and faithfulness.get("curve_steps") == 10
        and faithfulness.get("replacement") == "gaussian_blur"
        and faithfulness.get("deletion_insertion_order") == "signed_score"
        and faithfulness.get("visual_mas_density") == "positive_mass"
        and faithfulness.get("positive_only_ordering_sensitivity") is True
        and faithfulness.get("save_all_curves") is True
        and faithfulness.get("report_degenerate_curve_count") is True,
        "faithfulness protocol changed",
    )
    statistics = protocol.get("statistics")
    require(
        isinstance(statistics, Mapping)
        and statistics.get("comparison_subset")
        == "intersection_of_successful_sample_ids_across_requested_methods"
        and statistics.get("bootstrap_unit") == "paired_sample"
        and statistics.get("bootstrap_resamples_minimum") == 10_000
        and statistics.get("planned_bootstrap_resamples") == 50_000
        and statistics.get("confidence_interval") == 0.95
        and set(statistics.get("report") or [])
        == {"paired_difference", "wins_ties_losses", "gate_funnel"},
        "paired-bootstrap protocol changed",
    )

    experiments = {
        str(experiment.get("id")): (
            experiment.get("dataset"),
            experiment.get("task"),
        )
        for experiment in protocol.get("formal_experiments") or []
        if isinstance(experiment, Mapping)
    }
    require(
        experiments
        == {
            "E1": ("wiki_visa", "candidate_generation_and_ablation_audit"),
            "E2": ("vizwiz_lf", "candidate_generation_and_ablation_audit"),
            "E3": ("wiki_visa", "localization"),
            "E4": ("vizwiz_lf", "frozen_response_faithfulness"),
            "E5": ("wiki_visa", "frozen_response_faithfulness_appendix"),
        },
        "E1–E5 experiment registry changed",
    )
    require(
        protocol.get("frozen_id_artifact")
        == "results/strict/formal/frozen_ids.json",
        "frozen ID artifact path changed",
    )
    require(
        set(protocol.get("excluded_new_runs") or [])
        == {
            "k2_recursion",
            "rollout_grad_attention_tam_appendix",
            "position_equivariance",
            "visual_efficiency_scaling_grid",
            "vistaqa_expansion",
            "second_vlm",
        },
        "explicitly excluded experiment scope changed",
    )
    return issues


def _selection_replay_issues(
    *,
    formal_dir: Path,
    prefix: str,
    dataset_paths: list[Path],
    model_paths: list[Path],
    evaluation_paths: list[Path],
    exclusion_paths: list[Path],
    sample_size: int,
    balance_key: str | None,
    funnel_path: Path,
) -> tuple[list[str], list[str]]:
    """Replay fixed-seed selection and funnel construction from candidates."""

    required_paths = [
        *dataset_paths,
        *model_paths,
        *evaluation_paths,
        *exclusion_paths,
        funnel_path,
        formal_dir / f"{prefix}.dataset.jsonl",
        formal_dir / f"{prefix}.model.jsonl",
        formal_dir / f"{prefix}.generation_eval.jsonl",
    ]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        return [], missing

    excluded_ids = {
        str(record["sample_id"])
        for path in exclusion_paths
        for record in read_jsonl(path)
    }
    issues: list[str] = []
    try:
        selected = replay_strict_selection(
            dataset_paths,
            model_paths,
            evaluation_paths,
            sample_size=sample_size,
            balance_key=balance_key,
            seed=17,
            exclude_sample_ids=excluded_ids,
        )
    except (KeyError, TypeError, ValueError) as error:
        return [f"selection replay failed: {error}"], []

    frozen_records = (
        read_jsonl(formal_dir / f"{prefix}.dataset.jsonl"),
        read_jsonl(formal_dir / f"{prefix}.model.jsonl"),
        read_jsonl(formal_dir / f"{prefix}.generation_eval.jsonl"),
    )
    labels = ("dataset", "model", "generation evaluation")
    for label, replayed, frozen in zip(
        labels, selected, frozen_records, strict=True
    ):
        if replayed != frozen:
            issues.append(f"replayed {label} does not exactly match frozen artifact")

    datasets = {
        str(record["sample_id"]): record
        for path in dataset_paths
        for record in read_jsonl(path)
    }
    models = {
        str(record["sample_id"]): record
        for path in model_paths
        for record in read_jsonl(path)
    }
    evaluations = {
        str(record["sample_id"]): record
        for path in evaluation_paths
        for record in read_jsonl(path)
    }
    replayed_funnel = gate_funnel(
        datasets,
        models,
        evaluations,
        exclude_sample_ids=excluded_ids,
        frozen_sample_ids={
            str(record["sample_id"]) for record in selected[0]
        },
    )
    if replayed_funnel != _read_json(funnel_path):
        issues.append("replayed gate funnel does not exactly match stored artifact")
    return issues, []


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        passed: bool,
        *,
        status: str = "error",
        detail: Any = None,
    ) -> None:
        self.checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "failure_kind": None if passed else status,
                "detail": detail,
            }
        )

    def require_file(self, name: str, path: Path) -> bool:
        exists = path.is_file()
        self.add(name, exists, status="incomplete", detail=str(path))
        return exists

    def report(self) -> dict[str, Any]:
        errors = [check for check in self.checks if check["failure_kind"] == "error"]
        incomplete = [
            check for check in self.checks if check["failure_kind"] == "incomplete"
        ]
        return {
            "schema_version": 1,
            "complete": not errors and not incomplete,
            "protocol_violations": len(errors),
            "incomplete_checks": len(incomplete),
            "passed_checks": sum(check["passed"] for check in self.checks),
            "total_checks": len(self.checks),
            "checks": self.checks,
        }


def _resolved_revisions(models: Iterable[Mapping[str, Any]]) -> set[str]:
    revisions = set()
    for record in models:
        model = record.get("model") or {}
        revision = model.get("resolved_revision", model.get("revision"))
        if revision is not None:
            revisions.add(str(revision))
    return revisions


def _audit_frozen_input_hashes(
    audit: Audit,
    *,
    root: Path,
    formal_dir: Path,
    expected: Mapping[str, set[str]],
) -> None:
    hashes_path = formal_dir / "frozen_input_hashes.json"
    if not audit.require_file("frozen input hashes: artifact", hashes_path):
        return
    payload = _read_json(hashes_path)
    bundles = {
        str(bundle.get("manifest_path")): bundle
        for bundle in payload.get("manifests") or []
    }
    audit.add(
        "frozen input hashes: schema and exact formal manifests",
        payload.get("schema_version") == 1
        and payload.get("hash_algorithm") == "sha256"
        and set(bundles) == set(expected),
        detail={"manifests": sorted(bundles)},
    )
    for relative_manifest, expected_ids in expected.items():
        manifest_path = root / relative_manifest
        bundle = bundles.get(relative_manifest) or {}
        records = read_jsonl(manifest_path) if manifest_path.is_file() else []
        records_by_id = {
            str(record["sample_id"]): record for record in records
        }
        samples = bundle.get("samples") or []
        sample_ids = [str(sample.get("sample_id")) for sample in samples]
        samples_by_id = {
            str(sample.get("sample_id")): sample for sample in samples
        }
        joins_match = (
            len(records_by_id) == len(records)
            and len(samples_by_id) == len(samples)
            and set(records_by_id) == set(sample_ids) == expected_ids
            and int(bundle.get("sample_count", -1)) == len(expected_ids)
        )
        hashes_match = (
            manifest_path.is_file()
            and bundle.get("manifest_sha256") == _sha256(manifest_path)
        )
        mismatches: list[str] = []
        if joins_match and hashes_match:
            for sample_id in sorted(expected_ids):
                record = records_by_id[sample_id]
                sample = samples_by_id[sample_id]
                input_record = record.get("input") or {}
                image_value = str(input_record.get("I_IMAGE") or "")
                image_path = Path(image_value)
                resolved_image = (
                    (root / image_path).resolve()
                    if image_value and not image_path.is_absolute()
                    else image_path
                )
                valid_image = (
                    bool(image_value)
                    and not image_path.is_absolute()
                    and resolved_image.is_relative_to(root)
                    and resolved_image.is_file()
                )
                question_hash = hashlib.sha256(
                    str(input_record.get("I_QUESTION") or "").encode("utf-8")
                ).hexdigest()
                if (
                    sample.get("image_path") != image_value
                    or not valid_image
                    or sample.get("image_sha256") != _sha256(resolved_image)
                    or sample.get("question_sha256") != question_hash
                    or sample.get("dataset_record_sha256")
                    != canonical_record_sha256(record)
                ):
                    mismatches.append(sample_id)
        audit.add(
            f"frozen input hashes: {Path(relative_manifest).stem}",
            joins_match and hashes_match and not mismatches,
            detail={
                "expected_samples": len(expected_ids),
                "hashed_samples": len(samples),
                "mismatches": mismatches,
            },
        )


def _audit_frozen_response_hashes(
    audit: Audit,
    *,
    root: Path,
    formal_dir: Path,
    expected: Mapping[str, set[str]],
) -> None:
    hashes_path = formal_dir / "frozen_response_hashes.json"
    if not audit.require_file("frozen response hashes: artifact", hashes_path):
        return
    payload = _read_json(hashes_path)
    bundles = {
        str(bundle.get("model_output_path")): bundle
        for bundle in payload.get("model_outputs") or []
    }
    audit.add(
        "frozen response hashes: schema and exact formal model outputs",
        payload.get("schema_version") == 1
        and payload.get("hash_algorithm") == "sha256"
        and set(bundles) == set(expected),
        detail={"model_outputs": sorted(bundles)},
    )
    for relative_output, expected_ids in expected.items():
        model_path = root / relative_output
        bundle = bundles.get(relative_output) or {}
        records = read_jsonl(model_path) if model_path.is_file() else []
        records_by_id = {
            str(record["sample_id"]): record for record in records
        }
        samples = bundle.get("samples") or []
        sample_ids = [str(sample.get("sample_id")) for sample in samples]
        samples_by_id = {
            str(sample.get("sample_id")): sample for sample in samples
        }
        joins_match = (
            len(records_by_id) == len(records)
            and len(samples_by_id) == len(samples)
            and sample_ids == sorted(sample_ids)
            and set(records_by_id) == set(sample_ids) == expected_ids
            and int(bundle.get("sample_count", -1)) == len(expected_ids)
            and bundle.get("resolved_revisions") == [EXPECTED_REVISION]
        )
        file_hash_matches = (
            model_path.is_file()
            and bundle.get("model_output_sha256") == _sha256(model_path)
        )
        mismatches: list[str] = []
        if joins_match and file_hash_matches:
            for sample_id in sorted(expected_ids):
                record = records_by_id[sample_id]
                sample = samples_by_id[sample_id]
                metadata = record.get("generation_metadata") or {}
                generated_ids = metadata.get("original_generated_token_ids")
                teacher_forced_ids = metadata.get("teacher_forced_token_ids")
                model = record.get("model") or {}
                if (
                    not isinstance(generated_ids, list)
                    or generated_ids != teacher_forced_ids
                    or sample.get("model_record_sha256")
                    != canonical_record_sha256(record)
                    or sample.get("raw_response_sha256")
                    != hashlib.sha256(
                        str(record.get("raw_response") or "").encode("utf-8")
                    ).hexdigest()
                    or sample.get("thinking_sha256")
                    != hashlib.sha256(
                        str(record.get("THINKING") or "").encode("utf-8")
                    ).hexdigest()
                    or sample.get("output_sha256")
                    != hashlib.sha256(
                        str(record.get("OUTPUT") or "").encode("utf-8")
                    ).hexdigest()
                    or sample.get("generated_token_ids_sha256")
                    != token_ids_sha256(generated_ids)
                    or sample.get("teacher_forced_token_ids_sha256")
                    != token_ids_sha256(teacher_forced_ids)
                    or sample.get("resolved_revision") != EXPECTED_REVISION
                    or model.get("resolved_revision") != EXPECTED_REVISION
                ):
                    mismatches.append(sample_id)
        audit.add(
            f"frozen response hashes: {Path(relative_output).stem}",
            joins_match and file_hash_matches and not mismatches,
            detail={
                "expected_samples": len(expected_ids),
                "hashed_samples": len(samples),
                "mismatches": mismatches,
            },
        )


def _pilot_ids(paths: Iterable[Path]) -> set[str]:
    return {
        sample_id
        for path in paths
        if path.is_file()
        for sample_id in _ids(read_jsonl(path))
    }


def _audit_frozen_bundle(
    audit: Audit,
    formal_dir: Path,
    *,
    prefix: str,
    benchmark: str,
    count: int,
    pilot_manifests: Iterable[Path],
) -> set[str]:
    dataset_path = formal_dir / f"{prefix}.dataset.jsonl"
    model_path = formal_dir / f"{prefix}.model.jsonl"
    evaluation_path = formal_dir / f"{prefix}.generation_eval.jsonl"
    required = (
        audit.require_file(f"{prefix}: dataset", dataset_path),
        audit.require_file(f"{prefix}: model", model_path),
        audit.require_file(f"{prefix}: generation evaluation", evaluation_path),
    )
    if not all(required):
        return set()

    datasets = read_jsonl(dataset_path)
    models = read_jsonl(model_path)
    evaluations = read_jsonl(evaluation_path)
    dataset_ids = _ids(datasets)
    model_ids = _ids(models)
    evaluation_ids = _ids(evaluations)
    audit.add(f"{prefix}: exact sample count", len(datasets) == count, detail=len(datasets))
    audit.add(
        f"{prefix}: unique IDs",
        not _duplicates(dataset_ids)
        and not _duplicates(model_ids)
        and not _duplicates(evaluation_ids),
        detail={
            "dataset": _duplicates(dataset_ids),
            "model": _duplicates(model_ids),
            "evaluation": _duplicates(evaluation_ids),
        },
    )
    audit.add(
        f"{prefix}: artifact ID joins",
        set(dataset_ids) == set(model_ids) == set(evaluation_ids),
        detail={
            "dataset": len(set(dataset_ids)),
            "model": len(set(model_ids)),
            "evaluation": len(set(evaluation_ids)),
        },
    )
    record_protocol_issues = _frozen_record_protocol_issues(
        datasets,
        models,
        evaluations,
        benchmark=benchmark,
    )
    audit.add(
        f"{prefix}: exact frozen inputs, generation config, and hard gates",
        not record_protocol_issues,
        detail={"issues": record_protocol_issues},
    )
    ablation_paths = (
        sorted(formal_dir.glob("wiki_visa_candidates*.ablation.model.jsonl"))
        if benchmark == "wiki_visa"
        else [formal_dir / "vizwiz_lf_candidates.ablation.model.jsonl"]
    )
    ablation_provenance_issues = _ablation_provenance_issues(
        models,
        evaluations,
        paths=ablation_paths,
        benchmark=benchmark,
    )
    audit.add(
        f"{prefix}: separated raw ablation provenance reproduces gates",
        not ablation_provenance_issues,
        detail={
            "paths": [str(path) for path in ablation_paths],
            "issues": ablation_provenance_issues,
        },
    )
    audit.add(
        f"{prefix}: benchmark",
        all(record.get("benchmark") == benchmark for record in datasets),
    )
    audit.add(
        f"{prefix}: every row strict eligible",
        all(record.get("strict_eligible") is True for record in evaluations),
        detail=sum(record.get("strict_eligible") is True for record in evaluations),
    )
    revisions = _resolved_revisions(models)
    audit.add(
        f"{prefix}: frozen model revision",
        revisions == {EXPECTED_REVISION},
        detail=sorted(revisions),
    )
    overlap = set(dataset_ids) & _pilot_ids(pilot_manifests)
    audit.add(
        f"{prefix}: disjoint from pilots",
        not overlap,
        detail=sorted(overlap),
    )
    if benchmark == "wiki_visa":
        strata = Counter(
            str(record["evaluation"]["metadata"]["stratum"]) for record in datasets
        )
        audit.add(
            f"{prefix}: balanced Wiki strata",
            dict(strata) == EXPECTED_STRATA,
            detail=dict(sorted(strata.items())),
        )
    else:
        output_buckets = Counter(
            str(record["evaluation"]["metadata"].get("output_length_tercile"))
            for record in datasets
        )
        audit.add(
            f"{prefix}: output-length terciles recorded",
            set(output_buckets).issubset({"short", "medium", "long"})
            and sum(output_buckets.values()) == count
            and "None" not in output_buckets,
            detail=dict(sorted(output_buckets.items())),
        )
        question_types = Counter(
            str(record["evaluation"]["metadata"].get("question_type"))
            for record in datasets
        )
        audit.add(
            f"{prefix}: question types recorded",
            sum(question_types.values()) == count
            and set(question_types)
            == {"Identification", "Description", "Reading", "Others"},
            detail=dict(sorted(question_types.items())),
        )
    return set(dataset_ids)


def _audit_attribution(
    audit: Audit,
    root: Path,
    directory: Path,
    *,
    label: str,
    expected_ids: set[str],
    localization_required: bool,
) -> None:
    summary_path = directory / "summary.json"
    records_path = directory / "attribution_records.jsonl"
    if not (
        audit.require_file(f"{label}: attribution summary", summary_path)
        and audit.require_file(f"{label}: attribution records", records_path)
    ):
        return
    summary = _read_json(summary_path)
    prefix = directory.name.removesuffix("_methods")
    methods = set(summary.get("requested_methods") or [])
    common_ids = set(str(value) for value in summary.get("common_sample_ids") or [])
    records = read_jsonl(records_path)
    pairs = [
        (str(record.get("sample_id")), str(record.get("method")))
        for record in records
        if record.get("status") == "ok"
    ]
    audit.add(
        f"{label}: exact frozen input and model provenance",
        summary.get("schema_version") == 2
        and _recorded_path_matches(
            summary.get("dataset_manifest"),
            root=root,
            expected=directory.parent / f"{prefix}.dataset.jsonl",
        )
        and _recorded_path_matches(
            summary.get("model_output"),
            root=root,
            expected=directory.parent / f"{prefix}.model.jsonl",
        )
        and _recorded_path_matches(
            summary.get("generation_evaluation"),
            root=root,
            expected=directory.parent / f"{prefix}.generation_eval.jsonl",
        )
        and summary.get("model") == "Qwen/Qwen3-VL-8B-Thinking"
        and summary.get("revision") == EXPECTED_REVISION
        and summary.get("processor")
        == {"min_pixels": 200_704, "max_pixels": 2_007_040}
        and summary.get("eligible_samples") == len(expected_ids)
        and summary.get("comparison_protocol")
        == "common_paired_successful_subset",
    )
    audit.add(f"{label}: frozen eight methods", methods == EXPECTED_METHODS, detail=sorted(methods))
    audit.add(
        f"{label}: common success equals frozen set",
        common_ids == expected_ids and bool(expected_ids),
        detail={"common": len(common_ids), "expected": len(expected_ids)},
    )
    audit.add(f"{label}: no duplicate successful pairs", not _duplicates(map(repr, pairs)))
    audit.add(
        f"{label}: complete successful matrix",
        set(pairs) == {(sample_id, method) for sample_id in expected_ids for method in EXPECTED_METHODS},
        detail={"successful_pairs": len(set(pairs)), "expected_pairs": len(expected_ids) * 8},
    )
    audit.add(
        f"{label}: nearest whole-patch protocol",
        summary.get("spatial_resampling") == "nearest_patch"
        and summary.get("spatial_metric_unit") == "visual_patch"
        and summary.get("cutoff_tie_policy") == "expected_uniform",
    )
    native_shapes = {
        method: (summary.get("methods") or {})
        .get(method, {})
        .get("native_grid_shapes")
        or {}
        for method in EXPECTED_METHODS
    }
    audit.add(
        f"{label}: native grid resolution disclosed",
        all(
            sum(int(count) for count in shapes.values())
            == len(expected_ids)
            for shapes in native_shapes.values()
        ),
        detail=native_shapes,
    )
    shapes_by_sample: dict[str, dict[str, tuple[int, int]]] = {}
    for record in records:
        sample_id = str(record.get("sample_id"))
        method = str(record.get("method"))
        shape = record.get("visual_grid_shape")
        if (
            record.get("status") == "ok"
            and sample_id in expected_ids
            and method in LEARNED_GRID_METHODS
            and isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(value, int) for value in shape)
        ):
            shapes_by_sample.setdefault(sample_id, {})[method] = tuple(shape)
    native_shape_mismatches = {
        sample_id: {
            method: list(shape)
            for method, shape in sorted(shapes_by_sample.get(sample_id, {}).items())
        }
        for sample_id in sorted(expected_ids)
        if set(shapes_by_sample.get(sample_id, {})) != LEARNED_GRID_METHODS
        or len(set(shapes_by_sample.get(sample_id, {}).values())) != 1
    }
    audit.add(
        f"{label}: learned methods share native grid per sample",
        not native_shape_mismatches and bool(expected_ids),
        detail=native_shape_mismatches,
    )
    audit.add(
        f"{label}: formal timing and VRAM evidence complete",
        all(
            _finite_nonnegative(record.get("seconds"))
            and _finite_nonnegative(record.get("peak_vram_gb"))
            and _finite_nonnegative(record.get("incremental_peak_vram_gb"))
            and float(record["peak_vram_gb"])
            >= float(record["incremental_peak_vram_gb"])
            for record in records
            if record.get("status") == "ok"
        )
        and all(
            (summary.get("methods") or {})
            .get(method, {})
            .get("common_samples")
            == len(expected_ids)
            and _finite_nonnegative(
                (summary.get("methods") or {})
                .get(method, {})
                .get("mean_seconds")
            )
            and _finite_nonnegative(
                (summary.get("methods") or {})
                .get(method, {})
                .get("mean_peak_vram_gb")
            )
            for method in EXPECTED_METHODS
        ),
    )
    successful_records = [
        record for record in records if record.get("status") == "ok"
    ]
    audit.add(
        f"{label}: finite native attribution grids and localization values",
        len(successful_records) == len(expected_ids) * len(EXPECTED_METHODS)
        and all(_valid_visual_grid(record) for record in successful_records)
        and all(
            (
                isinstance(record.get("localization"), Mapping)
                and EXPECTED_LOCALIZATION_METRICS.issubset(
                    record["localization"]
                )
                and all(
                    isinstance(record["localization"][metric], (int, float))
                    and math.isfinite(
                        float(record["localization"][metric])
                    )
                    and 0.0
                    <= float(record["localization"][metric])
                    <= 1.0
                    for metric in EXPECTED_LOCALIZATION_METRICS
                )
            )
            if localization_required
            else record.get("localization") is None
            for record in successful_records
        ),
    )
    if localization_required:
        localized = sum(
            record.get("status") == "ok"
            and isinstance(record.get("localization"), Mapping)
            and EXPECTED_LOCALIZATION_METRICS.issubset(record["localization"])
            for record in records
        )
        audit.add(
            f"{label}: localization metrics present",
            localized == len(expected_ids) * 8,
            detail=localized,
        )


def _audit_faithfulness(
    audit: Audit,
    root: Path,
    directory: Path,
    *,
    label: str,
    expected_ids: set[str],
) -> None:
    summary_path = directory / "summary.json"
    records_path = directory / "faithfulness_records.jsonl"
    if not (
        audit.require_file(f"{label}: faithfulness summary", summary_path)
        and audit.require_file(f"{label}: faithfulness records", records_path)
    ):
        return
    summary = _read_json(summary_path)
    prefix = directory.name.removesuffix("_faithfulness")
    methods = set(summary.get("methods") or [])
    common_ids = set(str(value) for value in summary.get("common_sample_ids") or [])
    records = read_jsonl(records_path)
    successful = [
        record
        for record in records
        if record.get("status") == "ok"
        and record.get("sample_id") in expected_ids
        and record.get("method") in EXPECTED_METHODS
    ]
    audit.add(
        f"{label}: exact frozen input and model provenance",
        summary.get("schema_version") == 1
        and _recorded_path_matches(
            summary.get("dataset_manifest"),
            root=root,
            expected=directory.parent / f"{prefix}.dataset.jsonl",
        )
        and _recorded_path_matches(
            summary.get("model_output"),
            root=root,
            expected=directory.parent / f"{prefix}.model.jsonl",
        )
        and _recorded_path_matches(
            summary.get("attribution_dir"),
            root=root,
            expected=directory.parent / f"{prefix}_methods",
        )
        and summary.get("model") == "Qwen/Qwen3-VL-8B-Thinking"
        and summary.get("revision") == EXPECTED_REVISION
        and summary.get("processor")
        == {"min_pixels": 200_704, "max_pixels": 2_007_040}
        and summary.get("target_span") == "output_only"
        and summary.get("response_frozen") is True
        and summary.get("teacher_forced") is True
        and summary.get("comparison_protocol")
        == "common_paired_successful_subset"
        and summary.get("curve_normalization_policy")
        == CURVE_NORMALIZATION_POLICY,
    )
    audit.add(f"{label}: frozen eight methods", methods == EXPECTED_METHODS, detail=sorted(methods))
    audit.add(
        f"{label}: common success equals frozen set",
        common_ids == expected_ids and bool(expected_ids),
        detail={"common": len(common_ids), "expected": len(expected_ids)},
    )
    audit.add(
        f"{label}: 64-region/10-step budget",
        summary.get("target_regions") == 64 and summary.get("steps") == 10,
        detail={"target_regions": summary.get("target_regions"), "steps": summary.get("steps")},
    )
    audit.add(
        f"{label}: complete successful matrix",
        len(successful) == len(expected_ids) * 8
        and not _duplicates(
            repr((record["sample_id"], record["method"])) for record in successful
        ),
        detail={"successful_pairs": len(successful), "expected_pairs": len(expected_ids) * 8},
    )
    region_layouts = {
        method: (summary.get("methods") or {})
        .get(method, {})
        .get("region_layouts")
        or {}
        for method in EXPECTED_METHODS
    }
    audit.add(
        f"{label}: common perturbation resolution disclosed",
        all(
            sum(int(count) for count in layouts.values())
            == len(expected_ids)
            for layouts in region_layouts.values()
        )
        and len(
            {
                json.dumps(layouts, sort_keys=True)
                for layouts in region_layouts.values()
            }
        )
        == 1,
        detail=region_layouts,
    )
    layouts_by_sample: dict[str, dict[str, tuple[int, int]]] = {}
    for record in successful:
        layout = record["faithfulness"].get("region_layout")
        if (
            isinstance(layout, list)
            and len(layout) == 2
            and all(isinstance(value, int) for value in layout)
        ):
            layouts_by_sample.setdefault(str(record["sample_id"]), {})[
                str(record["method"])
            ] = tuple(layout)
    layout_mismatches = {
        sample_id: {
            method: list(layout)
            for method, layout in sorted(layouts_by_sample.get(sample_id, {}).items())
        }
        for sample_id in sorted(expected_ids)
        if set(layouts_by_sample.get(sample_id, {})) != EXPECTED_METHODS
        or len(set(layouts_by_sample.get(sample_id, {}).values())) != 1
    }
    audit.add(
        f"{label}: methods share perturbation layout per sample",
        not layout_mismatches and bool(expected_ids),
        detail=layout_mismatches,
    )
    curve_fields = {
        "region_scores",
        "fractions",
        "deletion_output_mean_logprob",
        "insertion_output_mean_logprob",
        "normalized_deletion",
        "normalized_insertion",
        "deletion_degenerate",
        "insertion_degenerate",
        "positive_only_ordering",
    }
    curves_complete = all(
        curve_fields.issubset(record["faithfulness"])
        and {
            "deletion_output_mean_logprob",
            "insertion_output_mean_logprob",
            "normalized_deletion",
            "normalized_insertion",
            "deletion_degenerate",
            "insertion_degenerate",
        }.issubset(record["faithfulness"]["positive_only_ordering"])
        for record in successful
    )
    summary_counts_complete = all(
        isinstance(
            (summary.get("methods") or {})
            .get(method, {})
            .get("degenerate_deletion_curves"),
            int,
        )
        and isinstance(
            (summary.get("methods") or {})
            .get(method, {})
            .get("degenerate_insertion_curves"),
            int,
        )
        and isinstance(
            (summary.get("methods") or {})
            .get(method, {})
            .get("positive_order_differs"),
            int,
        )
        for method in EXPECTED_METHODS
    )
    audit.add(
        f"{label}: curves, degenerate counts, and sign sensitivity saved",
        curves_complete and summary_counts_complete and bool(successful),
    )
    endpoint_pairs: dict[str, list[tuple[float, float]]] = {}
    curve_protocol_complete = True
    curve_protocol_failures: list[dict[str, Any]] = []
    for record in successful:
        faithfulness = record.get("faithfulness") or {}
        regions = faithfulness.get("regions")
        layout = faithfulness.get("region_layout")
        valid_regions = (
            isinstance(regions, int)
            and not isinstance(regions, bool)
            and regions in {63, 64}
            and isinstance(layout, list)
            and len(layout) == 2
            and all(isinstance(value, int) and value > 0 for value in layout)
            and int(layout[0]) * int(layout[1]) == regions
            and _finite_sequence(
                faithfulness.get("region_scores"),
                length=regions,
            )
        )
        valid_signed = (
            valid_regions
            and faithfulness.get("ordering_policy") == "signed_descending"
            and _valid_faithfulness_curve(
                faithfulness,
                regions=regions,
                expected_steps=10,
            )
        )
        positive = faithfulness.get("positive_only_ordering")
        valid_positive = valid_regions and _valid_faithfulness_curve(
            positive,
            regions=regions,
            expected_steps=10,
        )
        record_protocol = (
            record.get("target_span") == "output_only"
            and record.get("response_frozen") is True
            and record.get("teacher_forced") is True
        )
        curve_protocol_complete = (
            curve_protocol_complete
            and valid_signed
            and valid_positive
            and record_protocol
        )
        if not (valid_signed and valid_positive and record_protocol):
            curve_protocol_failures.append(
                {
                    "sample_id": record.get("sample_id"),
                    "method": record.get("method"),
                    "signed_curve": valid_signed,
                    "positive_only_curve": valid_positive,
                    "record_protocol": record_protocol,
                }
            )
        if valid_signed:
            deletion = faithfulness["deletion_output_mean_logprob"]
            endpoint_pairs.setdefault(str(record["sample_id"]), []).append(
                (float(deletion[0]), float(deletion[-1]))
            )
    endpoint_mismatches = {
        sample_id: values
        for sample_id, values in endpoint_pairs.items()
        if len(values) != len(EXPECTED_METHODS)
        or max(value[0] for value in values)
        - min(value[0] for value in values)
        > 1e-6
        or max(value[1] for value in values)
        - min(value[1] for value in values)
        > 1e-6
    }
    audit.add(
        f"{label}: exact 10-step curve protocol and shared frozen endpoints",
        curve_protocol_complete
        and len(endpoint_pairs) == len(expected_ids)
        and not endpoint_mismatches,
        detail={
            "valid_endpoint_samples": len(endpoint_pairs),
            "expected_endpoint_samples": len(expected_ids),
            "curve_protocol_failures": curve_protocol_failures,
            "endpoint_mismatches": endpoint_mismatches,
        },
    )
    audit.add(
        f"{label}: formal perturbation timing evidence complete",
        all(
            _finite_nonnegative(record.get("seconds"))
            for record in successful
        )
        and all(
            (summary.get("methods") or {})
            .get(method, {})
            .get("common_samples")
            == len(expected_ids)
            and _finite_nonnegative(
                (summary.get("methods") or {})
                .get(method, {})
                .get("mean_seconds")
            )
            for method in EXPECTED_METHODS
        ),
    )


def _audit_manual_protocol_review(
    audit: Audit,
    root: Path,
    formal_dir: Path,
    *,
    prefix: str,
    expected_ids: set[str],
    expected_reviewed: int,
) -> None:
    packet_path = formal_dir / f"{prefix}.protocol_audit.md"
    reviews_path = formal_dir / f"{prefix}.protocol_reviews.jsonl"
    ablation_paths = (
        sorted(formal_dir.glob("wiki_visa_candidates*.ablation.model.jsonl"))
        if prefix == "wiki_visa_n120"
        else [formal_dir / "vizwiz_lf_candidates.ablation.model.jsonl"]
    )
    packet_inputs = (
        formal_dir / f"{prefix}.dataset.jsonl",
        formal_dir / f"{prefix}.model.jsonl",
        formal_dir / f"{prefix}.generation_eval.jsonl",
        *ablation_paths,
        packet_path,
        reviews_path,
    )
    missing_packet_inputs = [
        str(path) for path in packet_inputs if not path.is_file()
    ]
    packet_issues: list[str] = []
    if not missing_packet_inputs:
        try:
            expected_packet, expected_template = prepare_protocol_audit(
                formal_dir / f"{prefix}.dataset.jsonl",
                formal_dir / f"{prefix}.model.jsonl",
                formal_dir / f"{prefix}.generation_eval.jsonl",
                ablation_paths,
                fraction=0.1,
                seed=17,
            )
            reviews = read_jsonl(reviews_path)
        except (KeyError, TypeError, ValueError) as error:
            packet_issues.append(f"protocol audit packet replay failed: {error}")
        else:
            expected_review_ids = [
                str(record["sample_id"]) for record in expected_template
            ]
            review_ids = [str(record.get("sample_id") or "") for record in reviews]
            if packet_path.read_text(encoding="utf-8") != expected_packet:
                packet_issues.append(
                    "review packet does not exactly match frozen source records"
                )
            if (
                len(reviews) != expected_reviewed
                or review_ids != expected_review_ids
                or _duplicates(review_ids)
            ):
                packet_issues.append(
                    "review template IDs differ from deterministic 10% sample"
                )
            required_fields = {
                "sample_id",
                "image_dependence",
                "thinking_quality",
                "reviewer",
                "reason",
            }
            if any(set(record) != required_fields for record in reviews):
                packet_issues.append("review template fields are not exact")
            for record in reviews:
                image_label = record.get("image_dependence")
                thinking_label = record.get("thinking_quality")
                reviewer = record.get("reviewer")
                reason = record.get("reason")
                if image_label is not None and image_label not in IMAGE_DEPENDENCE_LABELS:
                    packet_issues.append("review template has an invalid image label")
                    break
                if (
                    thinking_label is not None
                    and thinking_label not in THINKING_QUALITY_LABELS
                ):
                    packet_issues.append(
                        "review template has an invalid THINKING label"
                    )
                    break
                if reviewer is not None and not str(reviewer).strip():
                    packet_issues.append("review template has an empty reviewer")
                    break
                if reason is not None and not str(reason).strip():
                    packet_issues.append("review template has an empty reason")
                    break
            datasets = read_jsonl(formal_dir / f"{prefix}.dataset.jsonl")
            missing_images = [
                str(record["sample_id"])
                for record in datasets
                if str(record["sample_id"]) in set(expected_review_ids)
                and not (root / str(record["input"]["I_IMAGE"])).is_file()
            ]
            if missing_images:
                packet_issues.append(
                    f"review packet images are missing: {missing_images}"
                )
    audit.add(
        f"{prefix}: deterministic protocol review packet is reproducible",
        not missing_packet_inputs and not packet_issues,
        status="incomplete" if missing_packet_inputs else "error",
        detail={
            "missing": missing_packet_inputs,
            "issues": packet_issues,
        },
    )

    summary_path = formal_dir / f"{prefix}.protocol_audit_summary.json"
    if not audit.require_file(f"{prefix}: protocol manual audit", summary_path):
        return
    summary = _read_json(summary_path)
    reviewed_ids = set(str(value) for value in summary.get("audit_sample_ids") or [])
    audit.add(
        f"{prefix}: deterministic 10% protocol review complete",
        summary.get("complete") is True
        and summary.get("frozen_sample_count") == len(expected_ids)
        and summary.get("reviewed_count") == expected_reviewed
        and len(reviewed_ids) == expected_reviewed
        and reviewed_ids.issubset(expected_ids)
        and summary.get("selection_effect")
        == "caveat_only_no_frozen_id_changes",
        detail=summary,
    )


def _has_interval(value: Any) -> bool:
    return isinstance(value, Mapping) and {
        "mean",
        "ci95_low",
        "ci95_high",
    }.issubset(value)


def _finite_nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _finite_sequence(value: Any, *, length: int | None = None) -> bool:
    return (
        isinstance(value, list)
        and (length is None or len(value) == length)
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value
        )
    )


def _valid_visual_grid(record: Mapping[str, Any]) -> bool:
    shape = record.get("visual_grid_shape")
    grid = record.get("visual_grid")
    return (
        isinstance(shape, list)
        and len(shape) == 2
        and all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in shape
        )
        and isinstance(grid, list)
        and len(grid) == shape[0]
        and all(_finite_sequence(row, length=shape[1]) for row in grid)
    )


def _close(left: Any, right: Any, *, tolerance: float = 1e-6) -> bool:
    return (
        isinstance(left, (int, float))
        and isinstance(right, (int, float))
        and math.isfinite(float(left))
        and math.isfinite(float(right))
        and math.isclose(
            float(left),
            float(right),
            rel_tol=tolerance,
            abs_tol=tolerance,
        )
    )


def _valid_faithfulness_curve(
    curve: Any,
    *,
    regions: int,
    expected_steps: int,
) -> bool:
    if not isinstance(curve, Mapping):
        return False
    points = expected_steps + 1
    fractions = curve.get("fractions")
    deletion = curve.get("deletion_output_mean_logprob")
    insertion = curve.get("insertion_output_mean_logprob")
    normalized_deletion = curve.get("normalized_deletion")
    normalized_insertion = curve.get("normalized_insertion")
    density = curve.get("remaining_attribution_density")
    order = curve.get("region_order")
    if not (
        curve.get("steps") == expected_steps
        and _finite_sequence(fractions, length=points)
        and _finite_sequence(deletion, length=points)
        and _finite_sequence(insertion, length=points)
        and _finite_sequence(normalized_deletion, length=points)
        and _finite_sequence(normalized_insertion, length=points)
        and _finite_sequence(density, length=points)
        and isinstance(order, list)
        and len(order) == regions
        and all(isinstance(index, int) for index in order)
        and set(order) == set(range(regions))
    ):
        return False
    assert isinstance(fractions, list)
    assert isinstance(deletion, list)
    assert isinstance(insertion, list)
    assert isinstance(normalized_deletion, list)
    assert isinstance(normalized_insertion, list)
    assert isinstance(density, list)
    return (
        curve.get("normalization_policy") == CURVE_NORMALIZATION_POLICY
        and _close(fractions[0], 0.0)
        and _close(fractions[-1], 1.0)
        and all(
            float(left) <= float(right) + 1e-12
            for left, right in zip(fractions, fractions[1:])
        )
        and all(0.0 <= float(value) <= 1.0 for value in fractions)
        and all(
            -1e-9 <= float(value) <= 1.0 + 1e-9
            for value in normalized_deletion
        )
        and all(
            -1e-9 <= float(value) <= 1.0 + 1e-9
            for value in normalized_insertion
        )
        and all(
            -1e-9 <= float(value) <= 1.0 + 1e-9 for value in density
        )
        and all(
            float(left) + 1e-9 >= float(right)
            for left, right in zip(density, density[1:])
        )
        and _close(normalized_deletion[0], 1.0)
        and _close(normalized_deletion[-1], 0.0)
        and _close(normalized_insertion[0], 0.0)
        and _close(normalized_insertion[-1], 1.0)
        and _close(deletion[-1], insertion[0])
        and _close(insertion[-1], deletion[0])
        and _close(
            curve.get("deletion_endpoint_delta"),
            float(deletion[0]) - float(deletion[-1]),
        )
        and _close(
            curve.get("insertion_endpoint_delta"),
            float(insertion[-1]) - float(insertion[0]),
        )
        and isinstance(curve.get("deletion_degenerate"), bool)
        and isinstance(curve.get("insertion_degenerate"), bool)
        and all(
            _finite_nonnegative(curve.get(metric))
            for metric in (
                "deletion_auc",
                "insertion_auc",
                "visual_mas",
                "visual_rise",
                "visual_rise_plus_ap",
            )
        )
    )


def _valid_interval(value: Any) -> bool:
    if not _has_interval(value):
        return False
    numbers = [
        value.get("mean"),
        value.get("ci95_low"),
        value.get("ci95_high"),
    ]
    return (
        all(
            isinstance(number, (int, float))
            and not isinstance(number, bool)
            and math.isfinite(float(number))
            for number in numbers
        )
        and float(value["ci95_low"]) <= float(value["ci95_high"])
    )


def _valid_wtl(value: Any, expected: int) -> bool:
    return (
        _valid_interval(value)
        and all(
            isinstance(value.get(key), int)
            and not isinstance(value.get(key), bool)
            and int(value[key]) >= 0
            for key in ("wins", "ties", "losses")
        )
        and sum(int(value[key]) for key in ("wins", "ties", "losses"))
        == expected
    )


def _audit_analysis_payloads(
    audit: Audit,
    formal_dir: Path,
    *,
    wiki_ids: set[str],
    viz_ids: set[str],
) -> None:
    localization_path = formal_dir / "wiki_visa_n120_methods/analysis.json"
    wiki_diagnostics_path = formal_dir / "wiki_visa_n120_methods/diagnostics.json"
    viz_diagnostics_path = formal_dir / "vizwiz_lf_n100_methods/diagnostics.json"
    wiki_faith_path = formal_dir / "wiki_visa_n120_faithfulness/analysis.json"
    viz_faith_path = formal_dir / "vizwiz_lf_n100_faithfulness/analysis.json"

    if localization_path.is_file():
        analysis = _read_json(localization_path)
        estimates = analysis.get("estimates") or {}
        groups = analysis.get("per_group_paired") or {}
        paired = analysis.get("flashtrace_minus_baseline") or {}
        audit.add(
            "A5 localization: paired n=120 and 50k bootstrap",
            analysis.get("common_samples") == len(wiki_ids) == 120
            and analysis.get("bootstrap_draws", 0) >= 50_000,
        )
        audit.add(
            "E3 localization: all registered metrics and methods",
            all(
                set((estimates.get(metric) or {})) == EXPECTED_METHODS
                and all(
                    _has_interval(estimates[metric][method])
                    for method in EXPECTED_METHODS
                )
                for metric in EXPECTED_LOCALIZATION_METRICS
            ),
        )
        audit.add(
            "E3 localization: three balanced stratum analyses",
            set(groups) == set(EXPECTED_STRATA)
            and all(
                groups[stratum].get("samples") == count
                and set(
                    (groups[stratum].get("estimates") or {}).get(
                        "energy_in_mask", {}
                    )
                )
                == EXPECTED_METHODS
                for stratum, count in EXPECTED_STRATA.items()
            ),
        )
        audit.add(
            "A5 localization: primary paired deltas and W/T/L",
            all(
                set((paired.get(metric) or {}))
                == EXPECTED_METHODS - {"flashtrace"}
                and all(
                    sum(
                        int(paired[metric][method].get(key, 0))
                        for key in ("wins", "ties", "losses")
                    )
                    == 120
                    for method in EXPECTED_METHODS - {"flashtrace"}
                )
                for metric in ("energy_in_mask", "recovery_at_5pct")
            ),
        )
        group_sample_ids = [
            str(sample_id)
            for group in groups.values()
            for sample_id in group.get("sample_ids") or []
        ]
        audit.add(
            "A5 localization: exact sample partition and complete finite statistics",
            len(group_sample_ids) == len(set(group_sample_ids)) == len(wiki_ids)
            and set(group_sample_ids) == wiki_ids
            and all(
                _valid_interval(estimates[metric][method])
                for metric in EXPECTED_LOCALIZATION_METRICS
                for method in EXPECTED_METHODS
            )
            and all(
                set((paired.get(metric) or {}))
                == EXPECTED_METHODS - {"flashtrace"}
                and all(
                    _valid_wtl(paired[metric][method], len(wiki_ids))
                    for method in EXPECTED_METHODS - {"flashtrace"}
                )
                for metric in EXPECTED_LOCALIZATION_METRICS
            )
            and all(
                set((groups[stratum].get("estimates") or {}))
                == EXPECTED_LOCALIZATION_METRICS
                and all(
                    set(
                        groups[stratum]["estimates"].get(metric) or {}
                    )
                    == EXPECTED_METHODS
                    and all(
                        _valid_interval(
                            groups[stratum]["estimates"][metric][method]
                        )
                        for method in EXPECTED_METHODS
                    )
                    for metric in EXPECTED_LOCALIZATION_METRICS
                )
                and all(
                    set(
                        (
                            groups[stratum].get(
                                "flashtrace_minus_baseline"
                            )
                            or {}
                        ).get(metric)
                        or {}
                    )
                    == EXPECTED_METHODS - {"flashtrace"}
                    and all(
                        _valid_wtl(
                            groups[stratum][
                                "flashtrace_minus_baseline"
                            ][metric][method],
                            count,
                        )
                        for method in EXPECTED_METHODS - {"flashtrace"}
                    )
                    for metric in EXPECTED_LOCALIZATION_METRICS
                )
                for stratum, count in EXPECTED_STRATA.items()
            ),
        )

    for label, path, expected_ids, has_localization in (
        ("Wiki", wiki_diagnostics_path, wiki_ids, True),
        ("VizWiz", viz_diagnostics_path, viz_ids, False),
    ):
        if not path.is_file():
            continue
        diagnostics = _read_json(path)
        geometry = diagnostics.get("geometry") or {}
        audit.add(
            f"A2 {label}: cosine, recursive mass, and 50k bootstrap",
            diagnostics.get("common_samples") == len(expected_ids)
            and diagnostics.get("bootstrap_draws", 0) >= 50_000
            and _has_interval(diagnostics.get("exact_all_gen_cosine"))
            and _has_interval(diagnostics.get("direct_positive_fraction"))
            and _has_interval(diagnostics.get("recursive_positive_fraction"))
            and _has_interval(diagnostics.get("direct_absolute_fraction"))
            and _has_interval(diagnostics.get("recursive_absolute_fraction")),
        )
        audit.add(
            f"A3/A4 {label}: complete method geometry and signed diagnostics",
            set(geometry) == EXPECTED_METHODS
            and all(
                all(
                    _has_interval(geometry[method].get(metric))
                    for metric in (
                        "border_mass_ratio",
                        "top_row_mass_ratio",
                        "left_column_mass_ratio",
                        "heatmap_centroid_distance_to_center",
                        "negative_cell_fraction",
                    )
                )
                for method in EXPECTED_METHODS
            ),
        )
        sample_rows = diagnostics.get("samples") or []
        sample_ids = [
            str(sample.get("sample_id")) for sample in sample_rows
        ]
        bucket_counts = Counter(
            str(sample.get("thinking_bucket")) for sample in sample_rows
        )
        sample_count = len(expected_ids)
        expected_bucket_counts = Counter(
            ("short", "medium", "long")[
                min(2, index * 3 // sample_count)
            ]
            for index in range(sample_count)
        )
        scalar_geometry_metrics = (
            "border_mass_ratio",
            "top_row_mass_ratio",
            "left_column_mass_ratio",
            "heatmap_centroid_distance_to_center",
            "negative_cell_fraction",
        )
        audit.add(
            f"A2–A4 {label}: exact sample IDs, buckets, and finite diagnostics",
            len(sample_ids) == len(set(sample_ids)) == sample_count
            and set(sample_ids) == expected_ids
            and bucket_counts == expected_bucket_counts
            and all(
                set(sample.get("methods") or {}) == EXPECTED_METHODS
                and all(
                    isinstance(
                        sample["methods"][method].get(metric),
                        (int, float),
                    )
                    and math.isfinite(
                        float(sample["methods"][method][metric])
                    )
                    for method in EXPECTED_METHODS
                    for metric in scalar_geometry_metrics
                )
                and (
                    sample.get("ground_truth_centroid") is not None
                    if has_localization
                    else sample.get("ground_truth_centroid") is None
                )
                for sample in sample_rows
            )
            and all(
                _valid_interval(diagnostics.get(metric))
                for metric in (
                    "exact_all_gen_cosine",
                    "direct_positive_fraction",
                    "recursive_positive_fraction",
                    "direct_absolute_fraction",
                    "recursive_absolute_fraction",
                )
            )
            and all(
                _valid_interval(geometry[method].get(metric))
                for method in EXPECTED_METHODS
                for metric in scalar_geometry_metrics
            ),
        )
        if has_localization:
            buckets = diagnostics.get("recursion_by_thinking_bucket") or {}
            centroids = diagnostics.get("ground_truth_centroid_distance") or {}
            audit.add(
                "A1 Wiki: recursion gain in all THINKING buckets",
                set(buckets) == {"short", "medium", "long"}
                and all(
                    "ifr-span" in buckets[bucket]
                    and all(
                        _has_interval(buckets[bucket]["ifr-span"].get(metric))
                        for metric in ("energy_in_mask", "recovery_at_5pct")
                    )
                    for bucket in buckets
                ),
            )
            audit.add(
                "A3 Wiki: GT centroid distance by stratum",
                set(centroids) == set(EXPECTED_STRATA)
                and all(_has_interval(value) for value in centroids.values()),
            )

    for label, path, faith_expected_ids, require_fully_correct in (
        ("Wiki", wiki_faith_path, wiki_ids, False),
        ("VizWiz", viz_faith_path, viz_ids, True),
    ):
        if not path.is_file():
            continue
        expected_count = len(faith_expected_ids)
        analysis = _read_json(path)
        overall = analysis.get("overall") or {}
        positive = analysis.get("positive_only_ordering") or {}
        buckets = analysis.get("recursion_by_thinking_bucket") or {}
        audit.add(
            f"E4/E5 {label}: paired faithfulness analysis and 50k bootstrap",
            overall.get("samples") == expected_count
            and analysis.get("bootstrap_draws", 0) >= 50_000
            and set(overall.get("estimates") or {}) == EXPECTED_METHODS,
        )
        audit.add(
            f"A4 {label}: complete positive-only ordering sensitivity",
            analysis.get("positive_only_available") is True
            and positive.get("samples") == expected_count
            and set(positive.get("estimates") or {}) == EXPECTED_METHODS,
        )
        audit.add(
            f"A1 {label}: faithfulness recursion in all THINKING buckets",
            set(buckets) == {"short", "medium", "long"}
            and sum(
                int((buckets[bucket] or {}).get("samples", 0))
                for bucket in ("short", "medium", "long")
            )
            == expected_count
            and all(
                "ifr-span"
                in ((buckets[bucket] or {}).get("flashtrace_favorable_difference") or {})
                for bucket in ("short", "medium", "long")
            ),
        )
        overall_ids = [
            str(sample_id) for sample_id in overall.get("sample_ids") or []
        ]
        positive_ids = [
            str(sample_id) for sample_id in positive.get("sample_ids") or []
        ]
        bucket_ids = [
            str(sample_id)
            for bucket in ("short", "medium", "long")
            for sample_id in (buckets.get(bucket) or {}).get(
                "sample_ids"
            )
            or []
        ]
        paired_baselines = EXPECTED_METHODS - {"flashtrace"}
        audit.add(
            f"E4/E5 {label}: exact paired IDs, finite intervals, and W/T/L",
            len(overall_ids) == len(set(overall_ids)) == expected_count
            and set(overall_ids) == faith_expected_ids
            and len(positive_ids) == len(set(positive_ids)) == expected_count
            and set(positive_ids) == faith_expected_ids
            and len(bucket_ids) == len(set(bucket_ids)) == expected_count
            and set(bucket_ids) == faith_expected_ids
            and all(
                set((overall.get("estimates") or {}).get(method) or {})
                == {"deletion_auc", "insertion_auc", "visual_mas"}
                and all(
                    _valid_interval(
                        overall["estimates"][method][metric]
                    )
                    for metric in (
                        "deletion_auc",
                        "insertion_auc",
                        "visual_mas",
                    )
                )
                for method in EXPECTED_METHODS
            )
            and set(
                overall.get("flashtrace_favorable_difference") or {}
            )
            == paired_baselines
            and all(
                _valid_wtl(
                    overall["flashtrace_favorable_difference"][method][
                        metric
                    ],
                    expected_count,
                )
                for method in paired_baselines
                for metric in (
                    "deletion_auc",
                    "insertion_auc",
                    "visual_mas",
                )
            )
            and all(
                set((positive.get("estimates") or {}).get(method) or {})
                == {"deletion_auc", "insertion_auc", "visual_mas"}
                and all(
                    _valid_interval(
                        positive["estimates"][method][metric]
                    )
                    for metric in (
                        "deletion_auc",
                        "insertion_auc",
                        "visual_mas",
                    )
                )
                for method in EXPECTED_METHODS
            )
            and all(
                set((buckets.get(bucket) or {}).get("estimates") or {})
                == {"ifr-span", "flashtrace"}
                and set(
                    (
                        buckets.get(bucket) or {}
                    ).get("flashtrace_favorable_difference")
                    or {}
                )
                == {"ifr-span"}
                and all(
                    _valid_wtl(
                        buckets[bucket][
                            "flashtrace_favorable_difference"
                        ]["ifr-span"][metric],
                        int(buckets[bucket]["samples"]),
                    )
                    for metric in (
                        "deletion_auc",
                        "insertion_auc",
                        "visual_mas",
                    )
                )
                for bucket in ("short", "medium", "long")
            ),
        )
        if require_fully_correct:
            subset = analysis.get("fully_correct_subset") or {}
            subset_count = int(subset.get("samples", 0))
            reviewed_evaluation = (
                formal_dir / "vizwiz_lf_n100.reviewed.generation_eval.jsonl"
            )
            recorded_evaluation = analysis.get("generation_evaluation")
            recorded_path = (
                Path(str(recorded_evaluation))
                if recorded_evaluation
                else None
            )
            if recorded_path is not None and not recorded_path.is_absolute():
                repo_root = formal_dir.parents[4]
                recorded_path = repo_root / recorded_path
            reviewed_ready = reviewed_evaluation.is_file()
            provenance_matches = (
                recorded_path is not None
                and recorded_path.resolve() == reviewed_evaluation.resolve()
            )
            audit.add(
                "A8 VizWiz: fully-correct paired sensitivity",
                reviewed_ready
                and provenance_matches
                and 0 < subset_count <= expected_count
                and set(subset.get("estimates") or {}) == EXPECTED_METHODS
                and set(subset.get("flashtrace_favorable_difference") or {})
                == EXPECTED_METHODS - {"flashtrace"},
                status="error" if reviewed_ready else "incomplete",
                detail={
                    "samples": subset_count,
                    "reviewed_evaluation_ready": reviewed_ready,
                    "recorded_generation_evaluation": recorded_evaluation,
                },
            )


def _audit_semantic_provenance(
    audit: Audit,
    formal_dir: Path,
    *,
    expected_ids: set[str],
) -> None:
    llm_path = formal_dir / "vizwiz_lf_n100.semantic_judgments.llm.jsonl"
    reviews_path = formal_dir / "vizwiz_lf_n100.human_reviews.jsonl"
    judgments_path = formal_dir / "vizwiz_lf_n100.semantic_judgments.jsonl"
    packet_path = formal_dir / "vizwiz_lf_n100.human_audit.md"
    tasks_path = formal_dir / "vizwiz_lf_n100.semantic_tasks.jsonl"
    dataset_path = formal_dir / "vizwiz_lf_n100.dataset.jsonl"
    model_path = formal_dir / "vizwiz_lf_n100.model.jsonl"
    deterministic_ids = set(audit_sample_ids(sorted(expected_ids)))

    semantic_task_inputs = (dataset_path, model_path, tasks_path)
    missing_task_inputs = [
        str(path) for path in semantic_task_inputs if not path.is_file()
    ]
    task_issues: list[str] = []
    if not missing_task_inputs:
        try:
            expected_tasks = prepare_semantic_tasks(dataset_path, model_path)
            actual_tasks = read_jsonl(tasks_path)
        except (KeyError, TypeError, ValueError) as error:
            task_issues.append(f"semantic task replay failed: {error}")
        else:
            if actual_tasks != expected_tasks:
                task_issues.append(
                    "semantic tasks do not exactly match frozen dataset/model"
                )
    audit.add(
        "A8: semantic judgment tasks exactly replay frozen inputs",
        not missing_task_inputs and not task_issues,
        status="incomplete" if missing_task_inputs else "error",
        detail={"missing": missing_task_inputs, "issues": task_issues},
    )

    llm_by_id: dict[str, Mapping[str, Any]] = {}
    if audit.require_file("A8: complete LLM semantic judgments", llm_path):
        llm_rows = read_jsonl(llm_path)
        llm_ids = _ids(llm_rows)
        llm_by_id = {
            str(record.get("sample_id")): record for record in llm_rows
        }
        audit.add(
            "A8: LLM semantic judgment IDs and labels",
            len(llm_rows) == len(expected_ids) == 100
            and not _duplicates(llm_ids)
            and set(llm_ids) == expected_ids
            and all(
                record.get("label") in {"fully", "partial", "wrong"}
                and str(record.get("judge") or "").strip()
                and str(record.get("reason") or "").strip()
                and not record.get("human_reviewed")
                for record in llm_rows
            ),
            detail={
                "rows": len(llm_rows),
                "missing": sorted(expected_ids - set(llm_ids)),
                "extra": sorted(set(llm_ids) - expected_ids),
                "duplicates": _duplicates(llm_ids),
            },
        )

    if audit.require_file("A8: deterministic human audit packet", packet_path):
        packet = packet_path.read_text(encoding="utf-8")
        audit.add(
            "A8: human audit packet contains all deterministic samples",
            all(
                f"## {sample_id}" in packet
                and f"data/vizwiz_lf/images/"
                f"{sample_id.removeprefix('vizwiz-lf-')}.jpg" in packet
                for sample_id in deterministic_ids
            ),
            detail=sorted(deterministic_ids),
        )

    if audit.require_file("A8: 10% human semantic reviews", reviews_path):
        reviews = read_jsonl(reviews_path)
        review_ids = _ids(reviews)
        packet_replay_issues: list[str] = []
        if (
            dataset_path.is_file()
            and model_path.is_file()
            and llm_path.is_file()
            and packet_path.is_file()
        ):
            try:
                expected_packet, expected_template = (
                    prepare_semantic_human_review(
                        dataset_path,
                        model_path,
                        llm_path,
                        audit_fraction=0.1,
                        audit_seed=17,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                packet_replay_issues.append(
                    f"semantic audit packet replay failed: {error}"
                )
            else:
                if packet_path.read_text(encoding="utf-8") != expected_packet:
                    packet_replay_issues.append(
                        "semantic human packet differs from frozen sources"
                    )
                expected_pairs = [
                    (str(record["sample_id"]), str(record["llm_label"]))
                    for record in expected_template
                ]
                actual_pairs = [
                    (
                        str(record.get("sample_id") or ""),
                        str(record.get("llm_label") or ""),
                    )
                    for record in reviews
                ]
                if actual_pairs != expected_pairs:
                    packet_replay_issues.append(
                        "semantic review template IDs/LLM labels changed"
                    )
        else:
            packet_replay_issues.append(
                "semantic packet replay inputs are incomplete"
            )
        audit.add(
            "A8: deterministic semantic human packet is reproducible",
            not packet_replay_issues,
            status=(
                "incomplete"
                if "inputs are incomplete" in " ".join(packet_replay_issues)
                else "error"
            ),
            detail={"issues": packet_replay_issues},
        )
        complete_reviews = all(
            record.get("human_label") in {"fully", "partial", "wrong"}
            and str(record.get("human_reviewer") or "").strip()
            and str(record.get("human_reason") or "").strip()
            and (
                not llm_by_id
                or record.get("llm_label")
                == llm_by_id.get(str(record.get("sample_id")), {}).get("label")
            )
            for record in reviews
        )
        audit.add(
            "A8: deterministic 10% human semantic review complete",
            len(reviews) == len(deterministic_ids) == 10
            and not _duplicates(review_ids)
            and set(review_ids) == deterministic_ids
            and complete_reviews,
            status="incomplete",
            detail={
                "expected_ids": sorted(deterministic_ids),
                "review_ids": sorted(review_ids),
                "complete_rows": sum(
                    record.get("human_label") in {"fully", "partial", "wrong"}
                    and bool(str(record.get("human_reviewer") or "").strip())
                    and bool(str(record.get("human_reason") or "").strip())
                    for record in reviews
                ),
            },
        )

    if audit.require_file("A8: adjudicated semantic judgments", judgments_path):
        judgments = read_jsonl(judgments_path)
        judgment_ids = _ids(judgments)
        human_ids = {
            str(record["sample_id"])
            for record in judgments
            if record.get("human_reviewed") is True
        }
        audit.add(
            "A8: adjudication preserves LLM provenance and exact audit IDs",
            len(judgments) == len(expected_ids) == 100
            and not _duplicates(judgment_ids)
            and set(judgment_ids) == expected_ids
            and human_ids == deterministic_ids
            and all(
                record.get("label") in {"fully", "partial", "wrong"}
                and str(record.get("judge") or "").strip()
                and str(record.get("reason") or "").strip()
                and (
                    sample_id not in deterministic_ids
                    or (
                        str(record.get("human_reviewer") or "").strip()
                        and record.get("llm_label")
                        == llm_by_id.get(sample_id, {}).get("label")
                        and record.get("llm_reason")
                        == llm_by_id.get(sample_id, {}).get("reason")
                    )
                )
                for record in judgments
                for sample_id in (str(record.get("sample_id")),)
            ),
            detail={
                "rows": len(judgments),
                "human_ids": sorted(human_ids),
                "expected_human_ids": sorted(deterministic_ids),
            },
        )


def audit_formal(root: Path) -> dict[str, Any]:
    audit = Audit()
    protocol_path = root / "evaluations/multimodal/protocol.json"
    formal_dir = root / "evaluations/multimodal/results/strict/formal"
    if audit.require_file("protocol: file", protocol_path):
        protocol = _read_json(protocol_path)
        protocol_issues = _protocol_v2_issues(protocol)
        audit.add(
            "protocol: exact v2 frozen scope",
            not protocol_issues,
            detail={"issues": protocol_issues},
        )
        audit.add("protocol: schema v2", protocol.get("schema_version") == 2)
        audit.add(
            "protocol: revision",
            (protocol.get("model") or {}).get("revision") == EXPECTED_REVISION,
        )
        audit.add(
            "protocol: frozen eight methods",
            {method["id"] for method in protocol.get("methods") or []}
            == EXPECTED_METHODS,
        )
        faithfulness = protocol.get("faithfulness_protocol") or {}
        audit.add(
            "protocol: faithfulness budget",
            faithfulness.get("region_budget") == 64
            and faithfulness.get("curve_steps") == 10,
        )
        manual_audit = protocol.get("independent_manual_audit") or {}
        audit.add(
            "protocol: caveat-only 10% manual audit",
            manual_audit.get("fraction_per_dataset") == 0.1
            and manual_audit.get("selection_seed") == 17
            and set(manual_audit.get("dimensions") or [])
            == {"image_dependence", "thinking_quality"}
            and manual_audit.get("selection_effect")
            == "caveat_only_no_frozen_id_changes",
        )

    wiki_ids = _audit_frozen_bundle(
        audit,
        formal_dir,
        prefix="wiki_visa_n120",
        benchmark="wiki_visa",
        count=120,
        pilot_manifests=(
            root / "evaluations/multimodal/results/strict/final/wiki_visa_n18_2mp.dataset.jsonl",
            root / "evaluations/multimodal/results/strict/native_pilot/wiki_visa_n10.dataset.jsonl",
        ),
    )
    viz_ids = _audit_frozen_bundle(
        audit,
        formal_dir,
        prefix="vizwiz_lf_n100",
        benchmark="vizwiz_lf",
        count=100,
        pilot_manifests=(
            root / "evaluations/multimodal/results/strict/native_pilot/vizwiz_lf_n10.dataset.jsonl",
        ),
    )

    frozen_path = formal_dir / "frozen_ids.json"
    if audit.require_file("frozen IDs: artifact", frozen_path):
        frozen = _read_json(frozen_path)
        frozen_metadata_issues = _frozen_protocol_metadata_issues(frozen)
        audit.add(
            "frozen IDs: exact protocol metadata",
            not frozen_metadata_issues,
            detail={"issues": frozen_metadata_issues},
        )
        frozen_datasets = frozen.get("datasets") or {}
        frozen_wiki = {
            str(row["sample_id"])
            for row in (frozen_datasets.get("wiki_visa") or {}).get("samples") or []
        }
        frozen_viz = {
            str(row["sample_id"])
            for row in (frozen_datasets.get("vizwiz_lf") or {}).get("samples") or []
        }
        audit.add(
            "frozen IDs: Wiki join",
            bool(wiki_ids) and frozen_wiki == wiki_ids,
            status="error" if wiki_ids else "incomplete",
        )
        audit.add(
            "frozen IDs: VizWiz join",
            bool(viz_ids) and frozen_viz == viz_ids,
            status="error" if viz_ids else "incomplete",
        )
    _audit_frozen_input_hashes(
        audit,
        root=root,
        formal_dir=formal_dir,
        expected={
            (
                "evaluations/multimodal/results/strict/formal/"
                "wiki_visa_n120.dataset.jsonl"
            ): wiki_ids,
            (
                "evaluations/multimodal/results/strict/formal/"
                "vizwiz_lf_n100.dataset.jsonl"
            ): viz_ids,
        },
    )
    _audit_frozen_response_hashes(
        audit,
        root=root,
        formal_dir=formal_dir,
        expected={
            (
                "evaluations/multimodal/results/strict/formal/"
                "wiki_visa_n120.model.jsonl"
            ): wiki_ids,
            (
                "evaluations/multimodal/results/strict/formal/"
                "vizwiz_lf_n100.model.jsonl"
            ): viz_ids,
        },
    )

    for funnel, expected, benchmark_gate in (
        ("wiki_visa_funnel.json", 120, "whole_output_correct"),
        ("vizwiz_lf_funnel.json", 100, "output_non_refusal"),
    ):
        funnel_path = formal_dir / funnel
        if audit.require_file(f"gate funnel: {funnel}", funnel_path):
            payload = _read_json(funnel_path)
            stages = payload.get("stages") or []
            conservation_issues = _funnel_conservation_issues(payload)
            audit.add(
                f"gate funnel conserves stage populations: {funnel}",
                not conservation_issues,
                detail={"issues": conservation_issues},
            )
            audit.add(
                f"gate funnel reaches frozen n: {funnel}",
                payload.get("frozen_sample_count") == expected
                and bool(stages)
                and stages[-1].get("stage")
                == "unique_image_and_fixed_seed_freeze"
                and stages[-1].get("passed") == expected,
                detail={
                    "frozen_sample_count": payload.get("frozen_sample_count"),
                    "last_stage": stages[-1] if stages else None,
                },
            )
            marginal = payload.get("gate_marginal_counts") or {}
            required_gates = {
                "thinking_closed",
                "generated_teacher_forced_ids_match",
                "generation_stable",
                "positive_blur_logprob_drop",
                "generation_ablation_changes_output",
                benchmark_gate,
            }
            if benchmark_gate == "output_non_refusal":
                required_gates.update(
                    {"thinking_within_token_limit", "output_meets_min_tokens"}
                )
            audit.add(
                f"gate funnel has typed marginal counts: {funnel}",
                required_gates.issubset(marginal)
                and all(
                    sum(int(value) for value in marginal[gate].values())
                    == payload["candidate_count"]
                    - payload.get("excluded_prior_pilot_count", 0)
                    for gate in required_gates
                ),
                detail={"missing": sorted(required_gates - set(marginal))},
            )
            sequential_gate_stages = [
                stage
                for stage in stages
                if "not_evaluated_at_stage" in stage
            ]
            audit.add(
                f"gate funnel fully evaluates every reached gate: {funnel}",
                bool(sequential_gate_stages)
                and all(
                    int(stage["not_evaluated_at_stage"]) == 0
                    for stage in sequential_gate_stages
                ),
                detail={
                    str(stage.get("stage")): int(
                        stage.get("not_evaluated_at_stage", 0)
                    )
                    for stage in sequential_gate_stages
                },
            )

    wiki_replay_issues, wiki_replay_missing = _selection_replay_issues(
        formal_dir=formal_dir,
        prefix="wiki_visa_n120",
        dataset_paths=[
            formal_dir / "wiki_visa_candidates.dataset.jsonl",
            formal_dir
            / "wiki_visa_candidates_extension_seed31_n600.dataset.jsonl",
            formal_dir
            / (
                "wiki_visa_candidates_later_extension_seed47_n600_"
                "prefix_n100.dataset.jsonl"
            ),
        ],
        model_paths=[
            formal_dir / "wiki_visa_candidates.model.jsonl",
            formal_dir
            / "wiki_visa_candidates_extension_seed31_n600.model.jsonl",
            formal_dir
            / "wiki_visa_candidates_later_extension_seed47_n600.model.jsonl",
        ],
        evaluation_paths=[
            formal_dir
            / "wiki_visa_candidates.strict.generation_eval.jsonl",
            formal_dir
            / (
                "wiki_visa_candidates_extension_seed31_n600."
                "strict.generation_eval.jsonl"
            ),
            formal_dir
            / (
                "wiki_visa_candidates_later_extension_seed47_n600."
                "strict.generation_eval.jsonl"
            ),
        ],
        exclusion_paths=[
            root
            / (
                "evaluations/multimodal/results/strict/final/"
                "wiki_visa_n18_2mp.dataset.jsonl"
            ),
            root
            / (
                "evaluations/multimodal/results/strict/native_pilot/"
                "wiki_visa_n10.dataset.jsonl"
            ),
        ],
        sample_size=120,
        balance_key="stratum",
        funnel_path=formal_dir / "wiki_visa_funnel.json",
    )
    audit.add(
        "E1 Wiki: fixed-seed selection and funnel exactly replay candidates",
        not wiki_replay_issues and not wiki_replay_missing,
        status="incomplete" if wiki_replay_missing else "error",
        detail={
            "issues": wiki_replay_issues,
            "missing": wiki_replay_missing,
        },
    )
    viz_replay_issues, viz_replay_missing = _selection_replay_issues(
        formal_dir=formal_dir,
        prefix="vizwiz_lf_n100",
        dataset_paths=[formal_dir / "vizwiz_lf_candidates.dataset.jsonl"],
        model_paths=[formal_dir / "vizwiz_lf_candidates.model.jsonl"],
        evaluation_paths=[
            formal_dir / "vizwiz_lf_candidates.strict.generation_eval.jsonl"
        ],
        exclusion_paths=[
            root
            / (
                "evaluations/multimodal/results/strict/native_pilot/"
                "vizwiz_lf_n10.dataset.jsonl"
            )
        ],
        sample_size=100,
        balance_key=None,
        funnel_path=formal_dir / "vizwiz_lf_funnel.json",
    )
    audit.add(
        "E2 VizWiz: fixed-seed selection and funnel exactly replay candidates",
        not viz_replay_issues and not viz_replay_missing,
        status="incomplete" if viz_replay_missing else "error",
        detail={
            "issues": viz_replay_issues,
            "missing": viz_replay_missing,
        },
    )

    wiki_attribution = formal_dir / "wiki_visa_n120_methods"
    viz_attribution = formal_dir / "vizwiz_lf_n100_methods"
    _audit_attribution(
        audit,
        root,
        wiki_attribution,
        label="E3 Wiki",
        expected_ids=wiki_ids,
        localization_required=True,
    )
    _audit_attribution(
        audit,
        root,
        viz_attribution,
        label="E4 VizWiz maps",
        expected_ids=viz_ids,
        localization_required=False,
    )
    _audit_faithfulness(
        audit,
        root,
        formal_dir / "vizwiz_lf_n100_faithfulness",
        label="E4 VizWiz",
        expected_ids=viz_ids,
    )
    _audit_faithfulness(
        audit,
        root,
        formal_dir / "wiki_visa_n120_faithfulness",
        label="E5 Wiki",
        expected_ids=wiki_ids,
    )
    _audit_manual_protocol_review(
        audit,
        root,
        formal_dir,
        prefix="wiki_visa_n120",
        expected_ids=wiki_ids,
        expected_reviewed=12,
    )
    _audit_manual_protocol_review(
        audit,
        root,
        formal_dir,
        prefix="vizwiz_lf_n100",
        expected_ids=viz_ids,
        expected_reviewed=10,
    )
    _audit_analysis_payloads(
        audit,
        formal_dir,
        wiki_ids=wiki_ids,
        viz_ids=viz_ids,
    )
    ablation_reuse_path = (
        formal_dir / "vizwiz_lf_candidates.preview_ablation_reuse.json"
    )
    if audit.require_file(
        "E2 VizWiz: response-identical preview ablation provenance",
        ablation_reuse_path,
    ):
        reuse = _read_json(ablation_reuse_path)
        source_path = (
            root
            / "evaluations/multimodal/results/strict/formal_preview_n20/"
            "vizwiz_lf_candidates_n40.ablation.model.jsonl"
        )
        formal_ablation_path = (
            formal_dir / "vizwiz_lf_candidates.ablation.model.jsonl"
        )
        source_hashes = reuse.get("source_sha256") or {}
        recorded_source_hash = source_hashes.get(
            str(
                Path(
                    "evaluations/multimodal/results/strict/formal_preview_n20/"
                    "vizwiz_lf_candidates_n40.ablation.model.jsonl"
                )
            )
        )
        provenance_ids: set[str] = set()
        provenance_hashes: set[str] = set()
        if formal_ablation_path.is_file():
            for record in read_jsonl(formal_ablation_path):
                provenance = record.get("checkpoint_provenance") or {}
                if (
                    provenance.get("kind")
                    == "response_identical_formal_preview_candidate_ablation"
                ):
                    provenance_ids.add(str(record["sample_id"]))
                    provenance_hashes.add(
                        str(provenance.get("source_sha256"))
                    )
        reused_ids = set(
            str(value) for value in reuse.get("reused_sample_ids") or []
        )
        mismatched = set(
            str(value)
            for value in reuse.get("identity_mismatched_sample_ids") or []
        )
        overlap = int(reuse.get("overlap_candidates", -1))
        matched = int(reuse.get("identity_matched_candidates", -1))
        reusable = int(
            reuse.get("reusable_complete_ablation_records", -1)
        )
        reused = int(reuse.get("reused_ablation_records", -1))
        audit.add(
            "E2 VizWiz: preview ablation identity and record accounting",
            reuse.get("policy")
            == "reuse_only_response_token_model_and_ablation_config_identical"
            and matched + len(mismatched) == overlap
            and reused == reusable == len(reused_ids)
            and provenance_ids == reused_ids
            and source_path.is_file()
            and recorded_source_hash == _sha256(source_path)
            and provenance_hashes == {recorded_source_hash},
            detail={
                "overlap": overlap,
                "identity_matched": matched,
                "identity_mismatched": len(mismatched),
                "reusable": reusable,
                "reused": reused,
                "provenance_records": len(provenance_ids),
            },
        )
    for prefix, expected_ids in (
        ("wiki_visa_n120", wiki_ids),
        ("vizwiz_lf_n100", viz_ids),
    ):
        reuse_path = formal_dir / f"{prefix}.preview_reuse.json"
        if not audit.require_file(
            f"{prefix}: response-identical preview checkpoint provenance",
            reuse_path,
        ):
            continue
        reuse = _read_json(reuse_path)
        matched = set(str(value) for value in reuse.get("matched_sample_ids") or [])
        expected_pairs = len(matched) * len(EXPECTED_METHODS)
        audit.add(
            f"{prefix}: preview reuse identity and pair accounting",
            reuse.get("policy")
            == "reuse_only_response_identical_deterministic_gpu_records"
            and reuse.get("identity_mismatched_sample_ids") == []
            and reuse.get("identity_matched_samples") == len(matched)
            and matched.issubset(expected_ids)
            and reuse.get("reused_attribution_pairs") == expected_pairs
            and reuse.get("reused_faithfulness_pairs") == expected_pairs,
            detail={
                "matched_samples": len(matched),
                "expected_pairs": expected_pairs,
                "attribution_pairs": reuse.get("reused_attribution_pairs"),
                "faithfulness_pairs": reuse.get("reused_faithfulness_pairs"),
                "mismatched": reuse.get("identity_mismatched_sample_ids"),
            },
        )

    required_analysis = (
        wiki_attribution / "analysis.json",
        wiki_attribution / "diagnostics.json",
        viz_attribution / "diagnostics.json",
        formal_dir / "vizwiz_lf_n100_faithfulness/analysis.json",
        formal_dir / "wiki_visa_n120_faithfulness/analysis.json",
        formal_dir / "vizwiz_lf_n100.semantic_summary.json",
        formal_dir / "A6_LEGACY_DIAGNOSTICS.json",
        formal_dir / "RESULTS.md",
        formal_dir / "COMPUTE_BUDGET.md",
    )
    for path in required_analysis:
        audit.require_file(f"analysis/report: {path.name}", path)
    for path in (
        root / "paper/generated/visual_localization_rows.tex",
        root / "paper/generated/visual_faithfulness_rows.tex",
        root / "paper/generated/visual_appendix_results.tex",
        root / "paper/generated/visual_results_discussion.tex",
    ):
        if audit.require_file(f"paper table: {path.name}", path):
            text = path.read_text(encoding="utf-8")
            audit.add(
                f"paper table populated: {path.name}",
                r"\ph" not in text
                and r"\placeholder" not in text
                and "PLACEHOLDER" not in text,
                status="incomplete",
            )

    main_path = root / "paper/main.tex"
    if audit.require_file("paper integration: main.tex", main_path):
        main_text = main_path.read_text(encoding="utf-8")
        normalized_main = " ".join(main_text.split())
        audit.add(
            "paper integration: frozen visual protocol in main text",
            all(
                " ".join(fragment.split()) in normalized_main
                for fragment in (
                    EXPECTED_REVISION,
                    "whole visual patches",
                    "64 image regions",
                    "10 deletion/insertion steps",
                    "complete funnel",
                )
            )
            and (
                "2,007,040 pixels" in normalized_main
                or "2{,}007{,}040 pixels" in normalized_main
            )
            and (
                r"\input{generated/visual_results_discussion}" in normalized_main
                or (
                    "Visual LOO obtains the best mean" in normalized_main
                    and "same eight methods" in normalized_main
                )
            )
            and (
                r"\input{generated/visual_localization_rows}" in normalized_main
                or r"\primitiveinput generated/visual_localization_rows.tex"
                in normalized_main
            )
            and (
                r"\input{generated/visual_faithfulness_rows}" in normalized_main
                or r"\primitiveinput generated/visual_faithfulness_rows.tex"
                in normalized_main
            ),
        )
        audit.add(
            "paper integration: visual limitations preserve frozen scope",
            all(
                " ".join(fragment.split()) in normalized_main
                for fragment in (
                    "supporting HTML elements rather than exhaustive",
                    "prompted long-form answers",
                    "non-representative subset",
                    "Center prior",
                    "one recursive",
                    "one frozen Qwen3-VL revision",
                    "cross-model, cross-resolution",
                    "multi-hop-recursion generalization",
                )
            ),
        )

    appendix_path = root / "paper/appendix.tex"
    if audit.require_file("paper integration: appendix.tex", appendix_path):
        appendix_text = appendix_path.read_text(encoding="utf-8")
        normalized_appendix = " ".join(appendix_text.split())
        audit.add(
            "paper integration: retained diagnostics and formal appendix hook",
            all(
                " ".join(fragment.split()) in normalized_appendix
                for fragment in (
                    r"\input{generated/visual_appendix_results}",
                    "CLEVR-XAI uses the already frozen strict $n=20$ set",
                    "VISTAQA is likewise restricted to its existing native $n=10$",
                    "50,000 paired nonparametric resamples",
                    "approximately 64 aspect-ratio-preserving regions",
                    "ten cumulative fractions",
                    "Visual failure modes",
                    "2,007,040-pixel Wiki configuration",
                )
            ),
        )

    generated_requirements = {
        "visual_localization_rows.tex": (
            "Random",
            "Center prior",
            "Visual LOO",
            "Visual IG",
            "AttnLRP",
            r"\flashtrace{} (exact",
            "IFR-span",
            "all-generation",
        ),
        "visual_faithfulness_rows.tex": (
            "Random",
            "Center prior",
            "Visual LOO",
            "Visual IG",
            "AttnLRP",
            r"\flashtrace{} (exact",
            "IFR-span",
            "all-generation",
        ),
        "visual_appendix_results.tex": (
            "tab:visual_gate_funnels",
            "tab:visual_wiki_localization_supplemental",
            "tab:visual_wiki_primary_deltas",
            "tab:visual_spatial_resolution",
            "tab:visual_wiki_strata",
            "tab:visual_faithfulness_full",
            "tab:visual_faithfulness_deltas",
            "tab:visual_vizwiz_fully_correct",
            "tab:visual_vizwiz_fully_correct_deltas",
            "tab:visual_recursion_geometry",
            "tab:visual_geometry_bias",
            "tab:visual_gt_centroid_bias",
            "tab:visual_recursion_buckets",
            "tab:visual_observed_compute",
            r"\paragraph{Owner-approved protocol audit.}",
        ),
        "visual_results_discussion.tex": (
            "separates concentration from coverage",
            "On Wiki-VISA frozen-response faithfulness",
            "complete VizWiz-LF faithfulness panel",
            "Against Center prior",
            "Visual LOO obtained",
            "no across-metric winner is claimed",
        ),
    }
    for name, fragments in generated_requirements.items():
        path = root / "paper/generated" / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        populated = (
            r"\ph" not in text
            and r"\placeholder" not in text
            and "PLACEHOLDER" not in text
        )
        if populated:
            audit.add(
                f"paper generated coverage: {name}",
                all(fragment in text for fragment in fragments),
            )
            table_rows = [
                line
                for line in text.splitlines()
                if line.rstrip().endswith(r"\\")
            ]
            if name == "visual_localization_rows.tex":
                audit.add(
                    "paper main localization table has eight four-metric rows",
                    len(table_rows) == 8
                    and all(row.count("&") == 4 for row in table_rows)
                    and text.count("{") == text.count("}"),
                )
            if name == "visual_faithfulness_rows.tex":
                audit.add(
                    "paper main faithfulness table has eight three-metric rows",
                    len(table_rows) == 8
                    and all(row.count("&") == 3 for row in table_rows)
                    and text.count("{") == text.count("}"),
                )

    localization_analysis_path = wiki_attribution / "analysis.json"
    viz_faith_analysis_path = (
        formal_dir / "vizwiz_lf_n100_faithfulness/analysis.json"
    )
    wiki_faith_analysis_path = (
        formal_dir / "wiki_visa_n120_faithfulness/analysis.json"
    )
    semantic_summary_path = formal_dir / "vizwiz_lf_n100.semantic_summary.json"
    if (
        localization_analysis_path.is_file()
        and viz_faith_analysis_path.is_file()
        and wiki_faith_analysis_path.is_file()
        and semantic_summary_path.is_file()
    ):
        localization = _read_json(localization_analysis_path)
        viz_faith = _read_json(viz_faith_analysis_path)
        wiki_faith = _read_json(wiki_faith_analysis_path)
        semantic_summary = _read_json(semantic_summary_path)
        localization_rows = []
        for method in RENDER_METHODS:
            prefix = r"\rowcolor{cyan!10} " if method == "flashtrace" else ""
            cells = [
                _latex_cell(localization["estimates"][metric][method])
                for metric in (
                    "energy_in_mask",
                    "evidence_rank_auc",
                    "recovery_at_5pct",
                    "recovery_at_20pct",
                )
            ]
            localization_rows.append(
                prefix
                + LATEX_LABELS[method]
                + " & "
                + " & ".join(cells)
                + r" \\"
            )
            if method == "flashtrace":
                localization_rows.append(r"\midrule")
        faithfulness_rows = []
        for method in RENDER_METHODS:
            prefix = r"\rowcolor{cyan!10} " if method == "flashtrace" else ""
            estimates = viz_faith["overall"]["estimates"][method]
            cells = [
                _latex_cell(estimates[metric])
                for metric in ("deletion_auc", "insertion_auc", "visual_mas")
            ]
            faithfulness_rows.append(
                prefix
                + LATEX_LABELS[method]
                + " & "
                + " & ".join(cells)
                + r" \\"
            )
            if method == "flashtrace":
                faithfulness_rows.append(r"\midrule")
        expected_primary_artifacts = {
            "visual_localization_rows.tex": "\n".join(localization_rows) + "\n",
            "visual_faithfulness_rows.tex": "\n".join(faithfulness_rows) + "\n",
            "visual_results_discussion.tex": _visual_discussion_tex(
                localization,
                viz_faith,
                wiki_faith,
                semantic_summary,
            ),
        }
        mismatched_primary = [
            name
            for name, expected_text in expected_primary_artifacts.items()
            if not (root / "paper/generated" / name).is_file()
            or (root / "paper/generated" / name).read_text(encoding="utf-8")
            != expected_text
        ]
        audit.add(
            "paper generated primary artifacts exactly match analysis payloads",
            not mismatched_primary,
            detail={"mismatched": mismatched_primary},
        )

    results_path = formal_dir / "RESULTS.md"
    if results_path.is_file():
        results_text = results_path.read_text(encoding="utf-8")
        audit.add(
            "RESULTS: complete E1–E5/A1–A8 and disclosure sections",
            all(
                fragment in results_text
                for fragment in (
                    "## E1/E2 gate funnels",
                    "## E3: Wiki-VISA localization",
                    "## E4: VizWiz-LF frozen-response faithfulness",
                    "## E5: Wiki-VISA frozen-response faithfulness",
                    "## A1–A4: recursion and geometry diagnostics",
                    "## A8: VizWiz semantic correctness sensitivity",
                    "## Observed visual compute",
                    "## Spatial resolution disclosure",
                    "## Owner-approved frozen-sample protocol audits",
                    "## Scope and limitations",
                )
            )
            and all(
                method_label in results_text
                for method_label in (
                    "Random",
                    "Center prior",
                    "Visual LOO",
                    "Visual IG",
                    "AttnLRP",
                    "FlashTrace (exact, K=1)",
                    "IFR-span (K=0)",
                    "FlashTrace all-generation",
                )
            )
            and "paired bootstrap draws: 50000" in results_text,
        )

    complete_render_inputs = (
        formal_dir / "wiki_visa_n120_methods/analysis.json",
        formal_dir / "wiki_visa_n120_methods/diagnostics.json",
        formal_dir / "vizwiz_lf_n100_methods/diagnostics.json",
        formal_dir / "vizwiz_lf_n100_faithfulness/analysis.json",
        formal_dir / "wiki_visa_n120_faithfulness/analysis.json",
        formal_dir / "wiki_visa_n120_faithfulness/summary.json",
        formal_dir / "vizwiz_lf_n100.semantic_summary.json",
        formal_dir / "wiki_visa_n120.protocol_audit_summary.json",
        formal_dir / "vizwiz_lf_n100.protocol_audit_summary.json",
        formal_dir / "RESULTS.md",
    )
    if all(path.is_file() for path in complete_render_inputs):
        generated_paths = (
            results_path,
            root / "paper/generated/visual_localization_rows.tex",
            root / "paper/generated/visual_faithfulness_rows.tex",
            root / "paper/generated/visual_appendix_results.tex",
            root / "paper/generated/visual_results_discussion.tex",
        )
        try:
            expected_render = render_formal_results(formal_dir)
            mismatched_render = [
                str(path.relative_to(root))
                for path, expected_text in zip(
                    generated_paths, expected_render, strict=True
                )
                if not path.is_file()
                or path.read_text(encoding="utf-8") != expected_text
            ]
        except (KeyError, TypeError, ValueError) as error:
            mismatched_render = [f"renderer failed: {error}"]
        audit.add(
            "RESULTS and all paper fragments exactly match deterministic renderer",
            not mismatched_render,
            detail={"mismatched": mismatched_render},
        )

    _audit_semantic_provenance(
        audit,
        formal_dir,
        expected_ids=viz_ids,
    )
    semantic_path = formal_dir / "vizwiz_lf_n100.semantic_summary.json"
    if semantic_path.is_file():
        semantic = _read_json(semantic_path)
        audit.add(
            "A8: semantic labels and 10% review complete",
            semantic.get("complete") is True
            and semantic.get("eligible_samples") == 100
            and semantic.get("audit_reviewed", 0) >= 10,
            detail=semantic,
        )
    legacy_path = formal_dir / "A6_LEGACY_DIAGNOSTICS.json"
    if legacy_path.is_file():
        legacy = _read_json(legacy_path)
        audit.add(
            "A6: legacy diagnostics reused without new GPU inference",
            legacy.get("analysis_id") == "A6"
            and legacy.get("new_gpu_inference") is False
            and (legacy.get("clevr_xai") or {}).get("sample_count") == 20
            and (legacy.get("clevr_xai") or {})
            .get("mask_conventions", {})
            .get("primary")
            == "unique_first_nonempty"
            and (legacy.get("clevr_xai") or {})
            .get("mask_conventions", {})
            .get("sensitivity")
            == "union"
            and (legacy.get("vistaqa") or {}).get("manifest_sample_count") == 10,
        )
    compute_budget_path = formal_dir / "COMPUTE_BUDGET.md"
    if compute_budget_path.is_file():
        compute_budget = compute_budget_path.read_text(encoding="utf-8")
        audit.add(
            "A7: measured preview timing and A100 schedule documented",
            "pilot-disjoint formal-pipeline previews" in compute_budget
            and "64-region/10-step" in compute_budget
            and "NVIDIA A100 80GB PCIe" in compute_budget
            and "23.8-31.0" in compute_budget,
        )
    for path in required_analysis:
        if path.name != "analysis.json" or not path.is_file():
            continue
        analysis = _read_json(path)
        audit.add(
            f"statistics: 50k paired bootstrap ({path.parent.name})",
            int(analysis.get("bootstrap_draws", 0)) >= 50_000,
            detail=analysis.get("bootstrap_draws"),
        )
    report = audit.report()
    core_artifacts = [
        path
        for path in formal_dir.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".jsonl", ".md"}
        and path.name != "AUDIT.json"
        and not any(
            part.startswith(".") and part.endswith(".journal")
            for part in path.relative_to(formal_dir).parts
        )
        and "diagnostics/" not in path.relative_to(formal_dir).as_posix()
    ]
    core_artifacts.extend(
        [
            protocol_path,
            root / "paper/main.tex",
            root / "paper/appendix.tex",
            root / "paper/generated/visual_localization_rows.tex",
            root / "paper/generated/visual_faithfulness_rows.tex",
            root / "paper/generated/visual_appendix_results.tex",
            root / "paper/generated/visual_results_discussion.tex",
        ]
    )
    report["artifact_manifest"] = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(set(core_artifacts))
        if path.is_file()
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_formal(args.root.resolve())
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
