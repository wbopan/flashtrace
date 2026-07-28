"""Join candidate bundles and freeze a deterministic strict-eligible subset."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl, write_jsonl


def select(
    dataset_paths: list[Path],
    model_paths: list[Path],
    evaluation_paths: list[Path],
    *,
    sample_size: int,
    balance_key: str | None,
    seed: int = 17,
    exclude_sample_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not (len(dataset_paths) == len(model_paths) == len(evaluation_paths)):
        raise ValueError("dataset/model/evaluation path counts must match")
    datasets: dict[str, dict[str, Any]] = {}
    models: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for dataset_path, model_path, evaluation_path in zip(
        dataset_paths, model_paths, evaluation_paths, strict=True
    ):
        for record in read_jsonl(dataset_path):
            sample_id = record["sample_id"]
            if sample_id not in datasets:
                order.append(sample_id)
            datasets[sample_id] = record
        models.update({record["sample_id"]: record for record in read_jsonl(model_path)})
        evaluations.update(
            {record["sample_id"]: record for record in read_jsonl(evaluation_path)}
        )

    groups: dict[str, list[str]] = defaultdict(list)
    used_images: set[str] = set()
    excluded = exclude_sample_ids or set()
    for sample_id in order:
        if sample_id in excluded:
            continue
        evaluation = evaluations.get(sample_id, {})
        if (
            sample_id not in models
            or not evaluation.get("strict_eligible")
            or not evaluation.get("image_dependent_by_generation_ablation")
        ):
            continue
        generation_metadata = models[sample_id].get("generation_metadata", {})
        if generation_metadata.get(
            "original_generated_token_ids"
        ) != generation_metadata.get("teacher_forced_token_ids"):
            continue
        dataset = datasets[sample_id]
        group = (
            str(dataset["evaluation"]["metadata"][balance_key])
            if balance_key is not None
            else "__all__"
        )
        image_identity = str(
            dataset["evaluation"]["metadata"].get(
                "image_index", dataset["input"]["I_IMAGE"]
            )
        )
        if image_identity in used_images:
            continue
        groups[group].append(sample_id)
        used_images.add(image_identity)

    group_names = sorted(groups)
    if not group_names:
        raise ValueError("no strict-eligible groups found")
    base, remainder = divmod(sample_size, len(group_names))
    rng = random.Random(seed)
    selected: list[str] = []
    for index, group in enumerate(group_names):
        count = base + int(index < remainder)
        if len(groups[group]) < count:
            raise ValueError(
                f"group {group!r} has {len(groups[group])} eligible samples; needs {count}"
            )
        candidates = list(groups[group])
        rng.shuffle(candidates)
        selected.extend(candidates[:count])
    selected.sort()
    selected_datasets = [datasets[sample_id] for sample_id in selected]
    selected_models = [models[sample_id] for sample_id in selected]
    if selected_datasets and selected_datasets[0]["benchmark"] == "vizwiz_lf":
        terciles, cutpoints = _output_terciles(selected, models)
        for record in selected_datasets:
            metadata = record["evaluation"]["metadata"]
            metadata["output_length_tercile"] = terciles[record["sample_id"]]
            metadata["output_token_tercile_cutpoints"] = cutpoints
    return (
        selected_datasets,
        selected_models,
        [evaluations[sample_id] for sample_id in selected],
    )


def gate_funnel(
    datasets: Mapping[str, Mapping[str, Any]],
    models: Mapping[str, Mapping[str, Any]],
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    exclude_sample_ids: set[str] | None = None,
    frozen_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Summarize the preregistered gates without attribution information."""

    observed_gate_names: list[str] = []
    for sample_id in datasets:
        for gate_name in (evaluations.get(sample_id, {}).get("gates") or {}):
            if gate_name not in observed_gate_names:
                observed_gate_names.append(str(gate_name))
    preferred_order = (
        "thinking_closed",
        "generated_teacher_forced_ids_match",
        "thinking_within_token_limit",
        "output_meets_min_tokens",
        "output_non_refusal",
        "generation_stable",
        "whole_output_correct",
        "positive_blur_logprob_drop",
        "generation_ablation_changes_output",
    )
    gate_names = [
        gate_name
        for gate_name in preferred_order
        if gate_name in observed_gate_names
        or gate_name
        in {"thinking_closed", "generated_teacher_forced_ids_match"}
    ]
    gate_names.extend(
        gate_name
        for gate_name in observed_gate_names
        if gate_name not in gate_names
    )

    def gate_status(sample_id: str, gate_name: str) -> bool | None:
        evaluation = evaluations.get(sample_id, {})
        gates = evaluation.get("gates") or {}
        if gate_name in gates:
            return bool(gates[gate_name])
        error = str(evaluation.get("error") or "")
        if gate_name == "thinking_closed":
            if "no </think> terminator" in error or "empty THINKING" in error:
                return False
            if sample_id in models or "token IDs differ" in error:
                return True
        if gate_name == "generated_teacher_forced_ids_match":
            if "token IDs differ" in error:
                return False
            model = models.get(sample_id)
            if model is not None:
                metadata = model.get("generation_metadata") or {}
                return metadata.get(
                    "original_generated_token_ids"
                ) == metadata.get("teacher_forced_token_ids")
        return None

    cumulative_ids = set(datasets)
    stages = [
        {
            "stage": "candidate_manifest",
            "passed": len(cumulative_ids),
            "eliminated_at_stage": 0,
        }
    ]
    excluded = cumulative_ids.intersection(exclude_sample_ids or set())
    if exclude_sample_ids is not None:
        cumulative_ids -= excluded
        stages.append(
            {
                "stage": "prior_pilot_sample_exclusion",
                "passed": len(cumulative_ids),
                "eliminated_at_stage": len(excluded),
            }
        )
    for gate_name in ("thinking_closed", "generated_teacher_forced_ids_match"):
        if gate_name not in gate_names:
            continue
        evaluated = {
            sample_id
            for sample_id in cumulative_ids
            if gate_status(sample_id, gate_name) is not None
        }
        passed = {
            sample_id
            for sample_id in cumulative_ids
            if gate_status(sample_id, gate_name) is True
        }
        stages.append(
            {
                "stage": gate_name,
                "passed": len(passed),
                "eliminated_at_stage": len(cumulative_ids - passed),
                "not_evaluated_at_stage": len(cumulative_ids - evaluated),
            }
        )
        cumulative_ids = passed
    model_ids = cumulative_ids.intersection(models)
    stages.append(
        {
            "stage": "model_record_available",
            "passed": len(model_ids),
            "eliminated_at_stage": len(cumulative_ids - model_ids),
        }
    )
    cumulative_ids = model_ids
    for gate_name in gate_names:
        if gate_name in {"thinking_closed", "generated_teacher_forced_ids_match"}:
            continue
        evaluated = {
            sample_id
            for sample_id in cumulative_ids
            if gate_status(sample_id, gate_name) is not None
        }
        passed = {
            sample_id
            for sample_id in cumulative_ids
            if gate_status(sample_id, gate_name) is True
        }
        stages.append(
            {
                "stage": gate_name,
                "passed": len(passed),
                "eliminated_at_stage": len(cumulative_ids - passed),
                "not_evaluated_at_stage": len(cumulative_ids - evaluated),
            }
        )
        cumulative_ids = passed
    final_ids = {
        sample_id
        for sample_id in datasets
        if sample_id not in excluded
        and bool(evaluations.get(sample_id, {}).get("strict_eligible"))
    }
    stages.append(
        {
            "stage": "final_strict_eligible",
            "passed": len(final_ids),
            "eliminated_at_stage": len(cumulative_ids - final_ids),
        }
    )
    frozen_ids: set[str] | None = None
    if frozen_sample_ids is not None:
        frozen_ids = set(frozen_sample_ids)
        if not frozen_ids.issubset(final_ids):
            raise ValueError(
                "frozen sample IDs are not a subset of final strict eligibility: "
                f"{sorted(frozen_ids - final_ids)}"
            )
        stages.append(
            {
                "stage": "unique_image_and_fixed_seed_freeze",
                "passed": len(frozen_ids),
                "eliminated_at_stage": len(final_ids - frozen_ids),
            }
        )
    return {
        "candidate_count": len(datasets),
        "model_record_count": len(models),
        "evaluation_record_count": len(evaluations),
        "strict_eligible_count": len(final_ids),
        "frozen_sample_count": len(frozen_ids) if frozen_ids is not None else None,
        "excluded_prior_pilot_count": len(excluded),
        "gate_marginal_counts": {
            gate_name: {
                "passed": sum(
                    gate_status(sample_id, gate_name) is True
                    for sample_id in set(datasets) - excluded
                ),
                "failed": sum(
                    gate_status(sample_id, gate_name) is False
                    for sample_id in set(datasets) - excluded
                ),
                "not_evaluated": sum(
                    gate_status(sample_id, gate_name) is None
                    for sample_id in set(datasets) - excluded
                ),
            }
            for gate_name in gate_names
        },
        "stages": stages,
    }


def _output_terciles(
    selected_ids: list[str], models: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], list[int]]:
    lengths = sorted(
        int(models[sample_id]["generation_metadata"]["output_tokens"])
        for sample_id in selected_ids
    )
    if not lengths:
        return {}, []
    lower = lengths[(len(lengths) - 1) // 3]
    upper = lengths[(2 * (len(lengths) - 1)) // 3]
    labels = {}
    for sample_id in selected_ids:
        length = int(models[sample_id]["generation_metadata"]["output_tokens"])
        labels[sample_id] = (
            "short" if length <= lower else "medium" if length <= upper else "long"
        )
    return labels, [lower, upper]


def update_frozen_ids(
    path: Path,
    datasets: list[dict[str, Any]],
    models: list[dict[str, Any]],
    *,
    balance_key: str | None,
    seed: int,
) -> None:
    """Merge one frozen formal dataset selection into the shared ID artifact."""

    payload: dict[str, Any]
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {
            "schema_version": 1,
            "frozen_on": "2026-07-24",
            "selection_seed": seed,
            "datasets": {},
        }
    if int(payload.get("selection_seed", seed)) != seed:
        raise ValueError(
            f"{path} is already frozen with seed {payload.get('selection_seed')}"
        )
    model_by_id = {record["sample_id"]: record for record in models}
    selected_ids = [record["sample_id"] for record in datasets]
    terciles, cutpoints = _output_terciles(selected_ids, model_by_id)
    benchmark = str(datasets[0]["benchmark"]) if datasets else "empty"
    entries = []
    for dataset in datasets:
        sample_id = dataset["sample_id"]
        metadata = dataset["evaluation"]["metadata"]
        entry = {
            "sample_id": sample_id,
            "balance_group": (
                str(metadata[balance_key]) if balance_key is not None else None
            ),
            "output_tokens": int(
                model_by_id[sample_id]["generation_metadata"]["output_tokens"]
            ),
        }
        if benchmark == "vizwiz_lf":
            entry["question_type"] = str(metadata["question_type"])
            entry["output_length_tercile"] = terciles[sample_id]
        entries.append(entry)
    frozen_dataset = {
        "count": len(entries),
        "balance_key": balance_key,
        "selection_mode": (
            "balanced_fixed_seed"
            if balance_key is not None
            else "unstratified_fixed_seed"
        ),
        "output_token_tercile_cutpoints": cutpoints,
        "samples": entries,
    }
    existing_dataset = payload["datasets"].get(benchmark)
    if existing_dataset is not None:
        existing_ids = [
            str(sample["sample_id"]) for sample in existing_dataset.get("samples", [])
        ]
        new_ids = [str(sample["sample_id"]) for sample in entries]
        if existing_ids != new_ids:
            raise ValueError(
                f"{path} already freezes a different {benchmark} sample set"
            )
    payload["datasets"][benchmark] = frozen_dataset
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, action="append", required=True)
    parser.add_argument("--model-output", type=Path, action="append", required=True)
    parser.add_argument(
        "--generation-evaluation", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dataset", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output-evaluation", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument(
        "--balance-key",
        help=(
            "Balance equally across this metadata field. Omit for a fixed-seed "
            "sample from the full strict-eligible pool."
        ),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Exclude prior pilot IDs from the formal frozen subset.",
    )
    parser.add_argument(
        "--frozen-ids-output",
        type=Path,
        help="Merge selected IDs into the formal frozen_ids.json artifact.",
    )
    parser.add_argument(
        "--funnel-output",
        type=Path,
        help="Write candidate-to-eligibility gate counts as JSON.",
    )
    args = parser.parse_args()
    excluded_sample_ids = {
        record["sample_id"]
        for path in args.exclude_manifest
        for record in read_jsonl(path)
    }
    datasets, models, evaluations = select(
        args.dataset_manifest,
        args.model_output,
        args.generation_evaluation,
        sample_size=args.sample_size,
        balance_key=args.balance_key,
        seed=args.seed,
        exclude_sample_ids=excluded_sample_ids,
    )
    write_jsonl(datasets, args.output_dataset)
    write_jsonl(models, args.output_model)
    write_jsonl(evaluations, args.output_evaluation)
    if args.frozen_ids_output:
        update_frozen_ids(
            args.frozen_ids_output,
            datasets,
            models,
            balance_key=args.balance_key,
            seed=args.seed,
        )
    if args.funnel_output:
        all_datasets = {
            record["sample_id"]: record
            for path in args.dataset_manifest
            for record in read_jsonl(path)
        }
        all_models = {
            record["sample_id"]: record
            for path in args.model_output
            for record in read_jsonl(path)
        }
        all_evaluations = {
            record["sample_id"]: record
            for path in args.generation_evaluation
            for record in read_jsonl(path)
        }
        args.funnel_output.parent.mkdir(parents=True, exist_ok=True)
        args.funnel_output.write_text(
            json.dumps(
                gate_funnel(
                    all_datasets,
                    all_models,
                    all_evaluations,
                    exclude_sample_ids=excluded_sample_ids,
                    frozen_sample_ids={
                        record["sample_id"] for record in datasets
                    },
                ),
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    counts: dict[str, int] = defaultdict(int)
    for record in datasets:
        group = (
            str(record["evaluation"]["metadata"][args.balance_key])
            if args.balance_key is not None
            else "__all__"
        )
        counts[group] += 1
    print(
        json.dumps(
            {
                "records": len(datasets),
                "groups": dict(counts),
                "seed": args.seed,
                "frozen_ids_output": (
                    str(args.frozen_ids_output) if args.frozen_ids_output else None
                ),
                "funnel_output": (
                    str(args.funnel_output) if args.funnel_output else None
                ),
                "output_dataset": str(args.output_dataset),
                "output_model": str(args.output_model),
                "output_evaluation": str(args.output_evaluation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
