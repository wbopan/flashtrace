"""Join candidate bundles and materialize a balanced strict-eligible subset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl, write_jsonl


def select(
    dataset_paths: list[Path],
    model_paths: list[Path],
    evaluation_paths: list[Path],
    *,
    sample_size: int,
    balance_key: str,
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
    for sample_id in order:
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
        group = str(dataset["evaluation"]["metadata"][balance_key])
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
    selected: list[str] = []
    for index, group in enumerate(group_names):
        count = base + int(index < remainder)
        if len(groups[group]) < count:
            raise ValueError(
                f"group {group!r} has {len(groups[group])} eligible samples; needs {count}"
            )
        selected.extend(groups[group][:count])
    selected.sort()
    return (
        [datasets[sample_id] for sample_id in selected],
        [models[sample_id] for sample_id in selected],
        [evaluations[sample_id] for sample_id in selected],
    )


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
    parser.add_argument("--balance-key", required=True)
    args = parser.parse_args()
    datasets, models, evaluations = select(
        args.dataset_manifest,
        args.model_output,
        args.generation_evaluation,
        sample_size=args.sample_size,
        balance_key=args.balance_key,
    )
    write_jsonl(datasets, args.output_dataset)
    write_jsonl(models, args.output_model)
    write_jsonl(evaluations, args.output_evaluation)
    counts: dict[str, int] = defaultdict(int)
    for record in datasets:
        counts[str(record["evaluation"]["metadata"][args.balance_key])] += 1
    print(
        json.dumps(
            {
                "records": len(datasets),
                "groups": dict(counts),
                "output_dataset": str(args.output_dataset),
                "output_model": str(args.output_model),
                "output_evaluation": str(args.output_evaluation),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
