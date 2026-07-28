"""Recompute result-independent generation gates from saved strict records.

This is used when a gate implementation bug is fixed after generation. It
does not generate text, recompute log-probabilities, or alter frozen outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .strict_generation import (
    pre_ablation_gate,
    read_jsonl,
    write_jsonl,
)


def refresh(
    dataset_manifest: Path,
    model_output: Path,
    evaluation_output: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    datasets = {
        str(row["sample_id"]): row for row in read_jsonl(dataset_manifest)
    }
    models = {str(row["sample_id"]): row for row in read_jsonl(model_output)}
    evaluations = read_jsonl(evaluation_output)
    changed: list[str] = []
    refreshed: list[dict[str, Any]] = []
    for original in evaluations:
        row = dict(original)
        sample_id = str(row["sample_id"])
        model = models.get(sample_id)
        dataset = datasets.get(sample_id)
        if model is None or dataset is None or row.get("error") is not None:
            refreshed.append(row)
            continue
        metadata = model["generation_metadata"]
        gate = pre_ablation_gate(
            benchmark=str(dataset["benchmark"]),
            output=str(model["OUTPUT"]),
            output_correct_value=bool(row.get("reference_exact_match")),
            generation_stable=bool(row.get("generation_stable")),
            image_dependence_delta=float(row["image_dependence_delta"]),
            token_identity_stable=bool(
                row.get("generated_teacher_forced_ids_match")
            ),
            thinking_tokens=int(metadata["thinking_tokens"]),
            output_tokens=int(metadata["output_tokens"]),
        )
        old_gate = {
            "correctness_gate_required": row.get("correctness_gate_required"),
            "gates": row.get("gates"),
            "pre_ablation_eligible": row.get("pre_ablation_eligible"),
            "strict_eligible": row.get("strict_eligible"),
        }
        row.update(gate)
        row["strict_eligible"] = gate["pre_ablation_eligible"]
        new_gate = {
            "correctness_gate_required": row.get("correctness_gate_required"),
            "gates": row.get("gates"),
            "pre_ablation_eligible": row.get("pre_ablation_eligible"),
            "strict_eligible": row.get("strict_eligible"),
        }
        if old_gate != new_gate:
            changed.append(sample_id)
        refreshed.append(row)
    return refreshed, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--evaluation-output", type=Path, required=True)
    args = parser.parse_args()
    refreshed, changed = refresh(
        args.dataset_manifest, args.model_output, args.evaluation_output
    )
    write_jsonl(refreshed, args.evaluation_output)
    print(
        json.dumps(
            {
                "records": len(refreshed),
                "changed_count": len(changed),
                "changed_sample_ids": changed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
