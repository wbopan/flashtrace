"""Prepare and summarize the frozen protocol's independent manual audit.

The audit is descriptive only: reviewers assess image dependence and THINKING
quality on a deterministic 10% sample, and the result never changes frozen IDs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl, write_jsonl


IMAGE_DEPENDENCE_LABELS = {"supported", "borderline", "unsupported"}
THINKING_QUALITY_LABELS = {"good", "mixed", "poor"}


def audit_sample_ids(
    sample_ids: list[str], *, fraction: float = 0.1, seed: int = 17
) -> list[str]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("audit fraction must be in (0, 1]")
    ordered = sorted(sample_ids)
    count = min(len(ordered), max(1, math.ceil(len(ordered) * fraction)))
    rng = random.Random(seed)
    return sorted(rng.sample(ordered, count))


def prepare_audit(
    dataset_manifest: Path,
    model_output: Path,
    generation_evaluation: Path,
    ablation_model_outputs: list[Path],
    *,
    fraction: float = 0.1,
    seed: int = 17,
) -> tuple[str, list[dict[str, Any]]]:
    datasets = {row["sample_id"]: row for row in read_jsonl(dataset_manifest)}
    models = {row["sample_id"]: row for row in read_jsonl(model_output)}
    evaluations = {
        row["sample_id"]: row for row in read_jsonl(generation_evaluation)
    }
    ablations = {
        row["sample_id"]: row
        for path in ablation_model_outputs
        for row in read_jsonl(path)
    }
    common = sorted(set(datasets) & set(models) & set(evaluations) & set(ablations))
    if set(common) != set(datasets):
        raise ValueError(
            "manual audit inputs do not cover the complete frozen dataset: "
            f"dataset={len(datasets)}, common={len(common)}"
        )
    selected = audit_sample_ids(common, fraction=fraction, seed=seed)
    benchmark = str(datasets[selected[0]]["benchmark"]) if selected else "unknown"
    lines = [
        f"# {benchmark} deterministic protocol audit",
        "",
        f"Audit seed: {seed}; fraction: {fraction}; "
        f"samples: {len(selected)}/{len(common)}.",
        "",
        "This is a caveat-only review. Labels must not alter the frozen sample "
        "set. Inspect the image, question, original THINKING/OUTPUT, gate "
        "evidence, and both deterministic ablation generations.",
        "",
        "- `image_dependence`: `supported`, `borderline`, or `unsupported`",
        "- `thinking_quality`: `good`, `mixed`, or `poor`",
        "",
    ]
    template: list[dict[str, Any]] = []
    for sample_id in selected:
        dataset = datasets[sample_id]
        model = models[sample_id]
        evaluation = evaluations[sample_id]
        ablation = ablations[sample_id]
        ablation_outputs = {
            name: {
                "status": value.get("status"),
                "output": value.get("OUTPUT"),
            }
            for name, value in (ablation.get("ablations") or {}).items()
        }
        image_path = str(Path(dataset["input"]["I_IMAGE"]).resolve())
        lines.extend(
            [
                f"## {sample_id}",
                "",
                f"![{sample_id}]({image_path})",
                "",
                f"**Question:** {dataset['input']['I_QUESTION']}",
                "",
                f"**Reference:** {dataset['evaluation'].get('REFERENCE_OUTPUT')}",
                "",
                f"**THINKING:** {model.get('THINKING')}",
                "",
                f"**OUTPUT:** {model.get('OUTPUT')}",
                "",
                "**Gate evidence:** "
                + json.dumps(
                    {
                        "image_dependence_delta": evaluation.get(
                            "image_dependence_delta"
                        ),
                        "gates": evaluation.get("gates"),
                    },
                    ensure_ascii=False,
                ),
                "",
                "**Ablation generations:** "
                + json.dumps(ablation_outputs, ensure_ascii=False),
                "",
            ]
        )
        template.append(
            {
                "sample_id": sample_id,
                "image_dependence": None,
                "thinking_quality": None,
                "reviewer": None,
                "reason": None,
            }
        )
    return "\n".join(lines) + "\n", template


def summarize_reviews(
    dataset_manifest: Path,
    reviews_path: Path,
    *,
    fraction: float = 0.1,
    seed: int = 17,
) -> dict[str, Any]:
    dataset_ids = [str(row["sample_id"]) for row in read_jsonl(dataset_manifest)]
    expected = audit_sample_ids(dataset_ids, fraction=fraction, seed=seed)
    reviews = read_jsonl(reviews_path)
    by_id: dict[str, dict[str, Any]] = {}
    for row in reviews:
        sample_id = str(row.get("sample_id"))
        if sample_id in by_id:
            raise ValueError(f"duplicate manual review: {sample_id}")
        image_label = row.get("image_dependence")
        thinking_label = row.get("thinking_quality")
        if image_label not in IMAGE_DEPENDENCE_LABELS:
            raise ValueError(f"invalid image-dependence label for {sample_id}")
        if thinking_label not in THINKING_QUALITY_LABELS:
            raise ValueError(f"invalid THINKING-quality label for {sample_id}")
        if not str(row.get("reviewer") or "").strip():
            raise ValueError(f"missing reviewer for {sample_id}")
        if not str(row.get("reason") or "").strip():
            raise ValueError(f"missing review reason for {sample_id}")
        by_id[sample_id] = row
    if set(by_id) != set(expected):
        raise ValueError(
            "manual review IDs do not match deterministic audit sample: "
            f"missing={sorted(set(expected) - set(by_id))}, "
            f"extra={sorted(set(by_id) - set(expected))}"
        )
    image_counts = Counter(str(by_id[sid]["image_dependence"]) for sid in expected)
    thinking_counts = Counter(str(by_id[sid]["thinking_quality"]) for sid in expected)
    benchmark = (
        str(read_jsonl(dataset_manifest)[0]["benchmark"]) if dataset_ids else "unknown"
    )
    return {
        "schema_version": 1,
        "benchmark": benchmark,
        "complete": True,
        "audit_seed": seed,
        "audit_fraction": fraction,
        "frozen_sample_count": len(dataset_ids),
        "reviewed_count": len(expected),
        "audit_sample_ids": expected,
        "image_dependence_counts": dict(sorted(image_counts.items())),
        "thinking_quality_counts": dict(sorted(thinking_counts.items())),
        "selection_effect": "caveat_only_no_frozen_id_changes",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-manifest", type=Path, required=True)
    prepare.add_argument("--model-output", type=Path, required=True)
    prepare.add_argument("--generation-evaluation", type=Path, required=True)
    prepare.add_argument(
        "--ablation-model-output", type=Path, action="append", required=True
    )
    prepare.add_argument("--output-markdown", type=Path, required=True)
    prepare.add_argument("--review-template", type=Path, required=True)
    prepare.add_argument("--audit-fraction", type=float, default=0.1)
    prepare.add_argument("--audit-seed", type=int, default=17)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--dataset-manifest", type=Path, required=True)
    summarize.add_argument("--reviews", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--audit-fraction", type=float, default=0.1)
    summarize.add_argument("--audit-seed", type=int, default=17)
    args = parser.parse_args()

    if args.command == "prepare":
        markdown, template = prepare_audit(
            args.dataset_manifest,
            args.model_output,
            args.generation_evaluation,
            args.ablation_model_output,
            fraction=args.audit_fraction,
            seed=args.audit_seed,
        )
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
        write_jsonl(template, args.review_template)
        print(
            json.dumps(
                {
                    "packet": str(args.output_markdown),
                    "template": str(args.review_template),
                    "reviewed_count": len(template),
                },
                indent=2,
            )
        )
        return

    summary = summarize_reviews(
        args.dataset_manifest,
        args.reviews,
        fraction=args.audit_fraction,
        seed=args.audit_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
