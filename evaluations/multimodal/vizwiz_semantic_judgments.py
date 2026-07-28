"""Prepare, validate, and join VizWiz-LF semantic correctness judgments."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl, write_jsonl


ALLOWED_LABELS = frozenset({"fully", "partial", "wrong"})


def prepare_tasks(
    dataset_manifest: Path,
    model_output: Path,
    *,
    sample_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    datasets = {record["sample_id"]: record for record in read_jsonl(dataset_manifest)}
    models = {record["sample_id"]: record for record in read_jsonl(model_output)}
    selected = [
        sample_id
        for sample_id in datasets
        if sample_id in models and (sample_ids is None or sample_id in sample_ids)
    ]
    if sample_ids is not None and set(selected) != sample_ids:
        raise ValueError(
            f"missing requested judgment samples: {sorted(sample_ids - set(selected))}"
        )
    tasks = []
    for sample_id in selected:
        dataset = datasets[sample_id]
        if dataset["benchmark"] != "vizwiz_lf":
            raise ValueError(f"{sample_id} is not a VizWiz-LF sample")
        metadata = dataset["evaluation"]["metadata"]
        tasks.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "question": dataset["input"]["I_QUESTION"],
                "model_output": models[sample_id]["OUTPUT"],
                "expert_reference": dataset["evaluation"]["REFERENCE_OUTPUT"],
                "crowd_answers": list(metadata.get("crowd_answers") or []),
                "crowd_majority": metadata.get("crowd_majority"),
                "question_type": metadata.get("question_type"),
                "dataset_answerability": metadata.get("answerability"),
                "instruction": (
                    "Label the model output fully, partial, or wrong relative to "
                    "the visual question and references. Fully means all material "
                    "claims needed to answer are correct; partial means the core "
                    "answer is useful but incomplete, uncertain, or contains a "
                    "minor unsupported claim; wrong means the core answer is "
                    "incorrect or unusable. Return a concise evidence-based reason."
                ),
            }
        )
    return tasks


def audit_sample_ids(
    sample_ids: list[str], *, fraction: float = 0.1, seed: int = 17
) -> list[str]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("audit fraction must be in (0, 1]")
    shuffled = sorted(sample_ids)
    random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[: max(1, math.ceil(len(shuffled) * fraction))])


def validate_judgment(record: dict[str, Any]) -> None:
    required = {"sample_id", "label", "judge", "reason"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"judgment is missing fields: {sorted(missing)}")
    if record["label"] not in ALLOWED_LABELS:
        raise ValueError(
            f"invalid semantic label {record['label']!r}; "
            f"expected {sorted(ALLOWED_LABELS)}"
        )
    if not str(record["judge"]).strip() or not str(record["reason"]).strip():
        raise ValueError("judge and reason must be non-empty")
    confidence = record.get("confidence")
    if confidence is not None and not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between zero and one")
    if record.get("human_reviewed") and not str(
        record.get("human_reviewer", "")
    ).strip():
        raise ValueError("human-reviewed judgments need human_reviewer")


def prepare_human_review(
    dataset_manifest: Path,
    model_output: Path,
    judgments_path: Path,
    *,
    audit_fraction: float = 0.1,
    audit_seed: int = 17,
) -> tuple[str, list[dict[str, Any]]]:
    """Create a deterministic, image-linked human audit packet and template."""

    datasets = {record["sample_id"]: record for record in read_jsonl(dataset_manifest)}
    models = {record["sample_id"]: record for record in read_jsonl(model_output)}
    judgments: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(judgments_path):
        validate_judgment(record)
        sample_id = str(record["sample_id"])
        if sample_id in judgments:
            raise ValueError(f"duplicate semantic judgment: {sample_id}")
        judgments[sample_id] = record
    sample_ids = sorted(set(datasets) & set(models))
    missing = sorted(set(sample_ids) - set(judgments))
    if missing:
        raise ValueError(f"LLM judgments are incomplete before human audit: {missing}")
    audit_ids = audit_sample_ids(
        sample_ids, fraction=audit_fraction, seed=audit_seed
    )
    lines = [
        "# VizWiz-LF deterministic 10% semantic audit",
        "",
        f"Audit seed: {audit_seed}; fraction: {audit_fraction}; "
        f"samples: {len(audit_ids)}/{len(sample_ids)}.",
        "",
        "For each sample, compare the image, question, references, model output, "
        "and LLM judgment. Record `fully`, `partial`, or `wrong` in the JSONL "
        "review template and give a concise reason.",
        "",
    ]
    template = []
    for sample_id in audit_ids:
        dataset = datasets[sample_id]
        model = models[sample_id]
        judgment = judgments[sample_id]
        metadata = dataset["evaluation"]["metadata"]
        image_path = str(Path(dataset["input"]["I_IMAGE"]).resolve())
        lines.extend(
            [
                f"## {sample_id}",
                "",
                f"![{sample_id}]({image_path})",
                "",
                f"**Question:** {dataset['input']['I_QUESTION']}",
                "",
                f"**Expert reference:** "
                f"{dataset['evaluation']['REFERENCE_OUTPUT']}",
                "",
                f"**Crowd answers:** "
                f"{json.dumps(metadata.get('crowd_answers') or [], ensure_ascii=False)}",
                "",
                f"**Model output:** {model['OUTPUT']}",
                "",
                f"**LLM label:** `{judgment['label']}`",
                "",
                f"**LLM reason:** {judgment['reason']}",
                "",
            ]
        )
        template.append(
            {
                "sample_id": sample_id,
                "llm_label": judgment["label"],
                "human_label": None,
                "human_reviewer": None,
                "human_reason": None,
            }
        )
    return "\n".join(lines) + "\n", template


def apply_human_reviews(
    judgments_path: Path,
    reviews_path: Path,
    *,
    audit_fraction: float = 0.1,
    audit_seed: int = 17,
) -> list[dict[str, Any]]:
    """Adjudicate the deterministic audit rows while retaining LLM provenance."""

    judgments = read_jsonl(judgments_path)
    by_id: dict[str, dict[str, Any]] = {}
    for judgment in judgments:
        validate_judgment(judgment)
        sample_id = str(judgment["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"duplicate semantic judgment: {sample_id}")
        by_id[sample_id] = judgment
    required_ids = set(
        audit_sample_ids(
            list(by_id), fraction=audit_fraction, seed=audit_seed
        )
    )
    reviews: dict[str, dict[str, Any]] = {}
    for review in read_jsonl(reviews_path):
        sample_id = str(review.get("sample_id", ""))
        label = review.get("human_label")
        reviewer = str(review.get("human_reviewer") or "").strip()
        reason = str(review.get("human_reason") or "").strip()
        if sample_id in reviews:
            raise ValueError(f"duplicate human review: {sample_id}")
        if label not in ALLOWED_LABELS or not reviewer or not reason:
            raise ValueError(
                f"incomplete human review for {sample_id}: "
                "human_label, human_reviewer, and human_reason are required"
            )
        reviews[sample_id] = review
    missing = sorted(required_ids - set(reviews))
    unexpected = sorted(set(reviews) - required_ids)
    if missing or unexpected:
        raise ValueError(
            f"human review IDs do not match deterministic audit: "
            f"missing={missing}, unexpected={unexpected}"
        )

    output = []
    for judgment in judgments:
        sample_id = str(judgment["sample_id"])
        if sample_id not in reviews:
            output.append(judgment)
            continue
        review = reviews[sample_id]
        revised = dict(judgment)
        revised.update(
            {
                "llm_label": judgment["label"],
                "llm_reason": judgment["reason"],
                "label": review["human_label"],
                "reason": review["human_reason"],
                "human_reviewed": True,
                "human_reviewer": review["human_reviewer"],
            }
        )
        validate_judgment(revised)
        output.append(revised)
    return output


def join_judgments(
    generation_evaluation: Path,
    judgments_path: Path,
    *,
    audit_fraction: float = 0.1,
    audit_seed: int = 17,
    require_complete: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluations = read_jsonl(generation_evaluation)
    judgments: dict[str, dict[str, Any]] = {}
    for judgment in read_jsonl(judgments_path):
        validate_judgment(judgment)
        sample_id = str(judgment["sample_id"])
        if sample_id in judgments:
            raise ValueError(f"duplicate semantic judgment: {sample_id}")
        judgments[sample_id] = judgment

    eligible_ids = [
        record["sample_id"]
        for record in evaluations
        if record.get("benchmark") == "vizwiz_lf" and record.get("strict_eligible")
    ]
    audit_ids = set(
        audit_sample_ids(eligible_ids, fraction=audit_fraction, seed=audit_seed)
    )
    joined = []
    for evaluation in evaluations:
        record = dict(evaluation)
        sample_id = record["sample_id"]
        judgment = judgments.get(sample_id)
        if judgment is not None:
            record["semantic_correctness"] = {
                "status": "reviewed",
                "label": judgment["label"],
                "judge": judgment["judge"],
                "reason": judgment["reason"],
                "confidence": judgment.get("confidence"),
                "human_reviewed": bool(judgment.get("human_reviewed", False)),
                "human_reviewer": judgment.get("human_reviewer"),
                "llm_label": judgment.get("llm_label", judgment["label"]),
                "llm_reason": judgment.get("llm_reason", judgment["reason"]),
            }
        elif record.get("benchmark") == "vizwiz_lf":
            record["semantic_correctness"] = {
                "status": "unreviewed",
                "label": None,
            }
        record["semantic_audit_required"] = sample_id in audit_ids
        joined.append(record)

    eligible_missing = sorted(set(eligible_ids) - set(judgments))
    audit_missing = sorted(
        sample_id
        for sample_id in audit_ids
        if not judgments.get(sample_id, {}).get("human_reviewed")
    )
    if require_complete and (eligible_missing or audit_missing):
        raise ValueError(
            "semantic review is incomplete: "
            f"eligible_missing={eligible_missing}, audit_missing={audit_missing}"
        )
    return joined, {
        "eligible_samples": len(eligible_ids),
        "judged_eligible_samples": len(set(eligible_ids) & set(judgments)),
        "label_counts": {
            label: sum(
                judgments.get(sample_id, {}).get("label") == label
                for sample_id in eligible_ids
            )
            for label in sorted(ALLOWED_LABELS)
        },
        "audit_fraction": audit_fraction,
        "audit_seed": audit_seed,
        "audit_sample_ids": sorted(audit_ids),
        "audit_reviewed": len(audit_ids) - len(audit_missing),
        "eligible_missing": eligible_missing,
        "audit_missing": audit_missing,
        "complete": not eligible_missing and not audit_missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset-manifest", type=Path, required=True)
    prepare.add_argument("--model-output", type=Path, required=True)
    prepare.add_argument("--sample-id", action="append", dest="sample_ids")
    prepare.add_argument("--output", type=Path, required=True)

    join = subparsers.add_parser("join")
    join.add_argument("--generation-evaluation", type=Path, required=True)
    join.add_argument("--judgments", type=Path, required=True)
    join.add_argument("--output-evaluation", type=Path, required=True)
    join.add_argument("--summary-output", type=Path, required=True)
    join.add_argument("--audit-fraction", type=float, default=0.1)
    join.add_argument("--audit-seed", type=int, default=17)
    join.add_argument("--require-complete", action="store_true")

    packet = subparsers.add_parser("audit-packet")
    packet.add_argument("--dataset-manifest", type=Path, required=True)
    packet.add_argument("--model-output", type=Path, required=True)
    packet.add_argument("--judgments", type=Path, required=True)
    packet.add_argument("--output-markdown", type=Path, required=True)
    packet.add_argument("--review-template", type=Path, required=True)
    packet.add_argument("--audit-fraction", type=float, default=0.1)
    packet.add_argument("--audit-seed", type=int, default=17)

    review = subparsers.add_parser("apply-review")
    review.add_argument("--judgments", type=Path, required=True)
    review.add_argument("--reviews", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--audit-fraction", type=float, default=0.1)
    review.add_argument("--audit-seed", type=int, default=17)
    args = parser.parse_args()

    if args.command == "prepare":
        tasks = prepare_tasks(
            args.dataset_manifest,
            args.model_output,
            sample_ids=set(args.sample_ids) if args.sample_ids else None,
        )
        write_jsonl(tasks, args.output)
        print(json.dumps({"output": str(args.output), "tasks": len(tasks)}, indent=2))
        return

    if args.command == "audit-packet":
        markdown, template = prepare_human_review(
            args.dataset_manifest,
            args.model_output,
            args.judgments,
            audit_fraction=args.audit_fraction,
            audit_seed=args.audit_seed,
        )
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
        write_jsonl(template, args.review_template)
        print(
            json.dumps(
                {
                    "markdown": str(args.output_markdown),
                    "review_template": str(args.review_template),
                    "audit_samples": len(template),
                },
                indent=2,
            )
        )
        return

    if args.command == "apply-review":
        reviewed = apply_human_reviews(
            args.judgments,
            args.reviews,
            audit_fraction=args.audit_fraction,
            audit_seed=args.audit_seed,
        )
        write_jsonl(reviewed, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "judgments": len(reviewed),
                    "human_reviewed": sum(
                        record.get("human_reviewed") is True
                        for record in reviewed
                    ),
                },
                indent=2,
            )
        )
        return

    joined, summary = join_judgments(
        args.generation_evaluation,
        args.judgments,
        audit_fraction=args.audit_fraction,
        audit_seed=args.audit_seed,
        require_complete=args.require_complete,
    )
    write_jsonl(joined, args.output_evaluation)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
