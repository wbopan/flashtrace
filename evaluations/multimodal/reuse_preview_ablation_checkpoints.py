"""Seed formal ablation checkpoints from response-identical preview candidates.

The preview is allowed to save GPU work, but never to change the formal
selection.  A record is reused only when the dataset input, primary generated
response, token identity, resolved model revision, and ablation generation
configuration are all identical to the formal candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .reuse_preview_checkpoints import (
    _dataset_signature,
    _load_by_id,
    _model_signature,
    _sha256,
)
from .strict_generation import read_jsonl, write_jsonl


def _ablation_signature(
    record: Mapping[str, Any], primary_model: Mapping[str, Any]
) -> dict[str, Any]:
    ablation_model = record.get("model") or {}
    model = primary_model.get("model") or {}
    generation = model.get("generation") or {}
    return {
        "repo_id": ablation_model.get("repo_id"),
        "revision": ablation_model.get("revision"),
        "do_sample": ablation_model.get("do_sample"),
        "max_new_tokens": ablation_model.get("max_new_tokens"),
        "expected_repo_id": model.get("repo_id"),
        "expected_revision": model.get(
            "resolved_revision", model.get("revision")
        ),
        "expected_do_sample": generation.get("do_sample"),
        "expected_max_new_tokens": generation.get("max_new_tokens"),
    }


def _ablation_config_matches(
    record: Mapping[str, Any], primary_model: Mapping[str, Any]
) -> bool:
    signature = _ablation_signature(record, primary_model)
    return (
        signature["repo_id"] == signature["expected_repo_id"]
        and signature["revision"] == signature["expected_revision"]
        and signature["do_sample"] == signature["expected_do_sample"]
        and signature["max_new_tokens"]
        == signature["expected_max_new_tokens"]
    )


def _without_provenance(record: Mapping[str, Any]) -> dict[str, Any]:
    clean = dict(record)
    clean.pop("checkpoint_provenance", None)
    return clean


def reuse(
    *,
    formal_dataset: Path,
    formal_model: Path,
    formal_generation_evaluation: Path,
    preview_dataset: Path,
    preview_model: Path,
    preview_ablation_model: Path,
    formal_ablation_model: Path,
) -> dict[str, Any]:
    formal_datasets = _load_by_id(formal_dataset)
    formal_models = _load_by_id(formal_model)
    formal_evaluations = _load_by_id(formal_generation_evaluation)
    preview_datasets = _load_by_id(preview_dataset)
    preview_models = _load_by_id(preview_model)
    source_records = read_jsonl(preview_ablation_model)
    source_by_id = {
        str(record["sample_id"]): record for record in source_records
    }
    if len(source_by_id) != len(source_records):
        raise ValueError(f"duplicate sample IDs in {preview_ablation_model}")

    overlap = set(formal_datasets) & set(preview_datasets)
    identity_matched = {
        sample_id
        for sample_id in overlap
        if sample_id in formal_models
        and sample_id in preview_models
        and _dataset_signature(formal_datasets[sample_id])
        == _dataset_signature(preview_datasets[sample_id])
        and _model_signature(formal_models[sample_id])
        == _model_signature(preview_models[sample_id])
    }
    reusable = {
        sample_id
        for sample_id in identity_matched
        if formal_evaluations.get(sample_id, {}).get(
            "pre_ablation_eligible",
            formal_evaluations.get(sample_id, {}).get("strict_eligible"),
        )
        and sample_id in source_by_id
        and source_by_id[sample_id].get("status") == "complete"
        and source_by_id[sample_id].get("I_QUESTION")
        == formal_datasets[sample_id]["input"]["I_QUESTION"]
        and _ablation_config_matches(
            source_by_id[sample_id], formal_models[sample_id]
        )
    }

    existing = (
        read_jsonl(formal_ablation_model)
        if formal_ablation_model.is_file()
        else []
    )
    complete_existing = {
        str(record["sample_id"]): record
        for record in existing
        if record.get("status") == "complete"
    }
    additions = []
    source_hash = _sha256(preview_ablation_model)
    for sample_id in sorted(reusable):
        source = source_by_id[sample_id]
        prior = complete_existing.get(sample_id)
        if prior is not None:
            if _without_provenance(prior) != _without_provenance(source):
                raise ValueError(
                    f"{formal_ablation_model} already contains a different "
                    f"complete ablation record for {sample_id}"
                )
            continue
        seeded = dict(source)
        seeded["checkpoint_provenance"] = {
            "kind": "response_identical_formal_preview_candidate_ablation",
            "source": str(preview_ablation_model),
            "source_sha256": source_hash,
        }
        additions.append(seeded)
        complete_existing[sample_id] = seeded
    if additions:
        write_jsonl(existing + additions, formal_ablation_model)

    reused_ids = {
        str(record["sample_id"])
        for record in existing + additions
        if (record.get("checkpoint_provenance") or {}).get("kind")
        == "response_identical_formal_preview_candidate_ablation"
    }
    return {
        "schema_version": 1,
        "policy": (
            "reuse_only_response_token_model_and_ablation_config_identical"
        ),
        "formal_candidates": len(formal_datasets),
        "preview_candidates": len(preview_datasets),
        "overlap_candidates": len(overlap),
        "identity_matched_candidates": len(identity_matched),
        "identity_mismatched_sample_ids": sorted(overlap - identity_matched),
        "reusable_complete_ablation_records": len(reusable),
        "newly_seeded_ablation_records": len(additions),
        "reused_ablation_records": len(reused_ids),
        "reused_sample_ids": sorted(reused_ids),
        "source_sha256": {
            str(preview_ablation_model): source_hash,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dataset", type=Path, required=True)
    parser.add_argument("--formal-model", type=Path, required=True)
    parser.add_argument(
        "--formal-generation-evaluation", type=Path, required=True
    )
    parser.add_argument("--preview-dataset", type=Path, required=True)
    parser.add_argument("--preview-model", type=Path, required=True)
    parser.add_argument("--preview-ablation-model", type=Path, required=True)
    parser.add_argument("--formal-ablation-model", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = reuse(
        formal_dataset=args.formal_dataset,
        formal_model=args.formal_model,
        formal_generation_evaluation=args.formal_generation_evaluation,
        preview_dataset=args.preview_dataset,
        preview_model=args.preview_model,
        preview_ablation_model=args.preview_ablation_model,
        formal_ablation_model=args.formal_ablation_model,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
