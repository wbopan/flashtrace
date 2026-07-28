"""Seed formal GPU checkpoints from response-identical n=20 preview records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .strict_generation import read_jsonl, write_jsonl


EXPECTED_METHODS = {
    "random",
    "center",
    "visual-loo",
    "ifr-span",
    "visual-ig",
    "attnlrp",
    "flashtrace",
    "flashtrace-all-gen",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.get("generation_metadata") or {}
    model = record.get("model") or {}
    return {
        "I_IMAGE": record.get("I_IMAGE"),
        "I_QUESTION": record.get("I_QUESTION"),
        "raw_response": record.get("raw_response"),
        "THINKING": record.get("THINKING"),
        "OUTPUT": record.get("OUTPUT"),
        "THINKING_SPAN": record.get("THINKING_SPAN"),
        "OUTPUT_SPAN": record.get("OUTPUT_SPAN"),
        "original_generated_token_ids": metadata.get(
            "original_generated_token_ids"
        ),
        "teacher_forced_token_ids": metadata.get("teacher_forced_token_ids"),
        "resolved_revision": model.get("resolved_revision", model.get("revision")),
    }


def _dataset_signature(record: Mapping[str, Any]) -> dict[str, Any]:
    inputs = record.get("input") or {}
    return {
        "benchmark": record.get("benchmark"),
        "I_IMAGE": inputs.get("I_IMAGE"),
        "I_QUESTION": inputs.get("I_QUESTION"),
    }


def _load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl(path)
    by_id = {str(record["sample_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"duplicate sample IDs in {path}")
    return by_id


def _seed_records(
    source: Path,
    destination: Path,
    *,
    matched_ids: set[str],
    formal_order: Mapping[str, int],
) -> tuple[int, int]:
    existing = read_jsonl(destination) if destination.is_file() else []
    existing_pairs = {
        (str(record.get("sample_id")), str(record.get("method")))
        for record in existing
    }
    additions = []
    for record in read_jsonl(source):
        pair = (str(record.get("sample_id")), str(record.get("method")))
        if (
            record.get("status") != "ok"
            or pair[0] not in matched_ids
            or pair in existing_pairs
        ):
            continue
        seeded = dict(record)
        seeded["sample_index"] = formal_order[pair[0]]
        seeded["checkpoint_provenance"] = {
            "kind": "response_identical_formal_preview_n20",
            "source": str(source),
        }
        additions.append(seeded)
        existing_pairs.add(pair)
    if additions:
        write_jsonl(existing + additions, destination)
    total_reused = sum(
        (record.get("checkpoint_provenance") or {}).get("kind")
        == "response_identical_formal_preview_n20"
        for record in existing + additions
    )
    return len(additions), total_reused


def reuse(
    *,
    formal_dataset: Path,
    formal_model: Path,
    preview_dataset: Path,
    preview_model: Path,
    preview_attribution_dir: Path,
    preview_faithfulness_dir: Path,
    formal_attribution_dir: Path,
    formal_faithfulness_dir: Path,
) -> dict[str, Any]:
    formal_datasets = _load_by_id(formal_dataset)
    formal_models = _load_by_id(formal_model)
    preview_datasets = _load_by_id(preview_dataset)
    preview_models = _load_by_id(preview_model)
    attribution_summary = json.loads(
        (preview_attribution_dir / "summary.json").read_text(encoding="utf-8")
    )
    faithfulness_summary = json.loads(
        (preview_faithfulness_dir / "summary.json").read_text(encoding="utf-8")
    )
    if set(attribution_summary.get("requested_methods") or []) != EXPECTED_METHODS:
        raise ValueError("preview attribution is not the frozen eight-method panel")
    if set(faithfulness_summary.get("methods") or []) != EXPECTED_METHODS:
        raise ValueError("preview faithfulness is not the frozen eight-method panel")
    if (
        faithfulness_summary.get("target_regions") != 64
        or faithfulness_summary.get("steps") != 10
    ):
        raise ValueError("preview faithfulness does not use the formal 64/10 budget")

    overlap = set(formal_datasets) & set(preview_datasets)
    matched = {
        sample_id
        for sample_id in overlap
        if sample_id in formal_models
        and sample_id in preview_models
        and _dataset_signature(formal_datasets[sample_id])
        == _dataset_signature(preview_datasets[sample_id])
        and _model_signature(formal_models[sample_id])
        == _model_signature(preview_models[sample_id])
    }
    mismatched = sorted(overlap - matched)
    formal_order = {
        sample_id: index for index, sample_id in enumerate(formal_datasets)
    }
    formal_attribution_dir.mkdir(parents=True, exist_ok=True)
    formal_faithfulness_dir.mkdir(parents=True, exist_ok=True)
    source_attribution = preview_attribution_dir / "attribution_records.jsonl"
    source_faithfulness = preview_faithfulness_dir / "faithfulness_records.jsonl"
    newly_seeded_attribution, reused_attribution = _seed_records(
        source_attribution,
        formal_attribution_dir / "attribution_records.jsonl",
        matched_ids=matched,
        formal_order=formal_order,
    )
    newly_seeded_faithfulness, reused_faithfulness = _seed_records(
        source_faithfulness,
        formal_faithfulness_dir / "faithfulness_records.jsonl",
        matched_ids=matched,
        formal_order=formal_order,
    )
    return {
        "schema_version": 1,
        "policy": "reuse_only_response_identical_deterministic_gpu_records",
        "formal_samples": len(formal_datasets),
        "preview_samples": len(preview_datasets),
        "overlap_samples": len(overlap),
        "identity_matched_samples": len(matched),
        "identity_mismatched_sample_ids": mismatched,
        "matched_sample_ids": sorted(matched),
        "newly_seeded_attribution_pairs": newly_seeded_attribution,
        "newly_seeded_faithfulness_pairs": newly_seeded_faithfulness,
        "reused_attribution_pairs": reused_attribution,
        "reused_faithfulness_pairs": reused_faithfulness,
        "expected_pairs_per_matched_sample": len(EXPECTED_METHODS),
        "source_sha256": {
            str(source_attribution): _sha256(source_attribution),
            str(source_faithfulness): _sha256(source_faithfulness),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-dataset", type=Path, required=True)
    parser.add_argument("--formal-model", type=Path, required=True)
    parser.add_argument("--preview-dataset", type=Path, required=True)
    parser.add_argument("--preview-model", type=Path, required=True)
    parser.add_argument("--preview-attribution-dir", type=Path, required=True)
    parser.add_argument("--preview-faithfulness-dir", type=Path, required=True)
    parser.add_argument("--formal-attribution-dir", type=Path, required=True)
    parser.add_argument("--formal-faithfulness-dir", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    summary = reuse(
        formal_dataset=args.formal_dataset,
        formal_model=args.formal_model,
        preview_dataset=args.preview_dataset,
        preview_model=args.preview_model,
        preview_attribution_dir=args.preview_attribution_dir,
        preview_faithfulness_dir=args.preview_faithfulness_dir,
        formal_attribution_dir=args.formal_attribution_dir,
        formal_faithfulness_dir=args.formal_faithfulness_dir,
    )
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
