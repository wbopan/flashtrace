"""Materialize immutable hashes for frozen formal model responses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .freeze_formal_input_hashes import (
    canonical_record_sha256,
    sha256_file,
    sha256_text,
)
from .strict_generation import read_jsonl, validate_model_record


def token_ids_sha256(value: list[int]) -> str:
    canonical = json.dumps(value, separators=(",", ":"))
    return sha256_text(canonical)


def build_model_hashes(root: Path, model_output: Path) -> dict[str, Any]:
    root = root.resolve()
    model_output = model_output.resolve()
    if not model_output.is_relative_to(root):
        raise ValueError(f"model output escapes repository root: {model_output}")
    records = read_jsonl(model_output)
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("frozen model output contains duplicate sample IDs")

    samples = []
    revisions = set()
    for record in sorted(records, key=lambda value: str(value["sample_id"])):
        validate_model_record(record)
        metadata = record.get("generation_metadata") or {}
        generated_ids = metadata.get("original_generated_token_ids")
        teacher_forced_ids = metadata.get("teacher_forced_token_ids")
        if (
            not isinstance(generated_ids, list)
            or not generated_ids
            or generated_ids != teacher_forced_ids
            or not all(
                isinstance(token_id, int) and not isinstance(token_id, bool)
                for token_id in generated_ids
            )
        ):
            raise ValueError(
                f"{record['sample_id']} has invalid frozen token identity"
            )
        model = record.get("model") or {}
        revision = str(model.get("resolved_revision") or "")
        if not revision:
            raise ValueError(f"{record['sample_id']} has no resolved revision")
        revisions.add(revision)
        samples.append(
            {
                "sample_id": str(record["sample_id"]),
                "model_record_sha256": canonical_record_sha256(record),
                "raw_response_sha256": sha256_text(str(record["raw_response"])),
                "thinking_sha256": sha256_text(str(record["THINKING"])),
                "output_sha256": sha256_text(str(record["OUTPUT"])),
                "generated_token_ids_sha256": token_ids_sha256(generated_ids),
                "teacher_forced_token_ids_sha256": token_ids_sha256(
                    teacher_forced_ids
                ),
                "resolved_revision": revision,
            }
        )
    return {
        "model_output_path": model_output.relative_to(root).as_posix(),
        "model_output_sha256": sha256_file(model_output),
        "sample_count": len(samples),
        "resolved_revisions": sorted(revisions),
        "samples": samples,
    }


def build_payload(root: Path, model_outputs: list[Path]) -> dict[str, Any]:
    bundles = [
        build_model_hashes(root, model_output)
        for model_output in sorted(model_outputs, key=lambda path: str(path))
    ]
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "canonical_model_record": (
            "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False)"
        ),
        "token_ids_encoding": "json.dumps(separators=(',',':'))",
        "model_outputs": bundles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--model-output", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.root, args.model_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model_outputs": len(payload["model_outputs"]),
                "samples": sum(
                    int(bundle["sample_count"])
                    for bundle in payload["model_outputs"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
