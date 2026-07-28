"""Validate an exact frozen-sample by method evaluation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl


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


def validate(
    *,
    manifest: Path,
    evaluation_dir: Path,
    kind: str,
    expected_samples: int,
) -> dict[str, Any]:
    if kind not in {"attribution", "faithfulness"}:
        raise ValueError(f"unsupported matrix kind: {kind}")
    summary_path = evaluation_dir / "summary.json"
    records_path = evaluation_dir / (
        "attribution_records.jsonl"
        if kind == "attribution"
        else "faithfulness_records.jsonl"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    datasets = read_jsonl(manifest)
    dataset_ids = [str(record["sample_id"]) for record in datasets]
    if len(dataset_ids) != expected_samples or len(set(dataset_ids)) != len(
        dataset_ids
    ):
        raise ValueError(
            f"manifest is not a unique n={expected_samples} frozen set"
        )
    frozen_ids = set(dataset_ids)
    summary_methods = set(summary.get("methods") or {})
    summary_ids = {
        str(sample_id) for sample_id in summary.get("common_sample_ids") or []
    }
    if summary_methods != EXPECTED_METHODS:
        raise ValueError(
            f"summary methods differ from frozen panel: {sorted(summary_methods)}"
        )
    if (
        summary.get("common_samples") != expected_samples
        or summary_ids != frozen_ids
    ):
        raise ValueError("summary common sample set differs from frozen manifest")

    records = read_jsonl(records_path)
    pairs = [
        (str(record.get("sample_id")), str(record.get("method")))
        for record in records
    ]
    successful_pairs = {
        (str(record.get("sample_id")), str(record.get("method")))
        for record in records
        if record.get("status") == "ok"
    }
    expected_pairs = {
        (sample_id, method)
        for sample_id in frozen_ids
        for method in EXPECTED_METHODS
    }
    errors = [record for record in records if record.get("status") != "ok"]
    if len(pairs) != len(set(pairs)):
        raise ValueError("matrix contains duplicate sample/method records")
    if errors:
        raise ValueError(f"matrix contains {len(errors)} error records")
    if successful_pairs != expected_pairs or len(records) != len(expected_pairs):
        missing = expected_pairs - successful_pairs
        extra = successful_pairs - expected_pairs
        raise ValueError(
            "matrix is not the exact frozen Cartesian product: "
            f"missing={len(missing)}, extra={len(extra)}, records={len(records)}"
        )
    return {
        "schema_version": 1,
        "kind": kind,
        "manifest": str(manifest),
        "evaluation_dir": str(evaluation_dir),
        "samples": expected_samples,
        "methods": sorted(EXPECTED_METHODS),
        "successful_pairs": len(successful_pairs),
        "errors": 0,
        "duplicates": 0,
        "exact_cartesian_product": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=("attribution", "faithfulness"),
        required=True,
    )
    parser.add_argument("--expected-samples", type=int, required=True)
    args = parser.parse_args()
    result = validate(
        manifest=args.manifest,
        evaluation_dir=args.evaluation_dir,
        kind=args.kind,
        expected_samples=args.expected_samples,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
