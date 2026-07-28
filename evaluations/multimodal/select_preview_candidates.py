"""Freeze a deterministic, result-blind candidate subset for a preview run."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl, write_jsonl


def select_candidates(
    source_manifest: Path,
    *,
    sample_size: int,
    seed: int,
    exclude_manifests: list[Path] | None = None,
) -> list[dict[str, Any]]:
    records = read_jsonl(source_manifest)
    excluded = {
        str(row["sample_id"])
        for path in (exclude_manifests or [])
        for row in read_jsonl(path)
    }
    eligible = [row for row in records if str(row["sample_id"]) not in excluded]
    if len(eligible) < sample_size:
        raise ValueError(
            f"only {len(eligible)} candidates remain after exclusions; "
            f"requested {sample_size}"
        )
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(eligible)), sample_size))
    return [eligible[index] for index in selected_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--exclude-manifest", type=Path, action="append", default=[]
    )
    args = parser.parse_args()
    selected = select_candidates(
        args.source_manifest,
        sample_size=args.sample_size,
        seed=args.seed,
        exclude_manifests=args.exclude_manifest,
    )
    selected_ids = [str(row["sample_id"]) for row in selected]
    if args.output.is_file():
        existing_ids = [
            str(row["sample_id"]) for row in read_jsonl(args.output)
        ]
        if existing_ids != selected_ids:
            raise ValueError(
                f"{args.output} already contains a different frozen subset"
            )
    write_jsonl(selected, args.output)
    print(
        json.dumps(
            {
                "source_manifest": str(args.source_manifest),
                "output": str(args.output),
                "sample_size": len(selected),
                "seed": args.seed,
                "excluded_count": sum(
                    len(read_jsonl(path)) for path in args.exclude_manifest
                ),
                "sample_ids": selected_ids,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
