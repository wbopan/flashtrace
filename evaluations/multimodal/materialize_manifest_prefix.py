"""Materialize an immutable nested prefix of a frozen candidate manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .strict_generation import read_jsonl, write_jsonl


def materialize(source: Path, output: Path, count: int) -> list[dict]:
    records = read_jsonl(source)
    if count <= 0 or count > len(records):
        raise ValueError(
            f"prefix count must be in [1, {len(records)}], received {count}"
        )
    selected = records[:count]
    if output.is_file():
        existing = read_jsonl(output)
        if existing != selected:
            raise ValueError(f"immutable prefix artifact differs: {output}")
        return existing
    write_jsonl(selected, output)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = materialize(args.source, args.output, args.count)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "output": str(args.output),
                "records": len(records),
                "first_sample_id": records[0]["sample_id"],
                "last_sample_id": records[-1]["sample_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
