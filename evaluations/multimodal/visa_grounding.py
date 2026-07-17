"""Build a stratified Wiki-VISA manifest with explicit evidence boxes.

The official ``MrLight/wiki-visa`` test split contains historical Wikipedia
screenshots, Natural Questions answers, and the bounding box of the supporting
HTML element.  This module keeps dataset loading optional: record conversion
and tests require only the Python standard library, while the CLI uses the
``datasets`` package from FlashTrace's ``eval`` extra.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any


WIKI_VISA_DATASET = "MrLight/wiki-visa"


def visa_stratum(example: Mapping[str, Any]) -> str:
    """Return the split used by the official VISA evaluation."""

    answer_type = str(example.get("long_answer_type", ""))
    box = example.get("bounding_box") or []
    if answer_type == "p" and len(box) == 4:
        return "first_page_passage" if float(box[1]) < 980 else "later_page_passage"
    return "non_passage"


def build_visa_record(row_index: int, example: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one Hugging Face Wiki-VISA row to a portable manifest record."""

    image_size = example.get("image_size")
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        image = example.get("image")
        image_size = getattr(image, "size", None)
    if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        raise ValueError("Wiki-VISA row needs image_size=[width, height] or a PIL image")
    width, height = (int(value) for value in image_size)
    if width <= 0 or height <= 0:
        raise ValueError("Wiki-VISA image dimensions must be positive")

    box = example.get("bounding_box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("Wiki-VISA row needs bounding_box=[x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(value) for value in box)
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Wiki-VISA bounding box must have positive area after clipping")

    candidates = [str(value) for value in (example.get("candidates") or [])]
    positive_index = int(example.get("pos_idx", -1))
    return {
        "schema_version": 1,
        "benchmark": "wiki_visa_single_oracle",
        "sample_id": str(example.get("id", row_index)),
        "hf_dataset": WIKI_VISA_DATASET,
        "hf_split": "test",
        "hf_row_index": int(row_index),
        "question": str(example.get("question", "")),
        "reference_answer": str(example.get("short_answer", "")),
        "long_answer_type": str(example.get("long_answer_type", "")),
        "stratum": visa_stratum(example),
        "image_size": {"width": width, "height": height},
        "evidence_bbox_xyxy": [x1, y1, x2, y2],
        "evidence_bbox_xyxy_normalized": [x1 / width, y1 / height, x2 / width, y2 / height],
        "candidate_ids": candidates,
        "positive_candidate_index": positive_index,
        "has_positive_candidate": 0 <= positive_index < len(candidates),
        "source_url": str(example.get("url", "")),
    }


def iter_visa_records(examples: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield valid records while retaining the source dataset row index."""

    for row_index, example in enumerate(examples):
        try:
            yield build_visa_record(row_index, example)
        except ValueError:
            continue


def stratified_sample(
    records: Iterable[dict[str, Any]], sample_size: int, *, seed: int = 17
) -> list[dict[str, Any]]:
    """Sample approximately equal counts from the three VISA strata."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["stratum"])].append(record)

    expected = ("first_page_passage", "later_page_passage", "non_passage")
    if any(not groups[name] for name in expected):
        missing = [name for name in expected if not groups[name]]
        raise ValueError(f"cannot stratify because strata are empty: {missing}")

    rng = random.Random(seed)
    base, remainder = divmod(sample_size, len(expected))
    selected: list[dict[str, Any]] = []
    for index, name in enumerate(expected):
        count = base + int(index < remainder)
        if count > len(groups[name]):
            raise ValueError(f"requested {count} records from {name}, only {len(groups[name])} available")
        selected.extend(rng.sample(groups[name], count))
    selected.sort(key=lambda record: record["hf_row_index"])
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        from datasets import Image, load_dataset
    except ImportError as error:
        raise SystemExit("Install the evaluation dependencies with: pip install -e '.[eval]'") from error

    dataset = load_dataset(WIKI_VISA_DATASET, split="test").cast_column("image", Image(decode=False))
    records = stratified_sample(iter_visa_records(dataset), args.sample_size, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    strata: dict[str, int] = defaultdict(int)
    for record in records:
        strata[record["stratum"]] += 1
    print(json.dumps({"output": str(args.output), "records": len(records), "strata": strata}, indent=2))


if __name__ == "__main__":
    main()
