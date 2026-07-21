"""Verify the complete official evaluation data used by strict experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .strict_datasets import WIKI_VISA_TEST_SHARDS, wiki_test_shard_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(data_root: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    wiki_paths = wiki_test_shard_paths(data_root / "wiki_visa" / "test_parquet")
    wiki_rows = [pq.ParquetFile(path).metadata.num_rows for path in wiki_paths]
    clevr_root = data_root / "clevr_xai"
    annotations = clevr_root / "CLEVR-XAI_v1.0"
    media = clevr_root / "CLEVR-XAI_v1.0_images_masks"
    complex_payload = json.loads(
        (annotations / "CLEVR-XAI_complex_questions.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        "wiki_visa": {
            "source": "MrLight/wiki-visa",
            "split": "test",
            "expected_rows": 3_000,
            "rows": sum(wiki_rows),
            "shards": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "expected_bytes": WIKI_VISA_TEST_SHARDS[index][1],
                    "rows": wiki_rows[index],
                }
                for index, path in enumerate(wiki_paths)
            ],
        },
        "clevr_xai": {
            "release": "v1.0",
            "complex_questions": len(complex_payload["questions"]),
            "images": len(list((media / "images").glob("*.png"))),
            "object_masks": len(list((media / "masks").glob("*.png"))),
            "ground_truth_counts": {
                name: len(list((annotations / name).glob("*.npy")))
                for name in (
                    "ground_truth_complex_questions_unique",
                    "ground_truth_complex_questions_unique_firstnonempty",
                    "ground_truth_complex_questions_union",
                    "ground_truth_complex_questions_all_objects",
                )
            },
            "archives": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in (
                    clevr_root / "raw" / "CLEVR-XAI_v1.0.zip",
                    clevr_root / "raw" / "CLEVR-XAI_v1.0_images_masks.zip",
                )
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
