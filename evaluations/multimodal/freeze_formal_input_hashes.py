"""Materialize immutable hashes for frozen formal multimodal inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .strict_generation import read_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_record_sha256(record: dict[str, Any]) -> str:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(canonical)


def build_manifest_hashes(root: Path, manifest: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = manifest.resolve()
    records = read_jsonl(manifest)
    samples = []
    for record in sorted(records, key=lambda value: str(value["sample_id"])):
        input_record = record["input"]
        image_value = str(input_record["I_IMAGE"])
        image_path = Path(image_value)
        if image_path.is_absolute():
            raise ValueError(f"frozen image path must be relative: {image_path}")
        resolved_image = (root / image_path).resolve()
        if not resolved_image.is_relative_to(root):
            raise ValueError(f"frozen image escapes repository root: {image_path}")
        if not resolved_image.is_file():
            raise FileNotFoundError(resolved_image)
        samples.append(
            {
                "sample_id": str(record["sample_id"]),
                "image_path": image_path.as_posix(),
                "image_sha256": sha256_file(resolved_image),
                "question_sha256": sha256_text(str(input_record["I_QUESTION"])),
                "dataset_record_sha256": canonical_record_sha256(record),
            }
        )
    return {
        "manifest_path": manifest.relative_to(root).as_posix(),
        "manifest_sha256": sha256_file(manifest),
        "sample_count": len(samples),
        "samples": samples,
    }


def build_payload(root: Path, manifests: list[Path]) -> dict[str, Any]:
    bundles = [
        build_manifest_hashes(root, manifest)
        for manifest in sorted(manifests, key=lambda path: str(path))
    ]
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "canonical_dataset_record": (
            "json.dumps(sort_keys=True,separators=(',',':'),ensure_ascii=False)"
        ),
        "manifests": bundles,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_payload(args.root, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "manifests": len(payload["manifests"]),
                "samples": sum(
                    int(bundle["sample_count"])
                    for bundle in payload["manifests"]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
