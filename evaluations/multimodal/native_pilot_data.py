"""Materialize native VizWiz-LF and VISTAQA samples for strict pilots.

This adapter does not generate or rewrite questions and answers. It converts
official annotations into the repository's separated dataset-record schema and
decodes VISTAQA's official COCO RLE masks into lossless boolean arrays.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .strict_datasets import (
    SCHEMA_VERSION,
    _evaluation_record,
    _input_record,
    validate_dataset_record,
    write_jsonl,
)


def _compressed_coco_counts(value: str | bytes) -> list[int]:
    text = value.decode("ascii") if isinstance(value, bytes) else value
    counts: list[int] = []
    position = 0
    while position < len(text):
        number = 0
        shift = 0
        while True:
            code = ord(text[position]) - 48
            position += 1
            number |= (code & 0x1F) << (5 * shift)
            shift += 1
            if not (code & 0x20):
                if code & 0x10:
                    number |= -1 << (5 * shift)
                break
        if len(counts) > 2:
            number += counts[-2]
        counts.append(number)
    return counts


def decode_coco_rle(segmentation: Mapping[str, Any]) -> np.ndarray:
    """Decode compressed or uncompressed COCO RLE in column-major order."""

    size = segmentation.get("size")
    if not isinstance(size, Sequence) or len(size) != 2:
        raise ValueError("COCO RLE needs a two-element size")
    height, width = (int(size[0]), int(size[1]))
    raw_counts = segmentation.get("counts")
    if isinstance(raw_counts, (str, bytes)):
        counts = _compressed_coco_counts(raw_counts)
    elif isinstance(raw_counts, Sequence):
        counts = [int(value) for value in raw_counts]
    else:
        raise ValueError("COCO RLE counts must be compressed text or a sequence")

    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    foreground = False
    for run in counts:
        if run < 0 or offset + run > flat.size:
            raise ValueError("invalid COCO RLE run length")
        if foreground and run:
            flat[offset : offset + run] = 1
        offset += run
        foreground = not foreground
    if offset != flat.size:
        raise ValueError(f"COCO RLE covers {offset} pixels, expected {flat.size}")
    return flat.reshape((height, width), order="F").astype(bool)


def build_vistaqa_records(
    annotation_root: Path,
    images_root: Path,
    masks_root: Path,
) -> list[dict[str, Any]]:
    masks_root.mkdir(parents=True, exist_ok=True)
    annotations = sorted(
        annotation_root.glob("*.json"), key=lambda path: int(path.stem)
    )
    records: list[dict[str, Any]] = []
    for annotation_path in annotations:
        source = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_metadata = source["image"]
        image_id = int(image_metadata["image_id"])
        image_path = images_root / str(image_metadata["file_name"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as image:
            if image.size != (
                int(image_metadata["width"]),
                int(image_metadata["height"]),
            ):
                raise ValueError(f"VISTAQA image size mismatch for {image_id}")

        instance_masks = [
            decode_coco_rle(annotation["segmentation"])
            for annotation in source.get("annotations", [])
        ]
        if not instance_masks:
            raise ValueError(f"VISTAQA sample {image_id} has no evidence mask")
        union = np.logical_or.reduce(instance_masks)
        if not np.any(union):
            raise ValueError(f"VISTAQA sample {image_id} has an empty evidence mask")
        mask_path = masks_root / f"{image_id}.npy"
        np.save(mask_path, union)

        official_boxes = [
            [float(value) for value in annotation["bbox"]]
            for annotation in source["annotations"]
        ]
        xyxy_boxes = [
            [x, y, x + width, y + height]
            for x, y, width, height in official_boxes
        ]
        record = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": "vistaqa",
            "sample_id": f"vistaqa-{image_id:04d}",
            "input": _input_record(
                image_path=image_path,
                question=str(source["question"]),
            ),
            "evaluation": _evaluation_record(
                reference_output=str(source["answer"]),
                evidence_boxes=xyxy_boxes,
                evidence_masks={"primary": str(mask_path)},
                metadata={
                    "hf_dataset": "vista26/VistaQA",
                    "hf_split": "train",
                    "image_id": image_id,
                    "image_size": {
                        "width": int(image_metadata["width"]),
                        "height": int(image_metadata["height"]),
                    },
                    "task_type": str(source["task_type"]),
                    "task_domain": str(source["task_domain"]),
                    "num_instances": int(source["num_instances"]),
                    "hallucination": int(source["hallucination"]),
                    "official_bboxes_xywh": official_boxes,
                    "bbox_format": "xyxy_absolute_converted_from_official_xywh",
                    "official_annotation_path": str(annotation_path),
                    "prompt_profile": "concise",
                },
            ),
        }
        validate_dataset_record(record)
        records.append(record)
    return records


def build_vizwiz_lf_records(
    expert_json: Path,
    images_root: Path,
    sample_ids: Sequence[str],
) -> list[dict[str, Any]]:
    source = json.loads(expert_json.read_text(encoding="utf-8"))
    if not isinstance(source, Mapping):
        raise ValueError("VizWiz-LF expert annotations must be keyed by record ID")
    records: list[dict[str, Any]] = []
    for raw_id in sample_ids:
        record_id = str(int(raw_id))
        if record_id not in source:
            raise ValueError(f"VizWiz-LF record ID not found: {record_id}")
        item = source[record_id]
        if item.get("model") != "Expert":
            raise ValueError(f"VizWiz-LF record {record_id} is not an Expert answer")
        candidates = sorted(images_root.glob(f"{int(record_id):03d}.*"))
        if len(candidates) != 1:
            raise ValueError(
                f"expected one local image for VizWiz-LF {record_id}, found {candidates}"
            )
        image_path = candidates[0]
        with Image.open(image_path) as image:
            image_size = {"width": image.width, "height": image.height}

        record = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": "vizwiz_lf",
            "sample_id": f"vizwiz-lf-{int(record_id):03d}",
            "input": _input_record(
                image_path=image_path,
                question=str(item["question"]),
            ),
            "evaluation": _evaluation_record(
                reference_output=str(item["answer_paragraph"]),
                evidence_boxes=None,
                evidence_masks=None,
                metadata={
                    "source_dataset": "VizWiz-LF",
                    "official_record_id": record_id,
                    "source_model": "Expert",
                    "image_url": str(item["image_url"]),
                    "image_size": image_size,
                    "question_type": str(item["question_type"]),
                    "answerability": str(item["answerability"]),
                    "prompt_profile": "long_form",
                },
            ),
        }
        validate_dataset_record(record)
        records.append(record)
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("vistaqa", "vizwiz-lf"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--masks-root", type=Path)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.dataset == "vistaqa":
        if args.masks_root is None:
            raise ValueError("--masks-root is required for VISTAQA")
        records = build_vistaqa_records(
            args.source, args.images_root, args.masks_root
        )
    else:
        if not args.sample_id:
            raise ValueError("at least one --sample-id is required for VizWiz-LF")
        records = build_vizwiz_lf_records(
            args.source, args.images_root, args.sample_id
        )
    count = write_jsonl(records, args.output)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "output": str(args.output),
                "records": count,
                "sample_ids": [record["sample_id"] for record in records],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
