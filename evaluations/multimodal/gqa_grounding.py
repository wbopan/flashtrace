"""Build a grounded GQA evaluation manifest from official annotations.

GQA questions include word-to-object visual pointers and functional-program
arguments.  The corresponding scene graphs provide object bounding boxes.
This script joins those annotations so visual attribution methods can be scored
against the objects required by each reasoning problem.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


_OBJECT_ID_PATTERN = re.compile(r"\((\d+)\)")


def _annotation_object_ids(question: Mapping[str, Any]) -> dict[str, set[str]]:
    sources: dict[str, set[str]] = defaultdict(set)
    annotations = question.get("annotations") or {}
    for annotation_name in ("question", "answer", "fullAnswer"):
        pointers = annotations.get(annotation_name) or {}
        if not isinstance(pointers, Mapping):
            continue
        for object_id in pointers.values():
            if isinstance(object_id, (str, int)):
                sources[str(object_id)].add(f"annotation:{annotation_name}")

    for step_index, step in enumerate(question.get("semantic") or []):
        argument = str((step or {}).get("argument", ""))
        for object_id in _OBJECT_ID_PATTERN.findall(argument):
            sources[object_id].add(f"semantic:{step_index}")
    return sources


def _normalized_xyxy(obj: Mapping[str, Any], image_width: int, image_height: int) -> list[float]:
    x = float(obj["x"])
    y = float(obj["y"])
    width = float(obj["w"])
    height = float(obj["h"])
    if width < 0 or height < 0:
        raise ValueError("GQA object boxes must have non-negative width and height")

    x1 = min(max(x / image_width, 0.0), 1.0)
    y1 = min(max(y / image_height, 0.0), 1.0)
    x2 = min(max((x + width) / image_width, 0.0), 1.0)
    y2 = min(max((y + height) / image_height, 0.0), 1.0)
    return [x1, y1, x2, y2]


def build_grounded_record(
    question_id: str,
    question: Mapping[str, Any],
    scene_graph: Mapping[str, Any],
) -> dict[str, Any]:
    """Join one GQA question to its scene-graph evidence objects."""

    image_id = str(question["imageId"])
    image_width = int(scene_graph["width"])
    image_height = int(scene_graph["height"])
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"scene graph {image_id} has invalid image dimensions")

    sources = _annotation_object_ids(question)
    objects = scene_graph.get("objects") or {}
    evidence: list[dict[str, Any]] = []
    missing_object_ids: list[str] = []
    for object_id in sorted(sources, key=lambda value: (len(value), value)):
        obj = objects.get(object_id)
        if obj is None:
            missing_object_ids.append(object_id)
            continue
        evidence.append(
            {
                "object_id": object_id,
                "name": str(obj.get("name", "")),
                "bbox_xywh": [
                    float(obj["x"]),
                    float(obj["y"]),
                    float(obj["w"]),
                    float(obj["h"]),
                ],
                "bbox_xyxy_normalized": _normalized_xyxy(obj, image_width, image_height),
                "attributes": list(obj.get("attributes") or []),
                "sources": sorted(sources[object_id]),
            }
        )

    types = question.get("types") or {}
    return {
        "schema_version": 1,
        "benchmark": "gqa_balanced_grounded",
        "sample_id": str(question_id),
        "image_id": image_id,
        "image_file": f"{image_id}.jpg",
        "image_size": {"width": image_width, "height": image_height},
        "question": str(question.get("question", "")),
        "reference_answer": str(question.get("answer", "")),
        "full_answer": str(question.get("fullAnswer", "")),
        "is_balanced": bool(question.get("isBalanced", False)),
        "types": {
            "structural": types.get("structural"),
            "semantic": types.get("semantic"),
            "detailed": types.get("detailed"),
        },
        "program": list(question.get("semantic") or []),
        "program_steps": len(question.get("semantic") or []),
        "evidence": evidence,
        "evidence_object_ids": [item["object_id"] for item in evidence],
        "missing_object_ids": missing_object_ids,
    }


def iter_grounded_records(
    questions: Mapping[str, Mapping[str, Any]],
    scene_graphs: Mapping[str, Mapping[str, Any]],
    *,
    balanced_only: bool = True,
    min_program_steps: int = 2,
    min_evidence_objects: int = 1,
) -> Iterator[dict[str, Any]]:
    """Yield deterministic, filtered grounded records."""

    if min_program_steps < 0:
        raise ValueError("min_program_steps must be non-negative")
    if min_evidence_objects < 0:
        raise ValueError("min_evidence_objects must be non-negative")

    for question_id in sorted(questions, key=lambda value: (len(str(value)), str(value))):
        question = questions[question_id]
        if balanced_only and not question.get("isBalanced", False):
            continue
        if len(question.get("semantic") or []) < min_program_steps:
            continue
        image_id = str(question.get("imageId", ""))
        scene_graph = scene_graphs.get(image_id)
        if scene_graph is None:
            continue
        record = build_grounded_record(str(question_id), question, scene_graph)
        if len(record["evidence"]) < min_evidence_objects:
            continue
        yield record


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True, help="Official GQA questions JSON")
    parser.add_argument("--scene-graphs", type=Path, required=True, help="Official GQA scene graphs JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL manifest")
    parser.add_argument("--balanced-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-program-steps", type=int, default=2)
    parser.add_argument("--min-evidence-objects", type=int, default=1)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    questions = _load_json(args.questions)
    scene_graphs = _load_json(args.scene_graphs)
    records = list(
        iter_grounded_records(
            questions,
            scene_graphs,
            balanced_only=args.balanced_only,
            min_program_steps=args.min_program_steps,
            min_evidence_objects=args.min_evidence_objects,
        )
    )

    if args.sample_size is not None:
        if args.sample_size <= 0:
            raise SystemExit("--sample-size must be positive")
        if args.sample_size < len(records):
            records = random.Random(args.seed).sample(records, args.sample_size)
            records.sort(key=lambda item: (len(item["sample_id"]), item["sample_id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    missing = sum(bool(record["missing_object_ids"]) for record in records)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(records),
                "records_with_missing_object_ids": missing,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
