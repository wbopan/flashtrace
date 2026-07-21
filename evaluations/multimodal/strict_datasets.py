"""Strict dataset manifests for visual reasoning attribution.

Dataset records contain only model inputs and evaluation metadata.  Generated
``THINKING`` and ``OUTPUT`` fields belong in a separate model-record file and
must never be populated from dataset rationales or functional programs.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .visa_grounding import build_visa_record, stratified_sample


SCHEMA_VERSION = 2

CLEVR_FINAL_FAMILIES: dict[str, frozenset[str]] = {
    "count": frozenset({"count"}),
    "exist": frozenset({"exist"}),
    "compare_integer": frozenset({"equal_integer", "less_than", "greater_than"}),
    "compare_attribute": frozenset(
        {"equal_color", "equal_shape", "equal_size", "equal_material"}
    ),
    "query_attribute": frozenset(
        {"query_color", "query_shape", "query_size", "query_material"}
    ),
}

# The official Dataset Viewer exposes exactly these seven files for the 3,000
# row Wiki-VISA test split.  Pinning the split-specific list prevents
# ``load_dataset(repo_id, split="test")`` from downloading the 100 GB train
# split as part of repository resolution.
WIKI_VISA_TEST_SHARDS: tuple[tuple[str, int], ...] = (
    (
        "https://huggingface.co/datasets/MrLight/wiki-visa/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        453_613_894,
    ),
    (
        "https://huggingface.co/datasets/MrLight/wiki-visa/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0001.parquet",
        459_393_975,
    ),
    (
        "https://huggingface.co/datasets/MrLight/wiki-visa/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0002.parquet",
        517_280_056,
    ),
    (
        "https://huggingface.co/datasets/MrLight/wiki-visa/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0003.parquet",
        539_081_857,
    ),
    (
        "https://huggingface.co/datasets/MrLight/wiki-visa/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0004.parquet",
        513_087_252,
    ),
    (
        "https://huggingface.co/datasets/MrLight/wiki-visa/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0005.parquet",
        462_467_571,
    ),
    (
        "https://huggingface.co/datasets/MrLight/wiki-visa/resolve/"
        "refs%2Fconvert%2Fparquet/default/test/0006.parquet",
        464_238_259,
    ),
)

WIKI_METADATA_COLUMNS = (
    "id",
    "question",
    "long_answer_type",
    "url",
    "short_answer",
    "short_answer_type",
    "image_size",
    "candidates",
    "pos_idx",
    "bounding_box",
)


def _input_record(*, image_path: Path, question: str) -> dict[str, Any]:
    return {
        "I_IMAGE": str(image_path),
        "I_QUESTION": question,
    }


def _evaluation_record(
    *,
    reference_output: str,
    evidence_boxes: Sequence[Sequence[float]] | None = None,
    evidence_masks: Mapping[str, str | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "REFERENCE_OUTPUT": reference_output,
        "EVIDENCE_BOXES": (
            [list(map(float, box)) for box in evidence_boxes]
            if evidence_boxes is not None
            else None
        ),
        "EVIDENCE_MASKS": dict(evidence_masks) if evidence_masks is not None else None,
        "metadata": dict(metadata or {}),
    }


def validate_dataset_record(record: Mapping[str, Any]) -> None:
    """Validate that a manifest record contains no model-generated fields."""

    required = {"schema_version", "benchmark", "sample_id", "input", "evaluation"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"dataset record is missing keys: {sorted(missing)}")
    forbidden = {"THINKING", "OUTPUT", "THINKING_SPAN", "OUTPUT_SPAN"}
    leaked = forbidden.intersection(record)
    if leaked:
        raise ValueError(f"model fields leaked into dataset record: {sorted(leaked)}")
    inputs = record["input"]
    evaluation = record["evaluation"]
    if set(inputs) != {"I_IMAGE", "I_QUESTION"}:
        raise ValueError("input must contain exactly I_IMAGE and I_QUESTION")
    if not str(inputs["I_IMAGE"]) or not str(inputs["I_QUESTION"]):
        raise ValueError("I_IMAGE and I_QUESTION must be non-empty")
    if "REFERENCE_OUTPUT" not in evaluation:
        raise ValueError("evaluation metadata needs REFERENCE_OUTPUT")


def clevr_answer(value: Any) -> str:
    """Convert an official CLEVR answer into a stable output string."""

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def clevr_reasoning_family(question: Mapping[str, Any]) -> str:
    program = question.get("program") or []
    if not program:
        raise ValueError("CLEVR question has no functional program")
    final_operation = str(program[-1].get("type", ""))
    for family, operations in CLEVR_FINAL_FAMILIES.items():
        if final_operation in operations:
            return family
    raise ValueError(f"unsupported CLEVR final operation: {final_operation}")


def select_clevr_complex(
    questions: Iterable[Mapping[str, Any]],
    *,
    dataset_root: Path,
    images_root: Path,
    sample_size: int = 20,
    seed: int = 17,
    min_program_steps: int = 12,
    exclude_question_indices: set[int] | None = None,
    exclude_image_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Select a balanced, deterministic, reasoning-heavy CLEVR-XAI subset.

    The primary evidence is Unique First-non-empty.  Unique and Union are kept
    for sensitivity analysis, while All Objects is retained only as a sanity
    reference.  Functional programs are evaluation metadata and never input.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    family_names = tuple(CLEVR_FINAL_FAMILIES)
    base, remainder = divmod(sample_size, len(family_names))
    target_counts = {
        family: base + int(index < remainder)
        for index, family in enumerate(family_names)
    }

    excluded_questions = exclude_question_indices or set()
    excluded_images = exclude_image_indices or set()
    candidates: dict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for fallback_index, question in enumerate(questions):
        question_index = int(question.get("question_index", fallback_index))
        image_index = int(question.get("image_index", -1))
        if question_index in excluded_questions or image_index in excluded_images:
            continue
        program = question.get("program") or []
        if len(program) < min_program_steps:
            continue
        try:
            family = clevr_reasoning_family(question)
        except ValueError:
            continue
        candidates[family].append((question_index, question))

    missing = [
        family
        for family, count in target_counts.items()
        if len(candidates[family]) < count
    ]
    if missing:
        raise ValueError(f"not enough eligible CLEVR questions in families: {missing}")

    rng = random.Random(seed)
    selected: list[tuple[int, Mapping[str, Any], str]] = []
    used_images: set[int] = set()
    for family in family_names:
        pool = list(candidates[family])
        rng.shuffle(pool)
        family_selected: list[tuple[int, Mapping[str, Any], str]] = []
        for question_index, question in pool:
            image_index = int(question["image_index"])
            if image_index in used_images:
                continue
            # Validate evidence only for candidates that could actually enter
            # the subset. Loading all 100k masks before sampling turns a small
            # deterministic selection into minutes of unnecessary random I/O.
            primary_mask = (
                dataset_root
                / "ground_truth_complex_questions_unique_firstnonempty"
                / f"{question_index}.npy"
            )
            if not primary_mask.exists():
                continue
            mask = np.load(primary_mask, mmap_mode="r")
            if mask.ndim != 2 or not np.any(mask):
                continue
            family_selected.append((question_index, question, family))
            used_images.add(image_index)
            if len(family_selected) == target_counts[family]:
                break
        if len(family_selected) != target_counts[family]:
            raise ValueError(f"could not select unique images for CLEVR family {family}")
        selected.extend(family_selected)

    records: list[dict[str, Any]] = []
    for question_index, question, family in selected:
        image_path = images_root / str(question["image_filename"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        mask_paths = {
            "primary_unique_firstnonempty": str(
                dataset_root
                / "ground_truth_complex_questions_unique_firstnonempty"
                / f"{question_index}.npy"
            ),
            "sensitivity_unique": str(
                dataset_root
                / "ground_truth_complex_questions_unique"
                / f"{question_index}.npy"
            ),
            "sensitivity_union": str(
                dataset_root
                / "ground_truth_complex_questions_union"
                / f"{question_index}.npy"
            ),
            "sanity_all_objects": str(
                dataset_root
                / "ground_truth_complex_questions_all_objects"
                / f"{question_index}.npy"
            ),
        }
        for name, mask_path in list(mask_paths.items()):
            if not Path(mask_path).exists():
                mask_paths[name] = None

        record = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": "clevr_xai_complex",
            "sample_id": f"clevr-complex-{question_index:06d}",
            "input": _input_record(
                image_path=image_path,
                question=str(question["question"]),
            ),
            "evaluation": _evaluation_record(
                reference_output=clevr_answer(question["answer"]),
                evidence_masks=mask_paths,
                metadata={
                    "question_index": question_index,
                    "image_index": int(question["image_index"]),
                    "reasoning_family": family,
                    "program_steps": len(question["program"]),
                    "final_operation": str(question["program"][-1]["type"]),
                    "functional_program": question["program"],
                    "template_filename": str(question.get("template_filename", "")),
                    "question_family_index": int(
                        question.get("question_family_index", -1)
                    ),
                },
            ),
        }
        validate_dataset_record(record)
        records.append(record)

    records.sort(key=lambda record: record["sample_id"])
    return records


@dataclass(frozen=True)
class WikiRowLocation:
    shard_path: Path
    local_row_index: int


def wiki_test_shard_paths(parquet_root: Path) -> list[Path]:
    """Return and size-check the seven official Wiki-VISA test shards."""

    paths = []
    for url, expected_size in WIKI_VISA_TEST_SHARDS:
        path = parquet_root / url.rsplit("/", 1)[-1]
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; download the seven URLs in WIKI_VISA_TEST_SHARDS"
            )
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(
                f"{path} has {actual_size} bytes; expected {expected_size}"
            )
        paths.append(path)
    return paths


def _wiki_metadata_rows(
    shard_paths: Sequence[Path],
) -> tuple[list[dict[str, Any]], dict[int, WikiRowLocation]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Wiki-VISA preparation requires pyarrow") from error

    records: list[dict[str, Any]] = []
    locations: dict[int, WikiRowLocation] = {}
    global_row_index = 0
    for shard_path in shard_paths:
        table = pq.read_table(shard_path, columns=list(WIKI_METADATA_COLUMNS))
        for local_row_index, row in enumerate(table.to_pylist()):
            record = build_visa_record(global_row_index, row)
            records.append(record)
            locations[global_row_index] = WikiRowLocation(
                shard_path=shard_path,
                local_row_index=local_row_index,
            )
            global_row_index += 1
    return records, locations


def _decode_arrow_image(value: Any) -> tuple[bytes, str]:
    if isinstance(value, Mapping):
        image_bytes = value.get("bytes")
        source_name = str(value.get("path") or "")
    else:
        image_bytes = None
        source_name = ""
    if image_bytes is None:
        raise ValueError("Wiki-VISA image column has no embedded bytes")
    return bytes(image_bytes), source_name


def _read_wiki_images(
    locations: Mapping[int, WikiRowLocation],
    selected_indices: set[int],
) -> dict[int, tuple[bytes, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Wiki-VISA preparation requires pyarrow") from error

    by_shard: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for global_index in selected_indices:
        location = locations[global_index]
        by_shard[location.shard_path].append((global_index, location.local_row_index))

    images: dict[int, tuple[bytes, str]] = {}
    for shard_path, requested in by_shard.items():
        parquet_file = pq.ParquetFile(shard_path)
        row_group_offsets: list[tuple[int, int, int]] = []
        start = 0
        for row_group in range(parquet_file.num_row_groups):
            count = parquet_file.metadata.row_group(row_group).num_rows
            row_group_offsets.append((row_group, start, start + count))
            start += count
        for row_group, group_start, group_end in row_group_offsets:
            group_requests = [
                (global_index, local_index - group_start)
                for global_index, local_index in requested
                if group_start <= local_index < group_end
            ]
            if not group_requests:
                continue
            values = parquet_file.read_row_group(row_group, columns=["image"])["image"]
            for global_index, relative_index in group_requests:
                images[global_index] = _decode_arrow_image(values[relative_index].as_py())
    return images


def select_wiki_visa(
    *,
    parquet_root: Path,
    images_root: Path,
    sample_size: int = 20,
    seed: int = 17,
    max_reference_words: int | None = None,
    strata: set[str] | None = None,
    exclude_row_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Load the complete Wiki-VISA test metadata and materialize a fixed subset."""

    shard_paths = wiki_test_shard_paths(parquet_root)
    source_records, locations = _wiki_metadata_rows(shard_paths)
    if len(source_records) != 3_000:
        raise ValueError(
            f"expected 3,000 Wiki-VISA test rows, found {len(source_records)}"
        )
    if max_reference_words is not None:
        source_records = [
            record
            for record in source_records
            if 0 < len(str(record["reference_answer"]).split()) <= max_reference_words
        ]
    if strata:
        source_records = [
            record for record in source_records if str(record["stratum"]) in strata
        ]
    if exclude_row_indices:
        source_records = [
            record
            for record in source_records
            if int(record["hf_row_index"]) not in exclude_row_indices
        ]
    if strata:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in source_records:
            grouped[str(record["stratum"])].append(record)
        group_names = sorted(grouped)
        base, remainder = divmod(sample_size, len(group_names))
        rng = random.Random(seed)
        selected = []
        for index, name in enumerate(group_names):
            pool = list(grouped[name])
            rng.shuffle(pool)
            count = base + int(index < remainder)
            if len(pool) < count:
                raise ValueError(
                    f"Wiki-VISA stratum {name!r} has {len(pool)} rows; needs {count}"
                )
            selected.extend(pool[:count])
        selected.sort(key=lambda record: int(record["hf_row_index"]))
    else:
        selected = stratified_sample(source_records, sample_size, seed=seed)
    selected_indices = {int(record["hf_row_index"]) for record in selected}
    images = _read_wiki_images(locations, selected_indices)
    images_root.mkdir(parents=True, exist_ok=True)

    records = []
    for source in selected:
        row_index = int(source["hf_row_index"])
        image_bytes, source_name = images[row_index]
        suffix = Path(source_name).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
            try:
                from PIL import Image

                image_format = Image.open(io.BytesIO(image_bytes)).format
            except Exception:
                image_format = "PNG"
            suffix = f".{str(image_format).casefold()}"
        image_path = images_root / f"{row_index:04d}{suffix}"
        image_path.write_bytes(image_bytes)

        record = {
            "schema_version": SCHEMA_VERSION,
            "benchmark": "wiki_visa",
            "sample_id": f"wiki-visa-{row_index:04d}",
            "input": _input_record(
                image_path=image_path,
                question=source["question"],
            ),
            "evaluation": _evaluation_record(
                reference_output=source["reference_answer"],
                evidence_boxes=[source["evidence_bbox_xyxy"]],
                metadata={
                    "hf_dataset": source["hf_dataset"],
                    "hf_split": source["hf_split"],
                    "hf_row_index": row_index,
                    "stratum": source["stratum"],
                    "image_size": source["image_size"],
                    "bbox_format": "xyxy_absolute",
                    "long_answer_type": source["long_answer_type"],
                    "source_url": source["source_url"],
                },
            ),
        }
        validate_dataset_record(record)
        records.append(record)
    return records


def write_jsonl(records: Iterable[Mapping[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_dataset_record(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _load_clevr_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, Mapping) else None
    if not isinstance(questions, list):
        raise ValueError(f"{path} does not contain a questions list")
    return questions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("clevr-xai-complex", "wiki-visa"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-program-steps", type=int, default=12)
    parser.add_argument(
        "--max-reference-words",
        type=int,
        help="Wiki-VISA only: retain concise automatically scorable references.",
    )
    parser.add_argument(
        "--wiki-stratum",
        action="append",
        choices=("first_page_passage", "later_page_passage", "non_passage"),
        help="Wiki-VISA only: restrict selection to one or more strata.",
    )
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Exclude records already present in these manifests. CLEVR also "
            "excludes their image indices to avoid reusing a visual scene."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    excluded_question_indices: set[int] = set()
    excluded_image_indices: set[int] = set()
    excluded_wiki_rows: set[int] = set()
    for manifest in args.exclude_manifest:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record["evaluation"]["metadata"]
            if record.get("benchmark") == "clevr_xai_complex":
                excluded_question_indices.add(int(metadata["question_index"]))
                excluded_image_indices.add(int(metadata["image_index"]))
            elif record.get("benchmark") == "wiki_visa":
                excluded_wiki_rows.add(int(metadata["hf_row_index"]))
    if args.dataset == "clevr-xai-complex":
        dataset_root = args.data_root / "clevr_xai" / "CLEVR-XAI_v1.0"
        images_root = (
            args.data_root
            / "clevr_xai"
            / "CLEVR-XAI_v1.0_images_masks"
            / "images"
        )
        questions = _load_clevr_questions(
            dataset_root / "CLEVR-XAI_complex_questions.json"
        )
        records = select_clevr_complex(
            questions,
            dataset_root=dataset_root,
            images_root=images_root,
            sample_size=args.sample_size,
            seed=args.seed,
            min_program_steps=args.min_program_steps,
            exclude_question_indices=excluded_question_indices,
            exclude_image_indices=excluded_image_indices,
        )
    else:
        records = select_wiki_visa(
            parquet_root=args.data_root / "wiki_visa" / "test_parquet",
            images_root=args.data_root / "wiki_visa" / "images",
            sample_size=args.sample_size,
            seed=args.seed,
            max_reference_words=args.max_reference_words,
            strata=set(args.wiki_stratum or []),
            exclude_row_indices=excluded_wiki_rows,
        )

    count = write_jsonl(records, args.output)
    summary: dict[str, Any] = {
        "output": str(args.output),
        "records": count,
        "schema_version": SCHEMA_VERSION,
    }
    if args.dataset == "clevr-xai-complex":
        family_counts: dict[str, int] = defaultdict(int)
        for record in records:
            family_counts[record["evaluation"]["metadata"]["reasoning_family"]] += 1
        summary["reasoning_families"] = dict(family_counts)
    else:
        stratum_counts: dict[str, int] = defaultdict(int)
        for record in records:
            stratum_counts[record["evaluation"]["metadata"]["stratum"]] += 1
        summary["strata"] = dict(stratum_counts)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
