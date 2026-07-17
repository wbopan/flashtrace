"""Unified VQA-X and A-OKVQA adapters.

The adapters deliberately expose the same image + question -> rationale + answer
shape.  Human rationales are retained as metadata; model-generated reasoning is
the attribution target in the evaluation runner.
"""

from __future__ import annotations

import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class MultimodalExample:
    dataset: str
    split: str
    question_id: str
    image_id: int
    coco_split: str
    image_path: Path
    question: str
    answers: tuple[str, ...]
    rationales: tuple[str, ...]

    @property
    def majority_answer(self) -> str:
        normalized = [_normalize_answer(answer) for answer in self.answers]
        winner = Counter(normalized).most_common(1)[0][0]
        return next(
            answer for answer in self.answers if _normalize_answer(answer) == winner
        )


_CONTRACTIONS = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hes": "he's",
    "im": "i'm",
    "isnt": "isn't",
    "shouldnt": "shouldn't",
    "thats": "that's",
    "theyre": "they're",
    "wasnt": "wasn't",
    "werent": "weren't",
    "whats": "what's",
    "wheres": "where's",
    "wont": "won't",
    "wouldnt": "wouldn't",
    "youre": "you're",
}
_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_ARTICLES = {"a", "an", "the"}
_PUNCTUATION = set(string.punctuation) - {"'", "."}


def _normalize_answer(answer: str) -> str:
    """Approximate the official VQA answer normalization."""

    value = str(answer).casefold().replace("\n", " ").replace("\t", " ").strip()
    value = re.sub(r"(?<!\d)\.(?!\d)", " ", value)
    value = "".join(" " if char in _PUNCTUATION else char for char in value)
    words = []
    for word in value.split():
        word = _NUMBER_WORDS.get(word, word)
        word = _CONTRACTIONS.get(word, word)
        if word not in _ARTICLES:
            words.append(word)
    return " ".join(words)


def vqa_accuracy(prediction: str, answers: Iterable[str]) -> float:
    """Return official-style leave-one-out VQA consensus accuracy."""

    references = [_normalize_answer(answer) for answer in answers]
    if not references:
        return 0.0
    predicted = _normalize_answer(prediction)
    per_annotator = []
    for index in range(len(references)):
        matches = sum(
            reference == predicted
            for other_index, reference in enumerate(references)
            if other_index != index
        )
        per_annotator.append(min(1.0, matches / 3.0))
    return sum(per_annotator) / len(per_annotator)


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python -m evaluations.multimodal.prepare_data` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _vqa_x_examples(data_root: Path, split: str) -> list[MultimodalExample]:
    path = data_root / "vqa_x" / "nlxgpt" / f"vqaX_{split}.json"
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a question-id mapping in {path}")

    examples = []
    for question_id, row in payload.items():
        image_id = int(row["image_id"])
        image_name = row.get("image_name") or (
            f"COCO_{split}2014_{image_id:012d}.jpg"
        )
        image_name_match = re.match(r"COCO_([^_]+)_", image_name)
        coco_split = image_name_match.group(1) if image_name_match else f"{split}2014"
        answers = tuple(str(item["answer"]) for item in row.get("answers", []))
        examples.append(
            MultimodalExample(
                dataset="vqa_x",
                split=split,
                question_id=str(question_id),
                image_id=image_id,
                coco_split=coco_split,
                image_path=data_root / "coco" / coco_split / image_name,
                question=str(row["question"]),
                answers=answers,
                rationales=tuple(str(item) for item in row.get("explanation", [])),
            )
        )
    return examples


def _aokvqa_examples(data_root: Path, split: str) -> list[MultimodalExample]:
    path = data_root / "aokvqa" / f"aokvqa_v1p0_{split}.json"
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")

    examples = []
    for row in payload:
        image_id = int(row["image_id"])
        rationales = tuple(str(item) for item in row.get("rationales", []))
        direct_answers = tuple(str(item) for item in row.get("direct_answers", []))
        if not direct_answers and "correct_choice_idx" in row:
            direct_answers = (
                str(row["choices"][int(row["correct_choice_idx"])]),
            )
        examples.append(
            MultimodalExample(
                dataset="aokvqa",
                split=split,
                question_id=str(row["question_id"]),
                image_id=image_id,
                coco_split=f"{split}2017",
                image_path=(
                    data_root / "coco" / f"{split}2017" / f"{image_id:012d}.jpg"
                ),
                question=str(row["question"]),
                answers=direct_answers,
                rationales=rationales,
            )
        )
    return examples


def load_examples(
    dataset: str,
    data_root: str | Path = "data",
    *,
    split: str = "val",
    limit: int | None = None,
) -> list[MultimodalExample]:
    """Load either benchmark through one stable schema."""

    root = Path(data_root)
    normalized = dataset.casefold().replace("-", "_")
    if normalized in {"vqa_x", "vqax"}:
        examples = _vqa_x_examples(root, split)
    elif normalized in {"aokvqa", "a_okvqa"}:
        examples = _aokvqa_examples(root, split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if limit is not None:
        return examples[: max(0, int(limit))]
    return examples
