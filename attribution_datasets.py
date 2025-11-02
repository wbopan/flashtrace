from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

from llm_attr_eval import create_sentences


@dataclass
class AttributionExample:
    prompt: str
    target: Optional[str] = None
    indices_to_explain: Optional[List[int]] = None
    attr_mask_indices: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class AttributionDataset(Iterable[AttributionExample]):
    """Base iterable for attribution-ready datasets."""

    name: str = "dataset"

    def __init__(self) -> None:
        self.examples: List[AttributionExample] = []

    def __iter__(self) -> Iterator[AttributionExample]:
        return iter(self.examples)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, item):  # pragma: no cover - convenience
        return self.examples[item]


def _add_dummy_facts_to_prompt(text_sentences: Sequence[str]) -> List[str]:
    """
    Reproduces the original behaviour of interleaving dummy sentences with the
    provided text segments so attribution heads can be masked easily.
    """
    result: List[str] = []
    for sentence in text_sentences:
        result.append(sentence)
        result.append(" Unrelated Sentence.")
    return result


class MathAttributionDataset(AttributionDataset):
    """Dataset wrapper for synthetic math problems with dummy context facts."""

    name = "math"

    def __init__(self, path: str | Path, tokenizer: Any) -> None:
        super().__init__()
        data_path = Path(path)
        with data_path.open("r", encoding="utf-8") as f:
            raw_examples = json.load(f)

        for entry in raw_examples:
            question_text = entry["question"]
            sentences = create_sentences(question_text, tokenizer)
            if not sentences:
                continue

            context_sentences = sentences[:-1]
            question_sentence = sentences[-1]
            if question_sentence.startswith(" "):
                question_sentence = question_sentence[1:]

            context_with_dummy = _add_dummy_facts_to_prompt(context_sentences)
            question_with_dummy = _add_dummy_facts_to_prompt([question_sentence])

            prompt = "".join(context_with_dummy) + "\n" + "".join(question_with_dummy)
            total_sentences = len(context_with_dummy) + len(question_with_dummy)
            attr_mask_indices = list(range(0, total_sentences, 2))

            self.examples.append(
                AttributionExample(
                    prompt=prompt,
                    target=None,
                    indices_to_explain=[-2],
                    attr_mask_indices=attr_mask_indices,
                    metadata={"raw_question": question_text},
                )
            )


class FactsAttributionDataset(AttributionDataset):
    """Dataset wrapper for curated factual prompts with explicit gold attributions."""

    name = "facts"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        data_path = Path(path)
        with data_path.open("r", encoding="utf-8") as f:
            raw_examples = json.load(f)

        for entry in raw_examples:
            metadata = {
                key: value
                for key, value in entry.items()
                if key not in {"prompt", "target", "indices_to_explain", "attr_mask_indices"}
            }
            self.examples.append(
                AttributionExample(
                    prompt=entry["prompt"],
                    target=entry.get("target"),
                    indices_to_explain=entry.get("indices_to_explain"),
                    attr_mask_indices=entry.get("attr_mask_indices"),
                    metadata=metadata,
                )
            )


class MoreHopQAAttributionDataset(AttributionDataset):
    """Dataset wrapper for multi-hop QA prompts without explicit gold attribution."""

    name = "morehopqa"

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        data_path = Path(path)
        with data_path.open("r", encoding="utf-8") as f:
            raw_examples = json.load(f)

        for entry in raw_examples:
            context_chunks = ["".join(item[1]) for item in entry.get("context", [])]
            context = " ".join(context_chunks)
            prompt = context + "\n" + entry["question"]

            self.examples.append(
                AttributionExample(
                    prompt=prompt,
                    target=None,
                    indices_to_explain=[-2],
                    attr_mask_indices=None,
                    metadata={
                        "answer": entry.get("answer"),
                        "id": entry.get("_id"),
                        "original_context": entry.get("context"),
                    },
                )
            )
