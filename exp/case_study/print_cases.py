#!/usr/bin/env python3
"""
Pretty-print sampled attribution cases from exp/case_study/out/*.jsonl.
Usage:
  python exp/case_study/print_cases.py --topk 5 --attr_field seq_attr > exp/case_study/out/print.log
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Print attribution cases from JSONL files.")
    parser.add_argument(
        "--files",
        type=str,
        nargs="*",
        default=None,
        help="List of JSONL files to read. Defaults to all exp/case_study/out/*.jsonl.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of top sentences to show per sample.",
    )
    parser.add_argument(
        "--attr_field",
        type=str,
        default="seq_attr",
        choices=["seq_attr", "row_attr", "rec_attr"],
        help="Which attribution matrix to use when ranking sentences.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def column_scores(matrix: Sequence[Sequence[float]]) -> List[float]:
    if not matrix:
        return []
    n_cols = max(len(row) for row in matrix)
    scores = [0.0 for _ in range(n_cols)]
    for row in matrix:
        for j, val in enumerate(row):
            try:
                scores[j] += float(val)
            except Exception:
                continue
    return scores


def topk_sentences(sentences: Sequence[str], scores: Sequence[float], k: int) -> List[tuple[int, float, str]]:
    paired = [(idx, float(scores[idx])) for idx in range(min(len(sentences), len(scores)))]
    paired.sort(key=lambda x: x[1], reverse=True)
    result = []
    for idx, score in paired[:k]:
        sent = sentences[idx].replace("\n", "\\n")
        result.append((idx, score, sent))
    return result


def safe_get(record: Dict[str, Any], key: str) -> Optional[Any]:
    return record.get(key)


def print_sample(record: Dict[str, Any], attr_field: str, topk: int) -> None:
    sentences = record.get("all_sentences") or []
    prompt_sentences = record.get("prompt_sentences") or []
    generation_sentences = record.get("generation_sentences") or []

    # Normalize sink indices (default: [-2] => last non-EOS generation)
    raw_indices = record.get("indices_to_explain") or [-2]
    sink_rows: List[int] = []
    for idx in raw_indices:
        if idx < 0:
            idx = len(generation_sentences) + idx
        if 0 <= idx < len(generation_sentences):
            sink_rows.append(idx)
    sink_rows = sorted(set(sink_rows))

    matrix = record.get(attr_field) or []
    # If using full seq_attr, slice to only the sink rows; for row_attr/rec_attr keep as-is (already aggregated).
    if attr_field == "seq_attr" and matrix and sink_rows:
        matrix_slice = []
        for i in sink_rows:
            if 0 <= i < len(matrix):
                matrix_slice.append(matrix[i])
        matrix_for_scores = matrix_slice
    else:
        matrix_for_scores = matrix

    scores = column_scores(matrix_for_scores)
    top_items = topk_sentences(sentences, scores, topk)

    print(f"--- sample {record.get('example_idx')} (attr_func={record.get('attr_func')}) ---")
    print("prompt:")
    print(record.get("prompt") or "")
    print("===")
    print("generation:")
    print(record.get("generation") or "")
    print("===")

    if not top_items:
        print("top sentences: <no attribution data>")
    else:
        print("top sentences (score desc):")
        for rank, (idx, score, sent) in enumerate(top_items, start=1):
            print(f"  {rank}) {score:.4f} | idx={idx} | {sent}")
    print("===")
    if sink_rows:
        print("sink sentences (generation rows selected):")
        base_idx = len(prompt_sentences)
        for j in sink_rows:
            sent = generation_sentences[j] if j < len(generation_sentences) else ""
            print(f"  row={j} | idx={base_idx + j} | {sent}")
    else:
        print("sink sentences: <none>")

    coverage = safe_get(record, "coverage")
    if coverage is not None:
        print(f"coverage: {coverage}")
    faith = safe_get(record, "faithfulness")
    if faith is not None:
        print(f"faithfulness: {faith}")
    print()


def main() -> None:
    args = parse_args()
    if args.files:
        files = [Path(p) for p in args.files]
    else:
        files = sorted(Path("exp/case_study/out").glob("*.jsonl"))

    for path in files:
        print(f"=== FILE: {path} ===")
        for record in load_jsonl(path):
            print_sample(record, args.attr_field, args.topk)
        print()


if __name__ == "__main__":
    main()
