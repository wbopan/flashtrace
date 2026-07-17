from __future__ import annotations

import json

import pytest
from PIL import Image

from evaluations.multimodal.datasets import load_examples, vqa_accuracy
from evaluations.multimodal.run_smoke import (
    make_overlay,
    parse_response,
    perturb_region,
    summarize_grid,
)
from evaluations.multimodal.run_methods import (
    alignment_metrics,
    resample_grid,
    spearman_correlation,
)


def test_vqa_accuracy_matches_consensus_levels():
    assert vqa_accuracy("cat", ["cat"] * 10) == 1.0
    assert vqa_accuracy("cat", ["cat", "cat"] + ["dog"] * 8) == pytest.approx(0.6)
    assert vqa_accuracy("two", ["2"] * 10) == 1.0
    assert vqa_accuracy("a dog", ["dog"] * 10) == 1.0


def test_unified_dataset_adapters(tmp_path):
    vqax = tmp_path / "vqa_x" / "nlxgpt"
    vqax.mkdir(parents=True)
    (vqax / "vqaX_val.json").write_text(
        json.dumps(
            {
                "7": {
                    "question": "What is shown?",
                    "answers": [{"answer": "cat"}] * 10,
                    "image_id": "42",
                    "image_name": "COCO_val2014_000000000042.jpg",
                    "explanation": ["A cat is visible."],
                }
            }
        )
    )
    aok = tmp_path / "aokvqa"
    aok.mkdir()
    (aok / "aokvqa_v1p0_val.json").write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "image_id": 9,
                    "question": "Why?",
                    "direct_answers": ["safety"] * 10,
                    "rationales": ["It prevents injury."],
                }
            ]
        )
    )

    first = load_examples("vqa_x", tmp_path)[0]
    second = load_examples("aokvqa", tmp_path)[0]
    assert first.majority_answer == "cat"
    assert first.coco_split == "val2014"
    assert first.image_path.name == "COCO_val2014_000000000042.jpg"
    assert second.majority_answer == "safety"
    assert second.coco_split == "val2017"
    assert second.image_path.name == "000000000009.jpg"


def test_response_parser_and_spatial_summary():
    reasoning, answer = parse_response(
        "Reasoning: The person has a lit object in his mouth.\nFinal answer: cigarette"
    )
    assert reasoning.startswith("The person")
    assert answer == "cigarette"
    assert parse_response("Final answer: No, it is snowy.")[1] == "No"
    assert parse_response("No rain visible.\nNo") == ("No rain visible.", "No")
    assert parse_response("skateboarding\nskateboarding")[0] == ""
    summary = summarize_grid([[0.1, 0.2], [-0.1, 0.4]])
    assert summary["top_cell"] == [1, 1]
    assert summary["max_drop"] == 0.4
    assert 0 < summary["top_quartile_share"] <= 1


def test_perturb_region_changes_only_selected_cell():
    image = Image.new("RGB", (8, 8), "white")
    for x in range(4):
        for y in range(4):
            image.putpixel((x, y), (0, 0, 0))
    output = perturb_region(image, 0, 0, 2)
    assert output.getpixel((3, 3)) != image.getpixel((3, 3))
    assert output.getpixel((6, 6)) == image.getpixel((6, 6))


def test_rectangular_overlay_covers_the_full_image():
    image = Image.new("RGB", (40, 20), "white")
    grid = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    output = make_overlay(image, grid)
    assert output.getpixel((35, 15)) != image.getpixel((35, 15))
    assert output.getpixel((5, 5)) == image.getpixel((5, 5))


def test_method_grid_resampling_and_alignment_metrics():
    source = [[float(row * 8 + column) for column in range(8)] for row in range(8)]
    resized = resample_grid(source, 4)
    assert len(resized) == 4
    assert all(len(row) == 4 for row in resized)
    metrics = alignment_metrics(resized, resized)
    assert metrics["spearman_vs_loo"] == pytest.approx(1.0)
    assert metrics["loo_positive_mass_recall_at_25"] > 0.25
    assert metrics["top25_jaccard_vs_loo"] == 1.0
    assert metrics["top_cell_hit_vs_loo"] is True


def test_spearman_handles_reverse_order_and_ties():
    assert spearman_correlation([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman_correlation([1, 1, 1], [2, 3, 4]) == 0.0
