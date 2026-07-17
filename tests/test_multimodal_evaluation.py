import json
from pathlib import Path

import numpy as np
import pytest

from evaluations.multimodal.gqa_grounding import build_grounded_record, iter_grounded_records
from evaluations.multimodal.metrics import (
    binary_iou,
    curve_auc,
    energy_in_mask,
    evidence_recall_at_fraction,
    pointing_game,
    xywh_boxes_to_mask,
)
from evaluations.multimodal.visa_grounding import build_visa_record, stratified_sample


def test_curve_auc_normalizes_custom_x_range():
    assert curve_auc([0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert curve_auc([0.0, 1.0], fractions=[0.2, 0.8]) == pytest.approx(0.5)


def test_spatial_metrics_use_the_same_evidence_grid():
    attribution = np.array([[0.1, 0.2], [0.3, 0.9]])
    evidence = np.array([[False, False], [False, True]])

    assert pointing_game(attribution, evidence) == 1.0
    assert energy_in_mask(attribution, evidence) == pytest.approx(0.6)
    assert evidence_recall_at_fraction(attribution, evidence, fraction=0.25) == 1.0
    assert binary_iou(evidence, evidence) == 1.0


def test_xywh_boxes_to_mask_clips_boxes_to_image():
    mask = xywh_boxes_to_mask([[-1, 1, 3, 2], [3, 3, 5, 5]], height=4, width=4)

    expected = np.array(
        [
            [False, False, False, False],
            [True, True, False, False],
            [True, True, False, False],
            [False, False, False, True],
        ]
    )
    assert np.array_equal(mask, expected)


def _gqa_fixture():
    question = {
        "imageId": "2407890",
        "question": "Is there a red apple on the table?",
        "answer": "no",
        "fullAnswer": "No, there is an apple but it is green.",
        "isBalanced": True,
        "types": {"structural": "verify", "semantic": "relation", "detailed": "existAttrRel"},
        "annotations": {
            "question": {"4": "271881", "7": "279472"},
            "answer": {},
            "fullAnswer": {"4": "271881"},
        },
        "semantic": [
            {"operation": "select", "argument": "table (279472)"},
            {"operation": "relate", "argument": "on, subject, apple (271881)"},
            {"operation": "filter", "argument": "red"},
            {"operation": "exist", "argument": "?"},
        ],
    }
    scene_graph = {
        "width": 640,
        "height": 480,
        "objects": {
            "271881": {
                "name": "apple",
                "x": 220,
                "y": 310,
                "w": 50,
                "h": 80,
                "attributes": ["green"],
            },
            "279472": {
                "name": "table",
                "x": 100,
                "y": 280,
                "w": 400,
                "h": 180,
                "attributes": ["wooden"],
            },
        },
    }
    return question, scene_graph


def test_gqa_record_joins_visual_pointers_programs_and_boxes():
    question, scene_graph = _gqa_fixture()

    record = build_grounded_record("1238592", question, scene_graph)

    assert record["sample_id"] == "1238592"
    assert record["program_steps"] == 4
    assert record["evidence_object_ids"] == ["271881", "279472"]
    assert record["missing_object_ids"] == []
    assert record["evidence"][0]["sources"] == [
        "annotation:fullAnswer",
        "annotation:question",
        "semantic:1",
    ]
    assert record["evidence"][1]["bbox_xyxy_normalized"] == pytest.approx(
        [100 / 640, 280 / 480, 500 / 640, 460 / 480]
    )


def test_gqa_iterator_filters_unbalanced_and_ungrounded_questions():
    question, scene_graph = _gqa_fixture()
    unbalanced = dict(question, isBalanced=False)
    ungrounded = dict(question, annotations={}, semantic=[])
    records = list(
        iter_grounded_records(
            {"1": unbalanced, "2": ungrounded, "3": question},
            {"2407890": scene_graph},
        )
    )

    assert [record["sample_id"] for record in records] == ["3"]


def test_visa_record_preserves_row_reference_and_normalizes_box():
    record = build_visa_record(
        12,
        {
            "id": "wiki-12",
            "question": "Who wrote the book?",
            "short_answer": "Example Author",
            "long_answer_type": "p",
            "image_size": [980, 3920],
            "bounding_box": [49, 1200, 539, 1400],
            "candidates": ["negative", "wiki-12"],
            "pos_idx": 1,
        },
    )

    assert record["hf_row_index"] == 12
    assert record["stratum"] == "later_page_passage"
    assert record["has_positive_candidate"] is True
    assert record["evidence_bbox_xyxy_normalized"] == pytest.approx(
        [49 / 980, 1200 / 3920, 539 / 980, 1400 / 3920]
    )


def test_visa_sampling_is_balanced_and_deterministic():
    records = [
        {"hf_row_index": index, "stratum": stratum}
        for index, stratum in enumerate(
            ["first_page_passage"] * 3 + ["later_page_passage"] * 3 + ["non_passage"] * 3
        )
    ]

    selected = stratified_sample(records, 6, seed=17)
    assert selected == stratified_sample(records, 6, seed=17)
    assert {name: sum(item["stratum"] == name for item in selected) for name in {item["stratum"] for item in selected}} == {
        "first_page_passage": 2,
        "later_page_passage": 2,
        "non_passage": 2,
    }


def test_multimodal_protocol_has_two_primary_benchmarks_and_valid_ids():
    protocol_path = Path(__file__).parents[1] / "evaluations" / "multimodal" / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    primary = [benchmark["id"] for benchmark in protocol["benchmarks"] if benchmark["tier"] == "primary"]
    assert primary == ["coco_captions_grounded", "wiki_visa_single_oracle"]
    assert protocol["sample_budget"]["diagnostic_repope"] == 200
