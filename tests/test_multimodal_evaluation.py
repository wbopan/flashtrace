import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from evaluations.multimodal.analyze_attention_sink import (
    leave_one_out_priors,
    mask_top_fraction,
    normalized_positive,
    residualize_position_prior,
)
from evaluations.multimodal.gqa_grounding import build_grounded_record, iter_grounded_records
from evaluations.multimodal.metrics import (
    binary_iou,
    curve_auc,
    energy_in_mask,
    evidence_recall_at_fraction,
    evidence_rank_auc,
    patch_energy_in_mask,
    patch_evidence_rank_auc,
    patch_pointing_game,
    patch_recovery_at_fraction,
    pointing_game,
    top_evidence_iou,
    xywh_boxes_to_mask,
    xyxy_boxes_to_mask,
)
from evaluations.multimodal.native_pilot_data import decode_coco_rle
from evaluations.multimodal.render_strict_method_comparisons import _metric
from evaluations.multimodal.recompute_strict_spatial import (
    _restore_paper_flashtrace_composition,
)
from evaluations.multimodal.strict_datasets import (
    clevr_answer,
    clevr_reasoning_family,
    select_clevr_complex,
    validate_dataset_record,
)
from evaluations.multimodal.strict_generation import (
    model_record_prompt,
    normalized_output,
    output_correct,
    render_prompt,
    split_thinking_output,
    validate_model_record,
)
from evaluations.multimodal.strict_attribution import (
    _common_summary,
    _resample,
    _visual_grid_from_projected_scores,
    localization_metrics,
)
from evaluations.multimodal.strict_visual_faithfulness import (
    _normalize_deletion,
    _normalize_insertion,
    perturbation_pair,
    region_layout,
    visual_mas,
)
from evaluations.multimodal.visa_grounding import build_visa_record, stratified_sample


def test_curve_auc_normalizes_custom_x_range():
    assert curve_auc([0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert curve_auc([0.0, 1.0], fractions=[0.2, 0.8]) == pytest.approx(0.5)


def test_attention_sink_prior_is_leave_one_out_and_gt_independent():
    grids = [
        np.array([[9.0, 1.0], [0.0, 0.0]]),
        np.array([[8.0, 2.0], [0.0, 0.0]]),
        np.array([[7.0, 1.0], [0.0, 2.0]]),
    ]

    priors = leave_one_out_priors(grids)

    assert priors[0] == pytest.approx(
        (normalized_positive(grids[1]) + normalized_positive(grids[2])) / 2
    )
    assert priors[0].sum() == pytest.approx(1.0)


def test_attention_sink_corrections_remove_shared_position():
    grid = np.array([[0.7, 0.1], [0.1, 0.1]])
    prior = np.array([[0.6, 0.2], [0.1, 0.1]])

    masked = mask_top_fraction(grid, prior, 0.25)
    residual = residualize_position_prior(grid, prior)

    assert masked[0, 0] == 0.0
    assert masked[1, 1] == pytest.approx(0.1)
    np.testing.assert_allclose(residual, [[0.1, 0.0], [0.0, 0.0]])


def test_spatial_metrics_use_the_same_evidence_grid():
    attribution = np.array([[0.1, 0.2], [0.3, 0.9]])
    evidence = np.array([[False, False], [False, True]])

    assert pointing_game(attribution, evidence) == 1.0
    assert energy_in_mask(attribution, evidence) == pytest.approx(0.6)
    assert evidence_recall_at_fraction(attribution, evidence, fraction=0.25) == 1.0
    assert binary_iou(evidence, evidence) == 1.0
    assert evidence_rank_auc(attribution, evidence) == 1.0
    assert top_evidence_iou(attribution, evidence) == 1.0


def test_rank_auc_is_tie_aware_and_random_constant_is_half():
    attribution = np.ones((2, 3))
    evidence = np.array([[True, False, False], [True, False, False]])

    assert evidence_rank_auc(attribution, evidence) == pytest.approx(0.5)


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


def test_xyxy_boxes_to_mask_uses_box_edges_not_width_height():
    mask = xyxy_boxes_to_mask([[1, 1, 3, 4]], height=5, width=5)

    assert mask.sum() == 6
    assert mask[1:4, 1:3].all()
    assert not mask[4, 3]


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


def _clevr_question(question_index, image_index, final_operation, answer):
    program = [{"type": "scene", "inputs": [], "value_inputs": []}]
    program.extend(
        {"type": "filter_color", "inputs": [index], "value_inputs": ["red"]}
        for index in range(10)
    )
    program.append({"type": final_operation, "inputs": [10], "value_inputs": []})
    return {
        "split": "unique",
        "image_filename": f"CLEVR_unique_{image_index:06d}.png",
        "image_index": image_index,
        "question": f"Reasoning question {question_index}?",
        "program": program,
        "answer": answer,
        "question_index": question_index,
    }


def test_clevr_helpers_keep_official_boolean_and_group_final_operations():
    question = _clevr_question(0, 0, "greater_than", True)

    assert clevr_answer(question["answer"]) == "true"
    assert clevr_reasoning_family(question) == "compare_integer"


def test_clevr_selection_is_balanced_and_keeps_program_out_of_input(tmp_path):
    operations = [
        "count",
        "exist",
        "greater_than",
        "equal_color",
        "query_shape",
    ]
    questions = []
    dataset_root = tmp_path / "dataset"
    images_root = tmp_path / "images"
    images_root.mkdir(parents=True)
    primary_root = dataset_root / "ground_truth_complex_questions_unique_firstnonempty"
    primary_root.mkdir(parents=True)
    for directory in (
        "ground_truth_complex_questions_unique",
        "ground_truth_complex_questions_union",
        "ground_truth_complex_questions_all_objects",
    ):
        (dataset_root / directory).mkdir(parents=True)

    for index, operation in enumerate(operations):
        question = _clevr_question(index, index, operation, index)
        questions.append(question)
        (images_root / question["image_filename"]).write_bytes(b"image")
        for directory in (
            "ground_truth_complex_questions_unique_firstnonempty",
            "ground_truth_complex_questions_unique",
            "ground_truth_complex_questions_union",
            "ground_truth_complex_questions_all_objects",
        ):
            np.save(dataset_root / directory / f"{index}.npy", np.ones((2, 2), dtype=bool))

    records = select_clevr_complex(
        questions,
        dataset_root=dataset_root,
        images_root=images_root,
        sample_size=5,
        seed=17,
        min_program_steps=12,
    )

    assert len(records) == 5
    assert {
        record["evaluation"]["metadata"]["reasoning_family"] for record in records
    } == {
        "count",
        "exist",
        "compare_integer",
        "compare_attribute",
        "query_attribute",
    }
    assert all(set(record["input"]) == {"I_IMAGE", "I_QUESTION"} for record in records)
    assert all("functional_program" not in record["input"] for record in records)
    assert all("functional_program" in record["evaluation"]["metadata"] for record in records)
    for record in records:
        validate_dataset_record(record)


def test_dataset_record_validator_rejects_model_output_leakage():
    record = {
        "schema_version": 2,
        "benchmark": "fixture",
        "sample_id": "fixture-1",
        "input": {"I_IMAGE": "image.png", "I_QUESTION": "Question?"},
        "evaluation": {"REFERENCE_OUTPUT": "answer"},
        "THINKING": "human rationale",
    }

    with pytest.raises(ValueError, match="model fields leaked"):
        validate_dataset_record(record)


def test_strict_thinking_parser_preserves_the_whole_output():
    response = (
        "<think>\nFind the cylinder, follow left, then compare.\n</think>\n\n"
        "**Final answer:** cyan"
    )

    thinking, output, thinking_chars, output_chars = split_thinking_output(response)

    assert thinking == "Find the cylinder, follow left, then compare."
    assert output == "**Final answer:** cyan"
    assert response[slice(*thinking_chars)] == thinking
    assert response[slice(*output_chars)] == output


def test_strict_thinking_parser_rejects_missing_or_empty_spans():
    with pytest.raises(ValueError, match="no </think>"):
        split_thinking_output("Reasoning without a terminator")
    with pytest.raises(ValueError, match="empty THINKING"):
        split_thinking_output("</think>\nanswer")
    with pytest.raises(ValueError, match="empty OUTPUT"):
        split_thinking_output("reasoning</think>")


def test_correctness_normalization_does_not_change_saved_output():
    output = "**Final answer:** YES."

    assert normalized_output(output) == "yes"
    assert output_correct(output, "true", "clevr_xai_complex")
    assert output == "**Final answer:** YES."


def test_prompt_profile_preserves_native_long_form_task_instruction():
    prompt = render_prompt("What is this?", "long_form")

    assert "detailed, self-contained final answer" in prompt
    assert prompt.endswith("Question: What is this?")
    assert model_record_prompt(
        {
            "I_QUESTION": "What is this?",
            "generation_metadata": {"prompt_profile": "long_form"},
        }
    ) == prompt


def test_coco_rle_decoder_uses_column_major_order():
    mask = decode_coco_rle({"size": [2, 3], "counts": [1, 2, 3]})

    assert np.array_equal(
        mask,
        np.array([[False, True, False], [True, False, False]]),
    )


def test_model_record_validator_keeps_reference_metadata_separate():
    record = {
        "schema_version": 2,
        "benchmark": "clevr_xai_complex",
        "sample_id": "clevr-1",
        "I_IMAGE": "image.png",
        "I_QUESTION": "Question?",
        "THINKING": "Model reasoning.",
        "OUTPUT": "**Final answer:** red",
        "THINKING_SPAN": [0, 2],
        "OUTPUT_SPAN": [3, 7],
        "raw_response": "Model reasoning.</think>**Final answer:** red",
        "model": {"repo_id": "Qwen/Qwen3-VL-8B-Thinking"},
    }
    validate_model_record(record)

    record["REFERENCE_OUTPUT"] = "red"
    with pytest.raises(ValueError, match="evaluation fields leaked"):
        validate_model_record(record)


def test_model_record_validator_rejects_decode_reencode_token_drift():
    record = {
        "schema_version": 2,
        "benchmark": "wiki_visa",
        "sample_id": "wiki-1",
        "I_IMAGE": "image.png",
        "I_QUESTION": "Question?",
        "THINKING": "Model reasoning.",
        "OUTPUT": "answer",
        "THINKING_SPAN": [0, 2],
        "OUTPUT_SPAN": [3, 3],
        "raw_response": "Model reasoning.</think>answer",
        "model": {"repo_id": "Qwen/Qwen3-VL-8B-Thinking"},
        "generation_metadata": {
            "original_generated_token_ids": [1, 2, 3],
            "teacher_forced_token_ids": [1, 20, 3],
        },
    }

    with pytest.raises(ValueError, match="decode/re-encoded"):
        validate_model_record(record)


def test_strict_localization_uses_primary_mask_and_reports_multiple_thresholds(tmp_path):
    mask = np.array(
        [
            [False, False, False],
            [False, True, False],
            [False, False, False],
        ]
    )
    mask_path = tmp_path / "primary.npy"
    np.save(mask_path, mask)
    record = {
        "evaluation": {
            "EVIDENCE_MASKS": {
                "primary_unique_firstnonempty": str(mask_path),
                "sensitivity_unique": None,
                "sensitivity_union": None,
            }
        }
    }

    metrics = localization_metrics(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        record,
    )

    assert metrics["pointing_game"] == 1.0
    assert metrics["energy_in_mask"] == 1.0
    assert metrics["recovery_at_1pct"] == 1.0
    assert metrics["recovery_at_20pct"] == 1.0


def test_strict_localization_accepts_native_primary_mask(tmp_path):
    mask = np.array([[False, True], [False, False]])
    mask_path = tmp_path / "primary.npy"
    np.save(mask_path, mask)
    record = {"evaluation": {"EVIDENCE_MASKS": {"primary": str(mask_path)}}}

    metrics = localization_metrics([[0.0, 1.0], [0.0, 0.0]], record)

    assert metrics["pointing_game"] == 1.0
    assert metrics["energy_in_mask"] == 1.0


def test_strict_patch_resampling_preserves_raw_patch_boundaries():
    expanded = _resample([[1.0, 0.0], [0.0, 0.0]], (4, 4))

    assert set(np.unique(expanded)) == {0.0, 1.0}
    assert expanded[:2, :2].sum() == 4.0
    assert expanded[2:, :].sum() == 0.0
    assert expanded[:, 2:].sum() == 0.0


def test_strict_visual_grid_projects_cumulative_flashtrace_scores():
    grid = _visual_grid_from_projected_scores(
        [0.0, 1.0, 0.0, 2.0, 3.0, 0.0, 4.0],
        {
            "visual_grid_thw": [[1, 2, 2]],
            "visual_token_indices_prompt": [1, 3, 4, 6],
        },
    )

    assert grid == [[1.0, 2.0], [3.0, 4.0]]


def test_flashtrace_visual_map_keeps_direct_base_and_weighted_hops():
    record = {
        "method": "flashtrace",
        "visual_grid": [[99.0, 99.0]],
        "method_metadata": {
            "trace_metadata": {
                "multimodal": {
                    "visual_grid_thw": [[1, 1, 2]],
                    "visual_token_indices_prompt": [0, 2],
                },
                "ifr": {
                    "observation_projected": {
                        "base": [1.0, 0.0, 2.0],
                        "per_hop": [[3.0, 0.0, 4.0]],
                        "sum": [4.0, 0.0, 6.0],
                    }
                },
            },
        },
    }

    _restore_paper_flashtrace_composition(record)

    assert record["visual_grid"] == [[4.0, 6.0]]
    assert record["method_metadata"]["direct_base_included"] is True
    assert record["method_metadata"]["attribution_composition"] == (
        "direct_plus_weighted_reasoning_hops"
    )


def test_patch_metrics_keep_complete_patches_and_average_cutoff_ties():
    grid = [[1.0, 1.0], [0.0, 0.0]]
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True

    assert patch_recovery_at_fraction(grid, mask, fraction=0.25) == 0.5
    assert patch_energy_in_mask(grid, mask) == 0.5
    assert patch_pointing_game(grid, mask) == 0.5
    assert patch_evidence_rank_auc(grid, mask) == pytest.approx(5.0 / 6.0)


def test_method_comparison_metric_allows_missing_native_region_gt():
    assert np.isnan(_metric({"localization": None}, "energy_in_mask"))


def test_common_summary_does_not_compare_different_success_subsets():
    records = [
        {
            "status": "ok",
            "method": "a",
            "sample_id": sample,
            "seconds": 1,
            "incremental_peak_vram_gb": 1,
            "localization": {
                name: 1.0
                for name in (
                    "pointing_game",
                    "energy_in_mask",
                    "evidence_rank_auc",
                    "top_evidence_iou",
                    "recovery_at_1pct",
                    "recovery_at_5pct",
                    "recovery_at_10pct",
                    "recovery_at_20pct",
                )
            },
        }
        for sample in ("1", "2")
    ]
    records.append(
        {
            **records[0],
            "method": "b",
            "sample_id": "2",
            "localization": {
                **records[0]["localization"],
                "energy_in_mask": 0.25,
            },
        }
    )

    summary = _common_summary(records, ("a", "b"))

    assert summary["common_sample_ids"] == ["2"]
    assert summary["methods"]["a"]["common_samples"] == 1
    assert summary["methods"]["b"]["energy_in_mask"] == 0.25


def test_common_summary_allows_no_ground_truth_localization():
    records = [
        {
            "status": "ok",
            "method": method,
            "sample_id": "1",
            "seconds": 1.0,
            "incremental_peak_vram_gb": 0.5,
            "localization": None,
        }
        for method in ("a", "b")
    ]

    summary = _common_summary(records, ("a", "b"))

    assert summary["common_sample_ids"] == ["1"]
    assert summary["methods"]["a"]["localization_samples"] == 0
    assert summary["methods"]["a"]["energy_in_mask"] is None


def test_visual_faithfulness_grid_preserves_tall_page_geometry():
    tall = Image.new("RGB", (980, 3920))
    landscape = Image.new("RGB", (480, 320))

    assert region_layout(tall, 64) == (16, 4)
    assert region_layout(landscape, 64) == (7, 9)


def test_visual_deletion_and_insertion_use_the_same_region_mask():
    original = Image.new("RGB", (4, 4), "white")
    blurred = Image.new("RGB", (4, 4), "black")

    deleted, inserted = perturbation_pair(original, blurred, (2, 2), [0])

    assert deleted.getpixel((0, 0)) == (0, 0, 0)
    assert deleted.getpixel((3, 3)) == (255, 255, 255)
    assert inserted.getpixel((0, 0)) == (255, 255, 255)
    assert inserted.getpixel((3, 3)) == (0, 0, 0)


def test_visual_curve_normalization_enforces_monotone_envelopes():
    deletion, deletion_degenerate = _normalize_deletion(
        np.array([0.0, -0.6, -0.4, -1.0])
    )
    insertion, insertion_degenerate = _normalize_insertion(
        np.array([-1.0, -0.2, -0.4, 0.0])
    )

    assert not deletion_degenerate
    assert not insertion_degenerate
    assert deletion.tolist() == pytest.approx([1.0, 0.4, 0.4, 0.0])
    assert insertion.tolist() == pytest.approx([0.0, 0.8, 0.8, 1.0])
    assert visual_mas(deletion, np.array([1.0, 0.7, 0.3, 0.0]))[
        "visual_rise"
    ] == pytest.approx(0.4333333333)
