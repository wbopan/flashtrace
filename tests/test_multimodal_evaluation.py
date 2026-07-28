import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import evaluations.multimodal.strict_visual_faithfulness as faithfulness_module
from evaluations.multimodal.analyze_attention_sink import (
    leave_one_out_priors,
    mask_top_fraction,
    normalized_positive,
    residualize_position_prior,
)
from evaluations.multimodal.analyze_formal_faithfulness import (
    analyze as analyze_formal_faithfulness,
)
from evaluations.multimodal.analyze_strict_results import (
    METRICS as STRICT_LOCALIZATION_METRICS,
    analyze as analyze_strict_results,
)
from evaluations.multimodal.audit_formal_results import (
    Audit,
    EXPECTED_REVISION,
    EXPECTED_STRATA,
    _audit_analysis_payloads,
    _finite_nonnegative,
    _frozen_protocol_metadata_issues,
    _frozen_record_protocol_issues,
    _funnel_conservation_issues,
    _protocol_v2_issues,
    _valid_faithfulness_curve,
    _valid_interval,
    _valid_visual_grid,
    _valid_wtl,
)
from evaluations.multimodal.gqa_grounding import build_grounded_record, iter_grounded_records
from evaluations.multimodal.jsonl_checkpoint import PairJsonlCheckpoint
from evaluations.multimodal.formal_manual_audit import (
    audit_sample_ids as protocol_audit_sample_ids,
    summarize_reviews as summarize_protocol_reviews,
)
from evaluations.multimodal.freeze_formal_input_hashes import build_payload
from evaluations.multimodal.freeze_formal_response_hashes import (
    build_payload as build_response_hash_payload,
)
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
from evaluations.multimodal.materialize_manifest_prefix import materialize
from evaluations.multimodal.native_pilot_data import decode_coco_rle
from evaluations.multimodal.render_strict_method_comparisons import _metric
from evaluations.multimodal.refresh_generation_gates import refresh
from evaluations.multimodal.recompute_strict_spatial import (
    _restore_paper_flashtrace_composition,
)
from evaluations.multimodal.reuse_preview_checkpoints import (
    EXPECTED_METHODS as PREVIEW_REUSE_METHODS,
    reuse as reuse_preview_checkpoints,
)
from evaluations.multimodal.reuse_preview_ablation_checkpoints import (
    reuse as reuse_preview_ablation_checkpoints,
)
from evaluations.multimodal.select_strict_subset import (
    gate_funnel,
    select as select_strict_subset,
    update_frozen_ids,
)
from evaluations.multimodal.select_preview_candidates import select_candidates
from evaluations.multimodal.strict_datasets import (
    clevr_answer,
    clevr_reasoning_family,
    select_clevr_complex,
    select_vizwiz_lf,
    validate_dataset_record,
)
from evaluations.multimodal.strict_generation import (
    default_max_new_tokens,
    is_recorded_deterministic_generation_error,
    model_record_prompt,
    normalized_output,
    output_correct,
    output_is_refusal_or_unanswerable,
    pre_ablation_gate,
    read_jsonl,
    render_prompt,
    split_thinking_output,
    validate_model_record,
    write_jsonl,
)
from evaluations.multimodal.strict_attribution import (
    DEFAULT_METHODS,
    _common_summary,
    _resample,
    _visual_grid_from_projected_scores,
    localization_metrics,
)
from evaluations.multimodal.strict_visual_faithfulness import (
    CURVE_NORMALIZATION_POLICY,
    _summary as faithfulness_summary,
    _normalize_deletion,
    _normalize_insertion,
    perturbation_pair,
    refresh_derived_curve_metrics,
    region_layout,
    visual_mas,
    evaluate_grid,
)
from evaluations.multimodal.validate_paired_matrix import (
    EXPECTED_METHODS as MATRIX_METHODS,
    validate as validate_paired_matrix,
)
from evaluations.multimodal.visa_grounding import build_visa_record, stratified_sample
from evaluations.multimodal.vizwiz_semantic_judgments import (
    apply_human_reviews,
    audit_sample_ids,
    join_judgments,
    validate_judgment,
)


def test_curve_auc_normalizes_custom_x_range():
    assert curve_auc([0.0, 0.5, 1.0]) == pytest.approx(0.5)
    assert curve_auc([0.0, 1.0], fractions=[0.2, 0.8]) == pytest.approx(0.5)


def test_formal_audit_rejects_nonfinite_intervals_and_bad_wtl():
    interval = {"mean": 0.5, "ci95_low": 0.4, "ci95_high": 0.6}
    assert _valid_interval(interval)
    assert _valid_wtl(
        {**interval, "wins": 4, "ties": 1, "losses": 5},
        10,
    )
    assert not _valid_interval(
        {"mean": float("nan"), "ci95_low": 0.4, "ci95_high": 0.6}
    )
    assert not _valid_interval(
        {"mean": 0.5, "ci95_low": 0.7, "ci95_high": 0.6}
    )
    assert not _valid_wtl(
        {**interval, "wins": 4, "ties": 1, "losses": 4},
        10,
    )
    assert _finite_nonnegative(0.0)
    assert _finite_nonnegative(3.5)
    assert not _finite_nonnegative(-0.1)
    assert not _finite_nonnegative(float("inf"))


def test_formal_audit_validates_grids_and_faithfulness_curve_geometry():
    assert _valid_visual_grid(
        {
            "visual_grid_shape": [2, 2],
            "visual_grid": [[0.0, -1.0], [1.0, 2.0]],
        }
    )
    assert not _valid_visual_grid(
        {
            "visual_grid_shape": [2, 2],
            "visual_grid": [[0.0], [1.0, 2.0]],
        }
    )
    curve = {
        "steps": 1,
        "fractions": [0.0, 1.0],
        "region_order": [1, 0],
        "remaining_attribution_density": [1.0, 0.0],
        "deletion_output_mean_logprob": [-1.0, -2.0],
        "insertion_output_mean_logprob": [-2.0, -1.0],
        "normalization_policy": CURVE_NORMALIZATION_POLICY,
        "normalized_deletion": [1.0, 0.0],
        "normalized_insertion": [0.0, 1.0],
        "deletion_endpoint_delta": 1.0,
        "insertion_endpoint_delta": 1.0,
        "deletion_degenerate": False,
        "insertion_degenerate": False,
        "deletion_auc": 0.5,
        "insertion_auc": 0.5,
        "visual_mas": 0.5,
        "visual_rise": 0.5,
        "visual_rise_plus_ap": 0.5,
    }
    assert _valid_faithfulness_curve(curve, regions=2, expected_steps=1)
    assert not _valid_faithfulness_curve(
        {**curve, "region_order": [0, 0]},
        regions=2,
        expected_steps=1,
    )


def test_formal_faithfulness_analysis_records_semantic_provenance(tmp_path):
    faithfulness_dir = tmp_path / "faithfulness"
    faithfulness_dir.mkdir()
    (faithfulness_dir / "summary.json").write_text(
        json.dumps(
            {
                "methods": ["flashtrace"],
                "common_sample_ids": ["sample-1"],
            }
        )
    )
    metrics = {
        "deletion_auc": 0.25,
        "insertion_auc": 0.75,
        "visual_mas": 0.3,
    }
    write_jsonl(
        [
            {
                "status": "ok",
                "sample_id": "sample-1",
                "method": "flashtrace",
                "faithfulness": {
                    **metrics,
                    "positive_only_ordering": metrics,
                },
            }
        ],
        faithfulness_dir / "faithfulness_records.jsonl",
    )
    reviewed = tmp_path / "reviewed.generation_eval.jsonl"
    write_jsonl(
        [
            {
                "sample_id": "sample-1",
                "semantic_correctness": {"label": "fully"},
            }
        ],
        reviewed,
    )

    result = analyze_formal_faithfulness(
        faithfulness_dir,
        generation_evaluation=reviewed,
        draws=10,
    )

    assert result["generation_evaluation"] == str(reviewed)
    assert result["model_output"] is None
    assert result["fully_correct_subset"]["samples"] == 1


def test_jsonl_writes_replace_complete_file_without_temp_leak(tmp_path):
    output = tmp_path / "records.jsonl"
    write_jsonl([{"sample_id": "one"}], output)
    write_jsonl([{"sample_id": "two"}, {"sample_id": "three"}], output)

    assert [
        json.loads(line)["sample_id"] for line in output.read_text().splitlines()
    ] == ["two", "three"]
    assert not list(tmp_path.glob(".records.jsonl.*.tmp"))


def test_frozen_input_hashes_cover_manifest_record_question_and_image(tmp_path):
    image = tmp_path / "image.bin"
    image.write_bytes(b"frozen-image")
    manifest = tmp_path / "formal.dataset.jsonl"
    record = {
        "sample_id": "sample-1",
        "input": {
            "I_IMAGE": "image.bin",
            "I_QUESTION": "What is shown?",
        },
    }
    write_jsonl([record], manifest)

    payload = build_payload(tmp_path, [manifest])
    bundle = payload["manifests"][0]
    sample = bundle["samples"][0]

    assert bundle["manifest_path"] == "formal.dataset.jsonl"
    assert bundle["sample_count"] == 1
    assert sample["sample_id"] == "sample-1"
    assert sample["image_path"] == "image.bin"
    assert len(sample["image_sha256"]) == 64
    assert len(sample["question_sha256"]) == 64
    assert len(sample["dataset_record_sha256"]) == 64


def test_frozen_response_hashes_cover_record_text_and_token_identity(tmp_path):
    model_output = tmp_path / "formal.model.jsonl"
    record = {
        "schema_version": 2,
        "benchmark": "wiki_visa",
        "sample_id": "sample-1",
        "I_IMAGE": "image.png",
        "I_QUESTION": "question",
        "THINKING": "reasoning",
        "OUTPUT": "answer",
        "THINKING_SPAN": [0, 0],
        "OUTPUT_SPAN": [1, 1],
        "raw_response": "<think>reasoning</think>answer",
        "model": {"resolved_revision": EXPECTED_REVISION},
        "generation_metadata": {
            "original_generated_token_ids": [1, 2, 3],
            "teacher_forced_token_ids": [1, 2, 3],
        },
    }
    write_jsonl([record], model_output)

    payload = build_response_hash_payload(tmp_path, [model_output])
    bundle = payload["model_outputs"][0]
    sample = bundle["samples"][0]

    assert bundle["model_output_path"] == "formal.model.jsonl"
    assert bundle["sample_count"] == 1
    assert bundle["resolved_revisions"] == [EXPECTED_REVISION]
    assert sample["sample_id"] == "sample-1"
    assert sample["generated_token_ids_sha256"] == sample[
        "teacher_forced_token_ids_sha256"
    ]
    assert all(
        len(sample[field]) == 64
        for field in (
            "model_record_sha256",
            "raw_response_sha256",
            "thinking_sha256",
            "output_sha256",
            "generated_token_ids_sha256",
        )
    )


def test_paired_matrix_validator_requires_exact_cartesian_product(tmp_path):
    manifest = tmp_path / "dataset.jsonl"
    evaluation_dir = tmp_path / "methods"
    evaluation_dir.mkdir()
    sample_ids = ["sample-1", "sample-2"]
    write_jsonl(
        [{"sample_id": sample_id} for sample_id in sample_ids],
        manifest,
    )
    (evaluation_dir / "summary.json").write_text(
        json.dumps(
            {
                "methods": {method: {} for method in MATRIX_METHODS},
                "common_samples": 2,
                "common_sample_ids": sample_ids,
            }
        )
    )
    records = [
        {
            "sample_id": sample_id,
            "method": method,
            "status": "ok",
        }
        for sample_id in sample_ids
        for method in MATRIX_METHODS
    ]
    records_path = evaluation_dir / "attribution_records.jsonl"
    write_jsonl(records, records_path)

    result = validate_paired_matrix(
        manifest=manifest,
        evaluation_dir=evaluation_dir,
        kind="attribution",
        expected_samples=2,
    )
    assert result["successful_pairs"] == 16
    assert result["exact_cartesian_product"] is True

    write_jsonl(
        [
            *records,
            {
                "sample_id": "extra",
                "method": "random",
                "status": "error",
            },
        ],
        records_path,
    )
    with pytest.raises(ValueError, match="error records"):
        validate_paired_matrix(
            manifest=manifest,
            evaluation_dir=evaluation_dir,
            kind="attribution",
            expected_samples=2,
        )


def test_pair_jsonl_checkpoint_resumes_and_compacts_without_duplicates(tmp_path):
    output = tmp_path / "records.jsonl"
    write_jsonl(
        [
            {
                "sample_id": "sample-1",
                "method": "flashtrace",
                "status": "error",
            }
        ],
        output,
    )
    checkpoint = PairJsonlCheckpoint(output)
    checkpoint.put(
        {
            "sample_id": "sample-2",
            "method": "visual-ig",
            "status": "ok",
        }
    )

    # A resumed process sees the journal while the canonical snapshot remains
    # a valid, unchanged artifact.
    assert len(read_jsonl(output)) == 1
    resumed = PairJsonlCheckpoint(output)
    assert len(resumed.records()) == 2
    resumed.put(
        {
            "sample_id": "sample-1",
            "method": "flashtrace",
            "status": "ok",
        }
    )
    assert resumed.compact() == 2

    records = read_jsonl(output)
    assert {
        (record["sample_id"], record["method"], record["status"])
        for record in records
    } == {
        ("sample-1", "flashtrace", "ok"),
        ("sample-2", "visual-ig", "ok"),
    }
    assert not output.with_name(f".{output.name}.journal").exists()


@pytest.mark.parametrize(
    "error",
    [
        "strict Thinking response has no </think> terminator; generated_tokens=1024",
        "generated token IDs differ from decode/re-encoded teacher-forced IDs",
    ],
)
def test_resume_identifies_only_frozen_deterministic_generation_errors(error):
    assert is_recorded_deterministic_generation_error(
        {"status": "error", "error_type": "ValueError", "error": error}
    )
    assert not is_recorded_deterministic_generation_error(
        {"status": "error", "error_type": "OutOfMemoryError", "error": "CUDA OOM"}
    )
    assert not is_recorded_deterministic_generation_error(
        {"status": "error", "error_type": "ValueError", "error": "other input issue"}
    )


def test_protocol_manual_audit_is_deterministic_and_caveat_only(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    rows = [
        {"sample_id": f"sample-{index:02d}", "benchmark": "wiki_visa"}
        for index in range(20)
    ]
    write_jsonl(rows, dataset)
    selected = protocol_audit_sample_ids(
        [row["sample_id"] for row in rows], fraction=0.1, seed=17
    )
    write_jsonl(
        [
            {
                "sample_id": sample_id,
                "image_dependence": "supported",
                "thinking_quality": "good",
                "reviewer": "human",
                "reason": "Image and reasoning were inspected.",
            }
            for sample_id in selected
        ],
        reviews,
    )

    summary = summarize_protocol_reviews(dataset, reviews, fraction=0.1, seed=17)

    assert summary["complete"] is True
    assert summary["reviewed_count"] == 2
    assert summary["audit_sample_ids"] == selected
    assert summary["selection_effect"] == "caveat_only_no_frozen_id_changes"


def test_preview_candidate_selection_is_seeded_and_excludes_pilots(tmp_path):
    source = tmp_path / "source.jsonl"
    excluded = tmp_path / "excluded.jsonl"
    rows = [{"sample_id": f"sample-{index:02d}"} for index in range(12)]
    write_jsonl(rows, source)
    write_jsonl([rows[2], rows[7]], excluded)

    first = select_candidates(
        source, sample_size=5, seed=101, exclude_manifests=[excluded]
    )
    second = select_candidates(
        source, sample_size=5, seed=101, exclude_manifests=[excluded]
    )

    assert [row["sample_id"] for row in first] == [
        row["sample_id"] for row in second
    ]
    assert not {"sample-02", "sample-07"} & {
        row["sample_id"] for row in first
    }


def test_candidate_extension_prefixes_are_nested_and_immutable(tmp_path):
    source = tmp_path / "source.jsonl"
    prefix = tmp_path / "prefix.jsonl"
    rows = [{"sample_id": f"sample-{index:02d}"} for index in range(8)]
    write_jsonl(rows, source)

    assert materialize(source, prefix, 5) == rows[:5]
    assert materialize(source, prefix, 5) == rows[:5]
    with pytest.raises(ValueError, match="immutable prefix artifact differs"):
        materialize(source, prefix, 6)


def test_preview_checkpoint_reuse_requires_identical_frozen_response(tmp_path):
    formal_dataset = tmp_path / "formal.dataset.jsonl"
    formal_model = tmp_path / "formal.model.jsonl"
    preview_dataset = tmp_path / "preview.dataset.jsonl"
    preview_model = tmp_path / "preview.model.jsonl"
    preview_attr = tmp_path / "preview_attr"
    preview_faith = tmp_path / "preview_faith"
    formal_attr = tmp_path / "formal_attr"
    formal_faith = tmp_path / "formal_faith"
    preview_attr.mkdir()
    preview_faith.mkdir()
    dataset = {
        "sample_id": "same",
        "benchmark": "wiki_visa",
        "input": {"I_IMAGE": "image.jpg", "I_QUESTION": "question"},
    }
    model = {
        "sample_id": "same",
        "I_IMAGE": "image.jpg",
        "I_QUESTION": "question",
        "raw_response": "<think>reason</think>answer",
        "THINKING": "reason",
        "OUTPUT": "answer",
        "THINKING_SPAN": [0, 1],
        "OUTPUT_SPAN": [2, 2],
        "generation_metadata": {
            "original_generated_token_ids": [1, 2],
            "teacher_forced_token_ids": [1, 2],
        },
        "model": {"resolved_revision": "revision"},
    }
    write_jsonl([dataset], formal_dataset)
    write_jsonl([dataset], preview_dataset)
    write_jsonl([model], formal_model)
    write_jsonl([model], preview_model)
    (preview_attr / "summary.json").write_text(
        json.dumps({"requested_methods": sorted(PREVIEW_REUSE_METHODS)})
    )
    (preview_faith / "summary.json").write_text(
        json.dumps(
            {
                "methods": {method: {} for method in PREVIEW_REUSE_METHODS},
                "target_regions": 64,
                "steps": 10,
            }
        )
    )
    write_jsonl(
        [
            {
                "status": "ok",
                "sample_id": "same",
                "sample_index": 9,
                "method": method,
            }
            for method in PREVIEW_REUSE_METHODS
        ],
        preview_attr / "attribution_records.jsonl",
    )
    write_jsonl(
        [
            {
                "status": "ok",
                "sample_id": "same",
                "sample_index": 9,
                "method": method,
            }
            for method in PREVIEW_REUSE_METHODS
        ],
        preview_faith / "faithfulness_records.jsonl",
    )

    first = reuse_preview_checkpoints(
        formal_dataset=formal_dataset,
        formal_model=formal_model,
        preview_dataset=preview_dataset,
        preview_model=preview_model,
        preview_attribution_dir=preview_attr,
        preview_faithfulness_dir=preview_faith,
        formal_attribution_dir=formal_attr,
        formal_faithfulness_dir=formal_faith,
    )
    second = reuse_preview_checkpoints(
        formal_dataset=formal_dataset,
        formal_model=formal_model,
        preview_dataset=preview_dataset,
        preview_model=preview_model,
        preview_attribution_dir=preview_attr,
        preview_faithfulness_dir=preview_faith,
        formal_attribution_dir=formal_attr,
        formal_faithfulness_dir=formal_faith,
    )

    assert first["identity_matched_samples"] == 1
    assert first["newly_seeded_attribution_pairs"] == 8
    assert second["newly_seeded_attribution_pairs"] == 0
    assert second["reused_attribution_pairs"] == 8
    assert {
        row["sample_index"]
        for row in read_jsonl(formal_attr / "attribution_records.jsonl")
    } == {0}


def test_preview_ablation_reuse_requires_full_identity_and_is_idempotent(
    tmp_path,
):
    formal_dataset = tmp_path / "formal.dataset.jsonl"
    formal_model = tmp_path / "formal.model.jsonl"
    formal_evaluation = tmp_path / "formal.evaluation.jsonl"
    preview_dataset = tmp_path / "preview.dataset.jsonl"
    preview_model = tmp_path / "preview.model.jsonl"
    preview_ablation = tmp_path / "preview.ablation.jsonl"
    formal_ablation = tmp_path / "formal.ablation.jsonl"
    dataset = {
        "sample_id": "same",
        "benchmark": "vizwiz_lf",
        "input": {"I_IMAGE": "image.jpg", "I_QUESTION": "question"},
    }
    model = {
        "sample_id": "same",
        "I_IMAGE": "image.jpg",
        "I_QUESTION": "question",
        "raw_response": "<think>reason</think>answer",
        "THINKING": "reason",
        "OUTPUT": "answer",
        "THINKING_SPAN": [0, 1],
        "OUTPUT_SPAN": [2, 2],
        "generation_metadata": {
            "original_generated_token_ids": [1, 2],
            "teacher_forced_token_ids": [1, 2],
        },
        "model": {
            "repo_id": "model",
            "resolved_revision": "revision",
            "generation": {"do_sample": False, "max_new_tokens": 2048},
        },
    }
    ablation = {
        "schema_version": 1,
        "status": "complete",
        "benchmark": "vizwiz_lf",
        "sample_id": "same",
        "I_QUESTION": "question",
        "ablations": {
            "global_blur": {"status": "ok", "OUTPUT": "blurred"},
            "uniform_gray": {"status": "ok", "OUTPUT": "gray"},
        },
        "model": {
            "repo_id": "model",
            "revision": "revision",
            "do_sample": False,
            "max_new_tokens": 2048,
        },
    }
    write_jsonl([dataset], formal_dataset)
    write_jsonl([dataset], preview_dataset)
    write_jsonl([model], formal_model)
    write_jsonl([model], preview_model)
    write_jsonl(
        [{"sample_id": "same", "pre_ablation_eligible": True}],
        formal_evaluation,
    )
    write_jsonl([ablation], preview_ablation)

    kwargs = {
        "formal_dataset": formal_dataset,
        "formal_model": formal_model,
        "formal_generation_evaluation": formal_evaluation,
        "preview_dataset": preview_dataset,
        "preview_model": preview_model,
        "preview_ablation_model": preview_ablation,
        "formal_ablation_model": formal_ablation,
    }
    first = reuse_preview_ablation_checkpoints(**kwargs)
    second = reuse_preview_ablation_checkpoints(**kwargs)

    assert first["identity_matched_candidates"] == 1
    assert first["newly_seeded_ablation_records"] == 1
    assert second["newly_seeded_ablation_records"] == 0
    assert second["reused_ablation_records"] == 1
    seeded = read_jsonl(formal_ablation)
    assert len(seeded) == 1
    assert seeded[0]["checkpoint_provenance"]["kind"] == (
        "response_identical_formal_preview_candidate_ablation"
    )

    mismatched_model = dict(model)
    mismatched_model["raw_response"] = "<think>different</think>answer"
    write_jsonl([mismatched_model], formal_model)
    mismatch = reuse_preview_ablation_checkpoints(
        **{
            **kwargs,
            "formal_ablation_model": tmp_path / "mismatch.ablation.jsonl",
        }
    )
    assert mismatch["identity_matched_candidates"] == 0
    assert mismatch["newly_seeded_ablation_records"] == 0


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


def test_multimodal_protocol_matches_frozen_v2_main_experiment():
    protocol_path = Path(__file__).parents[1] / "evaluations" / "multimodal" / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    primary = [
        benchmark["id"]
        for benchmark in protocol["benchmarks"]
        if benchmark["tier"] == "primary"
    ]
    assert protocol["schema_version"] == 2
    assert protocol["status"] == "main_experiment_scope_frozen"
    assert primary == ["wiki_visa", "vizwiz_lf"]
    assert [experiment["id"] for experiment in protocol["formal_experiments"]] == [
        "E1",
        "E2",
        "E3",
        "E4",
        "E5",
    ]
    assert protocol["faithfulness_protocol"]["region_budget"] == 64
    assert protocol["faithfulness_protocol"]["curve_steps"] == 10
    assert protocol["frozen_id_artifact"].endswith("formal/frozen_ids.json")
    assert list(DEFAULT_METHODS) == [
        "random",
        "center",
        "visual-loo",
        "ifr-span",
        "visual-ig",
        "attnlrp",
        "flashtrace",
        "flashtrace-all-gen",
    ]


def test_vizwiz_selection_is_balanced_and_retains_crowd_metadata(tmp_path):
    source = {}
    images_root = tmp_path / "images"
    images_root.mkdir()
    for index, question_type in enumerate(
        ("Identification", "Description", "Reading", "Others"), start=1
    ):
        source[str(index)] = {
            "model": "Expert",
            "answerability": "answerable",
            "question": f"Question {index}?",
            "question_type": question_type,
            "image_url": f"https://example.invalid/{index}.jpg",
            "answer_paragraph": f"Expert answer {index}.",
            "answer_sentences": [f"Expert answer {index}."],
            "crowd_answers": ["answer", "answer"],
            "crowd_majority": "answer",
        }
        Image.new("RGB", (8, 6), "white").save(images_root / f"{index:03d}.jpg")
    expert_json = tmp_path / "expert.json"
    expert_json.write_text(json.dumps(source), encoding="utf-8")

    records = select_vizwiz_lf(
        expert_json=expert_json,
        images_root=images_root,
        sample_size=4,
        seed=17,
        download_missing=False,
    )

    assert len(records) == 4
    assert {
        record["evaluation"]["metadata"]["question_type"] for record in records
    } == {"Identification", "Description", "Reading", "Others"}
    assert all(
        record["evaluation"]["metadata"]["crowd_answers"] == ["answer", "answer"]
        for record in records
    )
    assert all(
        record["evaluation"]["metadata"]["semantic_correctness"]["status"]
        == "unreviewed"
        for record in records
    )
    assert all(
        record["evaluation"]["EVIDENCE_BOXES"] is None for record in records
    )


def test_vizwiz_gate_does_not_require_exact_match_but_requires_long_usable_output():
    usable = (
        "The label identifies this as tomato soup and lists heating instructions "
        "for two minutes, followed by a short standing period before serving."
    )
    gate = pre_ablation_gate(
        benchmark="vizwiz_lf",
        output=usable,
        output_correct_value=False,
        generation_stable=True,
        image_dependence_delta=0.1,
        token_identity_stable=True,
        thinking_tokens=200,
        output_tokens=24,
    )

    assert gate["correctness_gate_required"] is False
    assert gate["pre_ablation_eligible"] is True
    assert output_is_refusal_or_unanswerable("Unanswerable.") is True
    assert (
        output_is_refusal_or_unanswerable(
            "The image is highly blurred, making it impossible to clearly "
            "identify or read any text on the card."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The question about their color cannot be addressed based on the "
            "provided visual evidence."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The phone number is obscured and not legible, so it cannot be "
            "identified."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The image does not contain any readable text to answer the question."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The package has no instructions, so the image is insufficient "
            "to determine the cooking duration."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The number of calories cannot be determined from the visual evidence."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The sodium amount cannot be determined from the provided visual."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The specific DVD title cannot be determined with certainty based "
            "on the image alone."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "There is no visual evidence in the image to determine the color "
            "of the jeans."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The expiration date cannot be determined from this image."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The image is blurred, making it impossible to discern any "
            "readable text."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The image does not provide information about a page number."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The question cannot be addressed based on the provided visual "
            "evidence."
        )
        is True
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The exact garment cannot be definitively determined, but the "
            "visible seams clearly show that this is a piece of clothing."
        )
        is False
    )
    assert (
        output_is_refusal_or_unanswerable(
            "The image is blurry, but the largest visible word appears to be soup."
        )
        is False
    )

    refused = pre_ablation_gate(
        benchmark="vizwiz_lf",
        output="I cannot answer from this image.",
        output_correct_value=False,
        generation_stable=True,
        image_dependence_delta=0.1,
        token_identity_stable=True,
        thinking_tokens=200,
        output_tokens=24,
    )
    assert refused["pre_ablation_eligible"] is False


def test_refresh_generation_gates_updates_saved_unanswerable_output(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    models = tmp_path / "models.jsonl"
    evaluations = tmp_path / "evaluations.jsonl"
    write_jsonl(
        [{"sample_id": "viz-1", "benchmark": "vizwiz_lf"}], dataset
    )
    write_jsonl(
        [
            {
                "sample_id": "viz-1",
                "OUTPUT": (
                    "The image is blurred, making it impossible to identify "
                    "the card."
                ),
                "generation_metadata": {
                    "thinking_tokens": 100,
                    "output_tokens": 20,
                },
            }
        ],
        models,
    )
    write_jsonl(
        [
            {
                "sample_id": "viz-1",
                "generation_stable": True,
                "image_dependence_delta": 0.1,
                "generated_teacher_forced_ids_match": True,
                "reference_exact_match": False,
                "gates": {"output_non_refusal": True},
                "pre_ablation_eligible": True,
                "strict_eligible": True,
            }
        ],
        evaluations,
    )

    rows, changed = refresh(dataset, models, evaluations)

    assert changed == ["viz-1"]
    assert rows[0]["gates"]["output_non_refusal"] is False
    assert rows[0]["pre_ablation_eligible"] is False


def test_localization_gate_still_requires_correct_output():
    gate = pre_ablation_gate(
        benchmark="wiki_visa",
        output="wrong",
        output_correct_value=False,
        generation_stable=True,
        image_dependence_delta=0.1,
        token_identity_stable=True,
        thinking_tokens=100,
        output_tokens=1,
    )

    assert gate["correctness_gate_required"] is True
    assert gate["gates"]["whole_output_correct"] is False
    assert gate["pre_ablation_eligible"] is False


def test_formal_generation_budget_is_dataset_aware():
    assert default_max_new_tokens([{"benchmark": "wiki_visa"}]) == 1024
    assert default_max_new_tokens([{"benchmark": "vizwiz_lf"}]) == 2048
    with pytest.raises(ValueError, match="mixes benchmarks"):
        default_max_new_tokens(
            [{"benchmark": "wiki_visa"}, {"benchmark": "vizwiz_lf"}]
        )


def test_strict_subset_selection_is_seeded_and_can_freeze_ids(tmp_path):
    dataset_records = []
    model_records = []
    evaluation_records = []
    for index in range(8):
        sample_id = f"vizwiz-lf-{index:03d}"
        question_type = "Identification" if index < 4 else "Reading"
        dataset_records.append(
            {
                "schema_version": 2,
                "benchmark": "vizwiz_lf",
                "sample_id": sample_id,
                "input": {
                    "I_IMAGE": f"{index}.jpg",
                    "I_QUESTION": f"Question {index}?",
                },
                "evaluation": {
                    "REFERENCE_OUTPUT": f"Reference {index}",
                    "EVIDENCE_BOXES": None,
                    "EVIDENCE_MASKS": None,
                    "metadata": {
                        "official_record_id": str(index),
                        "question_type": question_type,
                    },
                },
            }
        )
        model_records.append(
            {
                "sample_id": sample_id,
                "generation_metadata": {
                    "original_generated_token_ids": [index],
                    "teacher_forced_token_ids": [index],
                    "output_tokens": index + 16,
                },
            }
        )
        evaluation_records.append(
            {
                "sample_id": sample_id,
                "strict_eligible": True,
                "image_dependent_by_generation_ablation": True,
                "gates": {
                    "generation_stable": True,
                    "generation_ablation_changes_output": True,
                },
            }
        )

    paths = []
    for name, records in (
        ("dataset", dataset_records),
        ("model", model_records),
        ("evaluation", evaluation_records),
    ):
        path = tmp_path / f"{name}.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        paths.append(path)

    selected = select_strict_subset(
        [paths[0]],
        [paths[1]],
        [paths[2]],
        sample_size=4,
        balance_key="question_type",
        seed=17,
        exclude_sample_ids={"vizwiz-lf-000"},
    )
    repeated = select_strict_subset(
        [paths[0]],
        [paths[1]],
        [paths[2]],
        sample_size=4,
        balance_key="question_type",
        seed=17,
        exclude_sample_ids={"vizwiz-lf-000"},
    )
    assert [record["sample_id"] for record in selected[0]] == [
        record["sample_id"] for record in repeated[0]
    ]
    assert "vizwiz-lf-000" not in {
        record["sample_id"] for record in selected[0]
    }
    assert {
        question_type: sum(
            record["evaluation"]["metadata"]["question_type"] == question_type
            for record in selected[0]
        )
        for question_type in ("Identification", "Reading")
    } == {"Identification": 2, "Reading": 2}

    unbalanced = select_strict_subset(
        [paths[0]],
        [paths[1]],
        [paths[2]],
        sample_size=4,
        balance_key=None,
        seed=17,
        exclude_sample_ids={"vizwiz-lf-000"},
    )
    assert all(
        record["evaluation"]["metadata"]["output_length_tercile"]
        in {"short", "medium", "long"}
        for record in unbalanced[0]
    )
    assert all(
        len(record["evaluation"]["metadata"]["output_token_tercile_cutpoints"])
        == 2
        for record in unbalanced[0]
    )

    frozen_path = tmp_path / "frozen_ids.json"
    update_frozen_ids(
        frozen_path,
        selected[0],
        selected[1],
        balance_key="question_type",
        seed=17,
    )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    assert frozen["datasets"]["vizwiz_lf"]["count"] == 4
    assert {
        sample["output_length_tercile"]
        for sample in frozen["datasets"]["vizwiz_lf"]["samples"]
    }.issubset({"short", "medium", "long"})

    unbalanced_frozen_path = tmp_path / "unbalanced_frozen_ids.json"
    update_frozen_ids(
        unbalanced_frozen_path,
        unbalanced[0],
        unbalanced[1],
        balance_key=None,
        seed=17,
    )
    unbalanced_frozen = json.loads(
        unbalanced_frozen_path.read_text(encoding="utf-8")
    )
    assert (
        unbalanced_frozen["datasets"]["vizwiz_lf"]["selection_mode"]
        == "unstratified_fixed_seed"
    )
    assert unbalanced_frozen["datasets"]["vizwiz_lf"]["balance_key"] is None

    funnel = gate_funnel(
        {record["sample_id"]: record for record in dataset_records},
        {record["sample_id"]: record for record in model_records},
        {record["sample_id"]: record for record in evaluation_records},
        exclude_sample_ids={"vizwiz-lf-000"},
        frozen_sample_ids={record["sample_id"] for record in unbalanced[0]},
    )
    assert funnel["candidate_count"] == 8
    assert funnel["strict_eligible_count"] == 7
    assert funnel["frozen_sample_count"] == 4
    assert funnel["stages"][-1] == {
        "stage": "unique_image_and_fixed_seed_freeze",
        "passed": 4,
        "eliminated_at_stage": 3,
    }
    assert funnel["excluded_prior_pilot_count"] == 1


def test_vizwiz_semantic_judgments_require_complete_deterministic_audit(tmp_path):
    evaluation_path = tmp_path / "evaluation.jsonl"
    evaluation_path.write_text(
        "".join(
            json.dumps(
                {
                    "schema_version": 2,
                    "benchmark": "vizwiz_lf",
                    "sample_id": f"vizwiz-lf-{index:03d}",
                    "strict_eligible": True,
                }
            )
            + "\n"
            for index in range(10)
        ),
        encoding="utf-8",
    )
    audit_ids = audit_sample_ids(
        [f"vizwiz-lf-{index:03d}" for index in range(10)],
        fraction=0.1,
        seed=17,
    )
    judgments = []
    for index in range(10):
        sample_id = f"vizwiz-lf-{index:03d}"
        judgments.append(
            {
                "sample_id": sample_id,
                "label": "fully" if index < 6 else "partial",
                "judge": "fixture-judge",
                "reason": "Fixture evidence.",
                "human_reviewed": sample_id in audit_ids,
                "human_reviewer": "fixture-human" if sample_id in audit_ids else None,
            }
        )
    judgments_path = tmp_path / "judgments.jsonl"
    judgments_path.write_text(
        "".join(json.dumps(record) + "\n" for record in judgments),
        encoding="utf-8",
    )

    joined, summary = join_judgments(
        evaluation_path,
        judgments_path,
        audit_fraction=0.1,
        audit_seed=17,
        require_complete=True,
    )

    assert summary["complete"] is True
    assert summary["audit_reviewed"] == 1
    assert summary["label_counts"] == {"fully": 6, "partial": 4, "wrong": 0}
    assert all(
        record["semantic_correctness"]["status"] == "reviewed"
        for record in joined
    )


def test_gate_funnel_separates_thinking_and_token_identity_errors():
    datasets = {
        sample_id: {"sample_id": sample_id}
        for sample_id in ("no-think", "token-mismatch", "complete")
    }
    models = {
        "complete": {
            "sample_id": "complete",
            "generation_metadata": {
                "original_generated_token_ids": [1],
                "teacher_forced_token_ids": [1],
            },
        }
    }
    evaluations = {
        "no-think": {
            "sample_id": "no-think",
            "status": "error",
            "error": "strict Thinking response has no </think> terminator",
        },
        "token-mismatch": {
            "sample_id": "token-mismatch",
            "status": "error",
            "error": (
                "generated token IDs differ from decode/re-encoded "
                "teacher-forced IDs"
            ),
        },
        "complete": {
            "sample_id": "complete",
            "strict_eligible": True,
            "gates": {
                "thinking_closed": True,
                "generated_teacher_forced_ids_match": True,
            },
        },
    }

    funnel = gate_funnel(datasets, models, evaluations)

    stages = {stage["stage"]: stage for stage in funnel["stages"]}
    assert stages["thinking_closed"]["passed"] == 2
    assert stages["thinking_closed"]["eliminated_at_stage"] == 1
    assert stages["generated_teacher_forced_ids_match"]["passed"] == 1
    assert stages["generated_teacher_forced_ids_match"][
        "eliminated_at_stage"
    ] == 1
    assert funnel["gate_marginal_counts"]["thinking_closed"] == {
        "passed": 2,
        "failed": 1,
        "not_evaluated": 0,
    }


def test_formal_audit_validates_funnel_population_conservation():
    payload = {
        "candidate_count": 10,
        "model_record_count": 8,
        "evaluation_record_count": 10,
        "strict_eligible_count": 6,
        "frozen_sample_count": 4,
        "excluded_prior_pilot_count": 1,
        "stages": [
            {
                "stage": "candidate_manifest",
                "passed": 10,
                "eliminated_at_stage": 0,
            },
            {
                "stage": "prior_pilot_sample_exclusion",
                "passed": 9,
                "eliminated_at_stage": 1,
            },
            {
                "stage": "final_strict_eligible",
                "passed": 6,
                "eliminated_at_stage": 3,
            },
            {
                "stage": "unique_image_and_fixed_seed_freeze",
                "passed": 4,
                "eliminated_at_stage": 2,
            },
        ],
    }
    assert _funnel_conservation_issues(payload) == []

    broken = json.loads(json.dumps(payload))
    broken["stages"][2]["eliminated_at_stage"] = 2
    broken["evaluation_record_count"] = 9
    issues = _funnel_conservation_issues(broken)
    assert any("population is not conserved" in issue for issue in issues)
    assert "evaluation_record_count must equal candidate_count" in issues


def test_formal_audit_validates_frozen_protocol_metadata():
    wiki_samples = [
        {
            "sample_id": f"wiki-{index:03d}",
            "balance_group": group,
            "output_tokens": index % 5 + 1,
        }
        for group_index, group in enumerate(EXPECTED_STRATA)
        for index in range(group_index * 40, (group_index + 1) * 40)
    ]
    viz_samples = [
        {
            "sample_id": f"viz-{index:03d}",
            "balance_group": None,
            "output_tokens": index + 1,
            "question_type": "Reading",
            "output_length_tercile": (
                "short" if index < 34 else "medium" if index < 67 else "long"
            ),
        }
        for index in range(100)
    ]
    frozen = {
        "schema_version": 1,
        "frozen_on": "2026-07-24",
        "selection_seed": 17,
        "datasets": {
            "wiki_visa": {
                "count": 120,
                "balance_key": "stratum",
                "selection_mode": "balanced_fixed_seed",
                "output_token_tercile_cutpoints": [2, 4],
                "samples": wiki_samples,
            },
            "vizwiz_lf": {
                "count": 100,
                "balance_key": None,
                "selection_mode": "unstratified_fixed_seed",
                "output_token_tercile_cutpoints": [34, 67],
                "samples": viz_samples,
            },
        },
    }
    assert _frozen_protocol_metadata_issues(frozen) == []

    frozen["selection_seed"] = 99
    frozen["datasets"]["wiki_visa"]["samples"][0]["balance_group"] = "wrong"
    issues = _frozen_protocol_metadata_issues(frozen)
    assert "selection_seed must be 17" in issues
    assert "wiki_visa balance groups must be exactly 40/40/40" in issues


def test_formal_audit_revalidates_frozen_record_joins_and_hard_gates():
    generated_ids = [11, 12, 13]
    dataset = {
        "benchmark": "wiki_visa",
        "sample_id": "wiki-001",
        "input": {"I_IMAGE": "image.png", "I_QUESTION": "question"},
        "evaluation": {"REFERENCE_OUTPUT": "answer"},
    }
    model = {
        "schema_version": 2,
        "benchmark": "wiki_visa",
        "sample_id": "wiki-001",
        "I_IMAGE": "image.png",
        "I_QUESTION": "question",
        "THINKING": "reasoning",
        "OUTPUT": "answer",
        "THINKING_SPAN": [0, 0],
        "OUTPUT_SPAN": [1, 1],
        "raw_response": "<think>reasoning</think>answer",
        "model": {
            "repo_id": "Qwen/Qwen3-VL-8B-Thinking",
            "requested_revision": EXPECTED_REVISION,
            "resolved_revision": EXPECTED_REVISION,
            "generation": {
                "do_sample": False,
                "max_new_tokens": 1024,
                "prompt_profile": "concise",
            },
        },
        "generation_metadata": {
            "prompt_profile": "concise",
            "original_generated_token_ids": generated_ids,
            "teacher_forced_token_ids": list(generated_ids),
            "original_generated_tokens_without_eos": 3,
            "output_tokens": 1,
        },
    }
    gates = {
        "generation_stable": True,
        "positive_blur_logprob_drop": True,
        "generated_teacher_forced_ids_match": True,
        "thinking_closed": True,
        "whole_output_correct": True,
        "generation_ablation_changes_output": True,
    }
    evaluation = {
        "sample_id": "wiki-001",
        "REFERENCE_OUTPUT": "answer",
        "gates": gates,
        "strict_eligible": True,
        "pre_ablation_eligible": True,
        "generation_stable": True,
        "generated_teacher_forced_ids_match": True,
        "image_dependent": True,
        "image_dependent_by_generation_ablation": True,
        "stability_repeats": 2,
        "original_output_mean_logprob": -0.1,
        "blurred_output_mean_logprob": -0.4,
        "image_dependence_delta": 0.3,
        "ablation_outputs": {
            "global_blur": {
                "status": "ok",
                "same_as_original_output": False,
            },
            "uniform_gray": {
                "status": "ok",
                "same_as_original_output": True,
            },
        },
        "correctness_gate_required": True,
        "output_correct": True,
        "reference_exact_match": True,
    }
    assert (
        _frozen_record_protocol_issues(
            [dataset], [model], [evaluation], benchmark="wiki_visa"
        )
        == []
    )

    broken_model = json.loads(json.dumps(model))
    broken_model["I_QUESTION"] = "different"
    broken_evaluation = json.loads(json.dumps(evaluation))
    broken_evaluation["gates"]["whole_output_correct"] = False
    issues = _frozen_record_protocol_issues(
        [dataset], [broken_model], [broken_evaluation], benchmark="wiki_visa"
    )
    assert any("model input does not match" in issue for issue in issues)
    assert any("hard-gate set is not exact and all true" in issue for issue in issues)


def test_formal_audit_locks_every_top_level_protocol_decision():
    protocol_path = Path("evaluations/multimodal/protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    assert _protocol_v2_issues(protocol) == []

    changed = json.loads(json.dumps(protocol))
    changed["benchmarks"][0]["primary_endpoints"] = ["evidence_rank_auc"]
    changed["excluded_new_runs"].remove("second_vlm")
    issues = _protocol_v2_issues(changed)
    assert any("Wiki role" in issue for issue in issues)
    assert "explicitly excluded experiment scope changed" in issues


def test_formal_audit_uses_wiki_ids_for_late_faithfulness_analysis(
    tmp_path,
):
    methods = {
        "random",
        "center",
        "visual-loo",
        "visual-ig",
        "attnlrp",
        "flashtrace",
        "ifr-span",
        "flashtrace-all-gen",
    }
    metrics = ("deletion_auc", "insertion_auc", "visual_mas")
    wiki_ids = {"wiki-a", "wiki-b", "wiki-c"}

    def interval():
        return {"mean": 0.5, "ci95_low": 0.4, "ci95_high": 0.6}

    def estimates(selected_methods):
        return {
            method: {metric: interval() for metric in metrics}
            for method in selected_methods
        }

    def comparisons(count, baselines):
        return {
            baseline: {
                metric: {
                    **interval(),
                    "wins": count,
                    "ties": 0,
                    "losses": 0,
                }
                for metric in metrics
            }
            for baseline in baselines
        }

    baselines = methods - {"flashtrace"}
    ordered_ids = sorted(wiki_ids)
    buckets = {}
    for bucket, sample_id in zip(
        ("short", "medium", "long"), ordered_ids, strict=True
    ):
        buckets[bucket] = {
            "samples": 1,
            "sample_ids": [sample_id],
            "estimates": estimates({"ifr-span", "flashtrace"}),
            "flashtrace_favorable_difference": comparisons(
                1, {"ifr-span"}
            ),
        }
    analysis = {
        "bootstrap_draws": 50_000,
        "positive_only_available": True,
        "overall": {
            "samples": 3,
            "sample_ids": ordered_ids,
            "estimates": estimates(methods),
            "flashtrace_favorable_difference": comparisons(3, baselines),
        },
        "positive_only_ordering": {
            "samples": 3,
            "sample_ids": ordered_ids,
            "estimates": estimates(methods),
        },
        "recursion_by_thinking_bucket": buckets,
    }
    faith_dir = tmp_path / "wiki_visa_n120_faithfulness"
    faith_dir.mkdir()
    (faith_dir / "analysis.json").write_text(
        json.dumps(analysis), encoding="utf-8"
    )

    audit = Audit()
    _audit_analysis_payloads(
        audit,
        tmp_path,
        wiki_ids=wiki_ids,
        viz_ids={"viz-a", "viz-b"},
    )
    checks = {check["name"]: check for check in audit.checks}

    assert checks[
        "E4/E5 Wiki: exact paired IDs, finite intervals, and W/T/L"
    ]["passed"]


def test_vizwiz_semantic_judgment_validator_rejects_bad_labels():
    with pytest.raises(ValueError, match="invalid semantic label"):
        validate_judgment(
            {
                "sample_id": "vizwiz-lf-001",
                "label": "mostly",
                "judge": "judge",
                "reason": "reason",
            }
        )


def test_vizwiz_human_review_adjudicates_only_deterministic_audit_rows(tmp_path):
    sample_ids = [f"vizwiz-lf-{index:03d}" for index in range(10)]
    audit_ids = audit_sample_ids(sample_ids, fraction=0.1, seed=17)
    judgments_path = tmp_path / "judgments.jsonl"
    judgments_path.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "label": "partial",
                    "judge": "fixture-llm",
                    "reason": "LLM reason.",
                }
            )
            + "\n"
            for sample_id in sample_ids
        ),
        encoding="utf-8",
    )
    reviews_path = tmp_path / "reviews.jsonl"
    reviews_path.write_text(
        json.dumps(
            {
                "sample_id": audit_ids[0],
                "human_label": "fully",
                "human_reviewer": "fixture-human",
                "human_reason": "The answer is materially correct.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    reviewed = apply_human_reviews(
        judgments_path,
        reviews_path,
        audit_fraction=0.1,
        audit_seed=17,
    )

    audited = next(
        record for record in reviewed if record["sample_id"] == audit_ids[0]
    )
    assert audited["label"] == "fully"
    assert audited["llm_label"] == "partial"
    assert audited["human_reviewed"] is True
    assert sum(record.get("human_reviewed") is True for record in reviewed) == 1


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
            "visual_grid_shape": [4, 8],
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
    assert summary["methods"]["a"]["native_grid_shapes"] == {"4x8": 1}
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


def test_strict_analysis_bootstraps_each_stratum_on_paired_samples(tmp_path):
    sample_ids = [f"sample-{index}" for index in range(4)]
    manifest = tmp_path / "dataset.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "evaluation": {
                        "metadata": {
                            "stratum": "first" if index < 2 else "later"
                        }
                    },
                }
            )
            + "\n"
            for index, sample_id in enumerate(sample_ids)
        ),
        encoding="utf-8",
    )
    attribution_dir = tmp_path / "attribution"
    attribution_dir.mkdir()
    (attribution_dir / "summary.json").write_text(
        json.dumps(
            {
                "requested_methods": ["baseline", "flashtrace"],
                "common_sample_ids": sample_ids,
            }
        ),
        encoding="utf-8",
    )
    records = []
    for index, sample_id in enumerate(sample_ids):
        for method in ("baseline", "flashtrace"):
            value = float(index + (1 if method == "flashtrace" else 0))
            records.append(
                {
                    "status": "ok",
                    "sample_id": sample_id,
                    "method": method,
                    "localization": {
                        metric: value for metric in STRICT_LOCALIZATION_METRICS
                    },
                }
            )
    (attribution_dir / "attribution_records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    analysis = analyze_strict_results(
        manifest,
        attribution_dir,
        draws=100,
        seed=17,
    )

    assert analysis["per_group_paired"]["first"]["samples"] == 2
    assert analysis["per_group_paired"]["later"]["samples"] == 2
    delta = analysis["per_group_paired"]["first"][
        "flashtrace_minus_baseline"
    ]["energy_in_mask"]["baseline"]
    assert delta["mean"] == 1.0
    assert (delta["wins"], delta["ties"], delta["losses"]) == (2, 0, 0)


def test_visual_faithfulness_grid_preserves_tall_page_geometry():
    tall = Image.new("RGB", (980, 3920))
    landscape = Image.new("RGB", (480, 320))

    assert region_layout(tall, 64) == (16, 4)
    assert region_layout(landscape, 64) == (7, 9)


def test_visual_faithfulness_summary_discloses_region_layouts():
    records = []
    for method in ("a", "b"):
        records.append(
            {
                "status": "ok",
                "sample_id": "sample",
                "method": method,
                "seconds": 1.0,
                "faithfulness": {
                    "region_layout": [8, 8],
                    "deletion_auc": 0.5,
                    "insertion_auc": 0.5,
                    "visual_rise": 0.5,
                    "visual_mas": 0.5,
                    "visual_rise_plus_ap": 0.5,
                    "deletion_endpoint_delta": 1.0,
                    "insertion_endpoint_delta": 1.0,
                    "deletion_degenerate": False,
                    "insertion_degenerate": False,
                    "positive_only_ordering": {
                        "deletion_auc": 0.5,
                        "insertion_auc": 0.5,
                        "visual_mas": 0.5,
                        "identical_to_signed_order": True,
                    },
                },
            }
        )

    summary = faithfulness_summary(records, ("a", "b"))

    assert summary["methods"]["a"]["region_layouts"] == {"8x8": 1}
    assert summary["methods"]["b"]["region_layouts"] == {"8x8": 1}


def test_visual_faithfulness_summary_only_never_loads_the_model(
    tmp_path, monkeypatch
):
    dataset_path = tmp_path / "dataset.jsonl"
    model_path = tmp_path / "model.jsonl"
    attribution_dir = tmp_path / "methods"
    output_dir = tmp_path / "faithfulness"
    attribution_dir.mkdir()
    output_dir.mkdir()
    write_jsonl(
        [{"sample_id": "sample", "benchmark": "wiki_visa"}],
        dataset_path,
    )
    write_jsonl(
        [
            {
                "schema_version": 2,
                "benchmark": "wiki_visa",
                "sample_id": "sample",
                "I_IMAGE": "image.png",
                "I_QUESTION": "question",
                "THINKING": "reasoning",
                "OUTPUT": "answer",
                "THINKING_SPAN": [0, 0],
                "OUTPUT_SPAN": [1, 1],
                "raw_response": "<think>reasoning</think>answer",
                "model": {"resolved_revision": EXPECTED_REVISION},
            }
        ],
        model_path,
    )
    (attribution_dir / "summary.json").write_text(
        json.dumps(
            {
                "requested_methods": ["method"],
                "common_sample_ids": ["sample"],
            }
        ),
        encoding="utf-8",
    )
    write_jsonl([], attribution_dir / "attribution_records.jsonl")
    faithfulness = {
        "region_layout": [8, 8],
        "deletion_auc": 0.5,
        "insertion_auc": 0.5,
        "visual_rise": 0.5,
        "visual_mas": 0.5,
        "visual_rise_plus_ap": 0.5,
        "deletion_endpoint_delta": 1.0,
        "insertion_endpoint_delta": 1.0,
        "deletion_degenerate": False,
        "insertion_degenerate": False,
        "positive_only_ordering": {
            "deletion_auc": 0.5,
            "insertion_auc": 0.5,
            "visual_mas": 0.5,
            "identical_to_signed_order": True,
        },
    }
    write_jsonl(
        [
            {
                "status": "ok",
                "sample_id": "sample",
                "method": "method",
                "seconds": 1.0,
                "faithfulness": faithfulness,
            }
        ],
        output_dir / "faithfulness_records.jsonl",
    )

    def fail_model_load(*args, **kwargs):
        raise AssertionError("summary-only must not load a model")

    monkeypatch.setattr(
        "flashtrace.load_vlm_and_processor",
        fail_model_load,
    )
    summary = faithfulness_module.run(
        dataset_manifest=dataset_path,
        model_output=model_path,
        attribution_dir=attribution_dir,
        output_dir=output_dir,
        methods=("method",),
        model_name="Qwen/Qwen3-VL-8B-Thinking",
        revision=EXPECTED_REVISION,
        device="cuda:0",
        min_pixels=200_704,
        max_pixels=2_007_040,
        steps=10,
        target_regions=64,
        sample_ids=None,
        summary_only=True,
    )

    assert summary["common_samples"] == 1
    assert summary["processor"] == {
        "min_pixels": 200_704,
        "max_pixels": 2_007_040,
    }


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


def test_visual_curve_normalization_marks_reversed_endpoints_degenerate():
    deletion, deletion_degenerate = _normalize_deletion(
        np.array([-0.4, -0.2])
    )
    insertion, insertion_degenerate = _normalize_insertion(
        np.array([-0.2, -0.4])
    )

    assert deletion_degenerate
    assert insertion_degenerate
    assert deletion.tolist() == pytest.approx([1.0, 0.0])
    assert insertion.tolist() == pytest.approx([0.0, 1.0])


def test_refresh_visual_curve_metrics_uses_saved_raw_observations():
    refreshed = refresh_derived_curve_metrics(
        {
            "fractions": [0.0, 0.5, 1.0],
            "remaining_attribution_density": [1.0, 0.4, 0.0],
            "deletion_output_mean_logprob": [-0.4, -0.3, -0.2],
            "insertion_output_mean_logprob": [-0.2, -0.3, -0.4],
            "normalized_deletion": [0.0, 0.0, 0.0],
            "normalized_insertion": [0.0, 0.0, 0.0],
        }
    )

    assert refreshed["normalization_policy"] == CURVE_NORMALIZATION_POLICY
    assert refreshed["deletion_degenerate"]
    assert refreshed["insertion_degenerate"]
    assert refreshed["normalized_deletion"] == pytest.approx([1.0, 0.5, 0.0])
    assert refreshed["normalized_insertion"] == pytest.approx([0.0, 0.5, 1.0])


def test_visual_faithfulness_saves_positive_only_order_sensitivity(monkeypatch):
    monkeypatch.setattr(
        faithfulness_module,
        "output_mean_logprob",
        lambda _model, _processor, image, _prompt, _response, _span: (
            np.asarray(image, dtype=np.float64).mean() / 255.0
        ),
    )
    image = Image.new("RGB", (4, 4), "white")

    result = evaluate_grid(
        model=object(),
        processor=object(),
        image=image,
        prompt="prompt",
        response="response",
        output_span=(0, 0),
        grid=[[-1.0, 1.0], [0.5, -2.0]],
        steps=4,
        target_regions=4,
        original_score=1.0,
        blurred_score=0.0,
    )

    assert result["ordering_policy"] == "signed_descending"
    assert result["region_scores"] == [-1.0, 1.0, 0.5, -2.0]
    assert (
        result["positive_only_ordering"]["identical_to_signed_order"] is False
    )
    assert result["positive_only_ordering"]["region_order"] != result["region_order"]
