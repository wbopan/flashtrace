"""Frozen-response visual deletion/insertion faithfulness for strict runs.

The complete model-generated THINKING + OUTPUT sequence is teacher-forced for
every perturbation. Only the saved OUTPUT_SPAN contributes to the score.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
import traceback
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

from .jsonl_checkpoint import PairJsonlCheckpoint
from .metrics import curve_auc
from .strict_attribution import _resample
from .strict_generation import (
    DEFAULT_MODEL,
    FORMAL_MAX_PIXELS,
    FROZEN_MODEL_REVISION,
    model_record_prompt,
    output_mean_logprob,
    read_jsonl,
    validate_model_record,
    write_jsonl,
)

CURVE_NORMALIZATION_POLICY = "directional_endpoint_span_nonpositive_is_degenerate"


def region_layout(image: Image.Image, target_regions: int) -> tuple[int, int]:
    """Return an approximately square-pixel grid with about target_regions cells."""

    if target_regions < 4:
        raise ValueError("target_regions must be at least 4")
    rows = max(1, round(math.sqrt(target_regions * image.height / image.width)))
    columns = max(1, round(target_regions / rows))
    return rows, columns


def _region_mask(
    size: tuple[int, int],
    layout: tuple[int, int],
    selected: Sequence[int],
) -> Image.Image:
    width, height = size
    rows, columns = layout
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for index in selected:
        row, column = divmod(int(index), columns)
        left = column * width // columns
        right = (column + 1) * width // columns
        top = row * height // rows
        bottom = (row + 1) * height // rows
        draw.rectangle((left, top, right - 1, bottom - 1), fill=255)
    return mask


def perturbation_pair(
    original: Image.Image,
    blurred: Image.Image,
    layout: tuple[int, int],
    selected: Sequence[int],
) -> tuple[Image.Image, Image.Image]:
    """Return deletion and insertion images for the same selected regions."""

    mask = _region_mask(original.size, layout, selected)
    deleted = Image.composite(blurred, original, mask)
    inserted = Image.composite(original, blurred, mask)
    return deleted, inserted


def _groups(order: np.ndarray, steps: int) -> list[np.ndarray]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    return [group for group in np.array_split(order, min(steps, order.size)) if group.size]


def _normalize_deletion(scores: np.ndarray) -> tuple[np.ndarray, bool]:
    denominator = float(scores[0] - scores[-1])
    if denominator <= 1e-8:
        return np.linspace(1.0, 0.0, scores.size), True
    normalized = np.clip((scores - scores[-1]) / denominator, 0.0, 1.0)
    return np.minimum.accumulate(normalized), False


def _normalize_insertion(scores: np.ndarray) -> tuple[np.ndarray, bool]:
    denominator = float(scores[-1] - scores[0])
    if denominator <= 1e-8:
        return np.linspace(0.0, 1.0, scores.size), True
    normalized = np.clip((scores - scores[0]) / denominator, 0.0, 1.0)
    return np.maximum.accumulate(normalized), False


def visual_mas(
    normalized_deletion: np.ndarray,
    remaining_density: np.ndarray,
) -> dict[str, float]:
    """Mirror the repository MAS/RISE correction on a visual deletion curve."""

    alignment_penalty = np.abs(normalized_deletion - remaining_density)
    corrected = np.clip(normalized_deletion + alignment_penalty, 0.0, 1.0)
    span = float(corrected.max() - corrected.min())
    if span > 1e-12:
        corrected = (corrected - corrected.min()) / span
    else:
        corrected = np.linspace(1.0, 0.0, corrected.size)
    fractions = np.linspace(0.0, 1.0, normalized_deletion.size)
    return {
        "visual_rise": curve_auc(normalized_deletion, fractions),
        "visual_mas": curve_auc(corrected, fractions),
        "visual_rise_plus_ap": curve_auc(
            normalized_deletion + alignment_penalty, fractions
        ),
    }


def _derived_curve_metrics(
    *,
    deletion: np.ndarray,
    insertion: np.ndarray,
    fractions: np.ndarray,
    remaining_density: np.ndarray,
) -> dict[str, Any]:
    """Derive reproducible metrics from saved perturbation observations."""

    normalized_deletion, deletion_degenerate = _normalize_deletion(deletion)
    normalized_insertion, insertion_degenerate = _normalize_insertion(insertion)
    return {
        "normalization_policy": CURVE_NORMALIZATION_POLICY,
        "normalized_deletion": normalized_deletion.tolist(),
        "normalized_insertion": normalized_insertion.tolist(),
        "deletion_auc": curve_auc(normalized_deletion, fractions),
        "insertion_auc": curve_auc(normalized_insertion, fractions),
        "deletion_endpoint_delta": float(deletion[0] - deletion[-1]),
        "insertion_endpoint_delta": float(insertion[-1] - insertion[0]),
        "deletion_degenerate": deletion_degenerate,
        "insertion_degenerate": insertion_degenerate,
        **visual_mas(normalized_deletion, remaining_density),
    }


def refresh_derived_curve_metrics(curve: Mapping[str, Any]) -> dict[str, Any]:
    """Refresh only metrics derivable from an already saved raw curve."""

    refreshed = dict(curve)
    deletion = np.asarray(
        refreshed["deletion_output_mean_logprob"], dtype=np.float64
    )
    insertion = np.asarray(
        refreshed["insertion_output_mean_logprob"], dtype=np.float64
    )
    fractions = np.asarray(refreshed["fractions"], dtype=np.float64)
    density = np.asarray(
        refreshed["remaining_attribution_density"], dtype=np.float64
    )
    if not (
        deletion.ndim
        == insertion.ndim
        == fractions.ndim
        == density.ndim
        == 1
        and deletion.size
        == insertion.size
        == fractions.size
        == density.size
        and deletion.size >= 2
        and np.isfinite(deletion).all()
        and np.isfinite(insertion).all()
        and np.isfinite(fractions).all()
        and np.isfinite(density).all()
    ):
        raise ValueError("cannot refresh malformed faithfulness curve")
    refreshed.update(
        _derived_curve_metrics(
            deletion=deletion,
            insertion=insertion,
            fractions=fractions,
            remaining_density=density,
        )
    )
    return refreshed


def refresh_record_metrics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Refresh signed and positive-only metrics without running the model."""

    refreshed = dict(record)
    if refreshed.get("status") != "ok":
        return refreshed
    faithfulness = dict(refreshed["faithfulness"])
    positive_only = refresh_derived_curve_metrics(
        faithfulness["positive_only_ordering"]
    )
    positive_only["identical_to_signed_order"] = faithfulness[
        "positive_only_ordering"
    ]["identical_to_signed_order"]
    faithfulness.update(refresh_derived_curve_metrics(faithfulness))
    faithfulness["positive_only_ordering"] = positive_only
    refreshed["faithfulness"] = faithfulness
    return refreshed


def _evaluate_order(
    *,
    model: Any,
    processor: Any,
    image: Image.Image,
    blurred: Image.Image,
    prompt: str,
    response: str,
    output_span: tuple[int, int],
    region_scores: np.ndarray,
    order: np.ndarray,
    layout: tuple[int, int],
    steps: int,
    original_score: float,
    blurred_score: float,
) -> dict[str, Any]:
    groups = _groups(order, steps)
    positive = np.clip(region_scores, 0.0, None)
    positive_total = float(positive.sum())

    deletion_scores = [float(original_score)]
    insertion_scores = [float(blurred_score)]
    fractions = [0.0]
    remaining_density = [1.0]
    selected: list[int] = []
    selected_positive = 0.0
    for group in groups:
        selected.extend(int(index) for index in group)
        selected_positive += float(positive[group].sum())
        deleted, inserted = perturbation_pair(image, blurred, layout, selected)
        deletion_scores.append(
            output_mean_logprob(
                model, processor, deleted, prompt, response, output_span
            )
        )
        insertion_scores.append(
            output_mean_logprob(
                model, processor, inserted, prompt, response, output_span
            )
        )
        fractions.append(len(selected) / order.size)
        if positive_total > 0.0:
            remaining_density.append(
                max(0.0, 1.0 - selected_positive / positive_total)
            )
        else:
            remaining_density.append(1.0 - len(selected) / order.size)

    deletion = np.asarray(deletion_scores, dtype=np.float64)
    insertion = np.asarray(insertion_scores, dtype=np.float64)
    x = np.asarray(fractions, dtype=np.float64)
    density = np.asarray(remaining_density, dtype=np.float64)
    return {
        "steps": len(groups),
        "fractions": x.tolist(),
        "region_order": order.tolist(),
        "remaining_attribution_density": density.tolist(),
        "deletion_output_mean_logprob": deletion.tolist(),
        "insertion_output_mean_logprob": insertion.tolist(),
        **_derived_curve_metrics(
            deletion=deletion,
            insertion=insertion,
            fractions=x,
            remaining_density=density,
        ),
    }


def evaluate_grid(
    *,
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    response: str,
    output_span: tuple[int, int],
    grid: Sequence[Sequence[float]],
    steps: int,
    target_regions: int,
    original_score: float | None = None,
    blurred_score: float | None = None,
) -> dict[str, Any]:
    layout = region_layout(image, target_regions)
    region_scores = _resample(grid, layout).reshape(-1)
    signed_order = np.argsort(region_scores, kind="stable")[::-1]
    positive_scores = np.clip(region_scores, 0.0, None)
    positive_order = np.argsort(positive_scores, kind="stable")[::-1]
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(image.size) / 12))
    if original_score is None:
        original_score = output_mean_logprob(
            model, processor, image, prompt, response, output_span
        )
    if blurred_score is None:
        blurred_score = output_mean_logprob(
            model, processor, blurred, prompt, response, output_span
        )

    signed = _evaluate_order(
        model=model,
        processor=processor,
        image=image,
        blurred=blurred,
        prompt=prompt,
        response=response,
        output_span=output_span,
        region_scores=region_scores,
        order=signed_order,
        layout=layout,
        steps=steps,
        original_score=float(original_score),
        blurred_score=float(blurred_score),
    )
    identical = np.array_equal(signed_order, positive_order)
    positive_only = (
        dict(signed)
        if identical
        else _evaluate_order(
            model=model,
            processor=processor,
            image=image,
            blurred=blurred,
            prompt=prompt,
            response=response,
            output_span=output_span,
            region_scores=region_scores,
            order=positive_order,
            layout=layout,
            steps=steps,
            original_score=float(original_score),
            blurred_score=float(blurred_score),
        )
    )
    positive_only["identical_to_signed_order"] = identical
    return {
        "region_layout": list(layout),
        "regions": int(signed_order.size),
        "region_scores": region_scores.tolist(),
        "ordering_policy": "signed_descending",
        **signed,
        "positive_only_ordering": positive_only,
    }


def _summary(records: list[dict[str, Any]], methods: tuple[str, ...]) -> dict[str, Any]:
    successful: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("status") == "ok":
            successful[record["method"]].add(record["sample_id"])
    common = set.intersection(*(successful[method] for method in methods))
    metric_names = (
        "deletion_auc",
        "insertion_auc",
        "visual_rise",
        "visual_mas",
        "visual_rise_plus_ap",
        "deletion_endpoint_delta",
        "insertion_endpoint_delta",
    )
    by_method: dict[str, Any] = {}
    for method in methods:
        paired = [
            record
            for record in records
            if record.get("status") == "ok"
            and record["method"] == method
            and record["sample_id"] in common
        ]
        region_layouts: dict[str, int] = defaultdict(int)
        for record in paired:
            layout = record["faithfulness"].get("region_layout")
            if (
                isinstance(layout, list)
                and len(layout) == 2
                and all(isinstance(value, int) for value in layout)
            ):
                region_layouts[f"{layout[0]}x{layout[1]}"] += 1
        by_method[method] = {
            "common_samples": len(paired),
            "region_layouts": dict(sorted(region_layouts.items())),
            **{
                metric: statistics.fmean(record["faithfulness"][metric] for record in paired)
                for metric in metric_names
            },
            "mean_seconds": statistics.fmean(record["seconds"] for record in paired),
            "degenerate_deletion_curves": sum(
                bool(record["faithfulness"]["deletion_degenerate"]) for record in paired
            ),
            "degenerate_insertion_curves": sum(
                bool(record["faithfulness"]["insertion_degenerate"]) for record in paired
            ),
            "positive_only_ordering": {
                metric: statistics.fmean(
                    record["faithfulness"]["positive_only_ordering"][metric]
                    for record in paired
                )
                for metric in ("deletion_auc", "insertion_auc", "visual_mas")
            },
            "positive_order_differs": sum(
                not record["faithfulness"]["positive_only_ordering"][
                    "identical_to_signed_order"
                ]
                for record in paired
            ),
        }
    return {
        "comparison_protocol": "common_paired_successful_subset",
        "common_sample_ids": sorted(common),
        "common_samples": len(common),
        "successful_samples_by_method": {
            method: len(successful[method]) for method in methods
        },
        "metric_direction": {
            "deletion_auc": "lower_is_better",
            "insertion_auc": "higher_is_better",
            "visual_rise": "lower_is_better",
            "visual_mas": "lower_is_better",
            "visual_rise_plus_ap": "lower_is_better",
        },
        "methods": by_method,
    }


def _write_summary(
    *,
    output_dir: Path,
    dataset_manifest: Path,
    model_output: Path,
    attribution_dir: Path,
    model_name: str,
    revision: str,
    min_pixels: int,
    max_pixels: int,
    steps: int,
    target_regions: int,
    methods: tuple[str, ...],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "schema_version": 1,
        "dataset_manifest": str(dataset_manifest),
        "model_output": str(model_output),
        "attribution_dir": str(attribution_dir),
        "model": model_name,
        "revision": revision,
        "processor": {
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
        "target_span": "output_only",
        "response_frozen": True,
        "teacher_forced": True,
        "curve_normalization_policy": CURVE_NORMALIZATION_POLICY,
        "steps": steps,
        "target_regions": target_regions,
        **_summary(records, methods),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def run(
    *,
    dataset_manifest: Path,
    model_output: Path,
    attribution_dir: Path,
    output_dir: Path,
    methods: tuple[str, ...] | None,
    model_name: str,
    revision: str | None,
    device: str,
    min_pixels: int,
    max_pixels: int,
    steps: int,
    target_regions: int,
    sample_ids: Sequence[str] | None,
    summary_only: bool = False,
    refresh_derived_metrics: bool = False,
) -> dict[str, Any]:
    datasets = {record["sample_id"]: record for record in read_jsonl(dataset_manifest)}
    models = {record["sample_id"]: record for record in read_jsonl(model_output)}
    attribution_summary = json.loads((attribution_dir / "summary.json").read_text())
    available_methods = tuple(attribution_summary["requested_methods"])
    methods = methods or available_methods
    unknown = set(methods) - set(available_methods)
    if unknown:
        raise ValueError(f"methods are absent from attribution run: {sorted(unknown)}")
    eligible_ids = list(attribution_summary["common_sample_ids"])
    if sample_ids:
        requested = set(sample_ids)
        eligible_ids = [sample_id for sample_id in eligible_ids if sample_id in requested]
        missing = requested - set(eligible_ids)
        if missing:
            raise ValueError(f"sample IDs are absent from paired attribution set: {sorted(missing)}")

    attribution_records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in read_jsonl(attribution_dir / "attribution_records.jsonl"):
        if record.get("status") == "ok":
            attribution_records[record["sample_id"]][record["method"]] = record
    for sample_id in eligible_ids:
        validate_model_record(models[sample_id])
    if revision is None:
        revisions = {
            str(models[sample_id]["model"]["resolved_revision"])
            for sample_id in eligible_ids
        }
        if len(revisions) != 1:
            raise ValueError(f"model outputs contain mixed revisions: {sorted(revisions)}")
        revision = next(iter(revisions))

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "faithfulness_records.jsonl"
    checkpoint = PairJsonlCheckpoint(results_path)
    existing = checkpoint.records()
    completed = {
        (record.get("sample_id"), record.get("method"))
        for record in existing
        if record.get("status") == "ok"
    }
    records = list(existing)
    if summary_only:
        expected_pairs = {
            (sample_id, method)
            for sample_id in eligible_ids
            for method in methods
        }
        if completed != expected_pairs:
            raise ValueError(
                "cannot summarize an incomplete faithfulness matrix: "
                f"complete={len(completed)}, expected={len(expected_pairs)}"
            )
        checkpoint.compact()
        records = checkpoint.records()
        if refresh_derived_metrics:
            records = [refresh_record_metrics(record) for record in records]
            write_jsonl(records, results_path)
        return _write_summary(
            output_dir=output_dir,
            dataset_manifest=dataset_manifest,
            model_output=model_output,
            attribution_dir=attribution_dir,
            model_name=model_name,
            revision=revision,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            steps=steps,
            target_regions=target_regions,
            methods=methods,
            records=records,
        )
    if refresh_derived_metrics:
        raise ValueError("--refresh-derived-metrics requires --summary-only")

    from flashtrace import load_vlm_and_processor

    model, processor = load_vlm_and_processor(
        model_name,
        revision=revision,
        dtype="bfloat16",
        device_map={"": device},
        processor_kwargs={
            "revision": revision,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        },
    )
    for sample_index, sample_id in enumerate(eligible_ids):
        model_record = models[sample_id]
        image = Image.open(model_record["I_IMAGE"]).convert("RGB")
        prompt = model_record_prompt(model_record)
        response = model_record["raw_response"]
        output_span = tuple(int(value) for value in model_record["OUTPUT_SPAN"])
        blurred = image.filter(ImageFilter.GaussianBlur(radius=max(image.size) / 12))
        original_score = output_mean_logprob(
            model, processor, image, prompt, response, output_span
        )
        blurred_score = output_mean_logprob(
            model, processor, blurred, prompt, response, output_span
        )
        for method in methods:
            if (sample_id, method) in completed:
                continue
            started = time.perf_counter()
            try:
                faithfulness = evaluate_grid(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=prompt,
                    response=response,
                    output_span=output_span,
                    grid=attribution_records[sample_id][method]["visual_grid"],
                    steps=steps,
                    target_regions=target_regions,
                    original_score=original_score,
                    blurred_score=blurred_score,
                )
                record = {
                    "schema_version": 1,
                    "status": "ok",
                    "benchmark": datasets[sample_id]["benchmark"],
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "method": method,
                    "target_span": "output_only",
                    "response_frozen": True,
                    "teacher_forced": True,
                    "faithfulness": faithfulness,
                    "seconds": time.perf_counter() - started,
                }
            except Exception as error:
                record = {
                    "schema_version": 1,
                    "status": "error",
                    "benchmark": datasets[sample_id]["benchmark"],
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "method": method,
                    "target_span": "output_only",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "seconds": time.perf_counter() - started,
                }
            checkpoint.put(record)
            records = checkpoint.records()
            print(
                f"[{sample_index + 1}/{len(eligible_ids)}] {sample_id} {method} "
                f"status={record['status']} seconds={record['seconds']:.2f}",
                flush=True,
            )

    checkpoint.compact()
    records = checkpoint.records()
    summary = _write_summary(
        output_dir=output_dir,
        dataset_manifest=dataset_manifest,
        model_output=model_output,
        attribution_dir=attribution_dir,
        model_name=model_name,
        revision=revision,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        steps=steps,
        target_regions=target_regions,
        methods=methods,
        records=records,
    )
    del processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=FROZEN_MODEL_REVISION)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=FORMAL_MAX_PIXELS)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--target-regions", type=int, default=64)
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Compact and summarize an already complete matrix without loading the model.",
    )
    parser.add_argument(
        "--refresh-derived-metrics",
        action="store_true",
        help=(
            "With --summary-only, recompute normalized curves and derived metrics "
            "from saved raw perturbation observations without loading the model."
        ),
    )
    args = parser.parse_args()
    summary = run(
        dataset_manifest=args.dataset_manifest,
        model_output=args.model_output,
        attribution_dir=args.attribution_dir,
        output_dir=args.output_dir,
        methods=tuple(args.methods) if args.methods else None,
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        steps=args.steps,
        target_regions=args.target_regions,
        sample_ids=args.sample_ids,
        summary_only=args.summary_only,
        refresh_derived_metrics=args.refresh_derived_metrics,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
