"""Run strict OUTPUT-only visual attribution on frozen Thinking responses.

The primary FlashTrace method uses ``OUTPUT_SPAN`` as the sink and exactly
``THINKING_SPAN`` as its single recursive bridge. Its final map follows the
paper: direct input attribution is added to recursively propagated input
attribution, scaled by the reasoning-mass ratio. The current public facade's
all-generation bridge is retained under the explicit ``flashtrace-all-gen``
ablation name. Every method receives the same frozen raw response and
output-token span. Localization is reported only on the common paired subset.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .metrics import (
    patch_energy_in_mask,
    patch_evidence_rank_auc,
    patch_pointing_game,
    patch_recovery_at_fraction,
    patch_top_evidence_iou,
    xyxy_boxes_to_mask,
)
from .strict_generation import (
    DEFAULT_MODEL,
    _messages,
    model_record_prompt,
    output_mean_logprob,
    read_jsonl,
    validate_model_record,
    write_jsonl,
)


DEFAULT_METHODS = (
    "random",
    "center",
    "visual-loo",
    "ifr-span",
    "attention-rollout",
    "grad-attention",
    "visual-ig",
    "attnlrp",
    "tam",
    "flashtrace",
    "flashtrace-all-gen",
)
TRACE_METHODS = frozenset(
    {"ifr-span", "ifr-tokenwise", "flashtrace", "flashtrace-all-gen"}
)
WHITEBOX_METHODS = frozenset(
    {"attention-rollout", "grad-attention", "visual-ig", "attnlrp"}
)
NO_MODEL_METHODS = frozenset({"random", "center"})
SUPPORTED_METHODS = NO_MODEL_METHODS | TRACE_METHODS | WHITEBOX_METHODS | {
    "visual-loo",
    "tam",
}


def _resample(grid: Any, shape: tuple[int, int]) -> np.ndarray:
    """Expand a visual-patch grid without inventing sub-patch gradients."""

    array = np.asarray(grid, dtype=np.float32)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"attribution grid must be non-empty 2-D, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("attribution grid contains non-finite values")
    tensor = torch.from_numpy(array)[None, None]
    return (
        F.interpolate(
            tensor,
            size=shape,
            mode="nearest",
        )[0, 0]
        .numpy()
        .astype(np.float64)
    )


def _evidence_masks(dataset_record: Mapping[str, Any]) -> dict[str, np.ndarray]:
    evaluation = dataset_record["evaluation"]
    mask_paths = evaluation.get("EVIDENCE_MASKS")
    if mask_paths:
        masks = {}
        for name, path in mask_paths.items():
            if path:
                mask = np.asarray(np.load(path), dtype=bool)
                if mask.ndim == 2 and np.any(mask):
                    masks[str(name)] = mask
        if not any(
            name in masks
            for name in ("primary_unique_firstnonempty", "primary", "primary_union")
        ):
            raise ValueError("sample has no non-empty primary evidence mask")
        return masks

    boxes = evaluation.get("EVIDENCE_BOXES")
    metadata = evaluation.get("metadata") or {}
    image_size = metadata.get("image_size") or {}
    if not boxes or not image_size:
        raise ValueError("sample has neither evidence mask nor evidence boxes")
    native_width = int(image_size["width"])
    native_height = int(image_size["height"])
    metric_height = 256
    metric_width = max(1, round(metric_height * native_width / native_height))
    normalized_boxes = [
        [
            float(box[0]) / native_width,
            float(box[1]) / native_height,
            float(box[2]) / native_width,
            float(box[3]) / native_height,
        ]
        for box in boxes
    ]
    return {
        "primary_bbox": xyxy_boxes_to_mask(
            normalized_boxes,
            height=metric_height,
            width=metric_width,
            normalized=True,
        )
    }


def _metric_bundle(grid: Any, mask: np.ndarray) -> dict[str, float]:
    values = {
        "pointing_game": patch_pointing_game(grid, mask),
        "energy_in_mask": patch_energy_in_mask(grid, mask),
        "evidence_rank_auc": patch_evidence_rank_auc(grid, mask),
        "top_evidence_iou": patch_top_evidence_iou(grid, mask),
    }
    for percentage in (1, 5, 10, 20):
        values[f"recovery_at_{percentage}pct"] = patch_recovery_at_fraction(
            grid,
            mask,
            fraction=percentage / 100.0,
        )
    values["evidence_area_fraction"] = float(mask.mean())
    return values


def localization_metrics(
    grid: Any,
    dataset_record: Mapping[str, Any],
) -> dict[str, float]:
    masks = _evidence_masks(dataset_record)
    primary_name = (
        "primary_unique_firstnonempty"
        if "primary_unique_firstnonempty" in masks
        else "primary"
        if "primary" in masks
        else "primary_union"
        if "primary_union" in masks
        else "primary_bbox"
    )
    metrics = _metric_bundle(grid, masks[primary_name])
    for name in ("sensitivity_unique", "sensitivity_union"):
        if name not in masks:
            continue
        for metric, value in _metric_bundle(grid, masks[name]).items():
            metrics[f"{name}.{metric}"] = value
    return metrics


def _visual_grid_from_projected_scores(
    scores: Sequence[Any], multimodal: Mapping[str, Any]
) -> list[list[float]]:
    grids = multimodal["visual_grid_thw"]
    if len(grids) != 1:
        raise ValueError(f"expected one visual grid, got {grids!r}")
    frames, height, width = (int(value) for value in grids[0])
    if frames != 1:
        raise ValueError(f"expected a still image, got {frames} frames")
    indices = [int(index) for index in multimodal["visual_token_indices_prompt"]]
    visual_scores = [float(scores[index]) for index in indices]
    if len(visual_scores) != height * width:
        raise ValueError(
            f"visual score count {len(visual_scores)} does not match "
            f"{height}x{width}"
        )
    return (
        np.asarray(visual_scores, dtype=np.float32)
        .reshape(height, width)
        .tolist()
    )


def _visual_grid_from_trace(result: Any) -> list[list[float]]:
    return _visual_grid_from_projected_scores(
        result.scores, result.metadata["multimodal"]
    )


def _seed(sample_id: str, method: str) -> int:
    digest = hashlib.sha256(f"{sample_id}:{method}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _random_grid(sample_id: str, size: int = 32) -> list[list[float]]:
    rng = np.random.default_rng(_seed(sample_id, "random"))
    return rng.random((size, size), dtype=np.float32).tolist()


def _center_grid(size: int = 32) -> list[list[float]]:
    coordinates = np.linspace(-1.0, 1.0, num=size, dtype=np.float32)
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    return np.exp(-0.5 * (x * x + y * y) / (0.35**2)).tolist()


def _perturb_cell(
    image: Image.Image,
    *,
    row: int,
    column: int,
    grid_size: int,
) -> Image.Image:
    width, height = image.size
    left, right = column * width // grid_size, (column + 1) * width // grid_size
    top, bottom = row * height // grid_size, (row + 1) * height // grid_size
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(image.size) / 12))
    output = image.copy()
    output.paste(blurred.crop((left, top, right, bottom)), (left, top))
    return output


def _visual_loo_grid(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    response: str,
    output_span: tuple[int, int],
    *,
    grid_size: int,
) -> list[list[float]]:
    base = output_mean_logprob(
        model, processor, image, prompt, response, output_span
    )
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for row in range(grid_size):
        for column in range(grid_size):
            perturbed = _perturb_cell(
                image,
                row=row,
                column=column,
                grid_size=grid_size,
            )
            perturbed_score = output_mean_logprob(
                model,
                processor,
                perturbed,
                prompt,
                response,
                output_span,
            )
            grid[row, column] = base - perturbed_score
    return grid.tolist()


@torch.inference_mode()
def _tam_frozen_grid(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    response: str,
    output_span: tuple[int, int],
    *,
    tam_source: Path,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Run official TAM while teacher-forcing the complete frozen response."""

    if not (tam_source / "tam.py").is_file():
        raise FileNotFoundError(f"missing official TAM source: {tam_source / 'tam.py'}")
    sys.path.insert(0, str(tam_source))
    try:
        from tam import TAM
    finally:
        sys.path.pop(0)

    batch = processor.apply_chat_template(
        _messages(prompt, image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(batch, "to"):
        batch = batch.to(model.device)
    inputs = dict(batch)
    prompt_length = int(inputs["input_ids"].shape[1])
    target_ids = processor.tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    full_ids = torch.cat((inputs["input_ids"], target_ids), dim=1)
    full_mask = torch.cat(
        (
            inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            torch.ones_like(target_ids),
        ),
        dim=1,
    )
    forward_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "position_ids", "cache_position"}
    }
    token_types = forward_inputs.get("mm_token_type_ids")
    if torch.is_tensor(token_types) and token_types.shape[-1] != full_ids.shape[-1]:
        forward_inputs["mm_token_type_ids"] = F.pad(
            token_types,
            (0, full_ids.shape[-1] - token_types.shape[-1]),
            value=0,
        )
    output = model(
        input_ids=full_ids,
        attention_mask=full_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
        **forward_inputs,
    )
    hidden = output.hidden_states[-1]
    # Match generate(..., output_hidden_states=True): the first item covers the
    # full prompt and each later item is the next single frozen target token.
    logit_list = [model.lm_head(hidden[:, :prompt_length])]
    for target_index in range(1, int(target_ids.shape[1])):
        position = prompt_length + target_index - 1
        logit_list.append(model.lm_head(hidden[:, position : position + 1]))

    input_ids = inputs["input_ids"][0].tolist()
    all_ids = full_ids[0].tolist()
    image_token_id = int(model.config.image_token_id)
    visual_indices = [
        index for index, token_id in enumerate(input_ids) if token_id == image_token_id
    ]
    if not visual_indices:
        raise RuntimeError("no Qwen3-VL image tokens found for TAM")
    merge = int(model.config.vision_config.spatial_merge_size)
    raw_grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
    vision_shape = (int(raw_grid[1]) // merge, int(raw_grid[2]) // merge)
    if len(visual_indices) != vision_shape[0] * vision_shape[1]:
        raise RuntimeError(
            f"visual token count {len(visual_indices)} != TAM grid {vision_shape}"
        )
    assistant_suffix = input_ids[-4:]
    special_ids = {
        "img_id": [image_token_id],
        "prompt_id": [[image_token_id], assistant_suffix],
        "answer_id": [assistant_suffix, -1],
    }
    raw_map_records: list[np.ndarray] = []
    maps = []
    # Official TAM's ECI stage indexes maps for every preceding prompt and
    # generated token. Build that fixed-response prefix state, but retain and
    # aggregate only OUTPUT_SPAN maps for the reported attribution.
    for target_token in range(0, output_span[1] + 1):
        activation_map = TAM(
            all_ids,
            vision_shape,
            logit_list,
            special_ids,
            [image],
            processor,
            "",
            target_token,
            raw_map_records,
            True,
        )
        if target_token >= output_span[0]:
            maps.append(np.asarray(activation_map, dtype=np.float32))
    if not maps:
        raise RuntimeError("TAM returned no OUTPUT-token maps")
    mean_map = np.stack(maps).mean(axis=0)
    if mean_map.shape != vision_shape:
        mean_map = np.asarray(mean_map).reshape(vision_shape)
    if not np.isfinite(mean_map).all():
        raise RuntimeError("TAM returned non-finite activation values")
    return mean_map.tolist(), {
        "target_span": "output_only",
        "response_frozen": True,
        "teacher_forced": True,
        "attributed_tokens": output_span[1] - output_span[0] + 1,
        "internal_prefix_tokens": output_span[0],
        "visual_grid_shape": list(vision_shape),
        "tam_raw_map_records": len(raw_map_records),
        "tam_source": str(tam_source),
    }


def _save_overlay(
    image: Image.Image,
    grid: Any,
    mask: np.ndarray | None,
    destination: Path,
) -> None:
    image = image.copy()
    image.thumbnail((700, 1400))
    heat = _resample(grid, (image.height, image.width))
    positive = np.clip(heat, 0.0, None)
    maximum = float(positive.max())
    if maximum > 0:
        positive /= maximum
    alpha = Image.fromarray(np.uint8(positive * 175), mode="L")
    red = Image.new("RGBA", image.size, (255, 20, 20, 0))
    red.putalpha(alpha)
    canvas = Image.alpha_composite(image.convert("RGBA"), red)

    if mask is not None:
        mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L").resize(
            image.size, Image.Resampling.NEAREST
        )
        dilated = mask_image.filter(ImageFilter.MaxFilter(7))
        eroded = mask_image.filter(ImageFilter.MinFilter(7))
        outline = ImageChops.subtract(dilated, eroded)
        draw_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw_layer.paste((20, 255, 40, 255), mask=outline)
        canvas = Image.alpha_composite(canvas, draw_layer)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, image.width - 1, image.height - 1), outline="white")
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(destination, quality=92)


def _timing_start() -> tuple[float, int]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = int(torch.cuda.memory_allocated())
    else:
        baseline = 0
    return time.perf_counter(), baseline


def _timing_finish(started: float, baseline: int) -> dict[str, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak = int(torch.cuda.max_memory_allocated())
    else:
        peak = 0
    return {
        "seconds": time.perf_counter() - started,
        "peak_vram_gb": peak / (1024**3),
        "incremental_peak_vram_gb": max(0, peak - baseline) / (1024**3),
    }


def _method_grid(
    method: str,
    *,
    dataset_record: Mapping[str, Any],
    model_record: Mapping[str, Any],
    image: Image.Image,
    model: Any,
    processor: Any,
    tracer: Any,
    ig_steps: int,
    loo_grid_size: int,
    tam_source: Path,
) -> tuple[list[list[float]], dict[str, Any]]:
    prompt = model_record_prompt(model_record)
    response = model_record["raw_response"]
    output_span = tuple(int(value) for value in model_record["OUTPUT_SPAN"])
    thinking_span = tuple(int(value) for value in model_record["THINKING_SPAN"])

    if method == "random":
        return _random_grid(str(model_record["sample_id"])), {"seeded": True}
    if method == "center":
        return _center_grid(), {"sigma": 0.35}
    if method == "visual-loo":
        return (
            _visual_loo_grid(
                model,
                processor,
                image,
                prompt,
                response,
                output_span,
                grid_size=loo_grid_size,
            ),
            {
                "grid_size": loo_grid_size,
                "target_span": "output_only",
                "response_frozen": True,
            },
        )
    if method == "tam":
        return _tam_frozen_grid(
            model,
            processor,
            image,
            prompt,
            response,
            output_span,
            tam_source=tam_source,
        )
    if method == "flashtrace":
        from flashtrace.attribution import LLMIFRAttribution
        from flashtrace.result import _jsonable

        engine = LLMIFRAttribution(
            model,
            processor.tokenizer,
            chunk_tokens=tracer.chunk_tokens,
            sink_chunk_tokens=tracer.sink_chunk_tokens,
            recompute_attention=tracer.recompute_attention,
            processor=processor,
            images=image,
        )
        raw = engine.calculate_ifr_multi_hop(
            prompt,
            target=response,
            sink_span=output_span,
            thinking_span=thinking_span,
            n_hops=1,
        )
        result = tracer._build_result(
            raw,
            method=method,
            output_span=output_span,
            reasoning_span=thinking_span,
        )
        # Preserve the FlashTrace definition from the paper: direct input
        # attribution from hop 0 plus every recursively propagated reasoning
        # hop, already scaled by its cumulative reasoning-mass ratio.  The VLM
        # extension changes only which prompt positions are visual sources and
        # how those scores are reshaped into a patch grid.
        return _visual_grid_from_trace(result), {
            "target_span": "output_only",
            "bridge_span": "thinking",
            "recursive_hops": 1,
            "attribution_composition": "direct_plus_weighted_reasoning_hops",
            "direct_base_included": True,
            "trace_metadata": _jsonable(result.metadata),
        }
    if method in TRACE_METHODS:
        from flashtrace.result import _jsonable

        trace_method = "flashtrace" if method == "flashtrace-all-gen" else method
        result = tracer.trace(
            prompt=prompt,
            images=image,
            target=response,
            output_span=output_span,
            reasoning_span=thinking_span,
            method=trace_method,
            hops=1,
        )
        return _visual_grid_from_trace(result), {
            "target_span": "output_only",
            "bridge_span": (
                "thinking_plus_output" if method == "flashtrace-all-gen" else None
            ),
            "recursive_hops": 1 if method == "flashtrace-all-gen" else 0,
            **(
                {
                    "attribution_composition": (
                        "direct_plus_weighted_reasoning_hops"
                    ),
                    "direct_base_included": True,
                }
                if method == "flashtrace-all-gen"
                else {}
            ),
            "trace_metadata": _jsonable(result.metadata),
        }

    from .visual_baselines import (
        attention_rollout,
        grad_attention,
        qwen3_vl_attnlrp,
        visual_integrated_gradients,
    )

    functions = {
        "attention-rollout": attention_rollout,
        "grad-attention": grad_attention,
        "visual-ig": visual_integrated_gradients,
        "attnlrp": qwen3_vl_attnlrp,
    }
    kwargs = {"steps": ig_steps} if method == "visual-ig" else {}
    output = functions[method](
        model,
        processor,
        _messages(prompt, image),
        response,
        output_span=output_span,
        **kwargs,
    )
    from flashtrace.result import _jsonable

    return output.grid, {
        "target_span": "output_only",
        **_jsonable(output.metadata),
    }


def _common_summary(
    records: list[dict[str, Any]],
    methods: tuple[str, ...],
) -> dict[str, Any]:
    successful: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.get("status") == "ok":
            successful[str(record["method"])].add(str(record["sample_id"]))
    common = (
        set.intersection(*(successful[method] for method in methods))
        if methods
        else set()
    )
    metric_names = (
        "pointing_game",
        "energy_in_mask",
        "evidence_rank_auc",
        "top_evidence_iou",
        "recovery_at_1pct",
        "recovery_at_5pct",
        "recovery_at_10pct",
        "recovery_at_20pct",
    )
    method_summary = {}
    for method in methods:
        paired = [
            record
            for record in records
            if record.get("status") == "ok"
            and record["method"] == method
            and record["sample_id"] in common
        ]
        localized = [
            record
            for record in paired
            if isinstance(record.get("localization"), Mapping)
        ]
        method_summary[method] = {
            "common_samples": len(paired),
            "localization_samples": len(localized),
            **{
                metric: (
                    statistics.fmean(
                        float(record["localization"][metric]) for record in localized
                    )
                    if localized
                    else None
                )
                for metric in metric_names
            },
            "mean_seconds": (
                statistics.fmean(float(record["seconds"]) for record in paired)
                if paired
                else None
            ),
            "mean_peak_vram_gb": (
                statistics.fmean(
                    float(record["incremental_peak_vram_gb"]) for record in paired
                )
                if paired
                else None
            ),
        }
    return {
        "comparison_protocol": "common_paired_successful_subset",
        "spatial_resampling": "nearest_patch",
        "spatial_metric_unit": "visual_patch",
        "cutoff_tie_policy": "expected_uniform",
        "common_sample_ids": sorted(common),
        "common_samples": len(common),
        "requested_methods": list(methods),
        "successful_samples_by_method": {
            method: len(successful[method]) for method in methods
        },
        "methods": method_summary,
    }


def run(
    *,
    dataset_manifest: Path,
    model_output: Path,
    generation_evaluation: Path,
    output_dir: Path,
    methods: tuple[str, ...],
    model_name: str,
    revision: str | None,
    device: str,
    min_pixels: int,
    max_pixels: int,
    ig_steps: int,
    loo_grid_size: int,
    tam_source: Path,
    include_ineligible: bool,
    allow_missing_evidence: bool,
    sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    unknown = set(methods).difference(SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"unsupported methods: {sorted(unknown)}")
    datasets = {record["sample_id"]: record for record in read_jsonl(dataset_manifest)}
    model_records = {record["sample_id"]: record for record in read_jsonl(model_output)}
    evaluations = {
        record["sample_id"]: record for record in read_jsonl(generation_evaluation)
    }
    selected_ids = [
        sample_id
        for sample_id in datasets
        if sample_id in model_records
        and sample_id in evaluations
        and (
            include_ineligible
            or bool(evaluations[sample_id].get("strict_eligible"))
        )
    ]
    if sample_ids:
        selected_ids = [
            sample_id for sample_id in selected_ids if sample_id in sample_ids
        ]
        missing = sample_ids.difference(selected_ids)
        if missing:
            raise ValueError(
                f"requested sample IDs are absent or ineligible: {sorted(missing)}"
            )
    if not selected_ids:
        raise ValueError("no generated samples satisfy the eligibility policy")
    for sample_id in selected_ids:
        validate_model_record(model_records[sample_id])

    if revision is None:
        revisions = {
            str(model_records[sample_id]["model"]["resolved_revision"])
            for sample_id in selected_ids
        }
        if len(revisions) != 1:
            raise ValueError(f"model records have mixed revisions: {sorted(revisions)}")
        revision = next(iter(revisions))
        if revision == "unknown":
            raise ValueError("model revision is unknown; pass --revision explicitly")

    needs_model = any(method not in NO_MODEL_METHODS for method in methods)
    model = processor = tracer = None
    if needs_model:
        import flashtrace.attribution as attribution
        from flashtrace import FlashTrace, load_vlm_and_processor

        attribution.multimodal_messages = _messages
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
        tracer = FlashTrace(
            model,
            processor,
            chunk_tokens=64,
            sink_chunk_tokens=8,
            recompute_attention=False,
        )

    results_path = output_dir / "attribution_records.jsonl"
    existing = read_jsonl(results_path) if results_path.exists() else []
    completed = {
        (str(record.get("sample_id")), str(record.get("method")))
        for record in existing
        if record.get("status") == "ok"
        and record.get("spatial_resampling") == "nearest_patch"
        and record.get("spatial_metric_unit") == "visual_patch"
    }
    records = list(existing)
    for sample_index, sample_id in enumerate(selected_ids):
        dataset_record = datasets[sample_id]
        model_record = model_records[sample_id]
        image = Image.open(model_record["I_IMAGE"]).convert("RGB")
        try:
            masks = _evidence_masks(dataset_record)
        except ValueError:
            if not allow_missing_evidence:
                raise
            masks = {}
        primary_mask = next(
            (
                masks[name]
                for name in (
                    "primary_unique_firstnonempty",
                    "primary",
                    "primary_union",
                    "primary_bbox",
                )
                if name in masks
            ),
            None,
        )
        for method in methods:
            if (sample_id, method) in completed:
                continue
            started, baseline = _timing_start()
            try:
                grid, metadata = _method_grid(
                    method,
                    dataset_record=dataset_record,
                    model_record=model_record,
                    image=image,
                    model=model,
                    processor=processor,
                    tracer=tracer,
                    ig_steps=ig_steps,
                    loo_grid_size=loo_grid_size,
                    tam_source=tam_source,
                )
                localization = (
                    localization_metrics(grid, dataset_record)
                    if primary_mask is not None
                    else None
                )
                timing = _timing_finish(started, baseline)
                overlay = (
                    output_dir
                    / "overlays"
                    / method
                    / str(dataset_record["benchmark"])
                    / f"{sample_id}.jpg"
                )
                _save_overlay(image, grid, primary_mask, overlay)
                record = {
                    "schema_version": 2,
                    "status": "ok",
                    "benchmark": dataset_record["benchmark"],
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "method": method,
                    "spatial_resampling": "nearest_patch",
                    "spatial_metric_unit": "visual_patch",
                    "cutoff_tie_policy": "expected_uniform",
                    "target_span": "output_only",
                    "bridge_span": (
                        "thinking"
                        if method == "flashtrace"
                        else "thinking_plus_output"
                        if method == "flashtrace-all-gen"
                        else None
                    ),
                    "visual_grid": grid,
                    "visual_grid_shape": [len(grid), len(grid[0])],
                    "localization": localization,
                    "overlay_path": str(overlay),
                    "method_metadata": metadata,
                    **timing,
                }
            except Exception as error:
                timing = _timing_finish(started, baseline)
                record = {
                    "schema_version": 2,
                    "status": "error",
                    "benchmark": dataset_record["benchmark"],
                    "sample_id": sample_id,
                    "sample_index": sample_index,
                    "method": method,
                    "spatial_resampling": "nearest_patch",
                    "spatial_metric_unit": "visual_patch",
                    "cutoff_tie_policy": "expected_uniform",
                    "target_span": "output_only",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    **timing,
                }
            # Replace stale protocol versions for this paired sample/method.
            # Keeping both would silently double count the aggregate.
            records = [
                item
                for item in records
                if (
                    str(item.get("sample_id")),
                    str(item.get("method")),
                )
                != (sample_id, method)
            ]
            records.append(record)
            write_jsonl(records, results_path)
            print(
                f"[{sample_index + 1}/{len(selected_ids)}] {sample_id} "
                f"{method} status={record['status']} seconds={record['seconds']:.2f}",
                flush=True,
            )
            # Methods such as decoder Grad×Attention and checkpointed visual IG
            # temporarily reserve tens of GB at document resolution.  Reclaim
            # method-local tensors before the next baseline so a resumable mixed
            # method run is not sensitive to allocator history.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = {
        "schema_version": 2,
        "dataset_manifest": str(dataset_manifest),
        "model_output": str(model_output),
        "generation_evaluation": str(generation_evaluation),
        "model": model_name,
        "revision": revision,
        "eligible_samples": len(selected_ids),
        **_common_summary(records, methods),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    del tracer, processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--generation-evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--ig-steps", type=int, default=8)
    parser.add_argument("--loo-grid-size", type=int, default=4)
    parser.add_argument("--tam-source", type=Path, default=Path("data/external/TAM"))
    parser.add_argument("--include-ineligible", action="store_true")
    parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help=(
            "Compute and save attribution maps without localization metrics when "
            "the dataset has no native evidence boxes or masks."
        ),
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        help="Run only the named eligible sample ID; may be repeated.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run(
        dataset_manifest=args.dataset_manifest,
        model_output=args.model_output,
        generation_evaluation=args.generation_evaluation,
        output_dir=args.output_dir,
        methods=tuple(args.methods),
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        ig_steps=args.ig_steps,
        loo_grid_size=args.loo_grid_size,
        tam_source=args.tam_source,
        include_ineligible=args.include_ineligible,
        allow_missing_evidence=args.allow_missing_evidence,
        sample_ids=set(args.sample_ids) if args.sample_ids else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
