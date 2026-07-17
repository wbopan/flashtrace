"""Compare multimodal attribution methods on frozen smoke-set responses.

Every method receives the same image, prompt, response, and target-token span.
The runner stores each native visual-token map and a 4x4 version aligned with
the existing visual leave-one-region-out reference.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if __package__:
    from .run_smoke import PROMPT, SYSTEM_PROMPT, save_overlay
else:  # `python /path/to/run_methods.py` against a separate method worktree.
    from run_smoke import PROMPT, SYSTEM_PROMPT, save_overlay


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SUPPORTED_METHODS = (
    "perturbation",
    "ifr-tokenwise",
    "ifr-span",
    "attention-rollout",
    "grad-attention",
    "visual-ig",
    "flashtrace",
    "tam",
    "attnlrp",
)


def exact_multimodal_messages(prompt: str, images: Any) -> list[dict[str, Any]]:
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes, bytearray)):
        image_list = list(images)
    else:
        image_list = [images]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": image} for image in image_list),
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _thumbnail(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def resample_grid(grid: list[list[float]] | np.ndarray, size: int = 4) -> list[list[float]]:
    array = np.asarray(grid, dtype=np.float32)
    if array.ndim != 2 or not array.size:
        raise ValueError(f"Expected a non-empty 2-D grid, got shape {array.shape}")
    tensor = torch.from_numpy(array)[None, None]
    resized = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    return resized.tolist()


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(first: list[float], second: list[float]) -> float:
    x = _rankdata(np.asarray(first, dtype=np.float64))
    y = _rankdata(np.asarray(second, dtype=np.float64))
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def alignment_metrics(
    method_grid: list[list[float]], reference_grid: list[list[float]]
) -> dict[str, float | bool]:
    method = np.asarray(method_grid, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_grid, dtype=np.float64).reshape(-1)
    if method.shape != reference.shape:
        raise ValueError(f"Grid mismatch: {method.shape} vs {reference.shape}")
    top_count = max(1, math.ceil(len(method) / 4))
    method_top = set(np.argsort(method)[-top_count:].tolist())
    reference_top = set(np.argsort(reference)[-top_count:].tolist())
    positive_reference = np.clip(reference, 0.0, None)
    denominator = float(positive_reference.sum())
    recall = (
        float(positive_reference[list(method_top)].sum() / denominator)
        if denominator
        else 0.0
    )
    union = method_top | reference_top
    return {
        "spearman_vs_loo": spearman_correlation(method.tolist(), reference.tolist()),
        "loo_positive_mass_recall_at_25": recall,
        "top25_jaccard_vs_loo": len(method_top & reference_top) / len(union),
        "top_cell_hit_vs_loo": bool(int(method.argmax()) == int(reference.argmax())),
    }


def _visual_grid_from_trace(result: Any) -> list[list[float]]:
    multimodal = result.metadata["multimodal"]
    grids = multimodal["visual_grid_thw"]
    if len(grids) != 1:
        raise ValueError(f"Expected one visual grid, got {grids!r}")
    frames, height, width = (int(value) for value in grids[0])
    if frames != 1:
        raise ValueError(f"Expected a single image frame, got {frames}")
    visual_indices = [int(index) for index in multimodal["visual_token_indices_prompt"]]
    scores = [float(result.scores[index]) for index in visual_indices]
    if len(scores) != height * width:
        raise ValueError(
            f"Visual score count {len(scores)} does not match {height}x{width}"
        )
    return np.asarray(scores, dtype=np.float32).reshape(height, width).tolist()


def _cuda_start() -> tuple[float, int]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        baseline = int(torch.cuda.memory_allocated())
    else:
        baseline = 0
    return time.perf_counter(), baseline


def _cuda_finish(started: float, baseline: int) -> dict[str, float]:
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


def _trace_records(
    rows: list[dict[str, Any]],
    methods: list[str],
    *,
    eval_root: Path,
    model_name: str,
    revision: str,
    max_image_side: int,
    grid_size: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    from flashtrace import FlashTrace, load_vlm_and_processor
    import flashtrace.attribution as attribution

    attribution.multimodal_messages = exact_multimodal_messages
    model, processor = load_vlm_and_processor(
        model_name,
        revision=revision,
        dtype="bfloat16",
        device_map="auto",
        processor_kwargs={"revision": revision},
    )
    tracer = FlashTrace(
        model,
        processor,
        chunk_tokens=64,
        sink_chunk_tokens=8,
        recompute_attention=True,
    )
    records = []
    for sample_index, row in enumerate(rows):
        image = _thumbnail(eval_root / row["image_path"], max_image_side)
        prompt = PROMPT.format(question=row["question"])
        response_token_count = len(
            processor.tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        )
        output_span = (0, response_token_count - 1)
        for method in methods:
            started, baseline = _cuda_start()
            try:
                result = tracer.trace(
                    prompt=prompt,
                    images=image,
                    target=row["response"],
                    output_span=output_span,
                    method=method,
                    hops=1,
                )
                timing = _cuda_finish(started, baseline)
                raw_grid = _visual_grid_from_trace(result)
                coarse_grid = resample_grid(raw_grid, grid_size)
                metrics = alignment_metrics(coarse_grid, row["visual_loo"]["grid"])
                overlay = (
                    output_dir
                    / "overlays"
                    / method
                    / row["dataset"]
                    / f"{row['question_id']}.jpg"
                )
                save_overlay(image, coarse_grid, overlay)
                record = {
                    "status": "ok",
                    "method": method,
                    "dataset": row["dataset"],
                    "question_id": row["question_id"],
                    "sample_index": sample_index,
                    "visual_grid": raw_grid,
                    "coarse_grid": coarse_grid,
                    "visual_grid_shape": [len(raw_grid), len(raw_grid[0])],
                    "overlay_path": str(overlay),
                    **timing,
                    **metrics,
                }
            except Exception as error:
                timing = _cuda_finish(started, baseline)
                record = {
                    "status": "error",
                    "method": method,
                    "dataset": row["dataset"],
                    "question_id": row["question_id"],
                    "sample_index": sample_index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    **timing,
                }
            records.append(record)
            print(
                f"[{method}/{row['dataset']}] {row['question_id']} "
                f"status={record['status']} seconds={record['seconds']:.3f}",
                flush=True,
            )
    del tracer, processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def _whitebox_records(
    rows: list[dict[str, Any]],
    methods: list[str],
    *,
    eval_root: Path,
    model_name: str,
    revision: str,
    max_image_side: int,
    grid_size: int,
    ig_steps: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    from flashtrace import load_vlm_and_processor

    if __package__:
        from .visual_baselines import (
            attention_rollout,
            grad_attention,
            qwen3_vl_attnlrp,
            visual_integrated_gradients,
        )
    else:
        from visual_baselines import (
            attention_rollout,
            grad_attention,
            qwen3_vl_attnlrp,
            visual_integrated_gradients,
        )

    method_functions = {
        "attention-rollout": attention_rollout,
        "grad-attention": grad_attention,
        "visual-ig": visual_integrated_gradients,
        "attnlrp": qwen3_vl_attnlrp,
    }
    model, processor = load_vlm_and_processor(
        model_name,
        revision=revision,
        dtype="bfloat16",
        device_map="auto",
        processor_kwargs={"revision": revision},
    )
    records = []
    for sample_index, row in enumerate(rows):
        image = _thumbnail(eval_root / row["image_path"], max_image_side)
        prompt = PROMPT.format(question=row["question"])
        response_token_count = len(
            processor.tokenizer(row["response"], add_special_tokens=False)["input_ids"]
        )
        output_span = (0, response_token_count - 1)
        messages = exact_multimodal_messages(prompt, image)
        for method in methods:
            started, baseline = _cuda_start()
            try:
                kwargs = {"steps": ig_steps} if method == "visual-ig" else {}
                output = method_functions[method](
                    model,
                    processor,
                    messages,
                    row["response"],
                    output_span=output_span,
                    **kwargs,
                )
                timing = _cuda_finish(started, baseline)
                raw_grid = output.grid
                coarse_grid = resample_grid(raw_grid, grid_size)
                metrics = alignment_metrics(coarse_grid, row["visual_loo"]["grid"])
                overlay = (
                    output_dir
                    / "overlays"
                    / method
                    / row["dataset"]
                    / f"{row['question_id']}.jpg"
                )
                save_overlay(image, coarse_grid, overlay)
                record = {
                    "status": "ok",
                    "method": method,
                    "dataset": row["dataset"],
                    "question_id": row["question_id"],
                    "sample_index": sample_index,
                    "visual_grid": raw_grid,
                    "coarse_grid": coarse_grid,
                    "visual_grid_shape": [len(raw_grid), len(raw_grid[0])],
                    "overlay_path": str(overlay),
                    "method_metadata": output.metadata,
                    **timing,
                    **metrics,
                }
            except Exception as error:
                timing = _cuda_finish(started, baseline)
                record = {
                    "status": "error",
                    "method": method,
                    "dataset": row["dataset"],
                    "question_id": row["question_id"],
                    "sample_index": sample_index,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    **timing,
                }
            records.append(record)
            print(
                f"[{method}/{row['dataset']}] {row['question_id']} "
                f"status={record['status']} seconds={record['seconds']:.3f}",
                flush=True,
            )
    del processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def _tam_inputs(
    model: Any, processor: Any, image: Image.Image, prompt: str
) -> dict[str, Any]:
    inputs = processor.apply_chat_template(
        exact_multimodal_messages(prompt, image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(inputs.to(model.device) if hasattr(inputs, "to") else inputs)


@torch.inference_mode()
def _tam_grid(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    expected_response: str,
    tam_function: Any,
) -> tuple[list[list[float]], dict[str, Any]]:
    inputs = _tam_inputs(model, processor, image, prompt)
    prompt_length = int(inputs["input_ids"].shape[1])
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    generated_ids = output.sequences[0, prompt_length:]
    response = processor.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    if response != expected_response:
        raise RuntimeError(
            "TAM greedy generation did not reproduce the frozen response: "
            f"expected={expected_response!r}, actual={response!r}"
        )

    logits = [model.lm_head(step[-1]) for step in output.hidden_states]
    input_ids = inputs["input_ids"][0].tolist()
    all_ids = output.sequences[0].tolist()
    image_token_id = int(model.config.image_token_id)
    visual_indices = [
        index for index, token_id in enumerate(input_ids) if token_id == image_token_id
    ]
    if not visual_indices:
        raise RuntimeError("No Qwen3-VL image tokens found for TAM")
    merge = int(model.config.vision_config.spatial_merge_size)
    grid = inputs["image_grid_thw"][0].detach().cpu().tolist()
    vision_shape = (int(grid[1]) // merge, int(grid[2]) // merge)
    if len(visual_indices) != vision_shape[0] * vision_shape[1]:
        raise RuntimeError(
            f"Visual token count {len(visual_indices)} != TAM grid {vision_shape}"
        )

    # The official TAM API locates regions through token-ID delimiters. Using
    # the actual image token and final assistant-header suffix avoids hard-coded
    # Qwen2-VL IDs while preserving the official TAM computation.
    assistant_suffix = input_ids[-4:]
    special_ids = {
        "img_id": [image_token_id],
        "prompt_id": [[image_token_id], assistant_suffix],
        "answer_id": [assistant_suffix, -1],
    }
    maps = []
    raw_map_records: list[np.ndarray] = []
    eos_id = processor.tokenizer.eos_token_id
    map_count = len(logits)
    if generated_ids.numel() and eos_id is not None and int(generated_ids[-1]) == int(eos_id):
        map_count -= 1
    for target_token in range(map_count):
        activation_map = tam_function(
            all_ids,
            vision_shape,
            logits,
            special_ids,
            [image],
            processor,
            "",
            target_token,
            raw_map_records,
            True,
        )
        maps.append(np.asarray(activation_map, dtype=np.float32))
    if not maps:
        raise RuntimeError("TAM returned no output-token maps")
    mean_map = np.stack(maps).mean(axis=0)
    if not np.isfinite(mean_map).all():
        raise RuntimeError("TAM returned non-finite activation values")
    return mean_map.tolist(), {
        "generated_tokens": int(generated_ids.numel()),
        "attributed_tokens": map_count,
        "visual_grid_shape": list(vision_shape),
        "tam_raw_map_records": len(raw_map_records),
    }


def _tam_records(
    rows: list[dict[str, Any]],
    *,
    eval_root: Path,
    tam_source: Path,
    model_name: str,
    revision: str,
    max_image_side: int,
    grid_size: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    if not (tam_source / "tam.py").is_file():
        raise FileNotFoundError(
            f"Missing {tam_source / 'tam.py'}; clone https://github.com/xmed-lab/TAM"
        )
    sys.path.insert(0, str(tam_source))
    try:
        from tam import TAM
    finally:
        sys.path.pop(0)

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, revision=revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    records = []
    for sample_index, row in enumerate(rows):
        image = _thumbnail(eval_root / row["image_path"], max_image_side)
        prompt = PROMPT.format(question=row["question"])
        started, baseline = _cuda_start()
        try:
            raw_grid, metadata = _tam_grid(
                model,
                processor,
                image,
                prompt,
                row["response"],
                TAM,
            )
            timing = _cuda_finish(started, baseline)
            coarse_grid = resample_grid(raw_grid, grid_size)
            metrics = alignment_metrics(coarse_grid, row["visual_loo"]["grid"])
            overlay = (
                output_dir
                / "overlays"
                / "tam"
                / row["dataset"]
                / f"{row['question_id']}.jpg"
            )
            save_overlay(image, coarse_grid, overlay)
            record = {
                "status": "ok",
                "method": "tam",
                "dataset": row["dataset"],
                "question_id": row["question_id"],
                "sample_index": sample_index,
                "visual_grid": raw_grid,
                "coarse_grid": coarse_grid,
                "overlay_path": str(overlay),
                **metadata,
                **timing,
                **metrics,
            }
        except Exception as error:
            timing = _cuda_finish(started, baseline)
            record = {
                "status": "error",
                "method": "tam",
                "dataset": row["dataset"],
                "question_id": row["question_id"],
                "sample_index": sample_index,
                "error_type": type(error).__name__,
                "error": str(error),
                **timing,
            }
        records.append(record)
        print(
            f"[tam/{row['dataset']}] {row['question_id']} "
            f"status={record['status']} seconds={record['seconds']:.3f}",
            flush=True,
        )
    del processor, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


def _perturbation_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for sample_index, row in enumerate(rows):
        reference = row["visual_loo"]["grid"]
        records.append(
            {
                "status": "reference",
                "method": "perturbation",
                "dataset": row["dataset"],
                "question_id": row["question_id"],
                "sample_index": sample_index,
                "coarse_grid": reference,
                "seconds": float(row["attribution_seconds"]),
                "peak_vram_gb": None,
                "incremental_peak_vram_gb": None,
                "spearman_vs_loo": 1.0,
                "loo_positive_mass_recall_at_25": 1.0,
                "top25_jaccard_vs_loo": 1.0,
                "top_cell_hit_vs_loo": True,
            }
        )
    return records


def _method_source_state(path: Path) -> dict[str, Any]:
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=path, text=True
            ).strip()
        )
        return {"path": str(path), "head": head, "dirty": dirty}
    except Exception as error:
        return {"path": str(path), "error": str(error)}


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    methods = sorted({record["method"] for record in records})
    for method in methods:
        summary[method] = {}
        datasets = sorted(
            {record["dataset"] for record in records if record["method"] == method}
        )
        for dataset in datasets:
            subset = [
                record
                for record in records
                if record["method"] == method
                and record["dataset"] == dataset
                and record["status"] in {"ok", "reference"}
            ]
            failures = [
                record
                for record in records
                if record["method"] == method
                and record["dataset"] == dataset
                and record["status"] == "error"
            ]
            item: dict[str, Any] = {
                "successful_samples": len(subset),
                "failed_samples": len(failures),
            }
            if subset:
                for key in (
                    "seconds",
                    "spearman_vs_loo",
                    "loo_positive_mass_recall_at_25",
                    "top25_jaccard_vs_loo",
                ):
                    values = [float(record[key]) for record in subset]
                    item[f"mean_{key}"] = statistics.fmean(values)
                    item[f"median_{key}"] = statistics.median(values)
                item["top_cell_hits"] = sum(
                    bool(record["top_cell_hit_vs_loo"]) for record in subset
                )
                peaks = [
                    float(record["peak_vram_gb"])
                    for record in subset
                    if record.get("peak_vram_gb") is not None
                ]
                if peaks:
                    item["max_peak_vram_gb"] = max(peaks)
            summary[method][dataset] = item
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("data/multimodal_smoke_final/results.jsonl"),
    )
    parser.add_argument("--eval-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/multimodal_methods_final")
    )
    parser.add_argument(
        "--methods",
        default=(
            "perturbation,ifr-tokenwise,attention-rollout,grad-attention,"
            "visual-ig,flashtrace,tam,attnlrp"
        ),
        help=f"Comma-separated subset of {SUPPORTED_METHODS}",
    )
    parser.add_argument("--tam-source", type=Path, default=Path("data/external/TAM"))
    parser.add_argument("--method-source", type=Path, default=Path.cwd())
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument("--ig-steps", type=int, default=20)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    rows = [
        json.loads(line)
        for line in args.reference_results.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        rows = rows[: max(0, args.limit)]
    if not rows:
        raise SystemExit("No reference samples selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    if "perturbation" in methods:
        records.extend(_perturbation_records(rows))
    trace_methods = [
        method
        for method in methods
        if method in {"ifr-tokenwise", "ifr-span", "flashtrace"}
    ]
    if trace_methods:
        records.extend(
            _trace_records(
                rows,
                trace_methods,
                eval_root=args.eval_root,
                model_name=args.model,
                revision=args.revision,
                max_image_side=args.max_image_side,
                grid_size=args.grid_size,
                output_dir=args.output_dir,
            )
        )
    whitebox_methods = [
        method
        for method in methods
        if method
        in {"attention-rollout", "grad-attention", "visual-ig", "attnlrp"}
    ]
    if whitebox_methods:
        records.extend(
            _whitebox_records(
                rows,
                whitebox_methods,
                eval_root=args.eval_root,
                model_name=args.model,
                revision=args.revision,
                max_image_side=args.max_image_side,
                grid_size=args.grid_size,
                ig_steps=args.ig_steps,
                output_dir=args.output_dir,
            )
        )
    if "tam" in methods:
        records.extend(
            _tam_records(
                rows,
                eval_root=args.eval_root,
                tam_source=args.tam_source,
                model_name=args.model,
                revision=args.revision,
                max_image_side=args.max_image_side,
                grid_size=args.grid_size,
                output_dir=args.output_dir,
            )
        )

    results_path = args.output_dir / "results.jsonl"
    results_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    summary = {
        "model": args.model,
        "revision": args.revision,
        "samples": len(rows),
        "methods_requested": methods,
        "method_source": _method_source_state(args.method_source),
        "tam_source": _method_source_state(args.tam_source),
        "unavailable_methods": {},
        "metrics_note": (
            "Method maps are bilinearly resampled to the same 4x4 grid as the "
            "visual LOO reference. Recall@25 has a random-selection expectation of 0.25."
        ),
        "methods": _summary(records),
        "results": str(results_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
