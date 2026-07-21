"""Confirm visual dependence by deterministic generation on ablated images.

Primary model records are never modified. Ablation generations are written to
a separate model-output file, while a revised evaluation file contains only
comparisons and eligibility metadata.
"""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFilter

from .strict_generation import (
    DEFAULT_MODEL,
    generate_response,
    model_record_prompt,
    normalized_output,
    read_jsonl,
    split_thinking_output,
    write_jsonl,
)


def _token_identity_stable(model_record: dict[str, Any]) -> bool:
    metadata = model_record.get("generation_metadata", {})
    return metadata.get("original_generated_token_ids") == metadata.get(
        "teacher_forced_token_ids"
    )


def _ablation_images(image: Image.Image) -> dict[str, Image.Image]:
    return {
        "global_blur": image.filter(
            ImageFilter.GaussianBlur(radius=max(image.size) / 12)
        ),
        "uniform_gray": Image.new("RGB", image.size, (127, 127, 127)),
    }


def _generate_ablation(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    response, token_ids = generate_response(
        model,
        processor,
        image,
        prompt,
        max_new_tokens=max_new_tokens,
    )
    record: dict[str, Any] = {
        "status": "ok",
        "raw_response": response,
        "generated_token_ids": token_ids,
    }
    try:
        thinking, output, _, _ = split_thinking_output(response)
        record.update({"THINKING": thinking, "OUTPUT": output})
    except ValueError as error:
        record.update(
            {
                "status": "parse_error",
                "error": str(error),
                "THINKING": None,
                "OUTPUT": None,
            }
        )
    return record


def run(
    *,
    dataset_manifest: Path,
    model_output: Path,
    generation_evaluation: Path,
    ablation_model_output: Path,
    revised_evaluation_output: Path,
    model_name: str,
    revision: str | None,
    device: str,
    min_pixels: int,
    max_pixels: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    datasets = {record["sample_id"]: record for record in read_jsonl(dataset_manifest)}
    models = {record["sample_id"]: record for record in read_jsonl(model_output)}
    evaluations = {
        record["sample_id"]: record for record in read_jsonl(generation_evaluation)
    }
    candidates = [
        sample_id
        for sample_id in datasets
        if sample_id in models
        and evaluations.get(sample_id, {}).get("output_correct")
        and evaluations[sample_id].get("generation_stable")
        and _token_identity_stable(models[sample_id])
    ]
    existing_models = (
        read_jsonl(ablation_model_output) if ablation_model_output.exists() else []
    )
    existing_by_id = {
        record["sample_id"]: record
        for record in existing_models
        if record.get("status") == "complete"
    }

    model = processor = None
    if any(sample_id not in existing_by_id for sample_id in candidates):
        if revision is None:
            revisions = {
                str(models[sample_id]["model"]["resolved_revision"])
                for sample_id in candidates
            }
            if len(revisions) != 1:
                raise ValueError(f"model outputs contain mixed revisions: {sorted(revisions)}")
            revision = next(iter(revisions))
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

    ablation_records = list(existing_models)
    for index, sample_id in enumerate(candidates):
        if sample_id in existing_by_id:
            continue
        assert model is not None and processor is not None
        dataset = datasets[sample_id]
        image = Image.open(dataset["input"]["I_IMAGE"]).convert("RGB")
        prompt = model_record_prompt(models[sample_id])
        ablations: dict[str, Any] = {}
        try:
            for name, ablated_image in _ablation_images(image).items():
                ablations[name] = _generate_ablation(
                    model,
                    processor,
                    ablated_image,
                    prompt,
                    max_new_tokens=max_new_tokens,
                )
            record = {
                "schema_version": 1,
                "status": "complete",
                "benchmark": dataset["benchmark"],
                "sample_id": sample_id,
                "I_QUESTION": dataset["input"]["I_QUESTION"],
                "ablations": ablations,
                "model": {
                    "repo_id": model_name,
                    "revision": revision,
                    "do_sample": False,
                    "max_new_tokens": max_new_tokens,
                },
            }
        except Exception as error:
            record = {
                "schema_version": 1,
                "status": "error",
                "benchmark": dataset["benchmark"],
                "sample_id": sample_id,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "ablations": ablations,
            }
        ablation_records.append(record)
        write_jsonl(ablation_records, ablation_model_output)
        print(
            f"[{index + 1}/{len(candidates)}] {sample_id} status={record['status']}",
            flush=True,
        )

    complete = {
        record["sample_id"]: record
        for record in ablation_records
        if record.get("status") == "complete"
    }
    revised: list[dict[str, Any]] = []
    for sample_id in datasets:
        original = dict(evaluations.get(sample_id, {}))
        # A targeted generation run may intentionally omit most manifest
        # samples. Keep those negative rows schema-valid so downstream joins
        # can distinguish "not generated" from an anonymous diagnostic row.
        original.setdefault("schema_version", 2)
        original.setdefault("benchmark", datasets[sample_id]["benchmark"])
        original.setdefault("sample_id", sample_id)
        original.setdefault(
            "REFERENCE_OUTPUT",
            datasets[sample_id]["evaluation"]["REFERENCE_OUTPUT"],
        )
        model_record = models.get(sample_id)
        identity = bool(model_record and _token_identity_stable(model_record))
        ablation = complete.get(sample_id)
        comparisons: dict[str, Any] = {}
        original_output = model_record.get("OUTPUT") if model_record else None
        for name, generated in (ablation or {}).get("ablations", {}).items():
            output = generated.get("OUTPUT")
            same = (
                output is not None
                and original_output is not None
                and normalized_output(output) == normalized_output(original_output)
            )
            comparisons[name] = {
                "status": generated.get("status"),
                "normalized_output": (
                    normalized_output(output) if output is not None else None
                ),
                "same_as_original_output": same,
            }
        confirmed = bool(comparisons) and any(
            not comparison["same_as_original_output"]
            for comparison in comparisons.values()
        )
        original.update(
            {
                "generated_teacher_forced_ids_match": identity,
                "ablation_outputs": comparisons,
                "image_dependent_by_generation_ablation": confirmed,
                "strict_eligible": bool(
                    original.get("output_correct")
                    and original.get("generation_stable")
                    and float(original.get("image_dependence_delta", 0.0)) > 0.0
                    and identity
                    and confirmed
                ),
            }
        )
        revised.append(original)
    write_jsonl(revised, revised_evaluation_output)
    summary = {
        "dataset_records": len(datasets),
        "ablation_candidates": len(candidates),
        "complete_ablation_records": len(complete),
        "strict_eligible": sum(record.get("strict_eligible", False) for record in revised),
        "token_identity_mismatches": sum(
            not record.get("generated_teacher_forced_ids_match", False)
            for record in revised
            if record.get("output_correct") and record.get("generation_stable")
        ),
        "ablation_model_output": str(ablation_model_output),
        "revised_evaluation_output": str(revised_evaluation_output),
    }
    if model is not None:
        del processor, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--generation-evaluation", type=Path, required=True)
    parser.add_argument("--ablation-model-output", type=Path, required=True)
    parser.add_argument("--revised-evaluation-output", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    summary = run(
        dataset_manifest=args.dataset_manifest,
        model_output=args.model_output,
        generation_evaluation=args.generation_evaluation,
        ablation_model_output=args.ablation_model_output,
        revised_evaluation_output=args.revised_evaluation_output,
        model_name=args.model,
        revision=args.revision,
        device=args.device,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
