"""Run Qwen3-VL generation and a visual leave-one-region-out baseline."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFilter

if __package__:
    from .datasets import MultimodalExample, load_examples, vqa_accuracy
else:  # Allow sibling tools to import this file when run as standalone scripts.
    from datasets import MultimodalExample, load_examples, vqa_accuracy


DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_REVISION = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
SYSTEM_PROMPT = """You are a visual-question-answering evaluator. Always return exactly two lines:
Reasoning: a complete sentence stating the visual evidence.
Final answer: a one-to-three-word answer.
Never omit either label and never add other text."""
PROMPT = """Answer the visual question using the image. Output exactly two lines and no other text:
Reasoning: one concise sentence describing the visual evidence.
Final answer: a one-to-three-word answer.

Question: {question}"""


def parse_response(response: str) -> tuple[str, str]:
    reasoning = re.search(r"(?im)^\s*(?:\*\*)?Reasoning\s*:\s*(?:\*\*)?\s*(.+?)\s*$", response)
    final = re.search(r"(?im)^\s*(?:\*\*)?Final answer\s*:\s*(?:\*\*)?\s*(.+?)\s*$", response)
    if final:
        answer = final.group(1).strip().strip("*`")
    else:
        nonempty = [line.strip() for line in response.splitlines() if line.strip()]
        answer = nonempty[-1] if nonempty else response.strip()
        answer = re.sub(r"(?i)^final\s+answer\s*:\s*", "", answer).strip()
    answer = answer.rstrip(". ")
    yes_or_no = re.match(r"(?i)^(yes|no)\b", answer)
    if yes_or_no:
        answer = yes_or_no.group(1)
    reasoning_text = reasoning.group(1).strip() if reasoning else ""
    if not reasoning_text:
        nonempty = [line.strip() for line in response.splitlines() if line.strip()]
        if len(nonempty) >= 2:
            fallback = re.sub(
                r"(?i)^reasoning\s*:\s*", "", nonempty[-2]
            ).strip()
            if fallback.casefold().rstrip(". ") != answer.casefold().rstrip(". "):
                reasoning_text = fallback
    return reasoning_text, answer


def _messages(prompt: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _model_inputs(model: Any, processor: Any, image: Image.Image, prompt: str) -> dict[str, Any]:
    inputs = processor.apply_chat_template(
        _messages(prompt, image),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    return dict(inputs)


@torch.inference_mode()
def generate(model: Any, processor: Any, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
    inputs = _model_inputs(model, processor, image, prompt)
    prompt_length = int(inputs["input_ids"].shape[1])
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    sequences = output.sequences if hasattr(output, "sequences") else output
    response_ids = sequences[:, prompt_length:]
    return processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


@torch.inference_mode()
def mean_response_logprob(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    response: str,
) -> float:
    inputs = _model_inputs(model, processor, image, prompt)
    response_ids = processor.tokenizer(
        response,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)
    if response_ids.numel() == 0:
        raise ValueError("Cannot score an empty response")
    prompt_length = int(inputs["input_ids"].shape[1])
    full_ids = torch.cat((inputs["input_ids"], response_ids), dim=1)
    full_mask = torch.cat(
        (
            inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
            torch.ones_like(response_ids),
        ),
        dim=1,
    )
    forward_inputs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }
    token_types = forward_inputs.get("mm_token_type_ids")
    if torch.is_tensor(token_types) and token_types.shape[-1] != full_ids.shape[-1]:
        forward_inputs["mm_token_type_ids"] = F.pad(
            token_types, (0, full_ids.shape[-1] - token_types.shape[-1]), value=0
        )
    output = model(input_ids=full_ids, attention_mask=full_mask, **forward_inputs)
    start = prompt_length - 1
    logits = output.logits[:, start : start + response_ids.shape[1], :].float()
    log_probs = logits.log_softmax(dim=-1)
    selected = log_probs.gather(-1, response_ids.unsqueeze(-1)).squeeze(-1)
    return float(selected.mean().item())


def perturb_region(image: Image.Image, row: int, column: int, grid_size: int) -> Image.Image:
    """Replace one grid cell with the corresponding heavily blurred pixels."""

    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    width, height = image.size
    left, right = column * width // grid_size, (column + 1) * width // grid_size
    top, bottom = row * height // grid_size, (row + 1) * height // grid_size
    blurred = image.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 12))
    output = image.copy()
    output.paste(blurred.crop((left, top, right, bottom)), (left, top))
    return output


def summarize_grid(grid: list[list[float]]) -> dict[str, Any]:
    flattened = [value for row in grid for value in row]
    if not flattened:
        raise ValueError("Attribution grid is empty")
    width = len(grid[0])
    top_index = max(range(len(flattened)), key=flattened.__getitem__)
    positive = [max(0.0, value) for value in flattened]
    denominator = sum(positive)
    top_count = max(1, math.ceil(len(positive) / 4))
    top_quartile_share = (
        sum(sorted(positive, reverse=True)[:top_count]) / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "top_cell": [top_index // width, top_index % width],
        "max_drop": flattened[top_index],
        "positive_mass": denominator,
        "top_quartile_share": top_quartile_share,
    }


@torch.inference_mode()
def visual_loo(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    response: str,
    *,
    grid_size: int,
) -> tuple[list[list[float]], float, float]:
    base = mean_response_logprob(model, processor, image, prompt, response)
    fully_blurred = image.filter(ImageFilter.GaussianBlur(radius=max(image.size) / 12))
    full_blur_drop = base - mean_response_logprob(
        model, processor, fully_blurred, prompt, response
    )
    grid = []
    for row in range(grid_size):
        values = []
        for column in range(grid_size):
            perturbed = perturb_region(image, row, column, grid_size)
            score = mean_response_logprob(model, processor, perturbed, prompt, response)
            values.append(base - score)
        grid.append(values)
    return grid, base, full_blur_drop


def make_overlay(image: Image.Image, grid: list[list[float]]) -> Image.Image:
    """Render a possibly rectangular attribution grid over an image."""

    if not grid or not grid[0]:
        raise ValueError("Attribution grid must be non-empty")
    rows = len(grid)
    columns = len(grid[0])
    if any(len(values) != columns for values in grid):
        raise ValueError("Attribution grid rows must have equal length")
    width, height = image.size
    positive = [max(0.0, value) for row in grid for value in row]
    maximum = max(positive) or 1.0
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for row, values in enumerate(grid):
        for column, value in enumerate(values):
            left, right = column * width // columns, (column + 1) * width // columns
            top, bottom = row * height // rows, (row + 1) * height // rows
            alpha = int(190 * max(0.0, value) / maximum)
            draw.rectangle((left, top, right - 1, bottom - 1), fill=(255, 30, 20, alpha), outline=(255, 255, 255, 180))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def save_overlay(image: Image.Image, grid: list[list[float]], destination: Path) -> None:
    """Write an image overlay whose red alpha tracks positive attribution."""

    composed = make_overlay(image, grid)
    destination.parent.mkdir(parents=True, exist_ok=True)
    composed.save(destination)


def _thumbnail(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def evaluate_example(
    model: Any,
    processor: Any,
    example: MultimodalExample,
    *,
    output_dir: Path,
    max_image_side: int,
    max_new_tokens: int,
    grid_size: int,
    response: str | None = None,
    generation_seconds: float | None = None,
) -> dict[str, Any]:
    image = _thumbnail(example.image_path, max_image_side)
    prompt = PROMPT.format(question=example.question)
    if response is None:
        started = time.perf_counter()
        response = generate(model, processor, image, prompt, max_new_tokens)
        generation_seconds = time.perf_counter() - started
    if generation_seconds is None:
        generation_seconds = 0.0
    reasoning, final_answer = parse_response(response)
    attribution_started = time.perf_counter()
    grid, base_logprob, full_blur_drop = visual_loo(
        model, processor, image, prompt, response, grid_size=grid_size
    )
    attribution_seconds = time.perf_counter() - attribution_started
    grid_summary = summarize_grid(grid)
    overlay_path = output_dir / "overlays" / example.dataset / f"{example.question_id}.jpg"
    save_overlay(image, grid, overlay_path)
    return {
        "dataset": example.dataset,
        "split": example.split,
        "question_id": example.question_id,
        "image_id": example.image_id,
        "image_path": str(example.image_path),
        "question": example.question,
        "reference_answers": list(example.answers),
        "majority_answer": example.majority_answer,
        "human_rationales": list(example.rationales),
        "response": response,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "vqa_accuracy": vqa_accuracy(final_answer, example.answers),
        "visual_loo": {
            "grid": grid,
            "grid_size": grid_size,
            "base_mean_logprob": base_logprob,
            "full_blur_drop": full_blur_drop,
            **grid_summary,
        },
        "overlay_path": str(overlay_path),
        "generation_seconds": generation_seconds,
        "attribution_seconds": attribution_seconds,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/multimodal_smoke"))
    parser.add_argument("--dataset", choices=("all", "vqa_x", "aokvqa"), default="all")
    parser.add_argument("--split", default="val")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--max-image-side", type=int, default=448)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--grid-size", type=int, default=4)
    parser.add_argument(
        "--correct-only",
        action="store_true",
        help="Attribute only correctly answered samples, matching the paper protocol.",
    )
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=25,
        help="Ordered candidate pool examined per dataset when --correct-only is set.",
    )
    parser.add_argument(
        "--minimum-vqa-accuracy",
        type=float,
        default=0.6,
        help="Consensus-score threshold used by --correct-only.",
    )
    parser.add_argument(
        "--minimum-reasoning-words",
        type=int,
        default=3,
        help="Minimum reasoning length; set zero to disable format filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if min(
        args.samples,
        args.max_image_side,
        args.max_new_tokens,
        args.grid_size,
        args.candidate_limit,
    ) <= 0:
        raise SystemExit("numeric arguments must be positive")
    if not 0 <= args.minimum_vqa_accuracy <= 1:
        raise SystemExit("--minimum-vqa-accuracy must be between zero and one")
    if args.minimum_reasoning_words < 0:
        raise SystemExit("--minimum-reasoning-words cannot be negative")

    from transformers import AutoModelForMultimodalLM, AutoProcessor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    datasets = ("vqa_x", "aokvqa") if args.dataset == "all" else (args.dataset,)
    records = []
    attempts = []
    result_path = args.output_dir / "results.jsonl"
    attempts_path = args.output_dir / "attempts.jsonl"
    with result_path.open("w", encoding="utf-8") as stream, attempts_path.open(
        "w", encoding="utf-8"
    ) as attempts_stream:
        for dataset in datasets:
            selected = 0
            limit = args.candidate_limit if args.correct_only else args.samples
            for candidate_index, example in enumerate(
                load_examples(dataset, args.data_root, split=args.split, limit=limit)
            ):
                image = _thumbnail(example.image_path, args.max_image_side)
                prompt = PROMPT.format(question=example.question)
                generation_started = time.perf_counter()
                response = generate(
                    model, processor, image, prompt, args.max_new_tokens
                )
                generation_seconds = time.perf_counter() - generation_started
                reasoning, final_answer = parse_response(response)
                accuracy = vqa_accuracy(final_answer, example.answers)
                reasoning_words = len(reasoning.split())
                meets_answer_threshold = accuracy >= args.minimum_vqa_accuracy
                meets_reasoning_threshold = (
                    reasoning_words >= args.minimum_reasoning_words
                )
                attempt = {
                    "dataset": dataset,
                    "candidate_index": candidate_index,
                    "question_id": example.question_id,
                    "question": example.question,
                    "majority_answer": example.majority_answer,
                    "response": response,
                    "reasoning": reasoning,
                    "final_answer": final_answer,
                    "vqa_accuracy": accuracy,
                    "reasoning_words": reasoning_words,
                    "selected": (
                        not args.correct_only
                        or (meets_answer_threshold and meets_reasoning_threshold)
                    ),
                }
                attempts.append(attempt)
                attempts_stream.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                attempts_stream.flush()
                if not attempt["selected"]:
                    print(
                        f"[{dataset}] skip {example.question_id} "
                        f"answer={final_answer!r} score={accuracy:.2f} "
                        f"reasoning_words={reasoning_words}",
                        flush=True,
                    )
                    continue
                record = evaluate_example(
                    model,
                    processor,
                    example,
                    output_dir=args.output_dir,
                    max_image_side=args.max_image_side,
                    max_new_tokens=args.max_new_tokens,
                    grid_size=args.grid_size,
                    response=response,
                    generation_seconds=generation_seconds,
                )
                record["candidate_index"] = candidate_index
                records.append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                print(
                    f"[{record['dataset']}] {record['question_id']} "
                    f"answer={record['final_answer']!r} "
                    f"score={record['vqa_accuracy']:.2f} "
                    f"blur_drop={record['visual_loo']['full_blur_drop']:.4f}",
                    flush=True,
                )
                selected += 1
                if selected >= args.samples:
                    break
            if selected < args.samples:
                raise RuntimeError(
                    f"Only found {selected}/{args.samples} eligible {dataset} examples "
                    f"within {limit} candidates"
                )

    by_dataset = {}
    for dataset in datasets:
        subset = [record for record in records if record["dataset"] == dataset]
        by_dataset[dataset] = {
            "samples": len(subset),
            "candidates_attempted": sum(
                attempt["dataset"] == dataset for attempt in attempts
            ),
            "mean_vqa_accuracy": sum(record["vqa_accuracy"] for record in subset) / len(subset),
            "parsed_reasoning": sum(bool(record["reasoning"]) for record in subset),
            "positive_full_blur_drop": sum(
                record["visual_loo"]["full_blur_drop"] > 0 for record in subset
            ),
            "mean_top_quartile_share": sum(
                record["visual_loo"]["top_quartile_share"] for record in subset
            ) / len(subset),
        }
    summary = {
        "model": args.model,
        "revision": args.revision,
        "samples_per_dataset": args.samples,
        "grid_size": args.grid_size,
        "max_image_side": args.max_image_side,
        "correct_only": args.correct_only,
        "minimum_vqa_accuracy": args.minimum_vqa_accuracy,
        "minimum_reasoning_words": args.minimum_reasoning_words,
        "datasets": by_dataset,
        "results": str(result_path),
        "attempts": str(attempts_path),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
