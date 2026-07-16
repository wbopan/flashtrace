"""Minimal spatial-grounding example for Qwen3-VL plus FlashTrace.

The script creates its own four-quadrant image, asks about the upper-left
quadrant, generates a two-line reasoning/final response, and traces only the
final answer through the generated reasoning.  It writes the source image and
an 8x8-style visual-token heatmap to disk for direct inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw

from flashtrace import FlashTrace, load_vlm_and_processor
from flashtrace.vlm import multimodal_messages


DEFAULT_PROMPT = """Look at the four-quadrant image. What colored shape is in the upper-left quadrant?
Reply in exactly two lines:
Reasoning: one short sentence describing the visual evidence you used.
Final answer: the color and shape only."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/flashtrace-qwen3-vl-quadrant"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless the answer says red circle and attribution peaks top-left.",
    )
    return parser.parse_args()


def make_quadrant_image(size: int = 224) -> Image.Image:
    """Create a deterministic four-shape image with no external assets."""

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    half = size // 2
    margin = size // 9
    shape_size = size // 4
    outline_width = max(2, size // 80)

    draw.line((half, 0, half, size), fill="#808080", width=outline_width)
    draw.line((0, half, size, half), fill="#808080", width=outline_width)

    # Upper-left: red circle (the queried evidence).
    draw.ellipse(
        (margin, margin, margin + shape_size, margin + shape_size),
        fill="#e53935",
        outline="#7f0000",
        width=outline_width,
    )

    # Upper-right: blue square.
    right_x = half + margin
    draw.rectangle(
        (right_x, margin, right_x + shape_size, margin + shape_size),
        fill="#1e88e5",
        outline="#003c8f",
        width=outline_width,
    )

    # Lower-left: green triangle.
    lower_y = half + margin
    draw.polygon(
        (
            (margin + shape_size // 2, lower_y),
            (margin, lower_y + shape_size),
            (margin + shape_size, lower_y + shape_size),
        ),
        fill="#43a047",
        outline="#00600f",
    )

    # Lower-right: yellow star.
    center_x = half + margin + shape_size // 2
    center_y = lower_y + shape_size // 2
    outer_radius = shape_size // 2
    inner_radius = max(1, outer_radius * 2 // 5)
    star: list[tuple[float, float]] = []
    for point in range(10):
        angle = -math.pi / 2 + point * math.pi / 5
        radius = outer_radius if point % 2 == 0 else inner_radius
        star.append(
            (center_x + radius * math.cos(angle), center_y + radius * math.sin(angle))
        )
    draw.polygon(star, fill="#fdd835", outline="#8d6e00")
    return image


@torch.inference_mode()
def generate_response(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    *,
    max_new_tokens: int,
) -> str:
    """Generate once using the same processor/template as the trace path."""

    messages = multimodal_messages(prompt, image)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(inputs, "to"):
        inputs = inputs.to(model.device)
    prompt_length = int(inputs["input_ids"].shape[1])
    generated = model.generate(
        **dict(inputs),
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    response_ids = sequences[:, prompt_length:]
    return processor.batch_decode(
        response_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def _token_span_for_chars(
    offsets: list[list[int]] | list[tuple[int, int]], start: int, end: int
) -> tuple[int, int] | None:
    indices = [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > start and token_start < end
    ]
    return (indices[0], indices[-1]) if indices else None


def response_parts_and_spans(
    tokenizer: Any, response: str
) -> tuple[str, str, tuple[int, int] | None, tuple[int, int]]:
    """Extract the two requested lines and their inclusive token spans."""

    reasoning_match = re.search(
        r"(?im)^\s*(?:\*\*)?Reasoning\s*:\s*(?:\*\*)?\s*(.+?)\s*$",
        response,
    )
    final_match = re.search(
        r"(?im)^\s*(?:\*\*)?Final answer\s*:\s*(?:\*\*)?\s*(.+?)\s*$",
        response,
    )
    if final_match is None:
        nonempty_lines = [line for line in response.splitlines() if line.strip()]
        fallback = nonempty_lines[-1] if nonempty_lines else response
        fallback_start = response.rfind(fallback)
        final_chars = (fallback_start, fallback_start + len(fallback))
        final_answer = fallback.strip()
    else:
        final_chars = final_match.span(1)
        final_answer = final_match.group(1).strip()

    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if torch.is_tensor(offsets):
        offsets = offsets.tolist()
    if offsets and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]

    final_span = _token_span_for_chars(offsets, *final_chars)
    if final_span is None:
        token_count = len(encoded["input_ids"])
        final_span = (0, max(0, token_count - 1))

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    reasoning_span = (
        _token_span_for_chars(offsets, *reasoning_match.span(1))
        if reasoning_match
        else None
    )
    return reasoning, final_answer, reasoning_span, final_span


def spatial_scores(result: Any) -> tuple[list[list[float]], dict[str, dict[str, float]]]:
    """Map projected visual-token scores back to a 2-D grid and four quadrants."""

    multimodal = result.metadata["multimodal"]
    grids = multimodal["visual_grid_thw"]
    if len(grids) != 1:
        raise ValueError(f"Expected one image grid, got {grids!r}")
    frames, height, width = (int(value) for value in grids[0])
    visual_indices = [int(index) for index in multimodal["visual_token_indices_prompt"]]
    expected = frames * height * width
    if len(visual_indices) != expected:
        raise ValueError(
            f"Visual-token count {len(visual_indices)} does not match grid {grids[0]}."
        )

    grid = [[0.0 for _ in range(width)] for _ in range(height)]
    for flattened, prompt_index in enumerate(visual_indices):
        spatial_index = flattened % (height * width)
        row, column = divmod(spatial_index, width)
        grid[row][column] += max(0.0, float(result.scores[prompt_index]))

    totals = {"top_left": 0.0, "top_right": 0.0, "bottom_left": 0.0, "bottom_right": 0.0}
    for row, values in enumerate(grid):
        for column, score in enumerate(values):
            vertical = "top" if row < height / 2 else "bottom"
            horizontal = "left" if column < width / 2 else "right"
            totals[f"{vertical}_{horizontal}"] += score
    denominator = sum(totals.values()) or 1.0
    summary = {
        name: {"score": score, "share": score / denominator}
        for name, score in totals.items()
    }
    return grid, summary


def save_heatmap(grid: list[list[float]], path: Path) -> None:
    """Save a dependency-free visual-token heatmap."""

    height = len(grid)
    width = len(grid[0])
    cell = 40
    maximum = max(max(row) for row in grid) or 1.0
    image = Image.new("RGB", (width * cell, height * cell), "white")
    draw = ImageDraw.Draw(image)
    for row, values in enumerate(grid):
        for column, score in enumerate(values):
            normalized = max(0.0, score) / maximum
            color = (
                int(35 + 220 * normalized),
                int(55 + 180 * normalized),
                int(150 * (1.0 - normalized)),
            )
            box = (
                column * cell,
                row * cell,
                (column + 1) * cell - 1,
                (row + 1) * cell - 1,
            )
            draw.rectangle(box, fill=color, outline="#202020")
    draw.line((width * cell // 2, 0, width * cell // 2, height * cell), fill="white", width=4)
    draw.line((0, height * cell // 2, width * cell, height * cell // 2), fill="white", width=4)
    image.save(path)


def main() -> None:
    args = _parse_args()
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_path = args.output_dir / "quadrants.png"
    heatmap_path = args.output_dir / "attribution_heatmap.png"
    result_path = args.output_dir / "result.json"

    image = make_quadrant_image()
    image.save(input_path)
    model, processor = load_vlm_and_processor(
        args.model,
        dtype="bfloat16",
        device_map="auto",
    )

    generated_started = time.perf_counter()
    response = generate_response(
        model,
        processor,
        image,
        DEFAULT_PROMPT,
        max_new_tokens=args.max_new_tokens,
    )
    generation_seconds = time.perf_counter() - generated_started
    reasoning, final_answer, reasoning_span, final_span = response_parts_and_spans(
        processor.tokenizer, response
    )

    tracer = FlashTrace(
        model,
        processor,
        chunk_tokens=64,
        sink_chunk_tokens=8,
        recompute_attention=True,
    )
    traced_started = time.perf_counter()
    result = tracer.trace(
        prompt=DEFAULT_PROMPT,
        images=image,
        target=response,
        reasoning_span=reasoning_span,
        output_span=final_span,
        method="flashtrace",
    )
    trace_seconds = time.perf_counter() - traced_started

    grid, quadrants = spatial_scores(result)
    save_heatmap(grid, heatmap_path)
    dominant_quadrant = max(quadrants, key=lambda name: quadrants[name]["score"])
    normalized_answer = final_answer.casefold()
    answer_is_red_circle = "red" in normalized_answer and "circle" in normalized_answer
    attribution_is_top_left = dominant_quadrant == "top_left"
    checks = {
        "answer_is_red_circle": answer_is_red_circle,
        "attribution_dominant_quadrant_is_top_left": attribution_is_top_left,
        "passed": answer_is_red_circle and attribution_is_top_left,
    }
    payload = {
        "model": args.model,
        "prompt": DEFAULT_PROMPT,
        "response": response,
        "reasoning": reasoning,
        "final_answer": final_answer,
        "reasoning_span": reasoning_span,
        "final_answer_span": final_span,
        "attention_mode": result.metadata["multimodal"]["attention_mode"],
        "visual_grid_thw": result.metadata["multimodal"]["visual_grid_thw"],
        "quadrant_attribution": quadrants,
        "dominant_quadrant": dominant_quadrant,
        "checks": checks,
        "generation_seconds": generation_seconds,
        "trace_seconds": trace_seconds,
        "artifacts": {
            "input_image": str(input_path),
            "attribution_heatmap": str(heatmap_path),
        },
    }
    result_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.strict and not checks["passed"]:
        raise SystemExit("Spatial-grounding checks failed; inspect the emitted artifacts.")


if __name__ == "__main__":
    main()
