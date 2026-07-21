"""Render strict dataset manifests with their evaluation-only evidence."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .strict_generation import read_jsonl


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _evidence_overlay(record: dict[str, Any], image: Image.Image) -> Image.Image:
    evaluation = record["evaluation"]
    mask_paths = evaluation.get("EVIDENCE_MASKS")
    if mask_paths:
        mask = np.load(mask_paths["primary_unique_firstnonempty"])
        mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L").resize(
            image.size, Image.Resampling.NEAREST
        )
        outline = mask_image.filter(ImageFilter.MaxFilter(9))
        layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        layer.paste((20, 255, 60, 150), mask=outline)
        return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")

    output = image.copy()
    draw = ImageDraw.Draw(output)
    metadata = evaluation["metadata"]
    native_width = metadata["image_size"]["width"]
    native_height = metadata["image_size"]["height"]
    for x1, y1, x2, y2 in evaluation["EVIDENCE_BOXES"]:
        draw.rectangle(
            (
                round(x1 * image.width / native_width),
                round(y1 * image.height / native_height),
                round(x2 * image.width / native_width),
                round(y2 * image.height / native_height),
            ),
            outline=(20, 255, 60),
            width=5,
        )
    return output


def _panel(record: dict[str, Any], *, width: int, image_height: int) -> Image.Image:
    original = Image.open(record["input"]["I_IMAGE"]).convert("RGB")
    original.thumbnail((width, image_height))
    original = _evidence_overlay(record, original)
    canvas_image = Image.new("RGB", (width, image_height), "#101318")
    left = (width - original.width) // 2
    top = (image_height - original.height) // 2
    canvas_image.paste(original, (left, top))

    metadata = record["evaluation"]["metadata"]
    family = metadata.get("reasoning_family", metadata.get("stratum", ""))
    steps = metadata.get("program_steps")
    title = f"{record['sample_id']}  ·  {family}"
    if steps is not None:
        title += f"  ·  {steps} program steps"
    question_lines = textwrap.wrap(record["input"]["I_QUESTION"], width=58)
    answer = f"Reference: {record['evaluation']['REFERENCE_OUTPUT']}"
    text_height = 42 + 30 * len(question_lines) + 44
    panel = Image.new("RGB", (width, image_height + text_height), "white")
    panel.paste(canvas_image, (0, 0))
    draw = ImageDraw.Draw(panel)
    draw.text((16, image_height + 10), title, fill="#111827", font=_font(19, bold=True))
    y = image_height + 40
    for line in question_lines:
        draw.text((16, y), line, fill="#1f2937", font=_font(18))
        y += 27
    draw.text((16, y + 4), answer, fill="#065f46", font=_font(18, bold=True))
    return panel


def render(manifest: Path, output: Path, *, limit: int) -> None:
    all_records = read_jsonl(manifest)
    if any(
        "reasoning_family" in record["evaluation"]["metadata"]
        for record in all_records
    ):
        records = []
        seen = set()
        for record in all_records:
            family = record["evaluation"]["metadata"].get("reasoning_family")
            if family in seen:
                continue
            seen.add(family)
            records.append(record)
            if len(records) == limit:
                break
    else:
        records = all_records[:limit]
    panels = [_panel(record, width=520, image_height=420) for record in records]
    columns = 2
    gutter = 18
    rows = (len(panels) + columns - 1) // columns
    row_heights = [
        max(
            (
                panels[index].height
                for index in range(row * columns, min((row + 1) * columns, len(panels)))
            ),
            default=0,
        )
        for row in range(rows)
    ]
    canvas = Image.new(
        "RGB",
        (
            columns * 520 + (columns - 1) * gutter,
            sum(row_heights) + max(0, rows - 1) * gutter,
        ),
        "#d1d5db",
    )
    y = 0
    for index, panel in enumerate(panels):
        row, column = divmod(index, columns)
        if column == 0 and row:
            y += row_heights[row - 1] + gutter
        canvas.paste(panel, (column * (520 + gutter), y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    render(args.manifest, args.output, limit=args.limit)
    print(json.dumps({"output": str(args.output), "samples": args.limit}, indent=2))


if __name__ == "__main__":
    main()
