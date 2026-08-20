#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy==2.2.6",
#   "pillow==12.3.0",
# ]
# ///
"""Render the compact qualitative visual-attribution comparison.

The renderer keeps the native attribution grids and applies one shared crop per
benchmark.  Metrics are read from the full-image evaluation records; the crop
changes only the presentation, not the reported score.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont


METHODS = ("flashtrace", "attnlrp", "visual-ig", "visual-loo")
METHOD_LABELS = {
    "flashtrace": "FlashTrace++",
    "attnlrp": "AttnLRP",
    "visual-ig": "Visual IG",
    "visual-loo": "Visual LOO",
}
ATTRIBUTION_COLOR = np.array([214, 39, 117], dtype=np.float32)
EVIDENCE_COLOR = (24, 214, 88)
INK = "#111827"
MUTED = "#4b5563"
ACCENT = "#087ea4"
ASSET_SHA256 = {
    "261.jpg": "dc712dcb3459662a959bebbdcd9a0a7b9c55d00498d6f2f8eca97b053e84d908",
    "1128.png": "afae63bedd552721f511365c10749ec4860ce27bbeb61e9d3876722890bda4b9",
    "vizwiz261.jsonl.gz": "5a294842b8fa270104e754fd2068a132217b27cfbc028ff16856b2f5d3703a46",
    "wiki1128.jsonl.gz": "59660c046ee7c3dd7500c72270c506cee106b89c8c2a49bdf164ebd21b86ae95",
    "vizwiz261-faith.jsonl.gz": "45f6e6b45dcbc3243c499ebf518e2e33a3fd27d1c6e67b535b140c43a4e063ef",
}


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise FileNotFoundError("Arial or DejaVu Sans is required to render the figure")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _verify_assets(raw_dir: Path) -> None:
    for name, expected in ASSET_SHA256.items():
        path = raw_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundled Fig. 6 asset: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"Checksum mismatch for {path}: {digest}")


def _records_by_method(path: Path, sample_id: str) -> dict[str, dict[str, Any]]:
    return {
        record["method"]: record
        for record in _read_jsonl(path)
        if record.get("sample_id") == sample_id
        and record.get("method") in METHODS
        and record.get("status") == "ok"
    }


def _positive_overlay(image: Image.Image, grid: list[list[float]]) -> Image.Image:
    """Overlay positive attribution using a robust, shared visual mapping.

    Each method is normalized by its 99th-percentile positive score, matching
    the per-method normalization used in the earlier figure while preventing a
    single outlier patch from making the remaining map invisible.
    """

    values = np.clip(np.asarray(grid, dtype=np.float32), 0.0, None)
    positive = values[values > 0]
    if positive.size:
        scale = float(np.quantile(positive, 0.99))
        if scale <= 0:
            scale = float(positive.max())
        normalized = np.clip(values / max(scale, 1e-12), 0.0, 1.0)
        # Suppress the faint background while keeping intermediate patches
        # visible.  Nearest-neighbor resizing preserves the native patch grid.
        strength = np.clip((normalized - 0.10) / 0.90, 0.0, 1.0) ** 0.85
    else:
        strength = np.zeros_like(values)

    alpha = Image.fromarray(np.uint8(strength * 218), mode="L").resize(
        image.size, Image.Resampling.NEAREST
    )
    base = ImageEnhance.Color(image.convert("RGB")).enhance(0.28)
    base = ImageEnhance.Contrast(base).enhance(0.92)
    base_array = np.asarray(base, dtype=np.float32)
    alpha_array = np.asarray(alpha, dtype=np.float32)[..., None] / 255.0
    output = base_array * (1.0 - alpha_array) + ATTRIBUTION_COLOR * alpha_array
    return Image.fromarray(np.uint8(np.clip(output, 0, 255)), mode="RGB")


def _crop_and_fit(
    image: Image.Image,
    crop: tuple[int, int, int, int],
    size: tuple[int, int],
) -> Image.Image:
    return image.crop(crop).resize(size, Image.Resampling.LANCZOS)


def _wiki_evidence_outline(
    image: Image.Image,
    boxes: list[list[float]],
    *,
    line_width: int = 12,
) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    for x1, y1, x2, y2 in boxes:
        draw.rectangle((x1, y1, x2, y2), outline=EVIDENCE_COLOR, width=line_width)
    return output


def _metric_text(method: str, value: float, *, kind: str) -> str:
    if kind == "deletion":
        return f"Deletion AUC {value:.2f}"
    if kind == "rank":
        return f"Rank AUC {value:.2f}"
    raise ValueError(kind)


def _draw_row(
    canvas: Image.Image,
    *,
    y: int,
    title: str,
    legend: str,
    image: Image.Image,
    records: dict[str, dict[str, Any]],
    crop: tuple[int, int, int, int],
    panel_width: int,
    image_height: int,
    gutter: int,
    margin: int,
    metrics: dict[str, float],
    metric_kind: str,
    evidence_boxes: list[list[float]] | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    group_font = _font(42, bold=True)
    legend_font = _font(24)
    method_font = _font(31)
    method_bold = _font(31, bold=True)
    metric_font = _font(25)
    metric_bold = _font(25, bold=True)

    draw.text((margin, y), title, fill=INK, font=group_font)
    legend_bbox = draw.textbbox((0, 0), legend, font=legend_font)
    draw.text(
        (canvas.width - margin - (legend_bbox[2] - legend_bbox[0]), y + 9),
        legend,
        fill=MUTED,
        font=legend_font,
    )

    label_y = y + 48
    image_y = label_y + 39
    metric_y = image_y + image_height + 8
    for index, method in enumerate(METHODS):
        x = margin + index * (panel_width + gutter)
        is_flashtrace = method == "flashtrace"
        draw.text(
            (x + 8, label_y),
            METHOD_LABELS[method],
            fill=ACCENT if is_flashtrace else INK,
            font=method_bold if is_flashtrace else method_font,
        )
        overlay = _positive_overlay(image, records[method]["visual_grid"])
        if evidence_boxes:
            overlay = _wiki_evidence_outline(overlay, evidence_boxes)
        panel = _crop_and_fit(overlay, crop, (panel_width, image_height))
        canvas.paste(panel, (x, image_y))
        border = ACCENT if is_flashtrace else "#d1d5db"
        draw.rectangle(
            (x, image_y, x + panel_width - 1, image_y + image_height - 1),
            outline=border,
            width=5 if is_flashtrace else 2,
        )
        metric = _metric_text(method, metrics[method], kind=metric_kind)
        draw.text(
            (x + 8, metric_y),
            metric,
            fill=ACCENT if is_flashtrace else MUTED,
            font=metric_bold if is_flashtrace else metric_font,
        )


def render(raw_dir: Path, output: Path) -> None:
    _verify_assets(raw_dir)
    vizwiz_id = "vizwiz-lf-261"
    wiki_id = "wiki-visa-1128"
    vizwiz_records = _records_by_method(raw_dir / "vizwiz261.jsonl.gz", vizwiz_id)
    wiki_records = _records_by_method(raw_dir / "wiki1128.jsonl.gz", wiki_id)
    vizwiz_faith = _records_by_method(
        raw_dir / "vizwiz261-faith.jsonl.gz", vizwiz_id
    )
    missing = set(METHODS) - set(vizwiz_records) | set(METHODS) - set(wiki_records)
    missing |= set(METHODS) - set(vizwiz_faith)
    if missing:
        raise ValueError(f"Missing selected records: {sorted(missing)}")

    vizwiz_metrics = {
        method: float(vizwiz_faith[method]["faithfulness"]["deletion_auc"])
        for method in METHODS
    }
    wiki_metrics = {
        method: float(wiki_records[method]["localization"]["evidence_rank_auc"])
        for method in METHODS
    }

    vizwiz_image = Image.open(raw_dir / "261.jpg").convert("RGB")
    wiki_image = Image.open(raw_dir / "1128.png").convert("RGB")
    # Shared semantic crop: the complete butterfly design plus enough shirt
    # boundary to make the scene legible.
    vizwiz_crop = (600, 760, 2520, 1498)
    # Shared evidence crop: the paragraph and neighboring section context.
    wiki_crop = (0, 2665, 980, 3042)
    wiki_boxes = [[24.0, 2836.0, 941.0, 2913.0]]

    width = 2400
    height = 758
    margin = 24
    gutter = 14
    panel_width = (width - 2 * margin - 3 * gutter) // 4
    image_height = 220
    canvas = Image.new("RGB", (width, height), "white")

    _draw_row(
        canvas,
        y=10,
        title="VizWiz-LF  ·  faithfulness  ·  shared crop",
        legend="magenta = positive attribution",
        image=vizwiz_image,
        records=vizwiz_records,
        crop=vizwiz_crop,
        panel_width=panel_width,
        image_height=image_height,
        gutter=gutter,
        margin=margin,
        metrics=vizwiz_metrics,
        metric_kind="deletion",
    )
    _draw_row(
        canvas,
        y=386,
        title="Wiki-VISA  ·  localization  ·  shared evidence crop",
        legend="magenta = attribution   ·   green = evidence",
        image=wiki_image,
        records=wiki_records,
        crop=wiki_crop,
        panel_width=panel_width,
        image_height=image_height,
        gutter=gutter,
        margin=margin,
        metrics=wiki_metrics,
        metric_kind="rank",
        evidence_boxes=wiki_boxes,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, compress_level=6, dpi=(300, 300))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.raw_dir, args.output)
    print(json.dumps({"output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
