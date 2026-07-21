"""Render side-by-side attribution overlays for strict multimodal runs."""

from __future__ import annotations

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from .strict_generation import read_jsonl


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _fit_image(path: Path, width: int, height: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#111827")
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def _primary_mask(record: dict[str, Any]) -> np.ndarray | None:
    mask_paths = record["evaluation"].get("EVIDENCE_MASKS") or {}
    path = mask_paths.get("primary_unique_firstnonempty")
    if not path:
        return None
    mask = np.asarray(np.load(path), dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        raise ValueError(f"Invalid primary evidence mask for {record['sample_id']}")
    return mask


def _patch_faithful_image(
    dataset: dict[str, Any],
    attribution: dict[str, Any],
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Render native attribution cells without inventing sub-patch structure."""

    image = Image.open(dataset["input"]["I_IMAGE"]).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)

    grid = np.asarray(attribution["visual_grid"], dtype=np.float32)
    if grid.ndim != 2 or not grid.size or not np.isfinite(grid).all():
        raise ValueError(
            f"Invalid attribution grid for {dataset['sample_id']} / "
            f"{attribution['method']}: {grid.shape}"
        )
    positive = np.clip(grid, 0.0, None)
    maximum = float(positive.max())
    if maximum > 0:
        positive /= maximum
    alpha = Image.fromarray(np.uint8(positive * 175), mode="L").resize(
        image.size, Image.Resampling.NEAREST
    )
    red = Image.new("RGBA", image.size, (255, 20, 20, 0))
    red.putalpha(alpha)
    overlay = Image.alpha_composite(image.convert("RGBA"), red)

    mask = _primary_mask(dataset)
    if mask is not None:
        mask_image = Image.fromarray(np.uint8(mask) * 255, mode="L").resize(
            image.size, Image.Resampling.NEAREST
        )
        radius = 5 if min(image.size) >= 200 else 3
        dilated = mask_image.filter(ImageFilter.MaxFilter(radius))
        eroded = mask_image.filter(ImageFilter.MinFilter(radius))
        outline = ImageChops.subtract(dilated, eroded)
        evidence_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        evidence_layer.paste((20, 255, 40, 255), mask=outline)
        overlay = Image.alpha_composite(overlay, evidence_layer)
    else:
        draw = ImageDraw.Draw(overlay)
        metadata = dataset["evaluation"].get("metadata") or {}
        native_size = metadata.get("image_size") or {}
        native_width = float(native_size.get("width", image.width))
        native_height = float(native_size.get("height", image.height))
        for x1, y1, x2, y2 in dataset["evaluation"].get("EVIDENCE_BOXES") or []:
            draw.rectangle(
                (
                    round(float(x1) * image.width / native_width),
                    round(float(y1) * image.height / native_height),
                    round(float(x2) * image.width / native_width),
                    round(float(y2) * image.height / native_height),
                ),
                outline=(20, 255, 40, 255),
                width=2,
            )

    # Draw the actual attribution-cell boundaries after compositing. This makes
    # the native method resolution explicit and prevents a smooth heat field
    # from being mistaken for sub-patch localization.
    draw = ImageDraw.Draw(overlay)
    rows, columns = grid.shape
    grid_color = (255, 255, 255, 72)
    for column in range(1, columns):
        x = round(column * image.width / columns)
        draw.line((x, 0, x, image.height - 1), fill=grid_color, width=1)
    for row in range(1, rows):
        y = round(row * image.height / rows)
        draw.line((0, y, image.width - 1, y), fill=grid_color, width=1)

    canvas = Image.new("RGB", (width, height), "#111827")
    canvas.paste(
        overlay.convert("RGB"),
        ((width - image.width) // 2, (height - image.height) // 2),
    )
    return canvas


def _metric(record: dict[str, Any], name: str) -> float:
    localization = record.get("localization")
    if not isinstance(localization, dict):
        return float("nan")
    value = localization.get(name)
    return float(value) if value is not None else float("nan")


def _method_panel(
    record: dict[str, Any],
    *,
    width: int,
    image_height: int,
    dataset: dict[str, Any] | None = None,
    patch_faithful: bool = False,
) -> Image.Image:
    label_height = 70
    panel = Image.new("RGB", (width, image_height + label_height), "white")
    if patch_faithful:
        if dataset is None:
            raise ValueError("dataset record is required for patch-faithful rendering")
        rendered_image = _patch_faithful_image(
            dataset, record, width=width, height=image_height
        )
    else:
        rendered_image = _fit_image(Path(record["overlay_path"]), width, image_height)
    panel.paste(rendered_image, (0, 0))
    draw = ImageDraw.Draw(panel)
    method_label = record["method"]
    if patch_faithful:
        grid = np.asarray(record["visual_grid"])
        method_label += f"  ·  native {grid.shape[0]}×{grid.shape[1]}"
    method_font_size = 13 if len(method_label) > 38 else 17
    draw.text(
        (9, image_height + 7),
        method_label,
        fill="#111827",
        font=_font(method_font_size, bold=True),
    )
    if isinstance(record.get("localization"), dict):
        metrics = (
            f"Energy {_metric(record, 'energy_in_mask'):.3f}  "
            f"RankAUC {_metric(record, 'evidence_rank_auc'):.3f}"
        )
        if patch_faithful:
            metrics = "Whole-patch eval · " + metrics
        recovery = f"Recovery@20 {_metric(record, 'recovery_at_20pct'):.3f}"
    else:
        metrics = "No native region GT"
        recovery = "Use frozen-response visual faithfulness"
    draw.text((9, image_height + 31), metrics, fill="#374151", font=_font(12))
    draw.text((9, image_height + 49), recovery, fill="#374151", font=_font(12))
    return panel


def _header(
    dataset: dict[str, Any],
    model: dict[str, Any],
    *,
    width: int,
    patch_faithful: bool = False,
) -> Image.Image:
    metadata = dataset["evaluation"].get("metadata", {})
    family = metadata.get("reasoning_family", metadata.get("stratum", ""))
    question = dataset["input"]["I_QUESTION"]
    reference = dataset["evaluation"]["REFERENCE_OUTPUT"]
    output = model.get("OUTPUT", "")
    thinking_tokens = model.get("generation_metadata", {}).get("thinking_tokens")
    title = f"{dataset['sample_id']}  ·  {family}"
    if thinking_tokens is not None:
        title += f"  ·  THINKING {thinking_tokens} tokens"
    question_lines = textwrap.wrap(question, width=max(60, width // 15))
    half_text_width = max(32, width // 30)
    reference_lines = textwrap.wrap(
        f"Reference: {reference}", width=half_text_width
    )
    output_lines = textwrap.wrap(
        f"Whole OUTPUT: {output}", width=half_text_width
    )
    comparison_lines = max(len(reference_lines), len(output_lines))
    height = 82 + 23 * len(question_lines) + 19 * comparison_lines + 38
    header = Image.new("RGB", (width, height), "#f9fafb")
    draw = ImageDraw.Draw(header)
    draw.text((14, 10), title, fill="#111827", font=_font(20, bold=True))
    y = 40
    for line in question_lines:
        draw.text((14, y), line, fill="#1f2937", font=_font(16))
        y += 23
    comparison_y = y + 5
    for index, line in enumerate(reference_lines):
        draw.text(
            (14, comparison_y + 19 * index),
            line,
            fill="#047857",
            font=_font(14, bold=True),
        )
    for index, line in enumerate(output_lines):
        draw.text(
            (width // 2, comparison_y + 19 * index),
            line,
            fill="#1d4ed8",
            font=_font(14, bold=True),
        )
    evaluation = dataset["evaluation"]
    evidence_masks = evaluation.get("EVIDENCE_MASKS") or {}
    has_evidence = bool(
        evaluation.get("EVIDENCE_MASK")
        or evaluation.get("EVIDENCE_BOXES")
        or any(evidence_masks.values())
    )
    legend = "Red = positive attribution"
    if has_evidence:
        legend = "Green outline = ground-truth evidence; " + legend.lower()
    if patch_faithful:
        legend += "; hard cells = native grid (no heatmap interpolation)"
        if has_evidence:
            legend += "; numbers below = whole-patch tie-aware metrics"
    legend_y = comparison_y + 19 * comparison_lines + 8
    draw.text((14, legend_y), legend, fill="#4b5563", font=_font(13))
    return header


def render_comparisons(
    manifest: Path,
    model_output: Path,
    attribution_dir: Path,
    output_dir: Path,
    *,
    sample_ids: list[str] | None = None,
    methods: list[str] | None = None,
    columns: int = 5,
    patch_faithful: bool = False,
    panel_width: int = 300,
    image_height: int = 220,
) -> list[dict[str, Any]]:
    datasets = {record["sample_id"]: record for record in read_jsonl(manifest)}
    models = {record["sample_id"]: record for record in read_jsonl(model_output)}
    results: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in read_jsonl(attribution_dir / "attribution_records.jsonl"):
        if record.get("status") == "ok":
            results[record["sample_id"]][record["method"]] = record

    summary = json.loads((attribution_dir / "summary.json").read_text())
    available_methods = summary["requested_methods"]
    methods = methods or available_methods
    unknown = set(methods) - set(available_methods)
    if unknown:
        raise ValueError(f"Methods are absent from attribution run: {sorted(unknown)}")
    ids = sample_ids or summary["common_sample_ids"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[dict[str, Any]] = []
    gutter = 12

    for sample_id in ids:
        available = results.get(sample_id, {})
        missing = [method for method in methods if method not in available]
        if missing:
            raise ValueError(f"{sample_id} is missing methods: {missing}")
        panels = [
            _method_panel(
                available[method],
                width=panel_width,
                image_height=image_height,
                dataset=datasets[sample_id],
                patch_faithful=patch_faithful,
            )
            for method in methods
        ]
        rows = (len(panels) + columns - 1) // columns
        grid_width = columns * panel_width + (columns - 1) * gutter
        header = _header(
            datasets[sample_id],
            models[sample_id],
            width=grid_width,
            patch_faithful=patch_faithful,
        )
        grid_height = rows * panels[0].height + max(0, rows - 1) * gutter
        canvas = Image.new("RGB", (grid_width, header.height + gutter + grid_height), "#d1d5db")
        canvas.paste(header, (0, 0))
        for index, panel in enumerate(panels):
            row, column = divmod(index, columns)
            canvas.paste(
                panel,
                (
                    column * (panel_width + gutter),
                    header.height + gutter + row * (panel.height + gutter),
                ),
            )
        suffix = ".png" if patch_faithful else ".jpg"
        path = output_dir / f"{sample_id}{suffix}"
        if patch_faithful:
            canvas.save(path, compress_level=6)
        else:
            canvas.save(path, quality=94)
        rendered.append({"sample_id": sample_id, "path": str(path)})

    (output_dir / "index.json").write_text(json.dumps(rendered, indent=2) + "\n")
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--method", action="append", dest="methods")
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--patch-faithful", action="store_true")
    parser.add_argument("--panel-width", type=int, default=300)
    parser.add_argument("--image-height", type=int, default=220)
    args = parser.parse_args()
    rendered = render_comparisons(
        args.manifest,
        args.model_output,
        args.attribution_dir,
        args.output_dir,
        sample_ids=args.sample_ids,
        methods=args.methods,
        columns=args.columns,
        patch_faithful=args.patch_faithful,
        panel_width=args.panel_width,
        image_height=args.image_height,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(rendered)}, indent=2))


if __name__ == "__main__":
    main()
