"""Render stored multimodal attribution maps at their native visual-token grids."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

if __package__:
    from .run_smoke import make_overlay
else:
    from run_smoke import make_overlay


METHOD_ORDER = (
    "ifr-tokenwise",
    "ifr-span",
    "attention-rollout",
    "grad-attention",
    "visual-ig",
    "flashtrace",
    "tam",
    "attnlrp",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _thumbnail(path: Path, max_side: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    return image


def _font(size: int = 19) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _comparison_sheet(
    panels: list[tuple[str, Image.Image]], destination: Path
) -> None:
    if not panels:
        raise ValueError("Expected at least one comparison panel")
    width, height = panels[0][1].size
    if any(panel.size != (width, height) for _, panel in panels):
        raise ValueError("Comparison panels must have identical dimensions")
    label_height = 38
    gutter = 10
    columns = math.ceil(math.sqrt(len(panels)))
    rows = math.ceil(len(panels) / columns)
    canvas = Image.new(
        "RGB",
        (
            columns * width + (columns - 1) * gutter,
            rows * (height + label_height) + (rows - 1) * gutter,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = _font()
    for index, (label, panel) in enumerate(panels):
        row, column = divmod(index, columns)
        left = column * (width + gutter)
        top = row * (height + label_height + gutter)
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (left + (width - text_width) // 2, top + 7),
            label,
            fill="black",
            font=font,
        )
        canvas.paste(panel, (left, top + label_height))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=95)


def render_native_maps(
    references_path: Path,
    methods_path: Path,
    *,
    eval_root: Path,
    output_dir: Path,
    max_image_side: int = 448,
) -> dict[str, Any]:
    references = _read_jsonl(references_path)
    methods = _read_jsonl(methods_path)
    reference_by_key = {
        (row["dataset"], str(row["question_id"])): row for row in references
    }
    methods_by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in methods:
        if row.get("status") != "ok" or row.get("method") not in METHOD_ORDER:
            continue
        key = (row["dataset"], str(row["question_id"]))
        methods_by_key.setdefault(key, {})[row["method"]] = row
    available_methods = tuple(
        method
        for method in METHOD_ORDER
        if any(method in rows for rows in methods_by_key.values())
    )

    rendered = []
    for key, reference in reference_by_key.items():
        method_rows = methods_by_key.get(key, {})
        if not available_methods or any(
            method not in method_rows for method in available_methods
        ):
            continue
        dataset, question_id = key
        image = _thumbnail(eval_root / reference["image_path"], max_image_side)
        loo_grid = reference["visual_loo"]["grid"]
        panels = [("Visual LOO (4x4)", make_overlay(image, loo_grid))]
        native_paths = {}
        for method in available_methods:
            grid = method_rows[method]["visual_grid"]
            rows, columns = len(grid), len(grid[0])
            overlay = make_overlay(image, grid)
            path = output_dir / "native_overlays" / method / dataset / f"{question_id}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            overlay.save(path, quality=95)
            native_paths[method] = str(path)
            panels.append((f"{method} ({rows}x{columns})", overlay))
        comparison_path = output_dir / "native_comparisons" / dataset / f"{question_id}.jpg"
        _comparison_sheet(panels, comparison_path)
        rendered.append(
            {
                "dataset": dataset,
                "question_id": question_id,
                "question": reference["question"],
                "response": reference["response"],
                "native_overlays": native_paths,
                "comparison": str(comparison_path),
            }
        )

    manifest = {
        "samples": len(rendered),
        "methods": list(available_methods),
        "native_overlays": len(rendered) * len(available_methods),
        "comparisons": len(rendered),
        "records": rendered,
    }
    manifest_path = output_dir / "native_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("data/multimodal_smoke_final/results.jsonl"),
    )
    parser.add_argument(
        "--method-results",
        type=Path,
        default=Path("data/multimodal_methods_final/results.jsonl"),
    )
    parser.add_argument("--eval-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/multimodal_methods_final")
    )
    parser.add_argument("--max-image-side", type=int, default=448)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = render_native_maps(
        args.reference_results,
        args.method_results,
        eval_root=args.eval_root,
        output_dir=args.output_dir,
        max_image_side=args.max_image_side,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
