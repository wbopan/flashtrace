"""Recompute strict spatial metrics and overlays from saved raw patch grids.

This intentionally performs no model forward pass. It restores FlashTrace's
paper-defined cumulative map from saved trace metadata, then recomputes metrics
when the spatial protocol changes, for example when replacing bilinear
interpolation with nearest-patch expansion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .strict_attribution import (
    _common_summary,
    _evidence_masks,
    _save_overlay,
    _visual_grid_from_projected_scores,
    localization_metrics,
)
from .strict_generation import read_jsonl, write_jsonl


def _primary_mask(masks: dict[str, Any]) -> Any | None:
    for name in (
        "primary_unique_firstnonempty",
        "primary",
        "primary_union",
        "primary_bbox",
    ):
        if name in masks:
            return masks[name]
    return None


def _restore_paper_flashtrace_composition(record: dict[str, Any]) -> None:
    if record.get("method") != "flashtrace":
        return
    method_metadata = record["method_metadata"]
    trace_metadata = method_metadata["trace_metadata"]
    observation = trace_metadata["ifr"]["observation_projected"]
    grid = _visual_grid_from_projected_scores(
        observation["sum"], trace_metadata["multimodal"]
    )
    record["visual_grid"] = grid
    record["visual_grid_shape"] = [len(grid), len(grid[0])]
    method_metadata["attribution_composition"] = (
        "direct_plus_weighted_reasoning_hops"
    )
    method_metadata["direct_base_included"] = True


def recompute(
    *,
    dataset_manifest: Path,
    attribution_dir: Path,
    allow_missing_evidence: bool,
) -> dict[str, Any]:
    datasets = {
        record["sample_id"]: record for record in read_jsonl(dataset_manifest)
    }
    records_path = attribution_dir / "attribution_records.jsonl"
    records = read_jsonl(records_path)

    for record in records:
        if record.get("status") != "ok":
            continue
        sample_id = str(record["sample_id"])
        dataset_record = datasets[sample_id]
        _restore_paper_flashtrace_composition(record)
        try:
            masks = _evidence_masks(dataset_record)
        except ValueError:
            if not allow_missing_evidence:
                raise
            masks = {}
        primary = _primary_mask(masks)
        record["localization"] = (
            localization_metrics(record["visual_grid"], dataset_record)
            if primary is not None
            else None
        )
        image = Image.open(dataset_record["input"]["I_IMAGE"]).convert("RGB")
        _save_overlay(image, record["visual_grid"], primary, Path(record["overlay_path"]))
        record["spatial_resampling"] = "nearest_patch"
        record["spatial_metric_unit"] = "visual_patch"
        record["cutoff_tie_policy"] = "expected_uniform"

    write_jsonl(records, records_path)
    summary_path = attribution_dir / "summary.json"
    summary = json.loads(summary_path.read_text())
    methods = tuple(str(method) for method in summary["requested_methods"])
    summary = {
        **summary,
        "spatial_resampling": "nearest_patch",
        "spatial_metric_unit": "visual_patch",
        "cutoff_tie_policy": "expected_uniform",
        **_common_summary(records, methods),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--attribution-dir", type=Path, required=True)
    parser.add_argument("--allow-missing-evidence", action="store_true")
    args = parser.parse_args()
    summary = recompute(
        dataset_manifest=args.dataset_manifest,
        attribution_dir=args.attribution_dir,
        allow_missing_evidence=args.allow_missing_evidence,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
