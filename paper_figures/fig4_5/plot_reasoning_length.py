from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import OUTPUT_DIR, SemanticPalette, box_axes, configure_style, save_figure


def load_method(directory: Path) -> list[dict[str, float | str]]:
    records = []
    for path in sorted(directory.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        span = np.asarray(data["span_cot"]).reshape(-1)
        records.append(
            {
                "id": path.stem,
                "cot_len": int(span[1] - span[0] + 1),
                "mas": float(data["mas"]),
            }
        )
    return records


def average_methods(base: Path, names: list[str]) -> list[dict[str, float | str]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for name in names:
        for record in load_method(base / name):
            grouped[str(record["id"])].append(record)
    return [
        {
            "id": sample_id,
            "cot_len": int(records[0]["cot_len"]),
            "mas": float(np.mean([float(record["mas"]) for record in records])),
        }
        for sample_id, records in sorted(grouped.items())
    ]


def percentile_bins(records: list[dict[str, float | str]], count: int = 5) -> list[tuple[int, int]]:
    if not records:
        raise ValueError("Cannot compute bins from an empty record set.")
    lengths = [int(record["cot_len"]) for record in records]
    thresholds = np.percentile(lengths, np.linspace(0, 100, count + 1))
    return [
        (int(thresholds[index]), int(thresholds[index + 1]) + 1)
        for index in range(count)
    ]


def binned_inverse_mas(
    records: list[dict[str, float | str]], bins: list[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    values: dict[int, list[float]] = defaultdict(list)
    for record in records:
        for index, (lower, upper) in enumerate(bins):
            if lower <= int(record["cot_len"]) < upper:
                values[index].append(float(record["mas"]))
                break
    means = []
    errors = []
    for index in range(len(bins)):
        current = np.asarray(values[index], dtype=float)
        if current.size < 2:
            means.append(np.nan)
            errors.append(np.nan)
            continue
        mas_mean = float(current.mean())
        mas_sem = float(current.std(ddof=0) / np.sqrt(current.size))
        means.append(1.0 / mas_mean)
        errors.append(mas_sem / mas_mean**2)
    return np.asarray(means), np.asarray(errors)


def bin_labels(bins: list[tuple[int, int]]) -> list[str]:
    return [
        f">{lower}" if index == len(bins) - 1 else f"{lower}-{upper - 1}"
        for index, (lower, upper) in enumerate(bins)
    ]


def require_records(name: str, records: list[dict[str, float | str]], source: Path) -> None:
    if not records:
        raise FileNotFoundError(f"No NPZ records for {name}: {source}")


def build(data_root: Path, output: Path) -> None:
    morehop_base = data_root / "morehopqa" / "qwen-8B"
    morehop = {
        "FlashTrace": load_method(morehop_base / "ifr_multi_hop_both_n1"),
        "AttnLRP": load_method(morehop_base / "attnlrp"),
        "Perturbation": average_methods(
            morehop_base,
            ["perturbation_all", "perturbation_CLP", "perturbation_REAGENT"],
        ),
    }
    for name, records in morehop.items():
        require_records(f"MoreHopQA/{name}", records, morehop_base)

    variable_tracking = {name: [] for name in morehop}
    for dataset in ("vt_h2_c3", "vt_h4_c1", "vt_h10_c1.jsonl"):
        base = data_root / dataset / "qwen-8B"
        variable_tracking["FlashTrace"].extend(load_method(base / "ifr_multi_hop_both_n1"))
        variable_tracking["AttnLRP"].extend(load_method(base / "attnlrp"))
        variable_tracking["Perturbation"].extend(
            average_methods(
                base,
                ["perturbation_all_fast", "perturbation_CLP_fast", "perturbation_REAGENT_fast"],
            )
        )
    for name, records in variable_tracking.items():
        require_records(f"VariableTracking/{name}", records, data_root)

    configure_style()
    palette = SemanticPalette()
    colors = {
        "FlashTrace": palette.green,
        "AttnLRP": palette.blue,
        "Perturbation": palette.orange,
    }
    markers = {"FlashTrace": "o", "AttnLRP": "^", "Perturbation": "s"}
    figure, axes = plt.subplots(1, 2, figsize=(3.25, 2.05))
    for ax, title, records_by_method in (
        (axes[0], "MoreHopQA", morehop),
        (axes[1], "Variable Tracking", variable_tracking),
    ):
        box_axes(ax, square=True)
        bins = percentile_bins(records_by_method["FlashTrace"])
        x = np.arange(len(bins))
        for method in ("FlashTrace", "AttnLRP", "Perturbation"):
            means, errors = binned_inverse_mas(records_by_method[method], bins)
            ax.errorbar(
                x,
                means,
                yerr=errors,
                color=colors[method],
                label=method,
                marker=markers[method],
                markersize=3,
                linewidth=1,
                capsize=2,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(x, bin_labels(bins), rotation=15, ha="right")
        ax.set_xlabel("CoT Length (tokens)", fontweight="bold")
        ax.xaxis.grid(False)
    axes[0].set_ylabel("1/MAS (higher is better)", fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "cot_faithfulness.pdf")
    args = parser.parse_args()
    build(args.data_root, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
