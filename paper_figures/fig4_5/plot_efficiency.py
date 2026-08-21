from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, NullFormatter

from common import (
    DATA_DIR,
    OUTPUT_DIR,
    SemanticPalette,
    box_axes,
    configure_style,
    load_json,
    save_figure,
)


METHODS = [
    "IG",
    "IG-Attn",
    "Perturbation",
    "REAGENT",
    "IFR",
    "AttnLRP",
    "CLP",
    "FlashTrace",
]
MARKERS = {
    "IG": "s",
    "IG-Attn": "^",
    "Perturbation": "v",
    "REAGENT": "D",
    "IFR": "p",
    "AttnLRP": "h",
    "CLP": "X",
    "FlashTrace": "o",
}
OOM_COLOR = SemanticPalette.red


def time_formatter(value: float, _: Any) -> str:
    if value < 60:
        return f"{value:.0f}s" if value >= 1 else f"{value:.1f}s"
    if value < 3600:
        return f"{value / 60:.0f}m"
    return f"{value / 3600:.0f}h"


def display_lengths(values: list[int]) -> list[str]:
    return [str(value) if value < 100 else f"{value / 10 ** int(np.log10(value)):.0f}e{int(np.log10(value))}" for value in values]


def plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    values: list[float | str | None],
    *,
    color: str,
    label: str,
    marker: str,
    oom_value: float,
) -> None:
    valid_x: list[float] = []
    valid_y: list[float] = []
    oom_index = None
    for index, value in enumerate(values):
        if value is None:
            break
        if value == "oom":
            oom_index = index
            break
        valid_x.append(float(x[index]))
        valid_y.append(float(value))
    ax.plot(
        valid_x,
        valid_y,
        color=color,
        label=label,
        marker=marker,
        markersize=3,
        linewidth=1,
    )
    if oom_index is not None and valid_x:
        ax.plot(
            [valid_x[-1], float(x[oom_index])],
            [valid_y[-1], oom_value],
            color=color,
            linewidth=1,
            linestyle="--",
            alpha=0.7,
        )
        if label in {"IG", "Perturbation"}:
            ax.annotate(
                "OOM",
                xy=(x[oom_index], oom_value),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=4,
                color=OOM_COLOR,
                ha="center",
            )


def build(data_path: Path, output: Path) -> None:
    data = load_json(data_path)
    configure_style()
    palette = SemanticPalette()
    colors = {
        "IG": palette.gray,
        "IG-Attn": palette.light_blue,
        "Perturbation": palette.orange,
        "REAGENT": palette.terracotta,
        "IFR": palette.purple,
        "AttnLRP": palette.blue,
        "CLP": palette.red,
        "FlashTrace": palette.green,
    }
    x = np.arange(len(data["lengths"]))
    labels = display_lengths(data["lengths"])
    figure, axes = plt.subplots(1, 5, figsize=(6.75, 2.0))

    panels = [
        ("time_input_seconds", "(a) Time vs Input", "Input Length", "time"),
        ("time_generation_seconds", "(b) Time vs Gen", "Generation Length", "time"),
        ("memory_input_gb", "(c) Mem vs Input", "Input Length", "memory"),
        ("memory_generation_gb", "(d) Mem vs Gen", "Generation Length", "memory"),
    ]
    for ax, (series_name, title, xlabel, kind) in zip(axes[:4], panels):
        box_axes(ax, square=True)
        for method in METHODS:
            plot_series(
                ax,
                x,
                data["series"][series_name][method],
                color=colors[method],
                label=method,
                marker=MARKERS[method],
                oom_value=40000 if kind == "time" else 500,
            )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_xticks(x, labels, rotation=45, ha="right")
        ax.set_yscale("log")
        ax.xaxis.grid(False)
        if kind == "time":
            ax.set_ylim(0.1, 50000)
            ax.set_yticks([1, 10, 60, 600, 3600])
            ax.yaxis.set_major_formatter(FuncFormatter(time_formatter))
        else:
            ax.set_ylim(25, 550)
            ax.set_yticks([30, 50, 100, 200, 400])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}GB"))
        ax.yaxis.set_minor_formatter(NullFormatter())

    pareto_ax = axes[4]
    box_axes(pareto_ax, facecolor=palette.highlight_bg, square=True)
    for method in METHODS:
        record = data["pareto_legacy"][method]
        pareto_ax.scatter(
            record["speed_normalized"],
            record["faithfulness"],
            color=colors[method],
            marker=MARKERS[method],
            s=18,
            zorder=3,
        )
    ours = data["pareto_legacy"]["FlashTrace"]
    pareto_ax.axvline(ours["speed_normalized"], color=palette.green, linestyle="--", alpha=0.5)
    pareto_ax.axhline(ours["faithfulness"], color=palette.green, linestyle="--", alpha=0.5)
    pareto_ax.set_title("(e) Speed vs Faith", fontweight="bold")
    pareto_ax.set_xlabel("Speed (norm.)", fontweight="bold")
    pareto_ax.set_xlim(0, 1.05)
    pareto_ax.set_ylim(0, 1.0)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 0.98))
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "efficiency.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "cost_comparison.pdf")
    args = parser.parse_args()
    build(args.data, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.png')}")
    print("WARNING: the legacy panel includes interpolated memory points and placeholder/hand-normalized Pareto values; see efficiency.json.")


if __name__ == "__main__":
    main()
