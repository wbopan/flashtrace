from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    DATA_DIR,
    OUTPUT_DIR,
    SemanticPalette,
    box_axes,
    configure_style,
    load_json,
    save_figure,
)


def build(data_path: Path, output: Path) -> None:
    data = load_json(data_path)
    samples = data["samples"]
    scaling = data["scaling"]
    if scaling["status"] != "synthetic":
        raise ValueError("The legacy scaling panel is expected to be explicitly marked synthetic.")

    configure_style()
    palette = SemanticPalette()
    figure, (weight_ax, recovery_ax, time_ax) = plt.subplots(
        1, 3, figsize=(3.25, 1.55)
    )

    labels = [sample["label"] for sample in samples]
    output_mass = np.asarray([sample["output_mass"] for sample in samples])
    other_mass = np.asarray(
        [sample["input_mass"] + sample["cot_mass"] for sample in samples]
    )
    total_mass = output_mass + other_mass
    output_fraction = output_mass / total_mass
    other_fraction = other_mass / total_mass
    positions = np.arange(len(samples))

    box_axes(weight_ax, square=True)
    weight_ax.bar(
        positions,
        output_fraction,
        0.6,
        color=palette.blue,
        edgecolor=palette.edge(palette.blue),
        linewidth=1,
    )
    weight_ax.bar(
        positions,
        other_fraction,
        0.6,
        bottom=output_fraction,
        color=palette.highlight_bg,
        edgecolor=palette.purple,
        linewidth=1,
    )
    for index, fraction in enumerate(output_fraction):
        weight_ax.text(
            index,
            fraction / 2,
            f"{int(fraction * 100)}%",
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
        )
    weight_ax.set_xticks(positions, labels)
    weight_ax.set_ylabel("Attr. Weight (%)", fontweight="bold")
    weight_ax.set_ylim(0, 0.4)
    weight_ax.xaxis.grid(False)

    box_axes(recovery_ax, square=True)
    recovery = np.asarray([sample["recovery_at_10pct"] for sample in samples])
    recovery_ax.bar(
        positions,
        recovery,
        0.6,
        color=[palette.blue, palette.orange],
        edgecolor=[palette.edge(palette.blue), palette.edge(palette.orange)],
        linewidth=1,
    )
    for index, rate in enumerate(recovery):
        recovery_ax.text(
            index,
            rate / 2,
            f"{int(rate * 100)}%",
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
        )
    recovery_ax.set_xticks(positions, labels)
    recovery_ax.set_ylabel("Recovery Rate (%)", fontweight="bold")
    recovery_ax.set_ylim(0, 0.35)
    recovery_ax.xaxis.grid(False)

    box_axes(time_ax, square=True)
    lengths = np.asarray(scaling["context_lengths"])
    time_ax.plot(
        lengths,
        scaling["baseline_seconds"],
        color=palette.blue,
        marker="d",
        markersize=2.5,
        linewidth=1,
    )
    time_ax.plot(
        lengths,
        scaling["flashtrace_seconds"],
        color=palette.green,
        marker="o",
        markersize=2.5,
        linewidth=1,
    )
    time_ax.text(100, 200, "  Baseline", color=palette.blue, fontsize=6, va="center")
    time_ax.text(
        2000, 50, "  FlashTrace", color=palette.green, fontsize=6, va="center"
    )
    time_ax.set_ylabel("Time (s)", fontweight="bold")
    ticks = [100, 2000, 5000, 10000]
    time_ax.set_xticks(ticks, [f"{tick:,}" for tick in ticks], rotation=45, ha="right")
    time_ax.xaxis.grid(False)

    figure.tight_layout()
    save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_DIR / "pilot.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "time_plot.pdf")
    args = parser.parse_args()
    build(args.data, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.png')}")
    print("WARNING: the right-hand scaling panel is synthetic legacy data, not a runtime benchmark.")


if __name__ == "__main__":
    main()
