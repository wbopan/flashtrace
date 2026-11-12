from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from common import OUTPUT_DIR, SemanticPalette, box_axes, configure_style, save_figure


def region_weight(values: np.ndarray, span: np.ndarray) -> float:
    total = float(values.sum())
    if total == 0:
        return 0.0
    start, end = [int(value) for value in np.asarray(span).reshape(-1)]
    return float(values[start : end + 1].sum() / total)


def load_records(input_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(input_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        if "hop" not in data.files:
            continue
        hops = np.asarray(data["hop"])
        if hops.shape[0] < 4:
            continue
        input_mass = region_weight(hops[0], data["span_in"])
        cot_mass = region_weight(hops[0], data["span_cot"])
        rows.extend(
            [
                {"hop": 0, "region": "Input Tokens", "weight": input_mass},
                {"hop": 0, "region": "CoT Tokens", "weight": cot_mass},
            ]
        )
        for hop_index in range(1, 4):
            hop_input = region_weight(hops[hop_index], data["span_in"])
            hop_cot = region_weight(hops[hop_index], data["span_cot"])
            input_mass = input_mass + hop_input * cot_mass
            cot_mass = hop_cot * cot_mass
            rows.extend(
                [
                    {"hop": hop_index, "region": "Input Tokens", "weight": input_mass},
                    {"hop": hop_index, "region": "CoT Tokens", "weight": cot_mass},
                ]
            )
    if not rows:
        raise FileNotFoundError(
            f"No usable four-hop NPZ records in {input_dir}. Expected the original MoreHopQA n3 directory."
        )
    return pd.DataFrame(rows)


def build(input_dir: Path, output: Path) -> None:
    frame = load_records(input_dir)
    configure_style()
    palette = SemanticPalette()
    cot_color, input_color = palette.orange, palette.blue
    cot_border, input_border = palette.edge(cot_color), palette.edge(input_color)
    figure, ax = plt.subplots(figsize=(3.25, 3.25 * 0.5))
    for region, color, border in (
        ("CoT Tokens", cot_color, cot_border),
        ("Input Tokens", input_color, input_border),
    ):
        sns.violinplot(
            data=frame[frame["region"] == region],
            x="hop",
            y="weight",
            color=color,
            linewidth=0.7,
            linecolor=border,
            inner="quart",
            ax=ax,
        )
    box_axes(ax)
    ax.set_xlabel("Recursive Hops", fontweight="bold")
    ax.set_ylabel("Attribution Weight", fontweight="bold")
    ax.set_ylim(0, None)
    ax.annotate(
        "Input tokens take majority\nof attr. weight after 3 hops",
        xy=(3, 0.65),
        xytext=(0.8, 0.75),
        fontsize=6,
        bbox={
            "boxstyle": "square,pad=0.3",
            "facecolor": palette.highlight_bg,
            "edgecolor": palette.purple,
        },
        arrowprops={
            "arrowstyle": "wedge",
            "color": palette.purple,
            "connectionstyle": "arc3,rad=-0.2",
        },
    )
    ax.annotate(
        "",
        xy=(3, 0.70),
        xytext=(0, 0.30),
        arrowprops={
            "arrowstyle": "->,head_width=0.4,head_length=0.4",
            "color": input_color,
            "connectionstyle": "arc3,rad=0.05",
            "linewidth": 1.2,
            "linestyle": "--",
        },
    )
    figure.legend(
        handles=[
            Patch(facecolor=cot_color, edgecolor=cot_border, label="CoT Tokens"),
            Patch(facecolor=input_color, edgecolor=input_border, label="Input Tokens"),
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.07),
    )
    figure.tight_layout()
    save_figure(figure, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "dist_fig.pdf")
    args = parser.parse_args()
    build(args.input_dir, args.output)
    print(f"Wrote {args.output} and {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
