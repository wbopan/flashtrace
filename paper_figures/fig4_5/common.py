from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


class SemanticPalette:
    """Natural semantic colors aligned with the paper overview figure."""

    blue = "#3F63A8"
    light_blue = "#79B6D2"
    green = "#6F9D4E"
    orange = "#E9904A"
    terracotta = "#B9654E"
    red = "#A23B49"
    purple = "#8A6BB5"
    gray = "#8B98A3"
    muted = "#D7DCE3"
    outline = "#343B43"

    # The TPAMI tables define cyan as #4F5BD5 and use cyan!10 for highlighted
    # rows. Mixing that color with 90% white yields this pale periwinkle.
    highlight_bg = "#EDEFFB"

    @staticmethod
    def darken(color: str, factor: float = 0.65) -> str:
        value = color.lstrip("#")
        rgb = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        return "#{:02x}{:02x}{:02x}".format(
            *[int(channel * factor * 255) for channel in rgb]
        )

    def edge(self, color: str) -> str:
        return self.darken(color, factor=0.72)


def box_axes(
    ax: plt.Axes, *, facecolor: str = "white", square: bool = False
) -> None:
    """Use a complete four-sided frame while keeping ticks on left/bottom."""

    ax.set_facecolor(facecolor)
    ax.set_axisbelow(True)
    if square:
        ax.set_box_aspect(1)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(SemanticPalette.outline)
        spine.set_linewidth(0.75)
    ax.tick_params(
        axis="both",
        which="both",
        top=False,
        right=False,
        color=SemanticPalette.outline,
        width=0.75,
    )


def configure_style() -> None:
    installed_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = "Libertinus Sans" if "Libertinus Sans" in installed_fonts else "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.color": "#C9CED6",
            "grid.alpha": 0.8,
            "grid.linewidth": 0.5,
            "axes.edgecolor": SemanticPalette.outline,
            "axes.linewidth": 0.75,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    if output.suffix.lower() != ".png":
        fig.savefig(output.with_suffix(".png"), dpi=300)
    plt.close(fig)
