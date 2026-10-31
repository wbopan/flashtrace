import json
import textwrap

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import FancyBboxPatch


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

GEN_PATH = "exp_for_span-to-span/results/case1_generation.json"
MATRIX_PATH = "exp_for_span-to-span/results/case1_span_matrix.json"

OUT_PATH = "exp_for_span-to-span/results/case1_recursive_span_figure.png"


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

with open(GEN_PATH, "r", encoding="utf-8") as f:
    generation = json.load(f)

with open(MATRIX_PATH, "r", encoding="utf-8") as f:
    matrix_data = json.load(f)


spans = matrix_data["reasoning_spans"]
labels = matrix_data["labels"]

n = len(spans)


# rows = target
# columns = source
A = np.array(
    [
        [np.nan if value is None else float(value) for value in row]
        for row in matrix_data["normalized_matrix"]
    ],
    dtype=float,
)


# ---------------------------------------------------------
# Hop 1:
# direct reasoning-span -> final-output attribution
# ---------------------------------------------------------

hop1 = A[-1, :n].copy()

hop1 = np.nan_to_num(hop1, nan=0.0)

if hop1.sum() > 0:
    hop1 = hop1 / hop1.sum()


# ---------------------------------------------------------
# Hop 2:
# recursively propagate Output attribution backward
#
# A[j, i] = attribution S_i -> S_j
#
# If S_j strongly contributes to O,
# and S_i strongly contributes to S_j,
# then S_i receives recursive attribution.
# ---------------------------------------------------------

hop2 = np.zeros(n, dtype=float)

for target_j in range(n):

    target_weight = hop1[target_j]

    if target_weight == 0:
        continue

    # Only spans before S_j can causally contribute to S_j.
    for source_i in range(target_j):

        edge = A[target_j, source_i]

        if np.isfinite(edge):
            hop2[source_i] += target_weight * edge


if hop2.sum() > 0:
    hop2 = hop2 / hop2.sum()


# ---------------------------------------------------------
# Difference
# ---------------------------------------------------------

delta = hop2 - hop1


# ---------------------------------------------------------
# Helper for drawing one panel
# ---------------------------------------------------------

def draw_panel(
    ax,
    title,
    scores,
    cmap,
    norm,
    show_target=True,
):
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.axis("off")

    y = 0.98

    # ---------------------------------------------
    # INPUT
    # ---------------------------------------------

    ax.text(
        0.01,
        y,
        "INPUT",
        fontsize=9,
        fontweight="bold",
        color="gray",
        transform=ax.transAxes,
    )

    y -= 0.035

    prompt = generation["prompt"]

    wrapped_prompt = textwrap.fill(
        prompt,
        width=70,
    )

    ax.text(
        0.01,
        y,
        wrapped_prompt,
        fontsize=7.5,
        va="top",
        transform=ax.transAxes,
    )

    # estimate vertical space used
    prompt_lines = len(wrapped_prompt.split("\n"))

    y -= 0.022 * prompt_lines + 0.035


    # ---------------------------------------------
    # REASONING
    # ---------------------------------------------

    ax.text(
        0.01,
        y,
        "REASONING",
        fontsize=9,
        fontweight="bold",
        color="gray",
        transform=ax.transAxes,
    )

    y -= 0.025

    for i, span in enumerate(spans):

        text = f"{span['id']}: {span['text']}"

        wrapped = textwrap.fill(
            text,
            width=65,
        )

        num_lines = max(
            1,
            len(wrapped.split("\n")),
        )

        height = 0.019 * num_lines + 0.009

        rgba = cmap(norm(scores[i]))

        # soften color so text stays readable
        facecolor = (
            rgba[0],
            rgba[1],
            rgba[2],
            0.55,
        )

        box = FancyBboxPatch(
            (0.01, y - height),
            0.97,
            height,
            boxstyle="round,pad=0.003",
            linewidth=0,
            facecolor=facecolor,
            transform=ax.transAxes,
            clip_on=False,
        )

        ax.add_patch(box)

        ax.text(
            0.018,
            y - 0.005,
            wrapped,
            fontsize=7.2,
            va="top",
            transform=ax.transAxes,
        )

        y -= height + 0.004


    # ---------------------------------------------
    # OUTPUT
    # ---------------------------------------------

    y -= 0.015

    ax.text(
        0.01,
        y,
        "OUTPUT",
        fontsize=9,
        fontweight="bold",
        color="gray",
        transform=ax.transAxes,
    )

    y -= 0.025

    final_text = generation["final_text"]

    wrapped_output = textwrap.fill(
        final_text,
        width=67,
    )

    output_lines = len(wrapped_output.split("\n"))
    output_height = 0.019 * output_lines + 0.01

    if show_target:

        target_box = FancyBboxPatch(
            (0.01, y - output_height),
            0.97,
            output_height,
            boxstyle="round,pad=0.005",
            linewidth=1.8,
            edgecolor="orange",
            facecolor="none",
            transform=ax.transAxes,
            clip_on=False,
        )

        ax.add_patch(target_box)

    ax.text(
        0.018,
        y - 0.005,
        wrapped_output,
        fontsize=7.2,
        va="top",
        transform=ax.transAxes,
    )


# ---------------------------------------------------------
# Colormaps
# ---------------------------------------------------------

max_attr = max(
    float(hop1.max()),
    float(hop2.max()),
)

attr_norm = colors.Normalize(
    vmin=0.0,
    vmax=max_attr if max_attr > 0 else 1.0,
)

attr_cmap = cm.Blues


max_delta = max(
    abs(float(delta.min())),
    abs(float(delta.max())),
)

if max_delta == 0:
    max_delta = 1.0

delta_norm = colors.TwoSlopeNorm(
    vmin=-max_delta,
    vcenter=0.0,
    vmax=max_delta,
)

# negative = red
# positive = green
delta_cmap = cm.RdYlGn


# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(23, 13),
)

draw_panel(
    axes[0],
    "Hop 1\nDirect Span Attribution",
    hop1,
    attr_cmap,
    attr_norm,
)

draw_panel(
    axes[1],
    "Hop 2\nRecursive Span Attribution",
    hop2,
    attr_cmap,
    attr_norm,
)

draw_panel(
    axes[2],
    "Δ (Hop 2 − Hop 1)",
    delta,
    delta_cmap,
    delta_norm,
    show_target=False,
)


# ---------------------------------------------------------
# Color bars
# ---------------------------------------------------------

attr_scalar = cm.ScalarMappable(
    norm=attr_norm,
    cmap=attr_cmap,
)

attr_scalar.set_array([])

cbar1 = fig.colorbar(
    attr_scalar,
    ax=axes[:2],
    fraction=0.015,
    pad=0.015,
)

cbar1.set_label(
    "Span Attribution",
    fontsize=10,
)


delta_scalar = cm.ScalarMappable(
    norm=delta_norm,
    cmap=delta_cmap,
)

delta_scalar.set_array([])

cbar2 = fig.colorbar(
    delta_scalar,
    ax=axes[2],
    fraction=0.025,
    pad=0.015,
)

cbar2.set_label(
    "Δ Attribution",
    fontsize=10,
)


# ---------------------------------------------------------
# Overall title
# ---------------------------------------------------------

fig.suptitle(
    "Recursive Span-to-Span Attribution — Case Study 1",
    fontsize=20,
    fontweight="bold",
    y=0.995,
)

plt.subplots_adjust(
    top=0.94,
    bottom=0.03,
    left=0.025,
    right=0.94,
    wspace=0.08,
)

plt.savefig(
    OUT_PATH,
    dpi=220,
    bbox_inches="tight",
)

print(f"Saved: {OUT_PATH}")


# ---------------------------------------------------------
# Print numerical results for sanity check
# ---------------------------------------------------------

print("\n===== HOP 1 =====")

for i in np.argsort(hop1)[::-1][:8]:
    print(
        f"{labels[i]}: "
        f"{hop1[i]:.4f} | "
        f"{spans[i]['text']}"
    )


print("\n===== HOP 2 =====")

for i in np.argsort(hop2)[::-1][:8]:
    print(
        f"{labels[i]}: "
        f"{hop2[i]:.4f} | "
        f"{spans[i]['text']}"
    )


print("\n===== BIGGEST POSITIVE Δ =====")

for i in np.argsort(delta)[::-1][:8]:
    print(
        f"{labels[i]}: "
        f"{delta[i]:+.4f} | "
        f"{spans[i]['text']}"
    )