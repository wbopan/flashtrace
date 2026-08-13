import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# =========================================================
# Paths
# =========================================================

BASE_DIR = Path(
    "/home/jgwak1/LLM_MCTS_Proj/"
    "circuit-span-to-span/flashtrace"
)

MATRIX_PATH = (
    BASE_DIR
    / "exp_for_span-to-span/results/case1_span_matrix.json"
)

ROUTES_PATH = (
    BASE_DIR
    / "exp_for_span-to-span/results/case1_flow_routes.json"
)

OUT_PATH = (
    BASE_DIR
    / "exp_for_span-to-span/results/"
    "case1_top10_case_study_walkthrough_slide.png"
)

TOP_K = 10


# =========================================================
# Load
# =========================================================

with MATRIX_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    matrix_data = json.load(f)

with ROUTES_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    route_data = json.load(f)


spans = matrix_data["reasoning_spans"]

span_ids = [
    span["id"]
    for span in spans
]

span_index = {
    sid: i
    for i, sid in enumerate(span_ids)
}


# =========================================================
# Resolve route list
# =========================================================

def find_route_list(data):

    for key in [
        "multi_hop_routes",
        "selected_routes",
        "final_routes",
        "routes",
        "all_routes",
    ]:

        value = data.get(key)

        if isinstance(value, list):
            return value

    raise RuntimeError(
        "Could not find route list. "
        f"Available keys: {list(data.keys())}"
    )


all_routes = find_route_list(
    route_data
)


def reasoning_span_count(route):

    return len(
        [
            node
            for node in route.get(
                "route",
                [],
            )
            if node.startswith("S")
        ]
    )


# Multi-hop = at least two reasoning spans.
multi_hop_routes = [
    route
    for route in all_routes
    if reasoning_span_count(route) >= 2
]


multi_hop_routes = sorted(
    multi_hop_routes,
    key=lambda route: float(
        route.get("flow", 0.0)
    ),
    reverse=True,
)


top_routes = (
    multi_hop_routes[:TOP_K]
)


if len(top_routes) < TOP_K:
    raise RuntimeError(
        f"Only {len(top_routes)} multi-hop routes found; "
        f"requested TOP_K={TOP_K}."
    )


for rank, route in enumerate(
    top_routes,
    start=1,
):
    route["_display_id"] = f"M{rank}"


# =========================================================
# Colors
# =========================================================

default_colors = (
    plt.rcParams[
        "axes.prop_cycle"
    ]
    .by_key()["color"]
)


route_colors = {
    route["_display_id"]:
    default_colors[
        i % len(default_colors)
    ]
    for i, route in enumerate(
        top_routes
    )
}


# =========================================================
# Helper: compact span text
# =========================================================

def compact_span_text(
    sid,
    text,
    width=48,
):

    wrapped = textwrap.wrap(
        f"{sid}: {text}",
        width=width,
    )

    if len(wrapped) <= 2:
        return "\n".join(wrapped)

    return (
        wrapped[0]
        + "\n"
        + wrapped[1].rstrip(" .")
        + " ..."
    )


# =========================================================
# Helper: draw route tag
# =========================================================

def draw_route_tag(
    ax,
    x,
    y,
    rid,
):

    color = route_colors[rid]

    w = 0.041
    h = 0.020

    tag = FancyBboxPatch(
        (
            x - w / 2,
            y - h / 2,
        ),
        w,
        h,
        boxstyle="round,pad=0.0015",
        linewidth=0,
        facecolor=color,
        transform=ax.transAxes,
        clip_on=False,
        zorder=5,
    )

    ax.add_patch(tag)

    ax.text(
        x,
        y,
        rid,
        fontsize=5.1,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        transform=ax.transAxes,
        zorder=6,
    )


# =========================================================
# Helper: draw one circuit card
# =========================================================

def draw_circuit_card(
    ax,
    route,
):

    ax.set_xlim(
        0.0,
        1.0,
    )

    ax.set_ylim(
        0.0,
        1.0,
    )

    ax.axis("off")


    rid = route["_display_id"]

    flow = float(
        route.get(
            "flow",
            0.0,
        )
    )

    path = route["route"]

    color = route_colors[rid]


    # ---------------------------------------------
    # Card
    # ---------------------------------------------

    card = FancyBboxPatch(
        (
            0.01,
            0.04,
        ),
        0.98,
        0.91,
        boxstyle="round,pad=0.010",
        fill=False,
        linewidth=0.9,
        edgecolor="#AAAAAA",
        transform=ax.transAxes,
    )

    ax.add_patch(card)


    # ---------------------------------------------
    # Header
    # ---------------------------------------------

    ax.text(
        0.035,
        0.84,
        rid,
        fontsize=9.0,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    ax.text(
        0.965,
        0.84,
        f"{flow:.4f}",
        fontsize=6.8,
        color="#333333",
        ha="right",
        va="center",
        transform=ax.transAxes,
    )


    # ---------------------------------------------
    # Nodes
    # ---------------------------------------------

    n_nodes = len(path)

    xs = [
        0.08
        + i
        * (
            0.84
            / max(
                n_nodes - 1,
                1,
            )
        )
        for i in range(n_nodes)
    ]

    y = 0.43


    # ---------------------------------------------
    # Directed circuit edges
    # ---------------------------------------------

    for i in range(
        n_nodes - 1
    ):

        arrow = FancyArrowPatch(
            (
                xs[i] + 0.026,
                y,
            ),
            (
                xs[i + 1] - 0.026,
                y,
            ),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.9,
            color=color,
            alpha=0.88,
            shrinkA=0,
            shrinkB=0,
            transform=ax.transAxes,
            zorder=2,
        )

        ax.add_patch(arrow)


    # ---------------------------------------------
    # Node boxes
    # ---------------------------------------------

    for x, node in zip(
        xs,
        path,
    ):

        is_terminal = (
            node in {"Q", "O"}
        )

        node_box = FancyBboxPatch(
            (
                x - 0.033,
                y - 0.073,
            ),
            0.066,
            0.146,
            boxstyle="round,pad=0.006",
            linewidth=(
                1.6
                if is_terminal
                else 1.3
            ),
            edgecolor=color,
            facecolor="white",
            transform=ax.transAxes,
            zorder=3,
        )

        ax.add_patch(
            node_box
        )

        ax.text(
            x,
            y,
            node,
            fontsize=6.8,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            transform=ax.transAxes,
            zorder=4,
        )


# =========================================================
# Figure
# =========================================================
#
# Qualitative case-study layout:
#
#   FULL INPUT (top, full width)
#
#   FULL REASONING SPANS      TOP-10 CIRCUITS
#   S1 ... S24                M1 ... M10
#
#   TRUNCATED OUTPUT (bottom, full width)
#
# Input and all reasoning spans are intentionally preserved
# for a qualitative case-study walkthrough.
# =========================================================

GEN_PATH = (
    BASE_DIR
    / "exp_for_span-to-span/results/case1_generation.json"
)

with GEN_PATH.open(
    "r",
    encoding="utf-8",
) as f:
    generation = json.load(f)


def truncate_output_for_slide(
    text,
    max_chars=650,
):
    text = " ".join(text.split())

    if len(text) <= max_chars:
        return text

    clipped = text[:max_chars]

    if " " in clipped:
        clipped = clipped.rsplit(
            " ",
            1,
        )[0]

    return clipped + " ..."


fig = plt.figure(
    figsize=(16, 9),
)

outer = fig.add_gridspec(
    3,
    1,
    height_ratios=[
        1.16,
        6.60,
        1.04,
    ],
    hspace=0.075,
)


# =========================================================
# TOP: full input
# =========================================================

ax_input = fig.add_subplot(
    outer[0]
)

ax_input.axis("off")

ax_input.text(
    0.01,
    0.94,
    "INPUT",
    fontsize=10,
    fontweight="bold",
    color="#666666",
    ha="left",
    va="top",
    transform=ax_input.transAxes,
)

wrapped_input = textwrap.fill(
    generation.get("prompt", ""),
    width=190,
)

input_box = FancyBboxPatch(
    (0.01, 0.05),
    0.98,
    0.72,
    boxstyle="round,pad=0.006",
    linewidth=0.8,
    edgecolor="#D0D0D0",
    facecolor="#FAFAFA",
    transform=ax_input.transAxes,
)

ax_input.add_patch(input_box)

ax_input.text(
    0.025,
    0.67,
    wrapped_input,
    fontsize=7.1,
    ha="left",
    va="top",
    color="#222222",
    transform=ax_input.transAxes,
)


# =========================================================
# MIDDLE: full reasoning spans + top-10 circuits
# =========================================================

middle = outer[
    1
].subgridspec(
    1,
    2,
    width_ratios=[
        0.94,
        1.34,
    ],
    wspace=0.045,
)

ax_routes = fig.add_subplot(
    middle[0]
)

circuit_grid = middle[
    1
].subgridspec(
    5,
    2,
    hspace=0.070,
    wspace=0.055,
)

circuit_axes = [
    fig.add_subplot(
        circuit_grid[
            row,
            col,
        ]
    )
    for row in range(5)
    for col in range(2)
]


fig.suptitle(
    "Case Study: Top-10 Multi-Hop Attribution Routes and Circuit Representations",
    fontsize=18,
    fontweight="bold",
    y=0.986,
)


# ---------------------------------------------------------
# LEFT: all reasoning spans
# ---------------------------------------------------------

ax_routes.set_xlim(
    0.0,
    1.0,
)

ax_routes.set_ylim(
    0.0,
    1.0,
)

ax_routes.axis("off")

ax_routes.text(
    0.015,
    0.992,
    "REASONING SPANS",
    fontsize=10.5,
    fontweight="bold",
    color="#444444",
    ha="left",
    va="top",
    transform=ax_routes.transAxes,
)


route_x = [
    0.650,
    0.686,
    0.722,
    0.758,
    0.794,
    0.830,
    0.866,
    0.902,
    0.938,
    0.974,
]

for x, route in zip(
    route_x,
    top_routes,
):
    rid = route["_display_id"]

    ax_routes.text(
        x,
        0.987,
        rid,
        fontsize=5.7,
        fontweight="bold",
        color=route_colors[rid],
        ha="center",
        va="top",
        transform=ax_routes.transAxes,
    )


top_y = 0.948
bottom_y = 0.016

row_h = (
    top_y
    - bottom_y
) / len(spans)


for i, span in enumerate(
    spans
):
    sid = span["id"]

    y_top = (
        top_y
        - i * row_h
    )

    y_bottom = (
        y_top
        - row_h * 0.91
    )

    y_center = (
        y_bottom
        + (
            y_top
            - y_bottom
        ) / 2
    )

    members = [
        route["_display_id"]
        for route in top_routes
        if sid in route["route"]
    ]

    facecolor = (
        "#EAF4FF"
        if members
        else "#FAFAFA"
    )

    row_box = FancyBboxPatch(
        (0.01, y_bottom),
        0.98,
        y_top - y_bottom,
        boxstyle="round,pad=0.0012",
        linewidth=0,
        facecolor=facecolor,
        transform=ax_routes.transAxes,
        zorder=0,
    )

    ax_routes.add_patch(row_box)

    wrapped_span = textwrap.fill(
        f"{sid}: {span['text']}",
        width=49,
    )

    ax_routes.text(
        0.020,
        y_center,
        wrapped_span,
        fontsize=5.15,
        ha="left",
        va="center",
        color="#222222",
        transform=ax_routes.transAxes,
        zorder=2,
        linespacing=0.90,
    )

    for x, route in zip(
        route_x,
        top_routes,
    ):
        rid = route["_display_id"]

        if rid not in members:
            continue

        draw_route_tag(
            ax_routes,
            x,
            y_center,
            rid,
        )


# ---------------------------------------------------------
# RIGHT: actual circuit cards
# ---------------------------------------------------------

fig.text(
    0.745,
    0.822,
    "TOP-10 CIRCUIT REPRESENTATIONS",
    fontsize=11.5,
    fontweight="bold",
    ha="center",
    va="center",
)

fig.text(
    0.745,
    0.806,
    "Each card shows one explicit Q → O attribution-flow route",
    fontsize=7.1,
    color="#555555",
    ha="center",
    va="center",
)

for ax, route in zip(
    circuit_axes,
    top_routes,
):
    draw_circuit_card(
        ax,
        route,
    )


# =========================================================
# BOTTOM: truncated output
# =========================================================

ax_output = fig.add_subplot(
    outer[2]
)

ax_output.axis("off")

ax_output.text(
    0.01,
    0.94,
    "OUTPUT",
    fontsize=10,
    fontweight="bold",
    color="#666666",
    ha="left",
    va="top",
    transform=ax_output.transAxes,
)

output_text = truncate_output_for_slide(
    generation.get(
        "final_text",
        "",
    ),
    max_chars=650,
)

wrapped_output = textwrap.fill(
    output_text,
    width=190,
)

output_box = FancyBboxPatch(
    (0.01, 0.05),
    0.98,
    0.70,
    boxstyle="round,pad=0.006",
    linewidth=1.2,
    edgecolor="#E69F00",
    facecolor="#FFFFFF",
    transform=ax_output.transAxes,
)

ax_output.add_patch(output_box)

ax_output.text(
    0.025,
    0.65,
    wrapped_output,
    fontsize=6.8,
    ha="left",
    va="top",
    color="#222222",
    transform=ax_output.transAxes,
)


# ---------------------------------------------------------
# Bottom note
# ---------------------------------------------------------

fig.text(
    0.745,
    0.038,
    (
        "Q = input prompt   |   O = final output   |   "
        "Si → Sj = extracted attribution-flow circuit edge"
    ),
    fontsize=6.7,
    ha="center",
    va="center",
)
# =========================================================
# Save
# =========================================================

plt.subplots_adjust(
    top=0.935,
    bottom=0.050,
    left=0.020,
    right=0.990,
)


plt.savefig(
    OUT_PATH,
    dpi=270,
    bbox_inches="tight",
)


print(
    f"Saved: {OUT_PATH}"
)


# =========================================================
# Console sanity check
# =========================================================

print(
    "\n===== TOP-10 MULTI-HOP ROUTES ====="
)


for route in top_routes:

    rid = route[
        "_display_id"
    ]

    flow = float(
        route.get(
            "flow",
            0.0,
        )
    )

    print(
        f"{rid}: "
        f"{' -> '.join(route['route'])} "
        f"[flow={flow:.8f}]"
    )
