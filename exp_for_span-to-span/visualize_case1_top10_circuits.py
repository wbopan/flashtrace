import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

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
    "case1_top10_circuit_edges_slide.png"
)

TOP_K = 10


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

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

span_text = {
    span["id"]: span["text"]
    for span in spans
}


# ---------------------------------------------------------
# Route loading
# ---------------------------------------------------------

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
        "Could not find a route list. "
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


top_routes = multi_hop_routes[:TOP_K]


if len(top_routes) < TOP_K:

    raise RuntimeError(
        f"Only {len(top_routes)} multi-hop routes found; "
        f"requested TOP_K={TOP_K}."
    )


for rank, route in enumerate(
    top_routes,
    start=1,
):

    route["_display_id"] = (
        f"M{rank}"
    )


# ---------------------------------------------------------
# Matplotlib default color cycle
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Helper: compact reasoning text
# ---------------------------------------------------------

def compact_reasoning_text(
    sid,
    text,
    width=54,
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


# ---------------------------------------------------------
# Helper: draw one circuit card
# ---------------------------------------------------------

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


    rid = route[
        "_display_id"
    ]

    flow = float(
        route.get(
            "flow",
            0.0,
        )
    )

    path = route[
        "route"
    ]

    color = route_colors[
        rid
    ]


    # Card border.
    card = FancyBboxPatch(
        (
            0.01,
            0.05,
        ),
        0.98,
        0.90,
        boxstyle="round,pad=0.012",
        fill=False,
        linewidth=1.0,
        alpha=0.55,
        transform=ax.transAxes,
    )

    ax.add_patch(
        card
    )


    # Route ID + attribution flow.
    ax.text(
        0.035,
        0.84,
        rid,
        fontsize=10,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        transform=ax.transAxes,
    )

    ax.text(
        0.965,
        0.84,
        f"flow = {flow:.4f}",
        fontsize=7.3,
        ha="right",
        va="center",
        transform=ax.transAxes,
    )


    # Even horizontal placement of route nodes.
    n_nodes = len(
        path
    )

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
        for i in range(
            n_nodes
        )
    ]

    y = 0.43


    # Arrows first so nodes appear on top.
    for i in range(
        n_nodes - 1
    ):

        arrow = FancyArrowPatch(
            (
                xs[i] + 0.028,
                y,
            ),
            (
                xs[i + 1] - 0.028,
                y,
            ),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=2.0,
            color=color,
            alpha=0.85,
            shrinkA=0,
            shrinkB=0,
            transform=ax.transAxes,
        )

        ax.add_patch(
            arrow
        )


    # Nodes.
    for x, node in zip(
        xs,
        path,
    ):

        if node in {
            "Q",
            "O",
        }:

            node_box = FancyBboxPatch(
                (
                    x - 0.031,
                    y - 0.075,
                ),
                0.062,
                0.15,
                boxstyle="round,pad=0.008",
                fill=False,
                linewidth=1.6,
                edgecolor=color,
                transform=ax.transAxes,
                zorder=3,
            )

        else:

            node_box = FancyBboxPatch(
                (
                    x - 0.035,
                    y - 0.075,
                ),
                0.070,
                0.15,
                boxstyle="round,pad=0.008",
                linewidth=1.4,
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
            fontsize=7.4,
            fontweight="bold",
            color=color,
            ha="center",
            va="center",
            transform=ax.transAxes,
            zorder=4,
        )


# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------
#
# 16:9 slide-friendly:
#
#   LEFT
#     reasoning span reference S1..S24
#
#   RIGHT
#     actual Top-10 circuit edge diagrams
#     arranged as 2 columns × 5 rows
#
# ---------------------------------------------------------

fig = plt.figure(
    figsize=(16, 9),
)


outer = fig.add_gridspec(
    1,
    2,
    width_ratios=[
        0.90,
        1.45,
    ],
    wspace=0.055,
)


ax_reasoning = fig.add_subplot(
    outer[0]
)


circuit_grid = outer[
    1
].subgridspec(
    5,
    2,
    hspace=0.11,
    wspace=0.07,
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


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

fig.suptitle(
    "Top-10 Circuit Edges from Recursive Span-to-Span Attribution",
    fontsize=18,
    fontweight="bold",
    y=0.985,
)

fig.text(
    0.5,
    0.956,
    "Greedy attribution-flow decomposition; each panel shows one explicit Q → O circuit",
    fontsize=8.5,
    ha="center",
    va="top",
)


# ---------------------------------------------------------
# Left: reasoning span reference
# ---------------------------------------------------------

ax_reasoning.set_xlim(
    0.0,
    1.0,
)

ax_reasoning.set_ylim(
    len(spans),
    -1,
)

ax_reasoning.axis(
    "off"
)


ax_reasoning.text(
    0.02,
    -0.62,
    "REASONING SPANS",
    fontsize=11,
    fontweight="bold",
    ha="left",
    va="center",
)


used_spans = {
    node
    for route in top_routes
    for node in route["route"]
    if node.startswith("S")
}


for i, span in enumerate(
    spans
):

    sid = span[
        "id"
    ]

    # Light visual distinction for spans that appear
    # in at least one Top-10 circuit.
    row = FancyBboxPatch(
        (
            0.01,
            i - 0.43,
        ),
        0.98,
        0.86,
        boxstyle="round,pad=0.002",
        linewidth=0,
        alpha=(
            0.11
            if sid in used_spans
            else 0.035
        ),
        transform=ax_reasoning.transData,
    )

    ax_reasoning.add_patch(
        row
    )


    text = compact_reasoning_text(
        sid,
        span["text"],
    )

    ax_reasoning.text(
        0.025,
        i,
        text,
        fontsize=6.6,
        ha="left",
        va="center",
    )


# ---------------------------------------------------------
# Right: actual circuits
# ---------------------------------------------------------

for ax, route in zip(
    circuit_axes,
    top_routes,
):

    draw_circuit_card(
        ax,
        route,
    )


# ---------------------------------------------------------
# Bottom note
# ---------------------------------------------------------

fig.text(
    0.74,
    0.028,
    "Q = input prompt   |   O = final output   |   edge direction follows the extracted attribution-flow route",
    fontsize=7.2,
    ha="center",
    va="center",
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.subplots_adjust(
    top=0.915,
    bottom=0.055,
    left=0.035,
    right=0.985,
)


plt.savefig(
    OUT_PATH,
    dpi=260,
    bbox_inches="tight",
)


print(
    f"Saved: {OUT_PATH}"
)


# ---------------------------------------------------------
# Console sanity check
# ---------------------------------------------------------

print(
    "\n===== TOP-10 CIRCUITS ====="
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
