import json
import textwrap

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

GEN_PATH = "exp_for_span-to-span/results/case1_generation.json"
MATRIX_PATH = "exp_for_span-to-span/results/case1_span_matrix.json"
ROUTES_PATH = "exp_for_span-to-span/results/case1_flow_routes.json"

OUT_PATH = (
    "exp_for_span-to-span/results/"
    "case1_recursive_top5_routes_slide.png"
)

TOP_K = 5


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

with open(GEN_PATH, "r", encoding="utf-8") as f:
    generation = json.load(f)

with open(MATRIX_PATH, "r", encoding="utf-8") as f:
    matrix_data = json.load(f)

with open(ROUTES_PATH, "r", encoding="utf-8") as f:
    route_data = json.load(f)


spans = matrix_data["reasoning_spans"]
labels = matrix_data["labels"]

span_ids = [span["id"] for span in spans]
span_text = {
    span["id"]: span["text"]
    for span in spans
}

label_to_index = {
    label: i
    for i, label in enumerate(labels)
}


A = np.array(
    [
        [
            np.nan if value is None else float(value)
            for value in row
        ]
        for row in matrix_data["normalized_matrix"]
    ],
    dtype=float,
)


Q_IDX = label_to_index["Q"]
O_IDX = label_to_index["O"]


# ---------------------------------------------------------
# Recursive span attribution
# ---------------------------------------------------------
#
# Start from the final output O and recursively propagate
# attribution through span-to-span dependencies.
#
# This is not restricted to a fixed two-hop depth.
# ---------------------------------------------------------

node_flow = np.zeros(len(labels), dtype=float)
node_flow[O_IDX] = 1.0

for target_idx in range(O_IDX, Q_IDX, -1):

    target_flow = node_flow[target_idx]

    if target_flow <= 0:
        continue

    for source_idx in range(target_idx):

        edge = A[target_idx, source_idx]

        if not np.isfinite(edge):
            continue

        if edge <= 0:
            continue

        node_flow[source_idx] += (
            target_flow * edge
        )


recursive_scores = np.array(
    [
        node_flow[label_to_index[sid]]
        for sid in span_ids
    ],
    dtype=float,
)

if recursive_scores.sum() > 0:
    recursive_display = (
        recursive_scores
        / recursive_scores.sum()
    )
else:
    recursive_display = recursive_scores.copy()


# ---------------------------------------------------------
# Load top-K multi-hop routes
# ---------------------------------------------------------

def find_route_list(data):

    preferred_keys = [
        "multi_hop_routes",
        "selected_routes",
        "final_routes",
        "routes",
        "all_routes",
    ]

    for key in preferred_keys:

        value = data.get(key)

        if isinstance(value, list):
            return value

    raise RuntimeError(
        "Could not find route list. "
        f"Available keys: {list(data.keys())}"
    )


all_routes = find_route_list(route_data)


def reasoning_span_count(route):

    return len(
        [
            node
            for node in route.get("route", [])
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


for rank, route in enumerate(
    top_routes,
    start=1,
):
    # Use simple rank labels for slide readability.
    route["_display_id"] = f"M{rank}"


# ---------------------------------------------------------
# Membership
# ---------------------------------------------------------

span_route_membership = {
    sid: []
    for sid in span_ids
}

for route in top_routes:

    rid = route["_display_id"]

    for node in route["route"]:

        if node in span_route_membership:
            span_route_membership[node].append(
                rid
            )


# ---------------------------------------------------------
# Route colors
# ---------------------------------------------------------

route_colors = {
    route["_display_id"]: color
    for route, color in zip(
        top_routes,
        [
            "#0072B2",
            "#E69F00",
            "#009E73",
            "#D55E00",
            "#CC79A7",
        ],
    )
}


# ---------------------------------------------------------
# Attribution color scale
# ---------------------------------------------------------

max_attr = float(
    recursive_display.max()
)

attr_norm = colors.Normalize(
    vmin=0.0,
    vmax=max_attr if max_attr > 0 else 1.0,
)

attr_cmap = cm.Blues


# ---------------------------------------------------------
# Compact text helper
# ---------------------------------------------------------

def compact_text(
    text,
    width,
    max_lines,
):
    """
    Wrap text for slide display and cap the number of lines.
    The full text is still in the experiment JSON; the figure
    intentionally abbreviates it for presentation readability.
    """
    wrapped = textwrap.wrap(
        text,
        width=width,
    )

    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)

    kept = wrapped[:max_lines]

    kept[-1] = (
        kept[-1].rstrip(" .")
        + " ..."
    )

    return "\n".join(kept)


# ---------------------------------------------------------
# Figure: 16:9 slide-friendly layout
# ---------------------------------------------------------

fig = plt.figure(
    figsize=(16, 9),
)

# Overall layout:
#
#   shared INPUT
#   ---------------------------------
#   recursive attr | top-5 routes
#   ---------------------------------
#   shared OUTPUT
#
gs = fig.add_gridspec(
    3,
    2,
    height_ratios=[
        1.05,
        6.8,
        1.40,
    ],
    width_ratios=[
        1.0,
        1.0,
    ],
    hspace=0.14,
    wspace=0.10,
)


ax_input = fig.add_subplot(
    gs[0, :]
)

ax_attr = fig.add_subplot(
    gs[1, 0]
)

ax_routes = fig.add_subplot(
    gs[1, 1]
)

ax_output = fig.add_subplot(
    gs[2, :]
)


for ax in [
    ax_input,
    ax_attr,
    ax_routes,
    ax_output,
]:
    ax.axis("off")


# ---------------------------------------------------------
# Overall title
# ---------------------------------------------------------

fig.suptitle(
    "Recursive Span Attribution and Top-5 Multi-Hop Routes — Case Study 1",
    fontsize=18,
    fontweight="bold",
    y=0.985,
)


# ---------------------------------------------------------
# Shared INPUT
# ---------------------------------------------------------

ax_input.text(
    0.0,
    0.92,
    "INPUT",
    fontsize=9,
    fontweight="bold",
    color="gray",
    transform=ax_input.transAxes,
)

prompt_display = compact_text(
    generation["prompt"],
    width=145,
    max_lines=4,
)

prompt_box = FancyBboxPatch(
    (0.0, 0.05),
    1.0,
    0.70,
    boxstyle="round,pad=0.008",
    linewidth=0.8,
    edgecolor="#D0D0D0",
    facecolor="#FAFAFA",
    transform=ax_input.transAxes,
)

ax_input.add_patch(
    prompt_box
)

ax_input.text(
    0.015,
    0.65,
    prompt_display,
    fontsize=7.7,
    va="top",
    transform=ax_input.transAxes,
)


# ---------------------------------------------------------
# Panel titles
# ---------------------------------------------------------

ax_attr.set_title(
    "Recursive Span Attribution",
    fontsize=14,
    fontweight="bold",
    pad=8,
)

ax_attr.text(
    0.5,
    1.005,
    "Attribution propagated through span-to-span dependencies",
    fontsize=8,
    color="#555555",
    ha="center",
    va="bottom",
    transform=ax_attr.transAxes,
)


ax_routes.set_title(
    "Top-5 Multi-Hop Routes",
    fontsize=14,
    fontweight="bold",
    pad=8,
)

ax_routes.text(
    0.5,
    1.005,
    "Greedy attribution-flow decomposition",
    fontsize=8,
    color="#555555",
    ha="center",
    va="bottom",
    transform=ax_routes.transAxes,
)


# ---------------------------------------------------------
# Shared reasoning-row geometry
# ---------------------------------------------------------

n_spans = len(spans)

top_y = 0.955
bottom_y = 0.025

available_h = (
    top_y
    - bottom_y
)

row_h = (
    available_h
    / n_spans
)

row_gap = 0.0018

box_h = (
    row_h
    - row_gap
)


# ---------------------------------------------------------
# Left panel:
# Recursive Span Attribution
# ---------------------------------------------------------

ax_attr.text(
    0.005,
    0.995,
    "REASONING",
    fontsize=8,
    fontweight="bold",
    color="gray",
    va="top",
    transform=ax_attr.transAxes,
)


for i, span in enumerate(spans):

    y_top = (
        top_y
        - i * row_h
    )

    y_bottom = (
        y_top
        - box_h
    )

    rgba = attr_cmap(
        attr_norm(
            recursive_display[i]
        )
    )

    facecolor = (
        rgba[0],
        rgba[1],
        rgba[2],
        0.62,
    )

    box = FancyBboxPatch(
        (
            0.005,
            y_bottom,
        ),
        0.915,
        box_h,
        boxstyle="round,pad=0.002",
        linewidth=0,
        facecolor=facecolor,
        transform=ax_attr.transAxes,
    )

    ax_attr.add_patch(
        box
    )

    label_text = (
        f"{span['id']}: "
        f"{span['text']}"
    )

    # One-line / compact two-line rendering.
    wrapped = compact_text(
        label_text,
        width=66,
        max_lines=2,
    )

    ax_attr.text(
        0.012,
        y_top - 0.004,
        wrapped,
        fontsize=6.1,
        va="top",
        transform=ax_attr.transAxes,
    )

    ax_attr.text(
        0.947,
        y_bottom + box_h / 2,
        f"{recursive_display[i]:.3f}",
        fontsize=6.3,
        ha="right",
        va="center",
        color="#333333",
        transform=ax_attr.transAxes,
    )


# ---------------------------------------------------------
# Right panel:
# Top-5 Multi-Hop Routes
# ---------------------------------------------------------

ax_routes.text(
    0.005,
    0.995,
    "REASONING",
    fontsize=8,
    fontweight="bold",
    color="gray",
    va="top",
    transform=ax_routes.transAxes,
)


# Route column headers.
tag_x_positions = np.linspace(
    0.71,
    0.955,
    TOP_K,
)

for x, route in zip(
    tag_x_positions,
    top_routes,
):

    rid = route["_display_id"]

    ax_routes.text(
        x,
        0.985,
        rid,
        fontsize=6.5,
        fontweight="bold",
        color=route_colors[rid],
        ha="center",
        va="top",
        transform=ax_routes.transAxes,
    )


for i, span in enumerate(spans):

    sid = span["id"]

    y_top = (
        top_y
        - i * row_h
    )

    y_bottom = (
        y_top
        - box_h
    )

    members = (
        span_route_membership[
            sid
        ]
    )

    if members:
        facecolor = (
            0.90,
            0.95,
            1.00,
            0.88,
        )
    else:
        facecolor = (
            0.97,
            0.97,
            0.97,
            0.72,
        )

    box = FancyBboxPatch(
        (
            0.005,
            y_bottom,
        ),
        0.985,
        box_h,
        boxstyle="round,pad=0.002",
        linewidth=0,
        facecolor=facecolor,
        transform=ax_routes.transAxes,
    )

    ax_routes.add_patch(
        box
    )

    label_text = (
        f"{sid}: "
        f"{span['text']}"
    )

    wrapped = compact_text(
        label_text,
        width=46,
        max_lines=2,
    )

    ax_routes.text(
        0.012,
        y_top - 0.004,
        wrapped,
        fontsize=6.0,
        va="top",
        transform=ax_routes.transAxes,
    )

    y_center = (
        y_bottom
        + box_h / 2
    )

    for x, route in zip(
        tag_x_positions,
        top_routes,
    ):

        rid = route[
            "_display_id"
        ]

        if rid not in members:
            continue

        tag = FancyBboxPatch(
            (
                x - 0.021,
                y_center - 0.009,
            ),
            0.042,
            0.018,
            boxstyle="round,pad=0.0015",
            linewidth=0,
            facecolor=route_colors[rid],
            transform=ax_routes.transAxes,
        )

        ax_routes.add_patch(
            tag
        )

        ax_routes.text(
            x,
            y_center,
            rid,
            fontsize=5.5,
            fontweight="bold",
            color="white",
            ha="center",
            va="center",
            transform=ax_routes.transAxes,
        )


# ---------------------------------------------------------
# Route legend with actual paths + flow
# ---------------------------------------------------------

legend_labels = []

for route in top_routes:

    rid = route["_display_id"]

    flow = float(
        route.get(
            "flow",
            0.0,
        )
    )

    path = "→".join(
        route["route"]
    )

    legend_labels.append(
        f"{rid}: {path}  ({flow:.4f})"
    )


legend_handles = [
    Patch(
        facecolor=route_colors[
            route["_display_id"]
        ],
        edgecolor="none",
    )
    for route in top_routes
]


ax_routes.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.025,
    ),
    ncol=1,
    fontsize=6.2,
    frameon=False,
    handlelength=1.2,
    handletextpad=0.45,
    borderaxespad=0.0,
)


# ---------------------------------------------------------
# Attribution colorbar
# ---------------------------------------------------------

attr_scalar = cm.ScalarMappable(
    norm=attr_norm,
    cmap=attr_cmap,
)

attr_scalar.set_array([])

cbar = fig.colorbar(
    attr_scalar,
    ax=ax_attr,
    fraction=0.022,
    pad=0.008,
)

cbar.ax.tick_params(
    labelsize=7
)

cbar.set_label(
    "Recursive Attribution",
    fontsize=8,
)


# ---------------------------------------------------------
# Shared OUTPUT
# ---------------------------------------------------------

ax_output.text(
    0.0,
    0.93,
    "OUTPUT",
    fontsize=9,
    fontweight="bold",
    color="gray",
    transform=ax_output.transAxes,
)


output_display = compact_text(
    generation["final_text"],
    width=150,
    max_lines=6,
)


output_box = FancyBboxPatch(
    (0.0, 0.05),
    1.0,
    0.72,
    boxstyle="round,pad=0.008",
    linewidth=1.4,
    edgecolor="#E69F00",
    facecolor="#FFFFFF",
    transform=ax_output.transAxes,
)

ax_output.add_patch(
    output_box
)

ax_output.text(
    0.015,
    0.68,
    output_display,
    fontsize=7.4,
    va="top",
    transform=ax_output.transAxes,
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

plt.subplots_adjust(
    top=0.93,
    bottom=0.045,
    left=0.035,
    right=0.975,
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

print("\n===== TOP-5 MULTI-HOP ROUTES =====")

for rank, route in enumerate(
    top_routes,
    start=1,
):

    flow = float(
        route.get(
            "flow",
            0.0,
        )
    )

    print(
        f"M{rank}: "
        f"{' -> '.join(route['route'])} "
        f"[flow={flow:.8f}]"
    )
