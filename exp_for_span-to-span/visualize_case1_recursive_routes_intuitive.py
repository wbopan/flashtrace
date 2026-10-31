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
ROUTES_PATH = "exp_for_span-to-span/results/case1_flow_routes.json"

OUT_PATH = (
    "exp_for_span-to-span/results/"
    "case1_recursive_routes_intuitive_slide.png"
)

TOP_K = 10


# ---------------------------------------------------------
# Load
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
span_index = {
    sid: i
    for i, sid in enumerate(span_ids)
}

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
# Recursive Span Attribution
# ---------------------------------------------------------
#
# Pairwise matrix:
#
#     A[target, source]
#
# Recursive attribution starts from the final output O and
# propagates attribution backward through the pairwise
# span-to-span dependencies.
#
# Because the graph is causal / acyclic, this reverse pass
# captures paths of arbitrary depth rather than a fixed Hop-2.
# ---------------------------------------------------------

node_flow = np.zeros(
    len(labels),
    dtype=float,
)

node_flow[O_IDX] = 1.0


for target_idx in range(
    O_IDX,
    Q_IDX,
    -1,
):

    target_flow = node_flow[
        target_idx
    ]

    if target_flow <= 0:
        continue

    for source_idx in range(
        target_idx
    ):

        edge = A[
            target_idx,
            source_idx,
        ]

        if not np.isfinite(edge):
            continue

        if edge <= 0:
            continue

        node_flow[source_idx] += (
            target_flow
            * edge
        )


recursive_scores = np.array(
    [
        node_flow[
            label_to_index[sid]
        ]
        for sid in span_ids
    ],
    dtype=float,
)


# Normalize across reasoning spans only for slide display.
if recursive_scores.sum() > 0:
    recursive_display = (
        recursive_scores
        / recursive_scores.sum()
    )
else:
    recursive_display = (
        recursive_scores.copy()
    )


# ---------------------------------------------------------
# Load Top-K multi-hop routes
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
        "Could not find route list. "
        f"Available keys: {list(data.keys())}"
    )


all_routes = find_route_list(
    route_data
)


def reasoning_count(route):

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
    if reasoning_count(route) >= 2
]

multi_hop_routes = sorted(
    multi_hop_routes,
    key=lambda route: float(
        route.get(
            "flow",
            0.0,
        )
    ),
    reverse=True,
)

top_routes = (
    multi_hop_routes[
        :TOP_K
    ]
)


if len(top_routes) < TOP_K:
    raise RuntimeError(
        f"Only {len(top_routes)} routes found."
    )


for rank, route in enumerate(
    top_routes,
    start=1,
):
    route["_display_id"] = (
        f"M{rank}"
    )


# ---------------------------------------------------------
# Route colors
# ---------------------------------------------------------

route_palette = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
]

route_colors = {
    route["_display_id"]: color
    for route, color in zip(
        top_routes,
        route_palette,
    )
}


# ---------------------------------------------------------
# Compact text
# ---------------------------------------------------------

def compact_text(
    text,
    width,
    max_lines,
):

    lines = textwrap.wrap(
        text,
        width=width,
    )

    if len(lines) <= max_lines:
        return "\n".join(lines)

    lines = lines[:max_lines]

    lines[-1] = (
        lines[-1].rstrip(" .")
        + " ..."
    )

    return "\n".join(lines)


# ---------------------------------------------------------
# Figure
# ---------------------------------------------------------
#
# Slide story:
#
#   INPUT
#
#   Pairwise span-to-span matrix
#        -> recursive propagation
#   Recursive Span Attribution
#        -> flow decomposition
#   Top-5 Multi-Hop Routes
#
#   OUTPUT
#
# ---------------------------------------------------------

fig = plt.figure(
    figsize=(16, 9),
)

outer = fig.add_gridspec(
    3,
    1,
    height_ratios=[
        1.0,
        7.0,
        1.35,
    ],
    hspace=0.13,
)


ax_input = fig.add_subplot(
    outer[0]
)

middle = outer[1].subgridspec(
    1,
    3,
    width_ratios=[
        1.15,
        0.78,
        1.72,
    ],
    wspace=0.13,
)

ax_matrix = fig.add_subplot(
    middle[0]
)

ax_recursive = fig.add_subplot(
    middle[1]
)

ax_routes = fig.add_subplot(
    middle[2]
)

ax_output = fig.add_subplot(
    outer[2]
)


ax_input.axis("off")
ax_output.axis("off")


# ---------------------------------------------------------
# Title
# ---------------------------------------------------------

fig.suptitle(
    "Recursive Span Attribution → Multi-Hop Reasoning Routes — Case Study 1",
    fontsize=18,
    fontweight="bold",
    y=0.985,
)


# ---------------------------------------------------------
# Shared INPUT
# ---------------------------------------------------------

ax_input.text(
    0.0,
    0.93,
    "INPUT",
    fontsize=9,
    fontweight="bold",
    color="gray",
    transform=ax_input.transAxes,
)

prompt_display = compact_text(
    generation["prompt"],
    width=150,
    max_lines=4,
)

input_box = FancyBboxPatch(
    (0.0, 0.05),
    1.0,
    0.72,
    boxstyle="round,pad=0.008",
    linewidth=0.8,
    edgecolor="#D0D0D0",
    facecolor="#FAFAFA",
    transform=ax_input.transAxes,
)

ax_input.add_patch(
    input_box
)

ax_input.text(
    0.015,
    0.66,
    prompt_display,
    fontsize=7.6,
    va="top",
    transform=ax_input.transAxes,
)


# ---------------------------------------------------------
# A. Pairwise Span-to-Span Attribution Matrix
# ---------------------------------------------------------

ax_matrix.set_title(
    "Pairwise Span-to-Span Attribution",
    fontsize=12.5,
    fontweight="bold",
    pad=10,
)

ax_matrix.text(
    0.5,
    1.01,
    "source → target attribution strength",
    fontsize=7.7,
    color="#555555",
    ha="center",
    va="bottom",
    transform=ax_matrix.transAxes,
)


# Mask invalid / causal-future cells.
masked_A = np.ma.masked_invalid(
    A
)

matrix_cmap = cm.Blues.copy()
matrix_cmap.set_bad(
    color="#F5F5F5"
)

finite_vals = A[
    np.isfinite(A)
]

matrix_vmax = (
    float(
        np.percentile(
            finite_vals,
            98,
        )
    )
    if finite_vals.size
    else 1.0
)

matrix_norm = colors.Normalize(
    vmin=0.0,
    vmax=(
        matrix_vmax
        if matrix_vmax > 0
        else 1.0
    ),
)


im = ax_matrix.imshow(
    masked_A,
    cmap=matrix_cmap,
    norm=matrix_norm,
    aspect="auto",
    interpolation="nearest",
)


# Show only a subset of ticks so the matrix stays legible.
tick_labels = [
    "Q",
    "S4",
    "S8",
    "S12",
    "S16",
    "S20",
    "S24",
    "O",
]

tick_indices = [
    label_to_index[label]
    for label in tick_labels
    if label in label_to_index
]

visible_tick_labels = [
    label
    for label in tick_labels
    if label in label_to_index
]


ax_matrix.set_xticks(
    tick_indices
)

ax_matrix.set_xticklabels(
    visible_tick_labels,
    fontsize=7,
    rotation=45,
    ha="right",
)

ax_matrix.set_yticks(
    tick_indices
)

ax_matrix.set_yticklabels(
    visible_tick_labels,
    fontsize=7,
)

ax_matrix.set_xlabel(
    "Source span",
    fontsize=8,
)

ax_matrix.set_ylabel(
    "Target span",
    fontsize=8,
)


# Small explanatory box.
ax_matrix.text(
    0.02,
    0.02,
    "Each cell = one pairwise dependency",
    fontsize=7,
    color="#444444",
    ha="left",
    va="bottom",
    transform=ax_matrix.transAxes,
    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="#DDDDDD",
        alpha=0.92,
    ),
)


matrix_cbar = fig.colorbar(
    im,
    ax=ax_matrix,
    fraction=0.035,
    pad=0.025,
)

matrix_cbar.ax.tick_params(
    labelsize=6.5
)

matrix_cbar.set_label(
    "Pairwise attribution",
    fontsize=7.5,
)


# ---------------------------------------------------------
# Visual transition:
# pairwise dependencies -> recursive attribution
# ---------------------------------------------------------

fig.text(
    0.402,
    0.675,
    "recursive\npropagation\nfrom O  →",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    color="#444444",
)


# ---------------------------------------------------------
# B. Recursive Span Attribution
# ---------------------------------------------------------

ax_recursive.set_title(
    "Recursive Span Attribution",
    fontsize=12.5,
    fontweight="bold",
    pad=10,
)

ax_recursive.text(
    0.5,
    1.01,
    "aggregated contribution through downstream spans",
    fontsize=7.7,
    color="#555555",
    ha="center",
    va="bottom",
    transform=ax_recursive.transAxes,
)


y_positions = np.arange(
    len(span_ids)
)


bar_cmap = cm.Blues

bar_norm = colors.Normalize(
    vmin=0.0,
    vmax=(
        float(
            recursive_display.max()
        )
        if recursive_display.max() > 0
        else 1.0
    ),
)

bar_colors = [
    bar_cmap(
        bar_norm(value)
    )
    for value in recursive_display
]


ax_recursive.barh(
    y_positions,
    recursive_display,
    height=0.74,
    color=bar_colors,
    edgecolor="none",
)


ax_recursive.set_yticks(
    y_positions
)

ax_recursive.set_yticklabels(
    span_ids,
    fontsize=7,
)

ax_recursive.invert_yaxis()

ax_recursive.set_xlabel(
    "Recursive attribution",
    fontsize=8,
)

ax_recursive.tick_params(
    axis="x",
    labelsize=7,
)

ax_recursive.grid(
    axis="x",
    linewidth=0.35,
    alpha=0.35,
)

ax_recursive.spines[
    "top"
].set_visible(False)

ax_recursive.spines[
    "right"
].set_visible(False)

ax_recursive.spines[
    "left"
].set_visible(False)


# Add values only for the strongest spans.
top_recursive_indices = set(
    np.argsort(
        recursive_display
    )[::-1][:6]
)

for i, value in enumerate(
    recursive_display
):

    if i not in top_recursive_indices:
        continue

    ax_recursive.text(
        value,
        i,
        f" {value:.3f}",
        fontsize=6.3,
        va="center",
        ha="left",
        color="#333333",
    )


# ---------------------------------------------------------
# Visual transition:
# recursive attribution -> routes
# ---------------------------------------------------------

fig.text(
    0.582,
    0.675,
    "greedy\nflow\n decomposition  →",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    color="#444444",
)


# ---------------------------------------------------------
# C. Top-5 Multi-Hop Routes
# ---------------------------------------------------------

ax_routes.set_title(
    "Top-5 Multi-Hop Routes",
    fontsize=12.5,
    fontweight="bold",
    pad=10,
)

ax_routes.text(
    0.5,
    1.01,
    "highest attribution-flow circuits",
    fontsize=7.7,
    color="#555555",
    ha="center",
    va="bottom",
    transform=ax_routes.transAxes,
)


ax_routes.set_xlim(
    0.0,
    1.0,
)

ax_routes.set_ylim(
    len(span_ids) - 0.5,
    -0.5,
)

ax_routes.axis("off")


# Text area + route lanes.
text_x = 0.01

lane_x = np.linspace(
    0.72,
    0.96,
    TOP_K,
)


# Route headers.
for x, route in zip(
    lane_x,
    top_routes,
):

    rid = route[
        "_display_id"
    ]

    flow = float(
        route.get(
            "flow",
            0.0,
        )
    )

    ax_routes.text(
        x,
        -0.82,
        rid,
        fontsize=7,
        fontweight="bold",
        color=route_colors[rid],
        ha="center",
        va="center",
        clip_on=False,
    )

    ax_routes.text(
        x,
        -0.25,
        f"{flow:.3f}",
        fontsize=5.7,
        color="#555555",
        ha="center",
        va="center",
        clip_on=False,
    )


# Row backgrounds + text.
for i, span in enumerate(
    spans
):

    membership = []

    for route in top_routes:

        if span["id"] in route[
            "route"
        ]:
            membership.append(
                route[
                    "_display_id"
                ]
            )

    facecolor = (
        "#EEF6FF"
        if membership
        else "#F8F8F8"
    )

    row_box = FancyBboxPatch(
        (
            0.0,
            i - 0.40,
        ),
        0.995,
        0.80,
        boxstyle="round,pad=0.002",
        linewidth=0,
        facecolor=facecolor,
        transform=ax_routes.transData,
        clip_on=True,
    )

    ax_routes.add_patch(
        row_box
    )

    text = (
        f"{span['id']}: "
        f"{span['text']}"
    )

    display = textwrap.shorten(
        text,
        width=64,
        placeholder=" ...",
    )

    ax_routes.text(
        text_x,
        i,
        display,
        fontsize=5.9,
        ha="left",
        va="center",
        color="#222222",
    )


# Draw each route as its own vertical lane.
for x, route in zip(
    lane_x,
    top_routes,
):

    rid = route[
        "_display_id"
    ]

    color = route_colors[
        rid
    ]

    route_spans = [
        node
        for node in route[
            "route"
        ]
        if node in span_index
    ]

    ys = [
        span_index[sid]
        for sid in route_spans
    ]


    # Connect only route-selected spans.
    for y1, y2 in zip(
        ys[:-1],
        ys[1:],
    ):

        ax_routes.plot(
            [
                x,
                x,
            ],
            [
                y1,
                y2,
            ],
            linewidth=2.0,
            color=color,
            alpha=0.85,
            solid_capstyle="round",
            zorder=4,
        )


    ax_routes.scatter(
        [x] * len(ys),
        ys,
        s=25,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
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
    width=155,
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
    top=0.925,
    bottom=0.045,
    left=0.045,
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
# Sanity checks
# ---------------------------------------------------------

print(
    "\n===== TOP RECURSIVE SPANS ====="
)

for i in np.argsort(
    recursive_display
)[::-1][:8]:

    print(
        f"{span_ids[i]}: "
        f"{recursive_display[i]:.4f} | "
        f"{span_text[span_ids[i]]}"
    )


print(
    "\n===== TOP-5 ROUTES ====="
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
