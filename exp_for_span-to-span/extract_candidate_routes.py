import json
import numpy as np

# IMPORTANT:
# Single-hop routes (Q -> Sj -> O) are retained in `all_routes`
# as valid direct attribution effects, but are excluded from the
# downstream circuit candidate set.
#
# This is intentional: our downstream objective is to study
# recursive, multi-hop information flow through intermediate
# reasoning spans, rather than direct span-to-output importance.
#
# Therefore, candidate circuits must contain at least two
# reasoning spans:
#
#     Q -> Sa -> Sb -> ... -> O
#
# We do NOT remove single-hop routes because they are invalid;
# we exclude them only because they fall outside the multi-hop
# search objective.


PATH = "/home/jgwak1/LLM_MCTS_Proj/circuit-span-to-span/flashtrace/exp_for_span-to-span/results/case1_span_matrix.json"
OUT = "/home/jgwak1/LLM_MCTS_Proj/circuit-span-to-span/flashtrace/exp_for_span-to-span/results/case1_flow_routes.json"

# Numerical tolerance only. This is NOT a route-selection threshold.
EPS = 1e-12


# =========================================================
# 1. Load Q, S1...Sn, O attribution matrix
# =========================================================

with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

labels = data["labels"]

A = np.array(
    [
        [0.0 if x is None else float(x) for x in row]
        for row in data["normalized_matrix"]
    ],
    dtype=float,
)

N = len(labels)
Q = 0
O = N - 1

assert labels[Q] == "Q"
assert labels[O] == "O"


# =========================================================
# 2. Convert incoming attribution for each target into
#    a backward transition distribution
#
# A[target, source] is the normalized attribution from an
# earlier source span to a later target span.
#
# P[target, source] is therefore the fraction of the target's
# incoming attribution assigned to that source.
# =========================================================

P = np.zeros_like(A)

for target in range(1, N):
    incoming = A[target, :target]
    total = incoming.sum()

    if total > EPS:
        P[target, :target] = incoming / total


# =========================================================
# 3. Propagate answer-conditioned flow backward from O
#
# Start with unit mass at the final output O and propagate it
# backward through the span-to-span attribution DAG.
#
# This produces an answer-conditioned edge-flow graph:
#     Q / earlier S  ->  later S / O
#
# No edge threshold is introduced here.
# =========================================================

node_mass = np.zeros(N, dtype=float)
node_mass[O] = 1.0

edge_flow = np.zeros((N, N), dtype=float)

for target in range(O, Q, -1):
    mass = node_mass[target]

    if mass <= EPS:
        continue

    for source in range(target):
        prob = P[target, source]

        if prob <= EPS:
            continue

        flow = mass * prob

        edge_flow[source, target] += flow
        node_mass[source] += flow


print(f"Flow reaching Q: {node_mass[Q]:.6f}")


# =========================================================
# 4. Decompose the conserved edge flow into explicit Q -> O
#    routes using greedy widest-path decomposition
#
# Why keep this step?
# -------------------
# The span matrix gives pairwise attribution edges, while the
# downstream MCTS prototype needs explicit multi-span circuit
# routes. The decomposition converts the conserved edge flow
# into such routes without introducing a hand-picked K,
# similarity cutoff, or cumulative-flow threshold.
#
# At each iteration:
#   1. Find the widest remaining Q -> O path.
#   2. Assign that path its bottleneck residual flow.
#   3. Subtract that flow from the path edges.
#   4. Repeat until no Q -> O residual flow remains.
#
# This is a deterministic decomposition rule for obtaining
# explicit routes from the flow graph. It does NOT decide how
# many routes are "important"; all decomposed routes are kept.
# =========================================================

residual = edge_flow.copy()
routes = []

while residual[Q].sum() > EPS:
    width = np.zeros(N, dtype=float)
    parent = np.full(N, -1, dtype=int)

    width[Q] = np.inf

    # Widest-path dynamic programming over the DAG.
    for source in range(Q, O):
        if width[source] <= EPS:
            continue

        for target in range(source + 1, N):
            capacity = residual[source, target]

            if capacity <= EPS:
                continue

            candidate = min(
                width[source],
                capacity,
            )

            if candidate > width[target] + EPS:
                width[target] = candidate
                parent[target] = source

    # No remaining Q -> O residual flow.
    if parent[O] == -1:
        break

    # Reconstruct the widest route.
    route = [O]
    current = O

    while current != Q:
        current = parent[current]

        if current == -1:
            raise RuntimeError("Broken flow path.")

        route.append(current)

    route.reverse()
    route_flow = float(width[O])

    # Remove extracted flow from the residual graph.
    for source, target in zip(route[:-1], route[1:]):
        residual[source, target] -= route_flow

        if residual[source, target] < EPS:
            residual[source, target] = 0.0

    routes.append(
        {
            "route": [labels[i] for i in route],
            "flow": route_flow,
        }
    )


# =========================================================
# 5. Sort ALL decomposed routes by explained flow
#
# This is descriptive ordering only. No route is selected or
# removed here based on its flow magnitude.
# =========================================================

routes.sort(
    key=lambda x: x["flow"],
    reverse=True,
)

total_flow = sum(r["flow"] for r in routes)
cumulative = 0.0

for i, route in enumerate(routes, start=1):
    cumulative += route["flow"]

    route["route_id"] = f"R{i}"
    route["cumulative_flow"] = cumulative
    route["cumulative_fraction"] = (
        cumulative / total_flow
        if total_flow > 0
        else 0.0
    )


# =========================================================
# 6. Separate direct, single-hop, and multi-hop routes
#
# IMPORTANT DESIGN CHOICE
# -----------------------
# Single-hop routes such as
#
#     Q -> Sj -> O
#
# are VALID direct attribution effects. We do NOT claim that
# they are wrong, weak, or redundant, and we keep them in
# `all_routes` and `single_hop_routes` for analysis.
#
# However, the downstream objective of this prototype is to
# search RECURRENT / MULTI-HOP information flow through
# intermediate reasoning spans -- the structure revealed by
# recursive attribution beyond a direct span-to-output effect.
#
# Therefore the MCTS candidate set contains only routes with
# at least TWO reasoning spans:
#
#     Q -> Sa -> Sb -> ... -> O
#
# This exclusion is based on the definition of the downstream
# search objective, NOT on a heuristic score threshold.
#
# Likewise, Q -> O is retained as a direct bypass route for
# bookkeeping but is not a multi-hop circuit candidate.
# =========================================================


def reasoning_nodes(route):
    return [
        node
        for node in route["route"]
        if node.startswith("S")
    ]


direct_routes = []
single_hop_routes = []
multi_hop_routes = []

for route in routes:
    num_reasoning_spans = len(reasoning_nodes(route))

    if num_reasoning_spans == 0:
        direct_routes.append(route)

    elif num_reasoning_spans == 1:
        single_hop_routes.append(route)

    else:
        multi_hop_routes.append(route)


# Give the downstream multi-hop candidates a compact candidate
# ID while preserving the original decomposition route_id.
multi_hop_total_flow = sum(
    route["flow"]
    for route in multi_hop_routes
)

multi_hop_cumulative = 0.0

for i, route in enumerate(multi_hop_routes, start=1):
    multi_hop_cumulative += route["flow"]

    route["candidate_id"] = f"M{i}"
    route["multi_hop_cumulative_flow"] = multi_hop_cumulative
    route["multi_hop_cumulative_fraction"] = (
        multi_hop_cumulative / multi_hop_total_flow
        if multi_hop_total_flow > 0
        else 0.0
    )


# =========================================================
# 7. Save
#
# `selected_routes` is retained for compatibility with the
# downstream scripts. It now means exactly:
#
#     all decomposed MULTI-HOP routes
#
# There is NO elbow selection, hand-picked K, flow threshold,
# or semantic cutoff in this extractor.
# =========================================================

result = {
    "method": (
        "answer-conditioned conserved flow + "
        "greedy widest-path flow decomposition + "
        "objective-defined multi-hop candidate extraction"
    ),

    "selection_policy": (
        "retain all decomposed routes for analysis; "
        "use only routes containing at least two reasoning spans "
        "as downstream multi-hop circuit candidates"
    ),

    "total_decomposed_routes": len(routes),
    "total_decomposed_flow": total_flow,

    "direct_route_count": len(direct_routes),
    "single_hop_route_count": len(single_hop_routes),
    "multi_hop_route_count": len(multi_hop_routes),
    "multi_hop_total_flow": multi_hop_total_flow,

    # Backward-compatible field used by downstream scripts.
    # This is NOT elbow-selected anymore.
    "selected_route_count": len(multi_hop_routes),
    "selected_routes": multi_hop_routes,

    # Explicit categories for inspection / analysis.
    "direct_routes": direct_routes,
    "single_hop_routes": single_hop_routes,
    "multi_hop_routes": multi_hop_routes,

    # Full mathematical flow decomposition, unchanged by the
    # downstream multi-hop objective.
    "all_routes": routes,
}


with open(
    OUT,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =========================================================
# 8. Print summary + multi-hop candidates
# =========================================================

print(f"\nTotal decomposed routes: {len(routes)}")
print(f"Direct Q -> O routes: {len(direct_routes)}")
print(f"Single-hop Q -> Sj -> O routes: {len(single_hop_routes)}")
print(f"Multi-hop circuit candidates: {len(multi_hop_routes)}")

if total_flow > 0:
    print(
        "Multi-hop share of decomposed flow: "
        f"{multi_hop_total_flow / total_flow:.3f}"
    )

print("\n=== MULTI-HOP CIRCUIT CANDIDATES ===")

for route in multi_hop_routes:
    print(
        f"{route['candidate_id']:>3}  "
        f"({route['route_id']})  "
        f"{' -> '.join(route['route'])}  "
        f"[flow={route['flow']:.4f}]"
    )

print(f"\nSaved: {OUT}")