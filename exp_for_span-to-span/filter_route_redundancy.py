import csv
import json
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# Paths
# =========================================================

FLOW_PATH = (
    "/home/jgwak1/LLM_MCTS_Proj/"
    "circuit-span-to-span/flashtrace/"
    "exp_for_span-to-span/results/"
    "case1_flow_routes.json"
)

MATRIX_PATH = (
    "/home/jgwak1/LLM_MCTS_Proj/"
    "circuit-span-to-span/flashtrace/"
    "exp_for_span-to-span/results/"
    "case1_span_matrix.json"
)

OUT_PATH = (
    "/home/jgwak1/LLM_MCTS_Proj/"
    "circuit-span-to-span/flashtrace/"
    "exp_for_span-to-span/results/"
    "case1_redundancy_filtered_routes.json"
)


# Directory containing all experiment outputs.
RESULTS_DIR = Path(OUT_PATH).parent


# =========================================================
# Models / configuration
# =========================================================

# ---------------------------------------------------------
# REDUNDANCY
# ---------------------------------------------------------
# Compare COMPLETE multi-hop routes with a sentence embedding
# model. Routes that are semantic near-duplicates are merged,
# and the higher-attribution-flow route is retained.
#
# This stage answers:
#   "Are two mechanically extracted circuits semantically
#    redundant?"
#
# It does NOT test logical consistency inside a route.
# ---------------------------------------------------------

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Explicit semantic near-duplicate criterion.
# Keep this separate from MCTS branch-out K.
REDUNDANCY_SIM_THRESHOLD = 0.90


# TP/FP validation artifact for redundancy.
#
# The filter only exports every removed pair.
# The labels are NOT required to run the filter.
# After the run, upload this CSV and it can be reviewed/labelled
# externally as:
#   TP = correctly removed as redundant
#   FP = incorrectly removed as redundant
#
# Filtering never uses these labels to make decisions.
REDUNDANCY_AUDIT_CSV = (
    RESULTS_DIR
    / "case1_redundancy_tp_fp_audit.csv"
)



# ---------------------------------------------------------
# CONSISTENCY
# ---------------------------------------------------------
# Lightweight, high-precision consistency filtering.
#
# NLI is intentionally NOT used here.
#
# For each later span Sj in a surviving route:
#
#   parent =
#       argmax attribution[Sj, previous_span]
#
# over previous reasoning spans in the SAME route.
#
# Then:
#
#   1) ASPECT GATE
#      Check whether parent/current actually discuss at least one
#      shared content aspect.
#
#   2) POLARITY-CONFLICT CHECK
#      Only inside that shared aspect, look for explicit semantic
#      opposition such as:
#
#          "X is ..."      vs "X is not ..."
#          "continue X"    vs "stop X"
#          "increase X"    vs "decrease X"
#
# A route is hard-dropped only when an explicit polarity conflict
# is found on a shared aspect.
#
# Different topics / topic transitions are NOT contradictions.
#
# This is deliberately conservative:
#   false negatives are preferable to hard-dropping a valid
#   attribution-supported circuit because two different reasoning
#   topics happen to be semantically unrelated.
#
# No NLI model.
# No LLM judge.
# No learned consistency threshold.
# No fine-tuning data.
# ---------------------------------------------------------

# Local token window around a shared semantic anchor.
POLARITY_WINDOW = 4


# Generic discourse / function words are excluded from the
# aspect gate. These are not useful semantic anchors.
ASPECT_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then",
    "next", "first", "finally", "also", "maybe", "so",
    "to", "of", "in", "on", "for", "with", "at", "from",
    "by", "as", "about", "into", "that", "this", "these",
    "those", "it", "its", "they", "their", "them", "user",
    "i", "we", "you", "he", "she", "his", "her",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "can", "may", "might",
    "need", "needs", "needed", "want", "wants", "wanted",
    "make", "makes", "making", "sure",
    "important", "crucial",
    "response", "reasoning", "step",
}


# Explicit negative / blocking polarity.
NEGATIVE_POLARITY_CUES = {
    "not", "no", "never", "without",
    "avoid", "prevent", "stop", "discontinue",
    "reject", "oppose", "forbid",
    "cannot", "cant", "shouldnt", "wont", "dont",
    "isnt", "arent", "wasnt", "werent",
}


# Opposing action/stance pairs.
# These are generic lexical oppositions, not case-specific labels.
OPPOSITE_CUE_PAIRS = (
    ({"continue", "keep", "maintain"},
     {"stop", "discontinue", "end"}),

    ({"increase", "raise", "more"},
     {"decrease", "reduce", "less"}),

    ({"accept", "allow", "support", "encourage"},
     {"reject", "forbid", "oppose", "discourage"}),

    ({"seek", "approach"},
     {"avoid"}),

    ({"safe"},
     {"unsafe", "dangerous", "harmful"}),

    ({"helpful", "beneficial"},
     {"harmful", "damaging"}),
)


# =========================================================
# Helpers
# =========================================================

def reasoning_span_ids(route):
    return [
        node
        for node in route["route"]
        if node.startswith("S")
    ]


def route_semantic_text(route, span_text):
    """Ordered semantic representation for route redundancy."""
    ids = reasoning_span_ids(route)

    return "\n".join(
        f"Step {i}: {span_text[sid].strip()}"
        for i, sid in enumerate(ids, start=1)
    )



def structural_overlap_coefficient(route_a, route_b):
    """
    Fraction of the shorter route's reasoning spans that are
    shared with the other route.

    Since all S_j spans come from one autoregressive trace,
    shared span IDs automatically preserve chronological order.
    """
    ids_a = reasoning_span_ids(route_a)
    ids_b = reasoning_span_ids(route_b)

    set_a = set(ids_a)
    set_b = set(ids_b)

    shorter = min(
        len(set_a),
        len(set_b),
    )

    if shorter == 0:
        return 0.0

    return (
        len(set_a & set_b)
        / shorter
    )



def reasoning_edges(route):
    """
    Consecutive directed reasoning-span edges inside a route.

    Q and O are intentionally excluded. The redundancy guard asks
    whether two candidate circuits share at least one actual
    reasoning transition S_i -> S_j.
    """
    ids = reasoning_span_ids(route)

    return {
        (ids[i], ids[i + 1])
        for i in range(len(ids) - 1)
    }


def shared_reasoning_edges(route_a, route_b):
    """Directed reasoning edges shared by both routes."""
    return (
        reasoning_edges(route_a)
        & reasoning_edges(route_b)
    )



def redundancy_audit_record(
    representative,
    member,
    similarity,
    span_text,
):
    """
    Diagnostic only.

    This does NOT alter redundancy decisions. It exposes:
      - semantic similarity used by the filter
      - route structure
      - shared / unique span IDs
      - actual reasoning text

    This lets us inspect whether a high cosine score corresponds
    to a genuinely redundant reasoning circuit rather than merely
    the same broad topic.
    """
    rep_ids = reasoning_span_ids(
        representative
    )

    member_ids = reasoning_span_ids(
        member
    )

    rep_set = set(
        rep_ids
    )

    member_set = set(
        member_ids
    )

    union = (
        rep_set
        | member_set
    )

    intersection = (
        rep_set
        & member_set
    )

    span_jaccard = (
        len(intersection)
        / len(union)
        if union
        else 1.0
    )

    return {
        "representative_candidate_id": (
            representative.get(
                "candidate_id"
            )
        ),
        "representative_route_id": (
            representative.get(
                "route_id"
            )
        ),
        "member_candidate_id": (
            member.get(
                "candidate_id"
            )
        ),
        "member_route_id": (
            member.get(
                "route_id"
            )
        ),
        "cosine_similarity": (
            float(
                similarity
            )
        ),
        "representative_flow": (
            float(
                representative[
                    "flow"
                ]
            )
        ),
        "member_flow": (
            float(
                member[
                    "flow"
                ]
            )
        ),
        "representative_path": (
            representative[
                "route"
            ]
        ),
        "member_path": (
            member[
                "route"
            ]
        ),
        "shared_span_ids": (
            sorted(
                intersection,
                key=lambda x: int(
                    x[1:]
                ),
            )
        ),
        "representative_only_span_ids": (
            [
                sid
                for sid in rep_ids
                if sid not in member_set
            ]
        ),
        "member_only_span_ids": (
            [
                sid
                for sid in member_ids
                if sid not in rep_set
            ]
        ),
        "span_set_jaccard": (
            float(
                span_jaccard
            )
        ),
        "shorter_route_coverage": (
            float(
                len(intersection)
                / min(
                    len(rep_set),
                    len(member_set),
                )
            )
            if min(
                len(rep_set),
                len(member_set),
            ) > 0
            else 0.0
        ),
        "shared_directed_reasoning_edges": [
            [src, dst]
            for src, dst in sorted(
                shared_reasoning_edges(
                    representative,
                    member,
                ),
                key=lambda edge: (
                    int(edge[0][1:]),
                    int(edge[1][1:]),
                ),
            )
        ],
        "representative_text": (
            route_semantic_text(
                representative,
                span_text,
            )
        ),
        "member_text": (
            route_semantic_text(
                member,
                span_text,
            )
        ),
    }




def normalize_token(token):
    """
    Lightweight normalization only.
    Avoids introducing another NLP model / dependency.
    """
    token = token.lower()
    token = token.replace("’", "'")

    contraction_map = {
        "can't": "cant",
        "cannot": "cannot",
        "don't": "dont",
        "doesn't": "dont",
        "didn't": "dont",
        "isn't": "isnt",
        "aren't": "arent",
        "wasn't": "wasnt",
        "weren't": "werent",
        "shouldn't": "shouldnt",
        "won't": "wont",
    }

    token = contraction_map.get(
        token,
        token,
    )

    token = re.sub(
        r"[^a-z0-9']+",
        "",
        token,
    )

    if not token:
        return ""

    # Very small morphology normalization.
    # Enough for cases such as dieting -> diet,
    # bullying -> bully, parents -> parent, thoughts -> thought.
    if len(token) > 5 and token.endswith("ying"):
        token = token[:-4] + "y"

    elif len(token) > 5 and token.endswith("ing"):
        token = token[:-3]

        # stopping -> stopp -> stop
        if (
            len(token) >= 2
            and token[-1] == token[-2]
        ):
            token = token[:-1]

    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"

    elif len(token) > 3 and token.endswith("s"):
        token = token[:-1]

    return token


def tokenize_normalized(text):
    raw_tokens = re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?",
        text.lower(),
    )

    return [
        token
        for token in (
            normalize_token(t)
            for t in raw_tokens
        )
        if token
    ]


def content_terms(text):
    """
    Semantic anchors used by the aspect gate.
    """
    return {
        token
        for token in tokenize_normalized(text)
        if (
            token not in ASPECT_STOPWORDS
            and len(token) >= 3
        )
    }


def shared_aspect_terms(text_a, text_b):
    """
    Aspect gate.

    If there is no shared content anchor, the two spans are treated
    as different topics / aspects rather than contradictory.
    """
    return (
        content_terms(text_a)
        & content_terms(text_b)
    )


def local_windows(tokens, anchor, radius=POLARITY_WINDOW):
    """
    Return token windows around every occurrence of an anchor.
    """
    windows = []

    for i, token in enumerate(tokens):
        if token != anchor:
            continue

        lo = max(
            0,
            i - radius,
        )

        hi = min(
            len(tokens),
            i + radius + 1,
        )

        windows.append(
            set(tokens[lo:hi])
        )

    return windows


def local_negative_polarity(tokens, anchor):
    """
    True when an explicit negative/blocking cue appears locally
    around the shared semantic anchor.
    """
    windows = local_windows(
        tokens,
        anchor,
    )

    return any(
        bool(
            window
            & NEGATIVE_POLARITY_CUES
        )
        for window in windows
    )


def local_opposite_actions(
    tokens_a,
    tokens_b,
    anchor,
):
    """
    Detect explicit opposing lexical actions around the same
    semantic anchor.
    """
    windows_a = local_windows(
        tokens_a,
        anchor,
    )

    windows_b = local_windows(
        tokens_b,
        anchor,
    )

    if not windows_a or not windows_b:
        return []

    cues_a = set().union(
        *windows_a
    )

    cues_b = set().union(
        *windows_b
    )

    conflicts = []

    for positive_set, negative_set in OPPOSITE_CUE_PAIRS:

        a_positive = bool(
            cues_a
            & positive_set
        )

        a_negative = bool(
            cues_a
            & negative_set
        )

        b_positive = bool(
            cues_b
            & positive_set
        )

        b_negative = bool(
            cues_b
            & negative_set
        )

        if (
            a_positive
            and b_negative
        ):
            conflicts.append(
                {
                    "type": "opposite_action",
                    "a_cues": sorted(
                        cues_a
                        & positive_set
                    ),
                    "b_cues": sorted(
                        cues_b
                        & negative_set
                    ),
                }
            )

        elif (
            a_negative
            and b_positive
        ):
            conflicts.append(
                {
                    "type": "opposite_action",
                    "a_cues": sorted(
                        cues_a
                        & negative_set
                    ),
                    "b_cues": sorted(
                        cues_b
                        & positive_set
                    ),
                }
            )

    return conflicts


def polarity_conflict_check(
    parent_text,
    current_text,
):
    """
    High-precision hard-consistency check.

    A relation is contradictory only when:
      1) parent/current share an explicit semantic aspect, AND
      2) at least one shared anchor has explicit opposite polarity.

    This intentionally does NOT call different topics contradictory.
    """
    shared_terms = sorted(
        shared_aspect_terms(
            parent_text,
            current_text,
        )
    )

    result = {
        "same_aspect": bool(
            shared_terms
        ),
        "shared_aspect_terms": (
            shared_terms
        ),
        "hard_contradiction": False,
        "conflicts": [],
    }

    if not shared_terms:
        return result

    parent_tokens = tokenize_normalized(
        parent_text
    )

    current_tokens = tokenize_normalized(
        current_text
    )

    conflicts = []

    for anchor in shared_terms:

        parent_negative = (
            local_negative_polarity(
                parent_tokens,
                anchor,
            )
        )

        current_negative = (
            local_negative_polarity(
                current_tokens,
                anchor,
            )
        )

        # Explicit polarity reversal:
        # same semantic anchor, but only one side negates/blocks it.
        if (
            parent_negative
            != current_negative
        ):
            conflicts.append(
                {
                    "anchor": anchor,
                    "type": (
                        "negation_polarity_mismatch"
                    ),
                    "parent_negative": (
                        parent_negative
                    ),
                    "current_negative": (
                        current_negative
                    ),
                }
            )

        # Generic opposite-action vocabulary around the same anchor.
        for conflict in local_opposite_actions(
            parent_tokens,
            current_tokens,
            anchor,
        ):
            conflict = dict(
                conflict
            )
            conflict["anchor"] = (
                anchor
            )
            conflicts.append(
                conflict
            )

    # Remove exact duplicate conflict records.
    unique_conflicts = []
    seen = set()

    for conflict in conflicts:
        key = json.dumps(
            conflict,
            sort_keys=True,
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        unique_conflicts.append(
            conflict
        )

    result["conflicts"] = (
        unique_conflicts
    )

    result["hard_contradiction"] = bool(
        unique_conflicts
    )

    return result





def _redundancy_pair_key(rep_route_id, member_route_id):
    return (
        str(rep_route_id),
        str(member_route_id),
    )


def load_existing_redundancy_tp_fp_labels(csv_path):
    """
    Keep labels already entered by the user across reruns.
    """
    labels = {}

    if not csv_path.exists():
        return labels

    with csv_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = (
                row.get("tp_fp_label", "")
                .strip()
                .upper()
            )

            if label not in {"TP", "FP"}:
                continue

            key = _redundancy_pair_key(
                row.get("representative_route_id", ""),
                row.get("member_route_id", ""),
            )

            labels[key] = label

    return labels


def write_redundancy_tp_fp_audit(audit_records, csv_path):
    """
    Export EVERY route that was removed by redundancy filtering.

    tp_fp_label:
        TP = correctly removed as redundant
        FP = should NOT have been removed

    Existing TP/FP labels are preserved on rerun.
    """
    old_labels = (
        load_existing_redundancy_tp_fp_labels(
            csv_path
        )
    )

    fieldnames = [
        "tp_fp_label",
        "representative_candidate_id",
        "representative_route_id",
        "member_candidate_id",
        "member_route_id",
        "cosine_similarity",
        "representative_flow",
        "member_flow",
        "shared_directed_edges",
        "shared_span_ids",
        "span_set_jaccard",
        "shorter_route_coverage",
        "representative_path",
        "member_path",
        "representative_text",
        "member_text",
    ]

    rows = []

    for item in audit_records:
        key = _redundancy_pair_key(
            item.get("representative_route_id", ""),
            item.get("member_route_id", ""),
        )

        shared_edges = "; ".join(
            f"{src}->{dst}"
            for src, dst
            in item.get(
                "shared_directed_reasoning_edges",
                [],
            )
        )

        row = {
            "tp_fp_label": old_labels.get(key, ""),
            "representative_candidate_id": (
                item.get("representative_candidate_id", "")
            ),
            "representative_route_id": (
                item.get("representative_route_id", "")
            ),
            "member_candidate_id": (
                item.get("member_candidate_id", "")
            ),
            "member_route_id": (
                item.get("member_route_id", "")
            ),
            "cosine_similarity": (
                f"{item.get('cosine_similarity', 0.0):.6f}"
            ),
            "representative_flow": (
                f"{item.get('representative_flow', 0.0):.10f}"
            ),
            "member_flow": (
                f"{item.get('member_flow', 0.0):.10f}"
            ),
            "shared_directed_edges": shared_edges,
            "shared_span_ids": "; ".join(
                item.get("shared_span_ids", [])
            ),
            "span_set_jaccard": (
                f"{item.get('span_set_jaccard', 0.0):.6f}"
            ),
            "shorter_route_coverage": (
                f"{item.get('shorter_route_coverage', 0.0):.6f}"
            ),
            "representative_path": " -> ".join(
                item.get("representative_path", [])
            ),
            "member_path": " -> ".join(
                item.get("member_path", [])
            ),
            "representative_text": (
                item.get("representative_text", "")
            ),
            "member_text": (
                item.get("member_text", "")
            ),
        }

        rows.append(row)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def summarize_redundancy_tp_fp(rows):
    """
    TP/FP summary over manually reviewed redundancy removals.

    precision = TP / (TP + FP)

    Unlabeled rows are excluded from precision.
    """
    tp = 0
    fp = 0

    for row in rows:
        label = (
            row.get("tp_fp_label", "")
            .strip()
            .upper()
        )

        if label == "TP":
            tp += 1
        elif label == "FP":
            fp += 1

    reviewed = tp + fp
    total = len(rows)
    unreviewed = total - reviewed

    precision = (
        tp / reviewed
        if reviewed > 0
        else None
    )

    return {
        "total_removed": total,
        "reviewed": reviewed,
        "tp": tp,
        "fp": fp,
        "unreviewed": unreviewed,
        "precision": precision,
    }



# =========================================================
# 1. Load routes + attribution matrix
# =========================================================

total_start = time.perf_counter()

with open(FLOW_PATH, "r", encoding="utf-8") as f:
    flow_data = json.load(f)

with open(MATRIX_PATH, "r", encoding="utf-8") as f:
    matrix_data = json.load(f)


routes = flow_data.get(
    "multi_hop_routes",
    flow_data.get("selected_routes", []),
)


labels = matrix_data["labels"]

label_to_index = {
    label: i
    for i, label in enumerate(labels)
}


attribution = np.array(
    [
        [
            0.0 if value is None else float(value)
            for value in row
        ]
        for row in matrix_data["normalized_matrix"]
    ],
    dtype=float,
)


span_text = {
    span["id"]: span["text"]
    for span in matrix_data["reasoning_spans"]
}


# Validate multi-hop input.
for route in routes:
    ids = reasoning_span_ids(route)

    if len(ids) < 2:
        raise ValueError(
            "Expected multi-hop route only, received: "
            f"{route.get('route_id')} "
            f"{' -> '.join(route['route'])}"
        )

    for sid in ids:
        if sid not in label_to_index:
            raise KeyError(
                f"{sid} missing from attribution labels."
            )

        if sid not in span_text:
            raise KeyError(
                f"{sid} missing from reasoning_spans."
            )


# Strongest flow first.
# This ordering is used only to choose the representative
# when redundant routes are found.
routes = sorted(
    routes,
    key=lambda r: float(r["flow"]),
    reverse=True,
)


print(f"Input multi-hop circuit routes: {len(routes)}")


# =========================================================
# 2. Route-level semantic REDUNDANCY
# =========================================================

redundancy_start = time.perf_counter()

print(f"\nLoading redundancy model: {EMBED_MODEL}")

embedding_model = SentenceTransformer(
    EMBED_MODEL
)


route_texts = [
    route_semantic_text(
        route,
        span_text,
    )
    for route in routes
]


print("Encoding route semantics...")

route_embeddings = embedding_model.encode(
    route_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
)


similarity_matrix = (
    route_embeddings
    @ route_embeddings.T
)


# Flow-prioritized semantic deduplication.
#
# No clustering / K / silhouette / elbow here.
# Each route is compared against already-kept representatives.
# If it is a semantic near-duplicate, it joins the most similar
# qualifying representative. Since representatives are created
# in descending flow order, the retained route is mechanically
# stronger.
clusters = []
representative_indices = []


for route_index, route in enumerate(routes):

    best_cluster_index = None
    best_similarity = -np.inf

    for cluster_index, rep_index in enumerate(
        representative_indices
    ):
        similarity = float(
            similarity_matrix[
                route_index,
                rep_index,
            ]
        )

        shared_edges = (
            shared_reasoning_edges(
                route,
                routes[rep_index],
            )
        )

        if (
            shared_edges
            and similarity
            >= REDUNDANCY_SIM_THRESHOLD
            and similarity > best_similarity
        ):
            best_similarity = similarity
            best_cluster_index = cluster_index


    if best_cluster_index is None:
        representative_indices.append(
            route_index
        )

        clusters.append(
            {
                "representative": route,
                "members": [
                    {
                        "route": route,
                        "similarity_to_representative": 1.0,
                    }
                ],
            }
        )

    else:
        clusters[
            best_cluster_index
        ]["members"].append(
            {
                "route": route,
                "similarity_to_representative": best_similarity,
            }
        )


for i, cluster in enumerate(
    clusters,
    start=1,
):
    cluster["cluster_id"] = f"C{i}"


representatives = [
    cluster["representative"]
    for cluster in clusters
]


# ---------------------------------------------------------
# REDUNDANCY AUDIT
# ---------------------------------------------------------
# Diagnostic only: no route is added/removed here.
#
# Every actual merge receives an audit record containing:
#   cosine similarity,
#   structural span overlap,
#   paths,
#   and complete reasoning text.
#
# We also print a small set of the structurally least-overlapping
# merges. Those are useful human spot-checks because they can
# reveal "same topic" merges that are not truly redundant.
redundancy_audit = []

for cluster in clusters:
    representative = cluster[
        "representative"
    ]

    for member in cluster[
        "members"
    ][1:]:

        redundancy_audit.append(
            redundancy_audit_record(
                representative=representative,
                member=member[
                    "route"
                ],
                similarity=member[
                    "similarity_to_representative"
                ],
                span_text=span_text,
            )
        )


redundancy_seconds = (
    time.perf_counter()
    - redundancy_start
)


print("\n========================================")
print("SEMANTIC REDUNDANCY")
print("========================================")
print(
    f"Routes before redundancy: "
    f"{len(routes)}"
)
print(
    f"Routes after redundancy: "
    f"{len(representatives)}"
)
print(
    f"Routes merged as redundant: "
    f"{len(routes) - len(representatives)}"
)
print(
    f"Redundancy stage time: "
    f"{redundancy_seconds:.2f}s"
)



print(
    f"Redundancy merges available for audit: "
    f"{len(redundancy_audit)}"
)


# Export ALL removed-as-redundant pairs for TP/FP validation.
redundancy_tp_fp_rows = (
    write_redundancy_tp_fp_audit(
        redundancy_audit,
        REDUNDANCY_AUDIT_CSV,
    )
)

redundancy_tp_fp_summary = (
    summarize_redundancy_tp_fp(
        redundancy_tp_fp_rows
    )
)

print(
    f"TP/FP audit CSV: "
    f"{REDUNDANCY_AUDIT_CSV}"
)

print(
    "\n=== REDUNDANCY TP/FP VALIDATION ==="
)

print(
    f"Total removed pairs: "
    f"{redundancy_tp_fp_summary['total_removed']}"
)

print(
    f"Reviewed: "
    f"{redundancy_tp_fp_summary['reviewed']}"
)

print(
    f"TP (correctly removed): "
    f"{redundancy_tp_fp_summary['tp']}"
)

print(
    f"FP (incorrectly removed): "
    f"{redundancy_tp_fp_summary['fp']}"
)

print(
    f"Unreviewed: "
    f"{redundancy_tp_fp_summary['unreviewed']}"
)

if (
    redundancy_tp_fp_summary[
        "precision"
    ]
    is None
):
    print(
        "Redundancy precision: "
        "N/A (audit CSV is ready for external TP/FP review)"
    )
else:
    print(
        "Redundancy precision: "
        f"{redundancy_tp_fp_summary['precision']:.3f}"
    )


if redundancy_audit:

    print(
        "\n=== REDUNDANCY AUDIT: "
        "LOWEST STRUCTURAL-OVERLAP MERGES ==="
    )

    # DISPLAY ONLY.
    # Does not affect filtering.
    audit_display = sorted(
        redundancy_audit,
        key=lambda item: (
            item[
                "span_set_jaccard"
            ],
            -item[
                "cosine_similarity"
            ],
        ),
    )[:12]

    for item in audit_display:

        print(
            "\n"
            f"{item['representative_candidate_id']} "
            f"({item['representative_route_id']}) "
            "<- "
            f"{item['member_candidate_id']} "
            f"({item['member_route_id']})"
        )

        print(
            f"  cosine="
            f"{item['cosine_similarity']:.4f}, "
            f"span_jaccard="
            f"{item['span_set_jaccard']:.3f}, "
            f"shorter_coverage="
            f"{item['shorter_route_coverage']:.3f}"
        )

        print(
            "  representative: "
            + " -> ".join(
                item[
                    "representative_path"
                ]
            )
        )

        print(
            "  member:         "
            + " -> ".join(
                item[
                    "member_path"
                ]
            )
        )

        print(
            "  shared spans: "
            + (
                ", ".join(
                    item[
                        "shared_span_ids"
                    ]
                )
                if item[
                    "shared_span_ids"
                ]
                else "(none)"
            )
        )


        print(
            "  shared directed edges: "
            + (
                ", ".join(
                    f"{src}->{dst}"
                    for src, dst
                    in item[
                        "shared_directed_reasoning_edges"
                    ]
                )
                if item[
                    "shared_directed_reasoning_edges"
                ]
                else "(none)"
            )
        )


# =========================================================
# 3. Attribution-guided semantic CONSISTENCY
# =========================================================
#
# Mechanical attribution chooses WHICH semantic relation matters.
#
# For each current span:
#
#   parent =
#       argmax attribution[current, previous_span]
#
# over previous reasoning spans actually present in the route.
#
# Then a lightweight semantic rule checks:
#
#   shared aspect?
#       no  -> KEEP
#       yes -> explicit polarity conflict?
#                  yes -> DROP route
#                  no  -> KEEP
#
# This prevents topic transitions such as:
#
#   "address body image"
#       -> "next, bullying"
#
# from being mislabeled as contradiction merely because the two
# spans discuss different subjects.
# =========================================================

consistency_start = time.perf_counter()

consistency_relations = []


for rep_index, route in enumerate(
    representatives
):

    ids = reasoning_span_ids(
        route
    )

    for current_position in range(
        1,
        len(ids),
    ):

        current_sid = ids[
            current_position
        ]

        previous_sids = ids[
            :current_position
        ]

        current_idx = label_to_index[
            current_sid
        ]


        parent_scores = [
            float(
                attribution[
                    current_idx,
                    label_to_index[
                        parent_sid
                    ],
                ]
            )
            for parent_sid
            in previous_sids
        ]


        dominant_position = int(
            np.argmax(
                parent_scores
            )
        )

        parent_sid = previous_sids[
            dominant_position
        ]

        parent_score = parent_scores[
            dominant_position
        ]


        parent_text = (
            span_text[
                parent_sid
            ].strip()
        )

        current_text = (
            span_text[
                current_sid
            ].strip()
        )


        semantic_check = (
            polarity_conflict_check(
                parent_text,
                current_text,
            )
        )


        consistency_relations.append(
            {
                "representative_index": (
                    rep_index
                ),
                "parent_span_id": (
                    parent_sid
                ),
                "current_span_id": (
                    current_sid
                ),
                "parent_attribution": (
                    float(
                        parent_score
                    )
                ),
                "candidate_parent_scores": {
                    sid: float(score)
                    for sid, score
                    in zip(
                        previous_sids,
                        parent_scores,
                    )
                },
                "parent_text": (
                    parent_text
                ),
                "current_text": (
                    current_text
                ),
                "same_aspect": (
                    semantic_check[
                        "same_aspect"
                    ]
                ),
                "shared_aspect_terms": (
                    semantic_check[
                        "shared_aspect_terms"
                    ]
                ),
                "hard_contradiction": (
                    semantic_check[
                        "hard_contradiction"
                    ]
                ),
                "conflicts": (
                    semantic_check[
                        "conflicts"
                    ]
                ),
            }
        )


consistency_records = [
    {
        "route": route,
        "is_consistent": True,
        "checks": [],
    }
    for route in representatives
]


for relation in consistency_relations:

    record = consistency_records[
        relation[
            "representative_index"
        ]
    ]

    check = {
        key: value
        for key, value
        in relation.items()
        if key != "representative_index"
    }

    record["checks"].append(
        check
    )

    if check[
        "hard_contradiction"
    ]:
        record[
            "is_consistent"
        ] = False


consistent_routes = [
    record["route"]
    for record in consistency_records
    if record["is_consistent"]
]


inconsistent_records = [
    record
    for record in consistency_records
    if not record["is_consistent"]
]


same_aspect_relation_count = sum(
    1
    for relation
    in consistency_relations
    if relation[
        "same_aspect"
    ]
)


hard_conflict_relation_count = sum(
    1
    for relation
    in consistency_relations
    if relation[
        "hard_contradiction"
    ]
)


consistency_seconds = (
    time.perf_counter()
    - consistency_start
)


print("\n========================================")
print("ATTRIBUTION-GUIDED CONSISTENCY")
print("========================================")
print(
    f"Representative routes checked: "
    f"{len(representatives)}"
)
print(
    f"Attribution-guided relations: "
    f"{len(consistency_relations)}"
)
print(
    f"Relations passing aspect gate: "
    f"{same_aspect_relation_count}"
)
print(
    f"Explicit polarity-conflict relations: "
    f"{hard_conflict_relation_count}"
)
print(
    f"Routes with contradiction: "
    f"{len(inconsistent_records)}"
)
print(
    f"Routes passing consistency: "
    f"{len(consistent_routes)}"
)
print(
    f"Consistency stage time: "
    f"{consistency_seconds:.4f}s"
)


if inconsistent_records:

    print(
        "\n=== INCONSISTENT ROUTES ==="
    )

    for record in inconsistent_records:

        route = record[
            "route"
        ]

        print(
            f"\n"
            f"{route.get('candidate_id')} "
            f"({route.get('route_id')})  "
            f"{' -> '.join(route['route'])}"
        )

        for check in record[
            "checks"
        ]:

            if not check[
                "hard_contradiction"
            ]:
                continue

            print(
                "  POLARITY CONFLICT: "
                f"{check['parent_span_id']} "
                f"-> "
                f"{check['current_span_id']} "
                f"[attr="
                f"{check['parent_attribution']:.6f}]"
            )

            print(
                "    shared aspect: "
                + ", ".join(
                    check[
                        "shared_aspect_terms"
                    ]
                )
            )

            print(
                "    parent: "
                f"{check['parent_text']}"
            )

            print(
                "    current: "
                f"{check['current_text']}"
            )

            for conflict in check[
                "conflicts"
            ]:
                print(
                    "    conflict: "
                    f"{conflict}"
                )


# =========================================================
# 4. Final set for downstream MCTS
# =========================================================
#
# IMPORTANT:
# We do NOT choose MCTS top-K here.
#
# This file performs semantic filtering only:
#
#   mechanical multi-hop circuits
#       -> redundancy consolidation
#       -> attribution-guided consistency validation
#
# The final routes remain sorted by original attribution flow.
#
# If the MCTS implementation uses a maximum branch factor K,
# THAT search-budget parameter should select:
#
#       selected_routes[:K]
#
# at the relevant expansion step.
#
# This keeps two ideas cleanly separated:
#
#   semantic filtering:
#       "Which circuits are distinct and coherent?"
#
#   MCTS branch budget:
#       "How many of the strongest surviving actions can
#        this search node afford to expand?"
#
# Thus K is justified by the MCTS search budget, rather than
# being introduced as another attribution / semantic threshold.
# =========================================================

consistent_routes = sorted(
    consistent_routes,
    key=lambda r: float(r["flow"]),
    reverse=True,
)


total_seconds = (
    time.perf_counter()
    - total_start
)


print("\n========================================")
print("FINAL CIRCUIT SET")
print("========================================")
print(
    f"Raw multi-hop circuits: "
    f"{len(routes)}"
)
print(
    f"After redundancy: "
    f"{len(representatives)}"
)
print(
    f"After consistency: "
    f"{len(consistent_routes)}"
)
print(
    f"Total filter time: "
    f"{total_seconds:.2f}s"
)


print(
    "\n=== TOP FINAL CIRCUITS "
    "(display only; MCTS K is NOT applied here) ==="
)

for route in consistent_routes[
    :30
]:

    print(
        f"{route.get('candidate_id', ''):>4}  "
        f"({route.get('route_id', '')})  "
        f"{' -> '.join(route['route'])}  "
        f"[flow="
        f"{float(route['flow']):.8f}]"
    )


# =========================================================
# 5. Save
# =========================================================

json_clusters = []

for cluster in clusters:
    json_clusters.append(
        {
            "cluster_id": (
                cluster[
                    "cluster_id"
                ]
            ),
            "representative": (
                cluster[
                    "representative"
                ]
            ),
            "members": [
                {
                    "route": member["route"],
                    "similarity_to_representative": float(
                        member[
                            "similarity_to_representative"
                        ]
                    ),
                }
                for member
                in cluster["members"]
            ],
        }
    )


result = {
    "method": (
        "shared-edge-gated SentenceTransformer redundancy "
        "+ attribution-guided aspect/polarity consistency"
    ),

    "design": {
        "redundancy": (
            "Two multi-hop routes are eligible for semantic "
            "deduplication only when they share at least one "
            "directed reasoning edge. Eligible routes are then "
            "compared with SentenceTransformer cosine similarity; "
            "a semantic near-duplicate retains the highest-flow "
            "representative."
        ),
        "consistency": (
            "For each later reasoning span, attribution argmax "
            "selects the mechanically dominant previous span. "
            "A lightweight aspect gate first checks whether the "
            "two spans share an explicit semantic content anchor. "
            "Only then is explicit negation/opposite-action "
            "polarity checked. A route is rejected only when "
            "a polarity conflict is found on a shared aspect."
        ),
        "mcts_handoff": (
            "No MCTS top-K is applied in this script. "
            "Final routes are sorted by flow so a downstream "
            "MCTS node with branch factor K can take the "
            "top K surviving circuit candidates."
        ),
    },

    "embedding_model": (
        EMBED_MODEL
    ),

    "redundancy_similarity_threshold": (
        REDUNDANCY_SIM_THRESHOLD
    ),

    "input_multi_hop_route_count": (
        len(routes)
    ),

    "after_redundancy_count": (
        len(representatives)
    ),

    "redundant_route_count": (
        len(routes)
        - len(representatives)
    ),

    "attribution_guided_relation_count": (
        len(consistency_relations)
    ),

    "same_aspect_relation_count": (
        same_aspect_relation_count
    ),

    "hard_polarity_conflict_relation_count": (
        hard_conflict_relation_count
    ),

    "inconsistent_route_count": (
        len(inconsistent_records)
    ),

    "final_route_count": (
        len(consistent_routes)
    ),

    # Downstream MCTS should consume this list.
    # It is sorted by attribution flow descending.
    "selected_routes": (
        consistent_routes
    ),

    "redundancy_clusters": (
        json_clusters
    ),

    # Diagnostic only. This does not affect which routes survive.
    "redundancy_audit": (
        redundancy_audit
    ),

    "redundancy_tp_fp_summary": (
        redundancy_tp_fp_summary
    ),


    "consistency_records": (
        consistency_records
    ),

    "timing_seconds": {
        "redundancy": float(
            redundancy_seconds
        ),
        "consistency": float(
            consistency_seconds
        ),
        "total": float(
            total_seconds
        ),
    },
}


with open(
    OUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False,
    )


print(f"\nSaved: {OUT_PATH}")
