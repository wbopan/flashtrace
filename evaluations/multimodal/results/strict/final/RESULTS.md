# Strict Wiki-VISA and CLEVR-XAI pilot results

These are small, controlled protocol-validation runs, not full-scale benchmark
claims. Dataset inputs, model generations, and evaluation metadata are stored
in separate JSONL files. Every attribution method teacher-forces the same full
model-generated `THINKING + OUTPUT` response and scores `OUTPUT_SPAN` only.
FlashTrace uses the paper's cumulative input attribution: the direct
`OUTPUT -> IMAGE` term plus the exact-THINKING recursive term weighted by the
attribution mass entering `THINKING`.

## Frozen evaluation sets

| dataset | n | balanced groups | Qwen THINKING tokens |
|---|---:|---|---:|
| Wiki-VISA | 18 | first-page / later-page / non-passage = 6 / 6 / 6 | 52-306 |
| CLEVR-XAI complex | 20 | query / attribute compare / count / exist / integer compare = 4 each | 78-1003 |

All 38 records are correct under the dataset normalizer, stable across two
greedy generations, exact generated-token matches for teacher forcing, positive
under the blur log-probability gate, and fail to reproduce the original whole
output under at least one deterministic blur/gray generation ablation. The
model is `Qwen/Qwen3-VL-8B-Thinking` at revision
`92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`.

The complete source data verified on disk is Wiki-VISA test (3,000 rows in
seven official parquet shards) and CLEVR-XAI complex (100,000 questions,
10,000 images/masks, and all four official evidence variants).

## Wiki-VISA localization

All 11 methods succeeded on the same 18 samples. Higher is better for every
metric below.

| method | Point | Energy | Rank AUC | Top IoU | R@5 | R@20 |
|---|---:|---:|---:|---:|---:|---:|
| FlashTrace, exact THINKING | 0.278 | 0.145 | **0.826** | 0.117 | **0.343** | **0.692** |
| FlashTrace, all-generation ablation | **0.333** | 0.135 | 0.809 | 0.102 | 0.311 | 0.662 |
| AttnLRP adaptation | 0.222 | **0.209** | 0.687 | **0.1173** | 0.299 | 0.588 |
| IFR-span | 0.222 | 0.119 | 0.764 | 0.082 | 0.244 | 0.580 |
| Grad x Attention | 0.056 | 0.084 | 0.773 | 0.069 | 0.238 | 0.588 |
| Visual LOO | 0.000 | 0.111 | 0.709 | 0.089 | 0.185 | 0.535 |
| Visual IG | 0.056 | 0.054 | 0.493 | 0.038 | 0.101 | 0.285 |
| Random | 0.000 | 0.038 | 0.496 | 0.017 | 0.043 | 0.191 |
| Center | 0.000 | 0.030 | 0.504 | 0.013 | 0.036 | 0.215 |

Paired 10,000-draw bootstrap differences support the exact bridge over
IFR-span for Rank AUC (`+0.0616`, 95% CI `[+0.0342,+0.0958]`) and R@5
(`+0.0995`, `[+0.0589,+0.1488]`). Exact THINKING also exceeds the broader
all-generation bridge in Rank AUC (`+0.0165`, `[+0.0085,+0.0271]`).

Independent manual audit inspected all 18 sheets. It supports FlashTrace as the
most consistent supporting-element coverage/ranking method and AttnLRP as the
stronger concentration method. Strong FlashTrace examples include `0072`,
`1115`, `1519`, `1722`, `2022`, `2606`, and `2889`; `1341`, `2086`, and
`2619` are clear weak or metric-inflated cases. Wiki boxes mark supporting HTML
elements rather than answer strings, may omit alternative valid answer
locations, and have a strong page-position prior. These scores must not be
described as strict word-level grounding.

## CLEVR-XAI complex localization

The official primary view is Unique First-nonempty. All 11 methods succeeded
on the same balanced 20 samples.

| method | Point | Energy | Rank AUC | Top IoU | R@5 | R@20 |
|---|---:|---:|---:|---:|---:|---:|
| Visual IG | **0.500** | **0.312** | 0.648 | **0.231** | **0.264** | 0.559 |
| Center | 0.150 | 0.203 | **0.821** | 0.160 | 0.134 | **0.574** |
| Visual LOO | 0.400 | 0.176 | 0.700 | 0.158 | 0.181 | 0.479 |
| AttnLRP adaptation | 0.100 | 0.175 | 0.675 | 0.152 | 0.163 | 0.465 |
| FlashTrace, exact THINKING | 0.050 | 0.057 | 0.621 | 0.014 | 0.014 | 0.184 |
| FlashTrace, all-generation ablation | 0.050 | 0.051 | 0.608 | 0.011 | 0.011 | 0.174 |
| Random | 0.000 | 0.087 | 0.494 | 0.043 | 0.048 | 0.186 |

Visual IG remains the semantic-localization winner in the union-mask
sensitivity view (Point `0.75`, Energy `0.496`, R@5 `0.203`), while center wins
Rank AUC (`0.850`) and R@20 (`0.555`). This center result is a geometry
diagnostic: 19/20 primary GT centroids and 20/20 union centroids lie within
0.25 normalized distance of the image center.

FlashTrace's primary Rank AUC is above random, but its Energy, Top IoU, and
R@20 are at or below random, especially for count/exist. Manual audit of all 20
images found that upper/left image-border attribution dominates many maps.
`query_attribute` is the only consistently credible family (primary FlashTrace
Energy/Rank/R@20 `0.106/0.695/0.396`). Exact THINKING has a small primary
advantage over all-generation attribution (Rank `+0.0131`, 95% CI
`[+0.0035,+0.0247]`), but the union Rank difference is not significant and the
two maps have mean cosine similarity `0.9969`. It would be incorrect to claim a
general CLEVR localization win for FlashTrace.

Unique First-nonempty can represent an intermediate program set rather than
all final evidence; union better represents chain completeness but is larger
and more center-confounded. Both views should remain in any paper table.

## Frozen-response visual faithfulness supplement

CLEVR uses a common approximately 64-region grid and 10 deletion/insertion
steps. The complete generated response is fixed and teacher-forced; only
`OUTPUT_SPAN` log probability is accumulated. All 20 x 11 pairs succeeded with
no degenerate curves. Lower deletion AUC / Visual-MAS and higher insertion AUC
are better.

| method | Deletion AUC | Insertion AUC | Visual-MAS |
|---|---:|---:|---:|
| Center | **0.362** | **0.854** | **0.454** |
| Visual LOO | 0.429 | 0.848 | 0.570 |
| AttnLRP adaptation | 0.447 | 0.808 | 0.586 |
| Visual IG | 0.465 | 0.850 | 0.607 |
| FlashTrace, exact THINKING | 0.497 | 0.746 | 0.615 |
| FlashTrace, all-generation ablation | 0.513 | 0.760 | 0.629 |
| Random | 0.707 | 0.661 | 0.814 |

Center's faithfulness result reinforces the dataset/model center bias rather
than establishing a semantic explanation method.

## Artifacts and caveats

- Wiki summary and bootstrap: `wiki_visa_n18_2mp_methods_v2/summary.json`,
  `analysis.md`.
- Wiki visual sheets: `wiki_visa_n18_2mp_method_comparisons_v2/`.
- CLEVR summary and primary/union bootstrap:
  `clevr_xai_complex_strict_n20_methods_v2/summary.json`,
  `analysis_primary.md`, and `analysis_union.md`.
- CLEVR visual sheets: `clevr_xai_complex_strict_n20_method_comparisons_v2/`.
- CLEVR frozen-response supplement:
  `clevr_xai_complex_strict_n20_faithfulness_v2/summary.json`.

The old `clevr_xai_complex_n20_methods/` directory predates the strict
generation-ablation gate and exact-THINKING bridge correction. It is diagnostic
history only and must be excluded from paper tables. Wiki attribution records
also retain two failed Grad-Attention OOM attempts before a successful retry;
all summaries and bootstrap analyses use the unique successful pair only.

The positive blur log-probability threshold is intentionally weak; minimum
deltas are `1.66e-5` (Wiki) and `2.82e-4` (CLEVR). Deterministic generation
ablations provide the stronger additional gate, but the results should still
be described as a small strict pilot (`n=18/20`), not a sufficiently powered
full benchmark.
