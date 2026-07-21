# Native-dataset n=10 fit pilot

Run date: 2026-07-18. This is an exploratory dataset-selection pilot, not a
paper-scale benchmark. Each set contains ten native, official questions; no
question, answer, rationale, or evidence annotation was synthesized or joined
from another dataset.

## Strict protocol used

- Qwen model: `Qwen/Qwen3-VL-8B-Thinking`, resolved revision
  `92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b`.
- Dataset, model generation, generation evaluation, attribution, and
  independent audit are separate files joined only by `sample_id`.
- `THINKING` and whole `OUTPUT` are model generated. Dataset explanations and
  programs never enter the prompt or model record.
- The attribution sink is the complete `OUTPUT_SPAN`.
- Primary `flashtrace` follows the paper exactly: direct `OUTPUT -> IMAGE`
  attribution is added to the `OUTPUT -> THINKING -> IMAGE` recursive hop,
  with the recursive contribution scaled by the fraction of attribution mass
  entering `THINKING`. Records use `observation_projected.sum`.
- Spatial metrics select complete visual patches. Native GT pixels are mapped
  to their containing patch, and cutoff ties receive expected credit under a
  uniform tie break. Nearest-neighbor expansion is used only to render hard
  patch overlays; bilinear heatmaps and partial-patch top-q selection are not
  used for the final scores below.
- VizWiz-LF has no native region GT. Its response is frozen and teacher-forced
  under every perturbation, while only `OUTPUT_SPAN` log probability is scored.

Wiki-VISA and VISTAQA used the concise task wrapper. VizWiz-LF used a generic
long-form answer instruction while preserving its native question unchanged;
its observed output-length result must therefore be described as prompted,
not as an unprompted property of the model.

## Generation and semantic audit

| dataset | first-pass parsed | 2048-token recovery | human whole-OUTPUT audit | median THINKING tokens | median OUTPUT tokens | OUTPUT >=32 tokens | native region GT |
|---|---:|---:|---|---:|---:|---:|---|
| Wiki-VISA | 8/10 | 0/2 recovered | 6 correct + 2 acceptable scope mismatches among 8 | 110.5 | 3 | 0/8 | element boxes |
| VISTAQA | 4/10 | 1/6 parsed, but wrong | 2 correct + 2 wrong among 4; 6 primary failures | 182 | 8.5 | 0/4 | pixel masks |
| VizWiz-LF | 9/10 | primary cap already 2048 | 6 fully correct + 3 partial + 1 run failure | 119 | 41 | 5/9 | none |

The recovery runs were written separately and did not overwrite primary
failures. Five of six VISTAQA failures still lacked `</think>` at 2048 tokens;
the sole recovered row used 1,027 THINKING tokens and incorrectly answered `2`
instead of `4`. Wiki row `0882` also remained unterminated at 2048; row `1531`
again failed exact generated-ID versus decode/re-encode identity.

Independent image review found only 2/8 genuinely non-trivial Wiki THINKING
traces and only 4/8 image-dependent outputs. For VizWiz-LF, 4/9 traces were
more than simple recognition/inventory; only sample `425` simultaneously had a
fully correct, long, image-dependent output, non-trivial OCR/package reasoning,
and a reasonably localizable positive target.

## Native-GT localization

The Wiki table is restricted to the four semantically acceptable, stable,
manually image-dependent rows (`1449`, `2114`, `2171`, `2870`). VISTAQA is
shown for its two semantically correct rows (`0656`, `0753`), but `0753`'s
official mask contains only the green answer object and omits the red comparison
object and robotic-arm base. Its values are therefore answer-object diagnostic
scores, not complete reasoning-evidence scores.

All metrics are higher-is-better.

### Wiki-VISA, audited paired n=4

| method | Recovery@5% | Recovery@20% | evidence rank AUC | energy in GT |
|---|---:|---:|---:|---:|
| random | 0.083 | 0.237 | 0.565 | 0.023 |
| center | 0.000 | 0.046 | 0.349 | 0.007 |
| Visual-IG | 0.081 | 0.278 | 0.510 | 0.023 |
| AttnLRP | 0.447 | 0.717 | 0.741 | **0.400** |
| FlashTrace | **0.560** | **0.883** | **0.927** | 0.259 |

FlashTrace covers and ranks the wide supporting passage bands more completely,
while AttnLRP concentrates more mass inside those bands. On paired bootstrap,
FlashTrace minus AttnLRP at Recovery@5% is `+0.113` with 95% CI
`[-0.073, +0.299]`; the small n does not establish a difference at the tight
budget. At Recovery@20% the delta is `+0.166`, 95% CI
`[+0.058,+0.273]`, and all four rows favor FlashTrace, but the boxes are only
thin, page-wide HTML elements, not word-level answer masks. AttnLRP retains
substantially greater energy concentration inside the boxes (`0.400` versus
`0.259`).

### VISTAQA, semantically correct paired n=2

| method | Recovery@5% | Recovery@20% | evidence rank AUC | energy in GT |
|---|---:|---:|---:|---:|
| random | 0.000 | 0.105 | 0.408 | 0.003 |
| center | 0.073 | 0.500 | 0.607 | 0.004 |
| Visual-IG | 0.368 | 0.381 | 0.684 | 0.008 |
| AttnLRP | 0.302 | 0.315 | 0.335 | 0.016 |
| FlashTrace | **0.775** | **0.979** | **0.936** | **0.122** |

The only clean complete-mask row is `0656` (two traffic signs). There,
FlashTrace Recovery@5% is `0.946`, and its maximum patch is visibly on the
`SLOW` sign. This one-row success is real but cannot support a dataset-level
claim. On `0753`, center reaches Recovery@10/20%=1 because the native mask is a
tiny centered answer object; the mask cannot evaluate the full comparison.
No post-hoc position removal is applied to the reported FlashTrace map.

## VizWiz-LF frozen-response visual faithfulness

This run uses the paper-aligned cumulative FlashTrace map and a common
approximately 36-region partition with five deletion/insertion steps. Lower is
better for deletion AUC and Visual-MAS; higher is better for insertion AUC.

| method | deletion AUC | insertion AUC | Visual-MAS |
|---|---:|---:|---:|
| random | 0.462 | 0.490 | 0.571 |
| center | 0.378 | **0.704** | 0.513 |
| Visual-IG | 0.436 | 0.499 | 0.550 |
| AttnLRP | 0.346 | 0.562 | 0.477 |
| FlashTrace | **0.318** | 0.597 | **0.440** |

FlashTrace's mean advantage over AttnLRP is small and statistically unresolved
at n=9:

| metric | favorable FlashTrace delta | paired 95% CI | W/T/L |
|---|---:|---:|---:|
| deletion AUC | +0.028 | [-0.020, +0.088] | 3/2/4 |
| insertion AUC | +0.035 | [-0.047, +0.125] | 3/1/5 |
| Visual-MAS | +0.037 | [-0.055, +0.155] | 3/1/5 |

Center's insertion result is consistent with a strong centered-subject prior in
these nine photographs. A second caveat is map sign: approximately 50% of
Visual-IG cells and 27% of AttnLRP cells are negative, whereas FlashTrace maps
are non-negative. Deletion/insertion ordering uses saved signed
scores, but the MAS density correction uses positive mass; Visual-MAS should
therefore remain secondary to the raw perturbation curves.

The confidence intervals cross zero for all three paired comparisons, so these
means remain directional pilot evidence rather than a universal method win.

## Dataset decision

No one of the three datasets satisfies all requirements.

- **Wiki-VISA: retain as a document/OCR localization diagnostic.** It has
  native boxes and mostly stable generation, but model OUTPUT is extremely
  short (median 3 tokens) and its apparent THINKING length is usually verbose
  lookup rather than deep reasoning.
- **VISTAQA: do not use as the long/open reasoning core.** It has official
  pixel masks and some relational questions, but Qwen Thinking termination is
  pathological even at 2048 tokens, outputs remain short, correctness is 2/10,
  and masks often annotate only the answer entity rather than all reasoning
  evidence.
- **VizWiz-LF: retain as the best of these three natural-image, long-OUTPUT
  stress tests.** It provides native open expert answers and median generated
  OUTPUT of 41 tokens, but it has no region GT, exact-match correctness is
  unusable, much of the long content is global negative evidence or uncertainty,
  and only 1/10 rows passes the full human reasoning/localizability gate.

Thus these three can play complementary diagnostic roles, but they do not yet
replace the missing benchmark: a popular, native natural-image task with
multi-token whole outputs, non-trivial model-generated reasoning, automatic
answer checking, and native complete evidence regions.

## Artifacts

- Dataset/model/evaluation records: `*_n10.dataset.jsonl`, `*_n10.model.jsonl`,
  and `*_n10.generation_eval.jsonl` in this directory.
- Non-overwriting recovery diagnostics:
  `vistaqa_n10.recovery2048.*` and `wiki_visa_n10.recovery2048.*`.
- Paper-aligned cumulative attribution records and summaries:
  `wiki_visa_n10_attribution/`, `vistaqa_n10_attribution/`, and
  `vizwiz_lf_n10_attribution/`.
- VizWiz faithfulness: `vizwiz_lf_n10_faithfulness/`.
- Judgment-only generation/semantic audits: `audits/`.
- Hard-cell, no-interpolation method comparisons: `visualizations/`.
- Explicit semantic/localization filter IDs: `analysis_subsets.json`.
- Paired 50,000-draw interval summary: `paired_bootstrap_summary.json`.
