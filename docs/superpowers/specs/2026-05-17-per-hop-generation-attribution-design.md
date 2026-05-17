# Per-Hop Generation-Side Attribution in the Traced View — Design

Date: 2026-05-17
Status: Approved (design); pending spec review

## Problem

In the traced-result view, every generation token is rendered grey
(`score=None, color=None`) and every hop tab highlights the same
`output_span`. Two things are wrong:

1. When the user traces a sub-span of the generation, generation tokens
   outside that span show grey — but in the FlashTrace "both" multi-hop
   method the recursive hops aggregate over the **full generation**, so those
   tokens do carry attribution weight.
2. Each hop has a different target: hop 0 targets the selected `sink_span`;
   recursive hops target `all_gen_span` (CoT + answer). The UI does not show
   this — all tabs look identical on the generation side.

## What already exists

- `improved.py::calculate_ifr_multi_hop_both` computes `multi_hop.raw_attributions`
  (one per hop); each carries `token_importance_total`, a per-token importance
  vector over the **full sequence** (prompt + generation). It is currently
  projected to the prompt only (`projected_per_hop` via `_project_vector`); the
  generation slice is discarded.
- The `ifr` metadata already records `sink_span_generation` (hop 0 target) and
  `all_gen_span_generation` (recursive-hop target), in generation-token indices.
- `per_hop_scores` in `TraceResult` carries the prompt-side per-hop vectors.

## Design

### 1. Core — `flashtrace/improved.py`

In `calculate_ifr_multi_hop_both`, alongside the existing prompt projection,
compute the **generation-side slice** of each hop's `token_importance_total`
(and of the aggregate `observation`), using the generation range from the
existing metadata. Add to the `ifr` metadata:

- `per_hop_generation`: list of per-hop generation-side weight vectors
  (length = number of generation tokens), aligned 1:1 with `per_hop_projected`.
- `observation_generation`: the aggregate generation-side weight vector.

`sink_span_generation` and `all_gen_span_generation` already exist — no change.

### 2. Core — `flashtrace/tracer.py` + `flashtrace/result.py`

`TraceResult` gains three fields, all with empty/`None` defaults so existing
consumers (`exp/`, `cli`, JSON export) are unaffected:

- `generation_scores: list[float]` — aggregate generation-side weights.
- `per_hop_generation_scores: list[list[float]]` — per-hop generation-side
  weights, aligned 1:1 with `per_hop_scores`.
- `per_hop_target_spans: list[tuple[int, int] | None]` — each hop's target
  span in generation-token indices, aligned 1:1 with `per_hop_scores`.
  `tracer.py` builds it from metadata: index 0 → `sink_span_generation`;
  indices 1.. → `all_gen_span_generation`.

`_build_result` reads `per_hop_generation` / `observation_generation` /
`sink_span_generation` / `all_gen_span_generation` from `ifr_meta` and
populates these fields. When the metadata is absent (methods `ifr-span` /
`ifr-matrix`, or no hops) the fields stay empty — generation tokens then
render uncoloured, exactly as today.

`to_dict` includes the new fields in the JSON export.

### 3. Demo — `demo/live/token_document.py`

`build_document_views(phase="traced", ...)` and `_build_trace_view` change so
each view colours and targets the generation region:

- **Aggregate view**: generation tokens coloured by `result.generation_scores`;
  target span = `result.output_span`.
- **Hop N view** (the i-th entry of `per_hop_scores`): generation tokens
  coloured by `result.per_hop_generation_scores[i]`; target span =
  `result.per_hop_target_spans[i]`.
- Generation-token colour uses `flashtrace/viz.py::_score_color` with per-view
  max-abs normalisation — the same gradient already used for prompt tokens.
  Each view normalises prompt and generation independently is acceptable;
  simplest is one max-abs over that view's generation scores.
- The `is_target` highlight is driven by the **per-view** target span, not a
  single document-wide `target_span`. The render model's per-view entry gains
  a `target_span` field; tokens whose `gen_index` falls in it get `is-target`.
- When a view has no generation scores (non-`flashtrace` methods), generation
  tokens render uncoloured — current behaviour preserved.

Tab naming stays `Aggregate` / `Hop 1..N` (no rename — avoids churn and keeps
existing tests; the per-hop target highlight now makes the distinction
visible).

## Render-model change

Each `view` in the traced render model gains `target_span` (its own inclusive
generation-index pair, or null). Generation-region tokens carry `score` and
`color` like prompt tokens already do. The top-level `target_span` stays for
back-compat but per-view `target_span` is what the renderer uses in the traced
phase.

## Frontend — `demo/live/static/app.js`

`makeToken` already applies `token.color` and the `is-target` class from the
render model. The traced phase now supplies colour + per-view target on
generation tokens, so the renderer needs at most a small change: when
switching tabs, re-apply `is-target` from the active view's `target_span`
rather than a single global span. Prompt-token rendering is unchanged.

## Testing

- `improved.py`: a hop's `per_hop_generation` vector has one entry per
  generation token and matches the generation slice of
  `token_importance_total`.
- `tracer.py` / `TraceResult`: `per_hop_generation_scores` aligns 1:1 with
  `per_hop_scores`; `per_hop_target_spans` is `[sink_span] + [all_gen_span]*`;
  `generation_scores` length equals the generation-token count.
- `token_document.py`: traced views give generation tokens non-null `color`
  when generation scores exist; each hop view's `is_target` tokens match that
  hop's target span; non-`flashtrace` methods leave generation tokens
  uncoloured (Aggregate-only).
- Frontend Playwright smoke: switching to a recursive-hop tab colours the
  generation region and highlights the full-generation target span.
- Full suite stays green; tiny Qwen2 model from `tests/helpers.py`.

## Risks

- `token_importance_total`'s generation-side values must be a meaningful
  per-token weight (not an internal artefact). The implementation verifies the
  generation slice indexing against `all_gen_span_absolute` /
  `sink_span_absolute` and asserts lengths in a test before relying on it.
- Per-view max-abs normalisation means colours are not comparable across tabs;
  acceptable — each tab answers "within this hop, which tokens matter".
