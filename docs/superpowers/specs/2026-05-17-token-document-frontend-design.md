# Token-Document Frontend Redesign — Design

Date: 2026-05-17
Status: Approved (design); pending spec review

## Goal

Replace the FlashTrace Live demo's prompt/target/inspector area with a single
**interactive token document**: a click-selectable, tokenized text view that
drives the Generate → select-target → Trace → inspect-attribution flow.

## Scope

- Rewrite `build_demo()` in `demo/live/app.py` and its event wiring.
- Add a new module `demo/live/token_document.py` for the render-model and HTML
  rendering (pure, testable).
- Keep the FlashTrace backend (`FlashTrace.trace`, `detect_sections`,
  `build_token_records`, `TraceResult`) unchanged.
- Out of scope: the FlashTrace attribution engine, the prompt/generation vs.
  FlashTrace-internal tokenization alignment (pre-existing; see Risks).

## Technical approach (chosen: A)

`gr.HTML` + a **global persistent JavaScript island**.

- The token document is one `gr.HTML` component. Python re-renders its content
  on each phase transition.
- The HTML content is **data-only**: token `<span>`s, a `<style>` block, and a
  `<script type="application/json">` data blob. It contains no behavior script,
  because scripts injected via `innerHTML` do not execute on Gradio updates.
- Interactive behavior lives in **one persistent script injected once** via
  `gr.Blocks(head=...)`. It uses event delegation on a stable container and a
  `MutationObserver` to re-read the data blob whenever Python swaps the HTML.
- Selection state is ferried to Python through a hidden `gr.Textbox`
  (`target_span_state`): the script writes `START:END` into it and dispatches an
  `input` event. Python only reads it when Trace is clicked — no per-click
  round-trip.

Rejected: Gradio custom component (B — needs a Node/Svelte build chain,
overkill); FastAPI + hand-written SPA (C — discards the Gradio demo and tests).

## Render-model contract (Python → JS)

The `gr.HTML` content embeds a JSON blob with this shape:

```json
{
  "phase": "prompt | generated | traced",
  "active_view": 0,
  "views": [
    {
      "name": "Document",
      "interactive": true,
      "tokens": [
        {
          "i": 0,
          "text": "Paris",
          "kind": "content | whitespace | special | template | control",
          "region": "prompt | generation",
          "gen_index": null,
          "selectable": true,
          "score": null
        }
      ]
    }
  ],
  "target_span": [start_gen_index, end_gen_index]
}
```

- `i` is the document-global token index; `gen_index` is the generation-token
  index (null for prompt-region tokens), used for target spans.
- `selectable` is true only for `kind == "content"` tokens in the generation
  region — these are the valid target-span endpoints.
- `score` is non-null only in traced-phase attribution views; it carries that
  view's attribution weight for prompt-region tokens.
- `target_span` is an inclusive generation-token index pair.

## Phases / state machine

**prompt phase** (initial)
- Document = prompt tokens only. One view, non-interactive.
- Buttons: Generate visible, Trace hidden.
- The Chat Template toggle re-renders the prompt tokens (see below).

**generated phase** (after Generate completes)
- Document = prompt tokens + appended generation tokens. One interactive view.
- `target_span` defaults to the **entire generation region** (all generation
  tokens). It is rendered with a distinct "target" highlight.
  (Decision: the spec says "默认整个输出的 span" — the whole output — so the
  default target is the full generation span, not the detected answer span.)
- The user clicks a selectable generation token (start), then a second
  (end); the new target span is `[min, max]` over their `gen_index` values and
  re-highlighted. The script writes `START:END` to `target_span_state`.
- `detect_sections()` still runs; its `thinking_token_span` is passed silently
  as `reasoning_span` to `FlashTrace.trace` (the user does not select it).
- Buttons: Generate hidden, Trace visible.

**traced phase** (after Trace completes)
- Document area becomes tabbed. Views: **"Aggregate"** plus one **"Hop N"** per
  entry in `TraceResult.per_hop_scores`.
- Each view renders `result.prompt_tokens` + `result.generation_tokens`
  (FlashTrace's own tokenization). Prompt-region tokens are colored by that
  view's score array; generation-region tokens keep the target highlight.
- All traced-phase views are read-only.
- Color: per-view normalization by max absolute score, reusing the gradient
  logic from `flashtrace/viz.py::_score_color`.
- Buttons: Generate visible (acts as "start over"), Trace hidden.
- A JSON trace artifact is still written and offered for download.

## Chat Template toggle

A single `gr.Checkbox` ("Show chat template") in the document area is the only
chat-template control. It does two things consistently:

1. Display — when on, the prompt is rendered through
   `tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True)`
   before tokenizing, so template tokens (`<|im_start|>` etc., `kind ==
   "template"`) appear in the document.
2. Tracing — its value is passed as `use_chat_template` to `FlashTrace`, so the
   displayed tokens match FlashTrace's internal tokenization.

The separate `use_chat_template` checkbox in the old control panel is removed.

## Component inventory

Kept: `model_name`, `prompt` (plain input textbox), `max_new_tokens`, `method`,
`hops`, `device_map`, `dtype`, `chunk_tokens`, `sink_chunk_tokens`, JSON file
output.

New: `token_doc` (`gr.HTML`), `target_span_state` (hidden `gr.Textbox`),
`chat_template_toggle` (`gr.Checkbox`), `generate_btn`, `trace_btn`.

Removed: `target` textbox, `output_span` / `reasoning_span` textboxes,
`sections_state` JSON, `raw_tokens` dataframe, `top_table` dataframe,
`generation_tokens` textbox, `heatmap` HTML, old `use_chat_template` checkbox,
`top_k` slider.

## Data flow

- **On Generate**: load model → `generate_with_qwen` → `detect_sections` →
  `build_token_records` for prompt and generation → build generated-phase
  render-model → render HTML. Outputs: `token_doc`, default `target_span_state`,
  button visibility.
- **On Trace**: read `target_span_state` → `FlashTrace.trace(prompt, target,
  output_span, reasoning_span, hops, method)` → build traced-phase render-model
  (Aggregate + per-hop) from `TraceResult` → render HTML, write JSON. Outputs:
  `token_doc`, `json_file`, button visibility.
- The chat-template toggle re-renders the prompt-phase document.

## Modules

- `demo/live/token_document.py` — pure functions:
  - `build_document_views(...)` → render-model dict (no Gradio, no I/O).
  - `render_document_html(render_model)` → HTML string (token spans + `<style>`
    + JSON data blob).
  - The persistent JS/CSS island as a module constant injected via `head=`.
- `demo/live/app.py` — `build_demo()` and thin handlers that orchestrate the
  backend calls and call into `token_document.py`.

## Error handling

- Empty prompt / oversize prompt: reuse existing `ValueError` validation paths.
- Selecting fewer than two endpoints: no commit; target span unchanged.
- A model that produces zero generation tokens: generated phase still renders,
  target span empty, Trace surfaces the existing validation error.

## Testing

- Unit-test `build_document_views` and `render_document_html` against the tiny
  Qwen2 model from `tests/helpers.py`: phase transitions, region tagging,
  `selectable` flags, target-span defaulting, per-hop view construction,
  score coloring bounds.
- `build_demo()` dry-run test (`FLASHTRACE_DEMO_DRY_RUN=1`) keeps passing.
- The JS↔Python bridge cannot be unit-tested; it is verified manually in a
  browser. A minimal "click a token → span reaches Python" spike is built and
  verified first, before the full document is wired.

## Risks

- **JS↔Python bridge** — the main uncertainty. Mitigation: global persistent
  script + data-only HTML + `MutationObserver`; prove the bridge with a minimal
  spike before building the full document.
- **Tokenization alignment** — the document's generation-token indices and
  FlashTrace's internal tokenization may not be 1:1; this is a pre-existing
  assumption in the demo. Attribution views sidestep it by rendering
  `result.prompt_tokens` / `result.generation_tokens` directly.
