# Token-Document Frontend Redesign — Design

Date: 2026-05-17
Status: Revised after Claude + Codex review (2026-05-17); pending spec review

## Goal

Replace the FlashTrace Live demo's prompt/target/inspector area with a single
**interactive token document**: a click-selectable, tokenized text view that
drives the Generate → select-target → Trace → inspect-attribution flow.

## Scope

- Rewrite `build_demo()` in `demo/live/app.py` and its event wiring.
- Add a new module `demo/live/token_document.py` for the render-model and HTML
  rendering (pure, testable).
- Add a token-ids-based record builder (see Tokenization contract) — a small,
  additive helper in `demo/live/token_overlay.py`.
- Keep the FlashTrace attribution engine unchanged.
- Out of scope: the attribution algorithm itself.

## Technical approach (chosen: A)

`gr.HTML` for rendering + a **persistent JavaScript island** for behavior.

- The token document is one `gr.HTML` component. Python re-renders its content
  on each phase transition.
- The HTML content is **data-only**: token `<span>`s (carrying `data-*`
  attributes), a `<style>` block, and a `<script type="application/json">` data
  blob. It contains no behavior script — scripts injected via `innerHTML` do not
  execute on Gradio updates.
- Interactive behavior lives in **one persistent script** loaded once. It uses
  event delegation on a stable `elem_id` container plus a `MutationObserver` to
  re-read the data blob whenever Python swaps the HTML.
- **Gradio 6 caveat:** `uv.lock` resolves Gradio to 6.14.0, and Gradio 6 moved
  app-level `head`/`js`/`css` off the `Blocks(...)` constructor. The exact
  injection point for the persistent island MUST be verified against the
  installed Gradio version before implementation — candidates: `demo.launch(...)`
  kwargs, `Blocks(js=...)`, or a one-time `gr.HTML` block rendered at the top of
  the layout. The spike (below) settles this.

### Selection → Python bridge

Selection state lives **entirely in the DOM** during the generated phase: the
persistent script highlights clicked tokens and records the pending span in
`data-*` attributes / a module variable. There is **no per-click sync** to
Python.

The span reaches Python **only when Trace is clicked**, via the Trace button's
`js=` function: `trace_btn.click(fn=on_trace, inputs=[..., target_span_input],
outputs=[...], js="(...) => { /* scrape current selection, return it as the
target_span_input value */ }")`. The `js=` function runs before `fn`, reads the
DOM selection, and substitutes the value of the hidden `target_span_input`
component. This is the documented Gradio path and avoids the unreliable
"set `.value` + dispatch `input` event" pattern.

Rejected: Gradio custom component (B — needs a Node/Svelte build chain);
FastAPI + hand-written SPA (C — discards the Gradio demo and its tests).

## Tokenization contract (critical)

**Problem (from review).** Three tokenizations are in play and they are not
guaranteed to agree:

1. `generate_with_qwen` returns the model's real generated token IDs
   (`GenerationOutput.token_ids`).
2. The old `run_generate_phase` ignores those IDs and re-tokenizes the *decoded*
   text via `build_token_records`/`detect_sections`.
3. `FlashTrace.trace` tokenizes the `target` string its own way
   (`target_response` ≈ `tokenizer(target) + eos`), and rejects spans that fall
   outside its non-EOS generation range.

A target span the user picks against tokenization (2) can be off-by-one or
out of range against (3), silently tracing the wrong span or raising
`Invalid sink_span`.

**Resolution — single source of truth.**

- The document's **generation region is built from the real generated token
  IDs** (`GenerationOutput.token_ids`), not from re-tokenizing decoded text. Add
  `build_token_records_from_ids(token_ids, tokenizer, ...)` to
  `token_overlay.py`: it maps each ID through `convert_ids_to_tokens` and derives
  `char_start/char_end` from the cumulative decoded text. Generation-token
  indices in the document are therefore the *true* generation indices.
- The generated text is carried in a `gr.State` and passed as `target` to
  `FlashTrace.trace`, so the engine never re-generates internally.
- **Equality is verified, not assumed.** Implementation MUST add a test that
  asserts the document's generation-token sequence equals
  `TraceResult.generation_tokens` (modulo the trailing EOS, which FlashTrace
  excludes). If the installed tokenizer's decode→encode round-trip breaks this
  equality, the fix is a **small, in-scope backend addition**: expose a
  FlashTrace helper that tokenizes a target the same way `target_response` does,
  and build the document generation region from that. Either way, the document
  and the engine share one tokenization.
- The default target span and any selection **exclude the trailing EOS token**
  and non-`content`/`special` tokens, matching FlashTrace's accepted range.

`reasoning_span` (from `detect_sections().thinking_token_span`) is mapped into
the same generation-index space. It is clamped to the valid range; if it cannot
be mapped it is dropped (passed as `None`) and a note is shown in the status
line — never passed unchecked.

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
- `selectable` is true only for `kind == "content"` generation-region tokens
  that are not the trailing EOS — the valid target-span endpoints.
- `score` is non-null only in traced-phase attribution views.
- `target_span` is an inclusive generation-token index pair.

## Phases / state machine

**prompt phase** (initial)
- Document = prompt tokens only. One non-interactive view.
- Buttons: Generate visible, Trace hidden.
- The Chat Template toggle re-renders the prompt tokens (see below).

**generated phase** (after Generate completes)
- Document = prompt tokens + appended generation tokens (built from real token
  IDs). One interactive view.
- `target_span` defaults to the **entire generation region excluding the
  trailing EOS** (decision: spec says "默认整个输出的 span" — the whole output —
  not the detected answer sub-span). Rendered with a distinct target highlight.
- The user clicks a selectable generation token (start), then a second (end);
  the new span is `[min, max]` over their `gen_index` values, re-highlighted.
- `detect_sections()` still runs; its `thinking_token_span`, after clamping, is
  passed as `reasoning_span` to `FlashTrace.trace`.
- The Chat Template toggle is **disabled** once the document contains a
  generation (it would change prompt tokenization mid-session).
- Buttons: Generate hidden, Trace visible.

**traced phase** (after Trace completes)
- Document area becomes tabbed. Views: **"Aggregate"** plus one **"Hop N"** per
  entry in `TraceResult.per_hop_scores`.
- **`per_hop_scores` is only populated for the `flashtrace` method.** For
  `ifr-span` / `ifr-matrix`, or when generation is empty, `per_hop_scores` is
  `[]` and the document shows the **Aggregate view only** — this is expected,
  not an error. Hop tabs appear only when hop data exists.
- Each view renders `result.prompt_tokens` + `result.generation_tokens`
  (FlashTrace's own tokenization — now equal to the document's by contract).
  Prompt-region tokens are colored by that view's score array; generation-region
  tokens keep the target highlight (`result.output_span`).
- All traced-phase views are read-only.
- Color: per-view normalization by max absolute score, reusing
  `flashtrace/viz.py::_score_color` (confirmed to exist and be importable).
- Buttons: Generate visible (acts as "start over"), Trace hidden.
- A JSON trace artifact is written and offered for download.

## Chat Template toggle

A single `gr.Checkbox` ("Show chat template") in the document area is the only
chat-template control. Its value is passed as `use_chat_template` to
`FlashTrace`. So the prompt-region tokens in the document are not free-form
display — they must show **exactly the prompt tokens FlashTrace will attribute**
for the chosen mode:

- toggle off → the user-prompt tokens FlashTrace attributes
  (`use_chat_template=False`).
- toggle on → the chat-template-formatted prompt tokens
  (`use_chat_template=True`; note FlashTrace also prepends a leading space and
  wraps content via its prompt template).

The prompt-phase document MUST be tokenized to match the chosen mode. If
matching the `use_chat_template=True` formatted-prompt tokenization requires
backend exposure, that exposure is in-scope (same principle as the Tokenization
contract). The separate `use_chat_template` checkbox in the old control panel is
removed.

## Component inventory

Kept: `model_name`, `prompt` (plain input textbox), `max_new_tokens`, `method`,
`hops`, `device_map`, `dtype`, `chunk_tokens`, `sink_chunk_tokens`, JSON file
output.

New:
- `token_doc` (`gr.HTML`, stable `elem_id`) — the document.
- `generated_text_state` (`gr.State`) — holds `GenerationOutput.text`; passed as
  `target` to Trace.
- `target_span_input` (hidden `gr.Textbox`) — carrier for the Trace button's
  `js=` function; not user-visible.
- `chat_template_toggle` (`gr.Checkbox`).
- `status` (`gr.Markdown`) — surfaces `ValueError`s (empty/oversize prompt,
  invalid span, dropped `reasoning_span`) and progress notes.
- `generate_btn`, `trace_btn`.

Removed: `target` textbox, `output_span` / `reasoning_span` textboxes,
`sections_state` JSON, `raw_tokens` dataframe, `top_table` dataframe,
`generation_tokens` textbox, `heatmap` HTML, old `use_chat_template` checkbox,
`top_k` slider.

## Data flow

- **On Generate**: load model → `generate_with_qwen` (keep `output.token_ids`
  and `output.text`) → `detect_sections` → `build_token_records` for the prompt
  + `build_token_records_from_ids` for the generation → build generated-phase
  render-model → render HTML. Outputs: `token_doc`, `generated_text_state`,
  default `target_span`, button visibility, `status`.
- **On Trace**: the Trace button's `js=` scrapes the DOM selection into
  `target_span_input` → `on_trace` reads `generated_text_state` +
  `target_span_input` → `FlashTrace.trace(prompt, target=generated_text,
  output_span, reasoning_span, hops, method, use_chat_template)` → build
  traced-phase render-model (Aggregate + per-hop) from `TraceResult` → render
  HTML, write JSON. Outputs: `token_doc`, `json_file`, button visibility,
  `status`.
- The chat-template toggle re-renders the prompt-phase document (prompt phase
  only; disabled afterward).

## Modules

- `demo/live/token_document.py` — pure functions:
  - `build_document_views(...)` → render-model dict (no Gradio, no I/O).
  - `render_document_html(render_model)` → HTML string (token spans + `<style>`
    + JSON data blob).
  - The persistent JS/CSS island as a module constant.
- `demo/live/token_overlay.py` — add `build_token_records_from_ids(...)`.
- `demo/live/app.py` — `build_demo()` and thin handlers (`on_generate`,
  `on_trace`) orchestrating backend calls and `token_document.py`.

## Error handling

- Empty / oversize prompt: existing `ValueError` validation; the message is
  shown in the `status` Markdown rather than raised into the UI uncaught.
- Fewer than two endpoints selected: no commit; target span unchanged.
- Zero generation tokens: generated phase renders with an empty target;
  Trace is disabled and `status` explains why.
- Out-of-range / unmappable `reasoning_span`: dropped to `None`, noted in
  `status`.

## Testing

- Unit-test `build_document_views`, `render_document_html`, and
  `build_token_records_from_ids` against the tiny Qwen2 model from
  `tests/helpers.py`: phase transitions, region tagging, `selectable` flags,
  EOS exclusion, target-span defaulting, per-hop view construction (including
  the empty-`per_hop_scores` Aggregate-only fallback), score-color bounds.
- **Tokenization-equality test**: assert the document's generation-token
  sequence equals `TraceResult.generation_tokens` (modulo trailing EOS) for the
  tiny model — this guards the Tokenization contract.
- **Browser smoke test** (Playwright or `gradio`'s test client): click two
  generated tokens, assert the scraped span, click Trace, assert the backend
  received the intended `output_span`. This covers the highest-risk surface
  (the JS↔Python bridge) that unit tests cannot.
- `build_demo()` dry-run test (`FLASHTRACE_DEMO_DRY_RUN=1`) keeps passing.

## Implementation gating

The JS↔Python bridge and the Gradio-6 injection point are the main unknowns.
**Before building the full document**, build a minimal spike: a persistent
script + a `gr.HTML` with two clickable tokens + a Trace-style button whose
`js=` scrapes the selection into Python. Only proceed once the spike works
against the installed Gradio version. This is a hard gate.

## Risks (residual)

- Gradio 6 asset-injection API differs from Gradio 5; resolved by the spike and
  version verification.
- Decode→encode round-trip could still break document/engine token equality;
  resolved by the equality test + the in-scope backend-exposure fallback.
