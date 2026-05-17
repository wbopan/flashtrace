# Replace Gradio with a FastAPI Backend — Design

Date: 2026-05-17
Status: Approved (design); pending spec review

## Goal

Replace the Gradio demo (`demo/live/app.py`) with a general-purpose Python
backend that serves the same Generate → select-target → Trace → inspect flow,
and that comfortably handles 2-3 concurrent users.

## Why this is cheap

The demo's core is already framework-agnostic: `token_overlay.py`,
`qwen_generation.py`, `span_parsers.py`, `build_document_views` (the
render-model builder in `token_document.py`), and the `flashtrace` engine import
no Gradio. The trace step is already stateless — `run_trace_document_phase`
takes `generated_text` + `target_span` as plain inputs. Only `build_demo()` and
its event wiring are Gradio-bound.

## Architecture

FastAPI application served by uvicorn (single process). Split into:

- `demo/live/service.py` — pure orchestration logic (load model, generate,
  build render-models). Extracted from the current `run_*_document_phase`
  functions, with the Gradio tuple-shaping removed; returns plain dicts. Accepts
  a `loader` injection (and `tracer_cls` injection for trace) so tests can pass
  the tiny model.
- `demo/live/server.py` — FastAPI app: routes, request/response models,
  concurrency control, static-file serving, uvicorn entry point.
- `demo/live/static/` — `index.html`, `app.js`, `styles.css`: the client.
- `demo/live/token_document.py` — keep `build_document_views`; **add a per-token
  `color` field to traced-phase views** (computed server-side via
  `flashtrace/viz.py::_score_color`, per-view max-abs normalization, so the
  gradient stays single-sourced). Remove the Gradio-era `render_document_html`,
  `TOKEN_DOCUMENT_JS`, `TRACE_SELECTION_JS`, `TOKEN_DOCUMENT_CSS`.
- Unchanged, reused: `token_overlay.py`, `qwen_generation.py`,
  `span_parsers.py`, `flashtrace/`.

## HTTP API (stateless)

The server keeps no per-user session. The client holds `prompt`,
`generated_text`, and `target_span` and sends them with each request.

- `GET /` → serves `static/index.html`. `GET /static/*` → static assets.
- `POST /api/tokenize` — body `{model, prompt, chat_template}` →
  `200 {render_model}` (phase `prompt`).
- `POST /api/generate` — body `{model, prompt, max_new_tokens, chat_template,
  device_map, dtype}` → `200 {render_model (phase generated), generated_text,
  target_span, reasoning_span, status}`.
- `POST /api/trace` — body `{model, prompt, generated_text, target_span,
  reasoning_span, method, hops, chat_template, device_map, dtype, chunk_tokens,
  sink_chunk_tokens}` → `200 {render_model (phase traced), trace_json, status}`.
  The full trace artifact is returned inline as `trace_json`; the client turns
  it into a downloadable Blob (no server-side file or file-id endpoint).

`render_model` is the existing `build_document_views` output (the contract is
unchanged and already tested). Spans use the inclusive `START:END` string form.

## Concurrency (2-3 users)

- FastAPI endpoints are `async def`. All model-touching work runs through a
  **single global `ThreadPoolExecutor(max_workers=1)`** via
  `loop.run_in_executor(...)`. This serializes inference (a shared PyTorch model
  is not safe under concurrent forward passes, and serialization also avoids
  memory contention) while keeping the event loop free to accept other
  connections.
- With 2-3 users, one request runs while the others queue in the executor;
  each is a normal synchronous request/response — a slow trace simply holds its
  connection until done. No async job/polling system (YAGNI at this scale).
- The model/tokenizer is loaded once and cached (the existing `_MODEL_CACHE`,
  keyed by `(model, device_map, dtype)`), shared across all users.
- The client shows a "running…" state for the duration of each `fetch`.

## Frontend (`static/`, no framework)

- `index.html` — static shell: control panel (model, prompt, max-new-tokens,
  method, hops, dtype, device map, chunk sizes), the chat-template checkbox, the
  token-document container, Generate/Trace buttons, a status line, a download
  link.
- `app.js` — plain ES modules / DOM. Responsibilities:
  - `fetch()` the three endpoints; render the returned `render_model` into token
    `<span>`s (rounded background, class by `kind`/`region`).
  - Generated phase: click a selectable generation token (start), then a second
    (end) → target span; highlight; hold the span in JS state.
  - Traced phase: render tabs (Aggregate + per hop), click to switch the active
    view; apply each token's server-provided `color`.
  - Generate/Trace buttons morph by phase; chat-template checkbox; status line;
    loading state; download the `trace_json` as a file.
  - No Gradio-era hacks — no `MutationObserver`, no hidden-textbox bridge, no
    injected-`js=`. Events are bound directly.
- `styles.css` — all styling, ported from the current
  `render_document_html` `<style>` block.

## Error handling

- Validation failures (empty/oversize prompt, malformed span, span out of
  range) → `400 {"error": msg}`.
- Model-load or trace failures → `500 {"error": msg}`.
- The client renders `error` in the status line; the document keeps its last
  good state.

## Removal / migration

- Delete `demo/live/app.py` (Gradio).
- `pyproject.toml` `demo` extra and `demo/live/requirements.txt`: drop `gradio`;
  add `fastapi`, `uvicorn[standard]`, and `httpx` (needed by FastAPI's
  `TestClient`).
- `demo/live/README.md`: rewrite run instructions.
- Drop Gradio-specific tests (`build_demo` dry-run, `test_gradio_6_js_bridge_*`,
  `run_trace_document_phase` UI-shaped tests) and replace with API tests.

## Run / hot-reload

- Run: `uv run --extra demo uvicorn demo.live.server:app --host 127.0.0.1
  --port 7860`.
- Hot-reload: add `--reload`.
- `server.py` exposes a module-level `app` and an `if __name__ == "__main__"`
  block that calls `uvicorn.run(...)`, with host/port from
  `FLASHTRACE_DEMO_HOST` / `FLASHTRACE_DEMO_PORT` (default `127.0.0.1:7860`).
- `server.py` keeps a `sys.path` bootstrap so it can also be run directly.

## Testing

- **API tests** (FastAPI `TestClient`): `/api/tokenize`, `/api/generate`,
  `/api/trace` exercised against the tiny Qwen2 model from `tests/helpers.py`
  via the `loader` (and `tracer_cls`) injection — assert phase, render-model
  shape, generated text, default span, traced views (Aggregate-only fallback
  for non-`flashtrace` methods).
- **Validation tests**: empty prompt, oversize prompt, malformed/out-of-range
  span → `400` with an `error` body.
- **Concurrency test**: fire 2-3 requests from threads against the app; assert
  all return correctly and the inference executor is configured single-worker.
- **Reused**: `build_document_views`, `build_token_records_from_ids`,
  tokenization-equality tests carry over.
- **Frontend**: a best-effort Playwright smoke test (click two tokens → span;
  switch a tab), `importorskip`-guarded.

## Risks

- A very slow trace holds its HTTP connection open; acceptable for the demo's
  model sizes and 2-3 users. If it became a problem the fix is an async job
  endpoint — explicitly out of scope now.
- `app.js` interaction is only verifiable in a browser; mitigated by the
  best-effort Playwright test and a manual check.
