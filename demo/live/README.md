# FlashTrace Live Demo

FastAPI backend and static frontend for the Generate -> select target span -> Trace flow.

## Local Run

```bash
uv run --extra demo uvicorn demo.live.server:app --host 127.0.0.1 --port 7860
```

Then open:

```text
http://127.0.0.1:7860
```

The default model (`Qwen/Qwen3-4B-Thinking-2507`) downloads on first use.

## Hot Reload

```bash
uv run --extra demo uvicorn demo.live.server:app --host 127.0.0.1 --port 7860 --reload
```

## API

- `POST /api/tokenize` with `{model, prompt}` returns a prompt render model.
- `POST /api/generate` with `{model, prompt, max_new_tokens, device_map, dtype}` returns generated text, default spans, status, and a generated render model.
- `POST /api/trace` with `{model, prompt, generated_text, target_span, reasoning_span, method, hops, device_map, dtype, chunk_tokens, sink_chunk_tokens}` returns a traced render model, inline `trace_json`, and status.

The server keeps no per-user session. The browser stores prompt, generated text, and selected span, then sends those values with each request.

## Concurrency

FastAPI endpoints are async, and model work runs through one global `ThreadPoolExecutor(max_workers=1)`. This keeps the event loop responsive while serializing shared PyTorch inference for small multi-user demos.

## Environment Variables

- `FLASHTRACE_DEMO_MODEL` — model id, default `Qwen/Qwen3-4B-Thinking-2507`.
- `FLASHTRACE_DEMO_DEVICE_MAP` — model `device_map`, default `auto`.
- `FLASHTRACE_DEMO_MAX_PROMPT_CHARS` — prompt cap, default `4000`.
- `FLASHTRACE_DEMO_HOST` — direct `server.py` host, default `127.0.0.1`.
- `FLASHTRACE_DEMO_PORT` — direct `server.py` port, default `7860`.

## Direct Script

```bash
uv run --extra demo python demo/live/server.py
```

The recommended command is the uvicorn invocation above.
