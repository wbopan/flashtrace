# FlashTrace Live Demo

Gradio demo that traces a generated answer back to the prompt tokens that shaped it.

## Local Run

```bash
uv run --extra demo python demo/live/app.py
```

Then open:

```text
http://127.0.0.1:7860
```

## Three-phase flow

1. **Generate** — Click *Generate*. The model produces reasoning + answer with deterministic settings (`do_sample=False`). The smoke model (`demo/paris-smoke`) returns a canned `<think>/<answer>` response without loading any weights.
2. **Inspect** — The generated text is parsed (`<think>/<answer>` → `\boxed{}` → last-paragraph) and the resulting answer/reasoning token spans are auto-filled into the span fields. The raw-token table shows every tokenizer token with its kind (content/whitespace/special/template/control).
3. **Trace** — Click *Trace selected answer*. The selected `output_span` and `reasoning_span` are passed to `FlashTrace.trace(...)` which produces the prompt-side attribution heatmap and JSON export. Override the auto-filled spans manually before tracing if the parser picked the wrong segment.

## Span format

Spans are inclusive generation-token index pairs in `START:END` format. The Generate phase fills these in based on the parser; manual overrides win.

## Switch to a real Qwen model

```bash
FLASHTRACE_DEMO_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
FLASHTRACE_DEMO_DEVICE_MAP=auto \
uv run --extra demo python demo/live/app.py
```

First run downloads the model. Subsequent runs reuse the in-process model cache.

## Hugging Face Space

Create a Gradio Space and copy these files into the Space repository:

- `app.py`
- `requirements.txt`

For a GPU Space, set `FLASHTRACE_DEMO_MODEL` to the target model id and choose an instance with enough VRAM for the selected model and prompt length.

## Environment variables

- `FLASHTRACE_DEMO_MODEL` — Model id (default `demo/paris-smoke`).
- `FLASHTRACE_DEMO_OUTPUT_DIR` — JSON trace output dir (default `demo/live/out`).
- `FLASHTRACE_DEMO_DEVICE_MAP` — `device_map` for `from_pretrained` (default `auto`).
- `FLASHTRACE_DEMO_MAX_PROMPT_CHARS` — Prompt-length cap (default 4000).

## Outputs

Each trace request writes a JSON artifact under `out/` next to `app.py` unless `FLASHTRACE_DEMO_OUTPUT_DIR` is set. The UI returns:

- top input tokens ranked by attribution score,
- generation token indices,
- standalone HTML heatmap,
- downloadable JSON trace,
- detected sections (parser, char/token spans),
- raw-token inspector with per-token kind and offsets.
