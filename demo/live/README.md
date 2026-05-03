# FlashTrace Live Demo

This directory contains a Gradio app for a live FlashTrace demo.

## Local Run

```bash
uv run --extra demo python demo/live/app.py
```

Then open:

```text
http://127.0.0.1:7860
```

The default `demo/paris-smoke` model is a deterministic local smoke path for instant UI validation. Real Hugging Face models load lazily on the first trace request. Override defaults with environment variables:

```bash
FLASHTRACE_DEMO_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
FLASHTRACE_DEMO_DEVICE_MAP=auto \
uv run --extra demo python demo/live/app.py
```

## Hugging Face Space

Create a Gradio Space and copy these files into the Space repository:

- `app.py`
- `requirements.txt`

For a GPU Space, set `FLASHTRACE_DEMO_MODEL` to the target model id and choose an instance with enough VRAM for the selected model and prompt length.

## Outputs

Each trace request writes a JSON artifact under `out/` next to `app.py` unless `FLASHTRACE_DEMO_OUTPUT_DIR` is set. The UI returns:

- top input tokens ranked by attribution score,
- generation token indices,
- standalone HTML heatmap,
- downloadable JSON trace.
