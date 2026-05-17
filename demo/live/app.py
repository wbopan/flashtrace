from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable


def _bootstrap_local_flashtrace() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "flashtrace").is_dir():
            sys.path.insert(0, str(parent))
            return


_bootstrap_local_flashtrace()

from demo.live.qwen_generation import generate_with_qwen
from demo.live.token_overlay import (
    GenerationSections,
    build_token_records,
    detect_sections,
)

if TYPE_CHECKING:
    from flashtrace.result import TraceResult

DEFAULT_MODEL = os.environ.get("FLASHTRACE_DEMO_MODEL", "Qwen/Qwen3-0.6B")
APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = """Context:
Paris is the capital of France.
Berlin is the capital of Germany.
Madrid is the capital of Spain.

Question: What is the capital of France?"""
DEFAULT_TARGET = "Paris"
DEFAULT_OUTPUT_DIR = Path(os.environ.get("FLASHTRACE_DEMO_OUTPUT_DIR", str(APP_DIR / "out")))
MAX_PROMPT_CHARS = int(os.environ.get("FLASHTRACE_DEMO_MAX_PROMPT_CHARS", "4000"))

_MODEL_CACHE: dict[tuple[str, str, str], tuple[object, object]] = {}


def _parse_optional_span(value: str | None) -> tuple[int, int] | None:
    text = (value or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        raise ValueError("Span must use START:END format.")
    try:
        start = int(parts[0])
        end = int(parts[1])
    except ValueError as exc:
        raise ValueError("Span bounds must be integers.") from exc
    if start < 0 or end < start:
        raise ValueError("Span must satisfy 0 <= START <= END.")
    return start, end


def _load_cached_model(
    model_name: str,
    *,
    device_map: str,
    dtype: str,
    loader: Callable | None = None,
):
    if loader is None:
        from flashtrace import load_model_and_tokenizer

        loader = load_model_and_tokenizer
    cache_key = (model_name, device_map, dtype)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = loader(model_name, device_map=device_map, dtype=dtype)
    return _MODEL_CACHE[cache_key]


def _top_rows(result: "TraceResult", top_k: int) -> list[list[object]]:
    return [[item.index, item.token, round(float(item.score), 6)] for item in result.topk_inputs(top_k)]


def _generation_token_text(result: TraceResult) -> str:
    if not result.generation_tokens:
        return ""
    return "\n".join(f"{index}: {token!r}" for index, token in enumerate(result.generation_tokens))


def run_trace(
    *,
    model_name: str,
    prompt: str,
    target: str,
    output_span: str,
    reasoning_span: str,
    method: str,
    hops: int,
    top_k: int,
    device_map: str,
    dtype: str,
    chunk_tokens: int,
    sink_chunk_tokens: int,
    use_chat_template: bool,
    loader: Callable | None = None,
    tracer_cls: type | None = None,
    work_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[list[object]], str, str, str]:
    model_id = model_name.strip()
    prompt_text = prompt.strip()
    target_text = target if target.strip() else None
    if not model_id:
        raise ValueError("Model is required.")
    if not prompt_text:
        raise ValueError("Prompt is required.")
    if len(prompt_text) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt must be at most {MAX_PROMPT_CHARS} characters.")

    if tracer_cls is None:
        from flashtrace import FlashTrace

        tracer_cls = FlashTrace
    from flashtrace.viz import render_trace_html

    model, tokenizer = _load_cached_model(model_id, device_map=device_map, dtype=dtype, loader=loader)
    tracer = tracer_cls(
        model,
        tokenizer,
        chunk_tokens=int(chunk_tokens),
        sink_chunk_tokens=int(sink_chunk_tokens),
        use_chat_template=bool(use_chat_template),
    )
    result = tracer.trace(
        prompt=prompt_text,
        target=target_text,
        output_span=_parse_optional_span(output_span),
        reasoning_span=_parse_optional_span(reasoning_span),
        hops=int(hops),
        method=method,
    )

    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"flashtrace-{uuid.uuid4().hex}.json"
    result.to_json(json_path)
    return _top_rows(result, int(top_k)), _generation_token_text(result), render_trace_html(result), str(json_path)


def run_trace_from_ui(
    model_name: str,
    prompt: str,
    target: str,
    output_span: str,
    reasoning_span: str,
    method: str,
    hops: int,
    top_k: int,
    device_map: str,
    dtype: str,
    chunk_tokens: int,
    sink_chunk_tokens: int,
    use_chat_template: bool,
    *,
    loader: Callable | None = None,
    tracer_cls: type | None = None,
    work_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[list[list[object]], str, str, str]:
    return run_trace(
        model_name=model_name,
        prompt=prompt,
        target=target,
        output_span=output_span,
        reasoning_span=reasoning_span,
        method=method,
        hops=hops,
        top_k=top_k,
        device_map=device_map,
        dtype=dtype,
        chunk_tokens=chunk_tokens,
        sink_chunk_tokens=sink_chunk_tokens,
        use_chat_template=use_chat_template,
        loader=loader,
        tracer_cls=tracer_cls,
        work_dir=work_dir,
    )


def _format_span(span: tuple[int, int] | None) -> str:
    if span is None:
        return ""
    return f"{span[0]}:{span[1]}"


def _sections_to_dict(sections: GenerationSections) -> dict[str, object]:
    return {
        "generation_text": sections.generation_text,
        "thinking_char_span": sections.thinking_char_span,
        "answer_char_span": sections.answer_char_span,
        "thinking_token_span": sections.thinking_token_span,
        "answer_token_span": sections.answer_token_span,
        "parser": sections.parser,
    }


def _raw_token_rows(
    generation_text: str,
    tokenizer,
    sections: GenerationSections,
) -> list[list[object]]:
    records = build_token_records(
        text=generation_text,
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )
    thinking_span = sections.thinking_token_span
    answer_span = sections.answer_token_span

    def label_for(idx: int) -> str:
        if thinking_span is not None and thinking_span[0] <= idx <= thinking_span[1]:
            return "thinking"
        if answer_span[0] <= idx <= answer_span[1]:
            return "answer"
        return "other"

    return [
        [
            f"{label_for(r.token_index)}#{r.token_index}",
            r.token_id,
            r.token_text,
            f"{r.char_start}:{r.char_end}",
            r.kind,
            "yes" if r.selectable else "no",
        ]
        for r in records
    ]


def run_generate_phase(
    *,
    model_name: str,
    prompt: str,
    device_map: str,
    dtype: str,
    max_new_tokens: int,
    loader: Callable | None = None,
) -> tuple[str, dict[str, object], list[list[object]], str, str]:
    model_id = model_name.strip()
    prompt_text = prompt.strip()
    if not model_id:
        raise ValueError("Model is required.")
    if not prompt_text:
        raise ValueError("Prompt is required.")
    if len(prompt_text) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt must be at most {MAX_PROMPT_CHARS} characters.")

    model, tokenizer = _load_cached_model(
        model_id, device_map=device_map, dtype=dtype, loader=loader
    )
    output = generate_with_qwen(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_new_tokens=int(max_new_tokens),
    )

    sections = detect_sections(text=output.text, tokenizer=tokenizer)
    raw_rows = _raw_token_rows(output.text, tokenizer, sections)
    return (
        output.text,
        _sections_to_dict(sections),
        raw_rows,
        _format_span(sections.answer_token_span),
        _format_span(sections.thinking_token_span),
    )


def build_demo():
    import gradio as gr

    with gr.Blocks(title="FlashTrace Live Demo") as demo:
        gr.Markdown(
            "# FlashTrace Live Demo\n"
            "Generate with Qwen, inspect token spans, then trace selected answer."
        )

        with gr.Row():
            with gr.Column(scale=2):
                model_name = gr.Textbox(label="Model", value=DEFAULT_MODEL)
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=10)
                with gr.Row():
                    max_new_tokens = gr.Number(label="Max new tokens", value=128, precision=0)
                    generate_btn = gr.Button("Generate", variant="secondary")
            with gr.Column(scale=1):
                method = gr.Dropdown(
                    label="Method",
                    choices=["flashtrace", "ifr-span", "ifr-matrix"],
                    value="flashtrace",
                )
                output_span = gr.Textbox(label="Output span (auto-filled by Generate)", value="0:0")
                reasoning_span = gr.Textbox(label="Reasoning span", value="")
                hops = gr.Slider(label="Hops", minimum=1, maximum=4, step=1, value=1)
                top_k = gr.Slider(label="Top K", minimum=1, maximum=50, step=1, value=20)
                device_map = gr.Textbox(
                    label="Device map",
                    value=os.environ.get("FLASHTRACE_DEMO_DEVICE_MAP", "auto"),
                )
                dtype = gr.Dropdown(
                    label="Dtype",
                    choices=["auto", "float16", "bfloat16", "float32"],
                    value="auto",
                )
                chunk_tokens = gr.Number(label="Chunk tokens", value=128, precision=0)
                sink_chunk_tokens = gr.Number(label="Sink chunk tokens", value=32, precision=0)
                use_chat_template = gr.Checkbox(label="Use chat template", value=False)
                submit = gr.Button("Trace selected answer", variant="primary")

        target = gr.Textbox(label="Generated response (used as trace target)", value=DEFAULT_TARGET, lines=6)
        sections_state = gr.JSON(label="Detected sections")
        raw_tokens = gr.Dataframe(
            headers=["section#idx", "token_id", "token_text", "char_span", "kind", "selectable"],
            label="Raw tokens",
        )

        top_table = gr.Dataframe(headers=["Index", "Token", "Score"], label="Top input tokens")
        generation_tokens = gr.Textbox(label="Generation tokens", lines=8)
        heatmap = gr.HTML(label="Trace heatmap")
        json_file = gr.File(label="JSON trace")

        generate_btn.click(
            fn=lambda model_name, prompt, device_map, dtype, max_new_tokens: run_generate_phase(
                model_name=model_name,
                prompt=prompt,
                device_map=device_map,
                dtype=dtype,
                max_new_tokens=max_new_tokens,
            ),
            inputs=[model_name, prompt, device_map, dtype, max_new_tokens],
            outputs=[target, sections_state, raw_tokens, output_span, reasoning_span],
        )

        submit.click(
            fn=run_trace_from_ui,
            inputs=[
                model_name,
                prompt,
                target,
                output_span,
                reasoning_span,
                method,
                hops,
                top_k,
                device_map,
                dtype,
                chunk_tokens,
                sink_chunk_tokens,
                use_chat_template,
            ],
            outputs=[top_table, generation_tokens, heatmap, json_file],
        )
    return demo


if __name__ == "__main__":
    if os.environ.get("FLASHTRACE_DEMO_DRY_RUN") == "1":
        print("FlashTrace live demo dry run OK")
        raise SystemExit(0)
    build_demo().queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
    )
