from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Callable


def _bootstrap_local_flashtrace() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "flashtrace").is_dir():
            sys.path.insert(0, str(parent))
            return


_bootstrap_local_flashtrace()

from demo.live.qwen_generation import generate_with_qwen
from demo.live.token_overlay import (
    TokenRecord,
    build_token_records,
    build_token_records_from_ids,
    detect_sections,
)
from demo.live.token_document import (
    TOKEN_DOCUMENT_CSS,
    TOKEN_DOCUMENT_ELEM_ID,
    TOKEN_DOCUMENT_JS,
    TRACE_SELECTION_JS,
    build_document_views,
    render_document_html,
)

DEFAULT_MODEL = os.environ.get("FLASHTRACE_DEMO_MODEL", "Qwen/Qwen3-0.6B")
APP_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT = """Context:
Paris is the capital of France.
Berlin is the capital of Germany.
Madrid is the capital of Spain.

Question: What is the capital of France?"""
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


def _format_span(span: tuple[int, int] | None) -> str:
    if span is None:
        return ""
    return f"{span[0]}:{span[1]}"


def _default_generation_span(records: list[TokenRecord]) -> tuple[int, int] | None:
    selectable = [
        int(record.token_index)
        for record in records
        if record.kind == "content" and record.selectable
    ]
    if not selectable:
        return None
    return min(selectable), max(selectable)


def _clamp_span(span: tuple[int, int] | None, max_index: int) -> tuple[int, int] | None:
    if span is None or max_index < 0:
        return None
    start, end = span
    start = max(0, min(int(start), int(max_index)))
    end = max(0, min(int(end), int(max_index)))
    if end < start:
        return None
    return start, end


def _prompt_text_for_document(prompt: str, tokenizer, *, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_prompt_records(prompt: str, tokenizer, *, use_chat_template: bool) -> list[TokenRecord]:
    return build_token_records(
        text=_prompt_text_for_document(
            prompt,
            tokenizer,
            use_chat_template=use_chat_template,
        ),
        tokenizer=tokenizer,
        section="prompt",
        role="user",
    )


def _strip_single_trailing_eos_text(text: str, tokenizer) -> str:
    eos = getattr(tokenizer, "eos_token", None)
    if eos and text.endswith(eos):
        return text[: -len(eos)]
    return text


def _empty_document_html() -> str:
    return render_document_html(build_document_views(phase="prompt", prompt_records=[]))


def run_prompt_document_phase(
    *,
    model_name: str,
    prompt: str,
    device_map: str,
    dtype: str,
    use_chat_template: bool,
    loader: Callable | None = None,
) -> tuple[str, str]:
    model_id = model_name.strip()
    prompt_text = prompt.strip()
    if not model_id:
        raise ValueError("Model is required.")
    if not prompt_text:
        raise ValueError("Prompt is required.")
    if len(prompt_text) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt must be at most {MAX_PROMPT_CHARS} characters.")
    _model, tokenizer = _load_cached_model(
        model_id, device_map=device_map, dtype=dtype, loader=loader
    )
    prompt_records = _build_prompt_records(
        prompt_text,
        tokenizer,
        use_chat_template=bool(use_chat_template),
    )
    html = render_document_html(
        build_document_views(phase="prompt", prompt_records=prompt_records)
    )
    return html, f"Prompt tokenized into {len(prompt_records)} tokens."


def run_generate_document_phase(
    *,
    model_name: str,
    prompt: str,
    device_map: str,
    dtype: str,
    max_new_tokens: int,
    use_chat_template: bool,
    loader: Callable | None = None,
) -> tuple[str, str, str, str, str]:
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
    prompt_records = _build_prompt_records(
        prompt_text,
        tokenizer,
        use_chat_template=bool(use_chat_template),
    )
    generation_records, generated_text = build_token_records_from_ids(
        token_ids=output.token_ids,
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )
    document = build_document_views(
        phase="generated",
        prompt_records=prompt_records,
        generation_records=generation_records,
    )
    target_span = document["target_span"]
    target_span_tuple = tuple(target_span) if target_span is not None else None
    max_index = target_span_tuple[1] if target_span_tuple is not None else -1
    reasoning_span = _clamp_span(sections.thinking_token_span, max_index)
    status = (
        f"Generated {len(generation_records)} tokens. "
        f"Default target span: {_format_span(target_span_tuple) or 'empty'}."
    )
    if sections.thinking_token_span is not None and reasoning_span is None:
        status += " Reasoning span was outside the attributable generation range and was dropped."
    if target_span_tuple is None:
        status += " Trace is disabled because there are no selectable generation tokens."
    return (
        render_document_html(document),
        generated_text,
        _format_span(target_span_tuple),
        _format_span(reasoning_span),
        status,
    )


def run_trace_document_phase(
    *,
    model_name: str,
    prompt: str,
    generated_text: str,
    target_span: str,
    reasoning_span: str,
    method: str,
    hops: int,
    device_map: str,
    dtype: str,
    chunk_tokens: int,
    sink_chunk_tokens: int,
    use_chat_template: bool,
    loader: Callable | None = None,
    tracer_cls: type | None = None,
    work_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> tuple[str, str, str]:
    model_id = model_name.strip()
    prompt_text = prompt.strip()
    if not model_id:
        raise ValueError("Model is required.")
    if not prompt_text:
        raise ValueError("Prompt is required.")
    if not generated_text:
        raise ValueError("Generate a response before tracing.")
    output_span = _parse_optional_span(target_span)
    if output_span is None:
        raise ValueError("Select at least two target endpoints before tracing.")
    if tracer_cls is None:
        from flashtrace import FlashTrace

        tracer_cls = FlashTrace

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
        target=_strip_single_trailing_eos_text(generated_text, tokenizer),
        output_span=output_span,
        reasoning_span=_parse_optional_span(reasoning_span),
        hops=int(hops),
        method=method,
    )
    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"flashtrace-{uuid.uuid4().hex}.json"
    result.to_json(json_path)
    document = build_document_views(phase="traced", result=result)
    view_count = len(document["views"])
    if view_count == 1:
        status = f"Trace complete with Aggregate view for {method}."
    else:
        status = f"Trace complete with Aggregate plus {view_count - 1} hop views."
    return render_document_html(document), str(json_path), status


def build_demo():
    import gradio as gr

    with gr.Blocks(title="FlashTrace Live Demo") as demo:
        gr.Markdown(
            "# FlashTrace Live Demo\n"
            "Generate with Qwen, inspect token spans, then trace selected answer."
        )

        generated_text_state = gr.State("")
        reasoning_span_state = gr.State("")
        target_span_input = gr.Textbox(value="", visible=False, label="Selected target span")

        with gr.Row():
            with gr.Column(scale=1):
                model_name = gr.Textbox(label="Model", value=DEFAULT_MODEL)
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT, lines=10)
                max_new_tokens = gr.Number(label="Max new tokens", value=128, precision=0)
                method = gr.Dropdown(
                    label="Method",
                    choices=["flashtrace", "ifr-span", "ifr-matrix"],
                    value="flashtrace",
                )
                hops = gr.Slider(label="Hops", minimum=1, maximum=4, step=1, value=1)
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
            with gr.Column(scale=2):
                chat_template_toggle = gr.Checkbox(label="Show chat template", value=False)
                token_doc = gr.HTML(value=_empty_document_html(), elem_id=TOKEN_DOCUMENT_ELEM_ID)
                status = gr.Markdown("Ready.")
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    trace_btn = gr.Button("Trace", variant="secondary", visible=False)

        json_file = gr.File(label="JSON trace")

        def on_prompt_preview(model_name, prompt, device_map, dtype, use_chat_template):
            try:
                return run_prompt_document_phase(
                    model_name=model_name,
                    prompt=prompt,
                    device_map=device_map,
                    dtype=dtype,
                    use_chat_template=use_chat_template,
                )
            except Exception as exc:
                return _empty_document_html(), f"Prompt preview failed: {exc}"

        def on_generate(model_name, prompt, device_map, dtype, max_new_tokens, use_chat_template):
            try:
                doc_html, generated_text, target_span, reasoning_span, message = run_generate_document_phase(
                    model_name=model_name,
                    prompt=prompt,
                    device_map=device_map,
                    dtype=dtype,
                    max_new_tokens=max_new_tokens,
                    use_chat_template=use_chat_template,
                )
                can_trace = bool(target_span)
                return (
                    doc_html,
                    generated_text,
                    target_span,
                    reasoning_span,
                    gr.update(visible=False),
                    gr.update(visible=can_trace),
                    gr.update(interactive=False),
                    message,
                    None,
                )
            except Exception as exc:
                return (
                    _empty_document_html(),
                    "",
                    "",
                    "",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    f"Generate failed: {exc}",
                    None,
                )

        def on_trace(
            target_span,
            model_name,
            prompt,
            generated_text,
            reasoning_span,
            method,
            hops,
            device_map,
            dtype,
            chunk_tokens,
            sink_chunk_tokens,
            use_chat_template,
            current_html,
        ):
            try:
                doc_html, json_path, message = run_trace_document_phase(
                    model_name=model_name,
                    prompt=prompt,
                    generated_text=generated_text,
                    target_span=target_span,
                    reasoning_span=reasoning_span,
                    method=method,
                    hops=hops,
                    device_map=device_map,
                    dtype=dtype,
                    chunk_tokens=chunk_tokens,
                    sink_chunk_tokens=sink_chunk_tokens,
                    use_chat_template=use_chat_template,
                )
                return (
                    doc_html,
                    json_path,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    message,
                )
            except Exception as exc:
                return (
                    current_html,
                    None,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(interactive=False),
                    f"Trace failed: {exc}",
                )

        chat_template_toggle.change(
            fn=on_prompt_preview,
            inputs=[model_name, prompt, device_map, dtype, chat_template_toggle],
            outputs=[token_doc, status],
        )
        prompt.change(
            fn=on_prompt_preview,
            inputs=[model_name, prompt, device_map, dtype, chat_template_toggle],
            outputs=[token_doc, status],
        )
        demo.load(
            fn=on_prompt_preview,
            inputs=[model_name, prompt, device_map, dtype, chat_template_toggle],
            outputs=[token_doc, status],
        )

        generate_btn.click(
            fn=on_generate,
            inputs=[model_name, prompt, device_map, dtype, max_new_tokens, chat_template_toggle],
            outputs=[
                token_doc,
                generated_text_state,
                target_span_input,
                reasoning_span_state,
                generate_btn,
                trace_btn,
                chat_template_toggle,
                status,
                json_file,
            ],
        )

        trace_btn.click(
            fn=on_trace,
            inputs=[
                target_span_input,
                model_name,
                prompt,
                generated_text_state,
                reasoning_span_state,
                method,
                hops,
                device_map,
                dtype,
                chunk_tokens,
                sink_chunk_tokens,
                chat_template_toggle,
                token_doc,
            ],
            outputs=[token_doc, json_file, generate_btn, trace_btn, chat_template_toggle, status],
            js=TRACE_SELECTION_JS,
        )
    return demo


if __name__ == "__main__":
    if os.environ.get("FLASHTRACE_DEMO_DRY_RUN") == "1":
        print("FlashTrace live demo dry run OK")
        raise SystemExit(0)
    build_demo().queue(max_size=8).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        js=TOKEN_DOCUMENT_JS,
        css=TOKEN_DOCUMENT_CSS,
    )
