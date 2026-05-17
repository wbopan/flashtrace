from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from flashtrace.result import TokenScore, TraceResult
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


def load_live_app_module():
    app_path = Path(__file__).resolve().parents[1] / "demo" / "live" / "app.py"
    spec = importlib.util.spec_from_file_location("flashtrace_live_demo_app", app_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTracer:
    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.kwargs = kwargs

    def trace(self, **kwargs):
        assert kwargs["prompt"] == "Context: Paris is in France.\nQuestion: Capital?"
        assert kwargs["target"] == "Paris"
        assert kwargs["output_span"] == (0, 0)
        assert kwargs["reasoning_span"] is None
        assert kwargs["hops"] == 1
        assert kwargs["method"] == "flashtrace"
        return TraceResult(
            prompt_tokens=["Context", "Paris", "France"],
            generation_tokens=["Paris"],
            scores=[0.1, 0.8, 0.4],
            output_span=(0, 0),
            method="flashtrace",
            metadata={"model": "fake-model"},
        )


def _extract_model(html: str) -> dict:
    import json
    import re

    match = re.search(
        r"<script[^>]+id=\"flashtrace-token-document-data\"[^>]*>(.*?)</script>",
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return json.loads(match.group(1))


def test_run_trace_document_phase_returns_token_document_and_json(tmp_path):
    module = load_live_app_module()

    def fake_loader(model_name, **kwargs):
        assert model_name == "fake-model"
        assert kwargs["device_map"] == "auto"
        return "model", "tokenizer"

    html, json_path, status = module.run_trace_document_phase(
        model_name="fake-model",
        prompt="Context: Paris is in France.\nQuestion: Capital?",
        generated_text="Paris",
        target_span="0:0",
        reasoning_span="",
        method="flashtrace",
        hops=1,
        device_map="auto",
        dtype="auto",
        chunk_tokens=16,
        sink_chunk_tokens=4,
        use_chat_template=False,
        loader=fake_loader,
        tracer_cls=FakeTracer,
        work_dir=tmp_path,
    )

    assert _extract_model(html)["phase"] == "traced"
    assert "Trace complete" in status
    assert Path(json_path).exists()


def test_default_model_is_a_real_qwen():
    module = load_live_app_module()

    assert module.DEFAULT_MODEL == "Qwen/Qwen3-0.6B"


def test_run_trace_document_phase_validates_required_prompt(tmp_path):
    module = load_live_app_module()

    try:
        module.run_trace_document_phase(
            model_name="fake-model",
            prompt="",
            generated_text="Paris",
            target_span="0:0",
            reasoning_span="",
            method="flashtrace",
            hops=1,
            device_map="auto",
            dtype="auto",
            chunk_tokens=16,
            sink_chunk_tokens=4,
            use_chat_template=False,
            work_dir=tmp_path,
        )
    except ValueError as exc:
        assert "Prompt is required" in str(exc)
    else:
        raise AssertionError("run_trace_document_phase should reject an empty prompt")


def test_run_trace_document_phase_strips_single_trailing_eos_before_trace(tmp_path):
    module = load_live_app_module()
    _model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        return "model", tokenizer

    module.run_trace_document_phase(
        model_name="fake-model",
        prompt="Context: Paris is in France.\nQuestion: Capital?",
        generated_text="Paris" + tokenizer.eos_token,
        target_span="0:0",
        reasoning_span="",
        method="flashtrace",
        hops=1,
        device_map="auto",
        dtype="auto",
        chunk_tokens=16,
        sink_chunk_tokens=4,
        use_chat_template=False,
        loader=fake_loader,
        tracer_cls=FakeTracer,
        work_dir=tmp_path,
    )

    assert Path(tmp_path).exists()


def test_example_token_scores_are_ranked_for_display():
    result = TraceResult(
        prompt_tokens=["a", "b", "c"],
        generation_tokens=["x"],
        scores=[0.2, 0.5, 0.4],
    )

    assert [
        TokenScore(index=item.index, token=item.token, score=item.score)
        for item in result.topk_inputs(2)
    ] == [
        TokenScore(index=1, token="b", score=0.5),
        TokenScore(index=2, token="c", score=0.4),
    ]


def test_demo_dependency_extra_and_space_requirements_are_declared():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    demo_extra = pyproject["project"]["optional-dependencies"]["demo"]
    assert any(item.startswith("gradio") for item in demo_extra)

    requirements = (repo_root / "demo" / "live" / "requirements.txt").read_text(encoding="utf-8")
    assert "gradio" in requirements
    assert "flashtrace" in requirements


def test_gradio_6_js_bridge_spike_config():
    import inspect

    import gradio as gr

    assert gr.__version__ == "6.14.0"
    assert "js" in inspect.signature(gr.Blocks.launch).parameters
    assert "css" in inspect.signature(gr.Blocks.launch).parameters

    bridge_js = """
    (value) => {
      const root = document.querySelector('#spike-doc');
      return [root ? root.getAttribute('data-selected') : value];
    }
    """

    def echo(value):
        return value

    with gr.Blocks(title="FlashTrace JS Spike") as demo:
        gr.HTML("<div id='spike-doc' data-selected='0:1'></div>")
        carrier = gr.Textbox(value="", visible=False)
        out = gr.Textbox()
        trace = gr.Button("Trace")
        trace.click(fn=echo, inputs=[carrier], outputs=[out], js=bridge_js)

    config = demo.get_config_file()
    deps_with_js = [dep for dep in config["dependencies"] if dep.get("js")]
    assert len(deps_with_js) == 1
    assert "data-selected" in deps_with_js[0]["js"]


def test_live_app_script_entrypoint_imports_local_package():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["FLASHTRACE_DEMO_DRY_RUN"] = "1"

    result = subprocess.run(
        [sys.executable, "demo/live/app.py"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "FlashTrace live demo dry run OK" in result.stdout


class RecordingTracer:
    last_kwargs: dict | None = None

    def __init__(self, model, tokenizer, **kwargs):
        self.model = model
        self.tokenizer = tokenizer

    def trace(self, **kwargs):
        RecordingTracer.last_kwargs = kwargs
        return TraceResult(
            prompt_tokens=["a", "b", "c"],
            generation_tokens=["x"],
            scores=[0.1, 0.5, 0.3],
            output_span=kwargs.get("output_span"),
            reasoning_span=kwargs.get("reasoning_span"),
            method=kwargs.get("method", "flashtrace"),
        )


def test_run_prompt_document_phase_renders_prompt_tokens():
    module = load_live_app_module()
    _model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        return "model", tokenizer

    html, status = module.run_prompt_document_phase(
        model_name="tiny-qwen2",
        prompt="t10 t20",
        device_map="auto",
        dtype="auto",
        use_chat_template=False,
        loader=fake_loader,
    )
    model = _extract_model(html)

    assert status == "Prompt tokenized into 2 tokens."
    assert [token["text"] for token in model["views"][0]["tokens"]] == ["t10", "t20"]


def test_run_prompt_document_phase_chat_template_renders_template_tokens():
    module = load_live_app_module()
    _model, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|im_start|>", "<|im_end|>", "<|assistant|>"]}
    )
    tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
        "{{ m['content'] }}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )

    def fake_loader(model_name, **kwargs):
        return "model", tokenizer

    html, _status = module.run_prompt_document_phase(
        model_name="tiny-qwen2",
        prompt="t10",
        device_map="auto",
        dtype="auto",
        use_chat_template=True,
        loader=fake_loader,
    )
    tokens = _extract_model(html)["views"][0]["tokens"]

    assert any(token["text"] == "<|im_start|>" and token["kind"] == "template" for token in tokens)
    assert any(token["text"] == "<|im_end|>" and token["kind"] == "template" for token in tokens)
    assert any(token["text"] == "<|assistant|>" and token["kind"] == "template" for token in tokens)


def test_full_token_document_pipeline_generate_then_trace(tmp_path):
    module = load_live_app_module()
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        return model, tokenizer

    doc_html, text, output_span_text, reasoning_span_text, _status = module.run_generate_document_phase(
        model_name="tiny-qwen2",
        prompt="t10 t20 t30 t40",
        device_map="auto",
        dtype="auto",
        max_new_tokens=8,
        use_chat_template=False,
        loader=fake_loader,
    )

    html, json_path, _status = module.run_trace_document_phase(
        model_name="tiny-qwen2",
        prompt="t10 t20 t30 t40",
        generated_text=text,
        target_span=output_span_text,
        reasoning_span=reasoning_span_text,
        method="flashtrace",
        hops=1,
        device_map="auto",
        dtype="auto",
        chunk_tokens=16,
        sink_chunk_tokens=4,
        use_chat_template=False,
        loader=fake_loader,
        tracer_cls=RecordingTracer,
        work_dir=tmp_path,
    )

    assert _extract_model(doc_html)["phase"] == "generated"
    assert RecordingTracer.last_kwargs["output_span"] == module._parse_optional_span(output_span_text)
    assert RecordingTracer.last_kwargs["reasoning_span"] == module._parse_optional_span(reasoning_span_text)
    assert _extract_model(html)["phase"] == "traced"
    assert Path(json_path).exists()


def test_build_demo_wires_prompt_load_and_change_events():
    module = load_live_app_module()
    demo = module.build_demo()
    config = demo.get_config_file()
    targets = [
        target
        for dependency in config["dependencies"]
        for target in dependency.get("targets", [])
    ]

    assert any(event == "load" for _component, event in targets)
    assert any(event == "change" for _component, event in targets)
