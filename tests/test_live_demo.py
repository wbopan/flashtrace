from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from flashtrace.result import TokenScore, TraceResult


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


def test_run_trace_returns_table_html_and_json(tmp_path):
    module = load_live_app_module()

    def fake_loader(model_name, **kwargs):
        assert model_name == "fake-model"
        assert kwargs["device_map"] == "auto"
        return "model", "tokenizer"

    top_rows, generation_tokens, html, json_path = module.run_trace(
        model_name="fake-model",
        prompt="Context: Paris is in France.\nQuestion: Capital?",
        target="Paris",
        output_span="0:0",
        reasoning_span="",
        method="flashtrace",
        hops=1,
        top_k=2,
        device_map="auto",
        dtype="auto",
        chunk_tokens=16,
        sink_chunk_tokens=4,
        use_chat_template=False,
        loader=fake_loader,
        tracer_cls=FakeTracer,
        work_dir=tmp_path,
    )

    assert top_rows == [
        [1, "Paris", 0.8],
        [2, "France", 0.4],
    ]
    assert generation_tokens == "0: 'Paris'"
    assert "<html" in html
    assert "Paris" in html
    assert Path(json_path).exists()


def test_paris_smoke_trace_ranks_context_paris_first(tmp_path):
    module = load_live_app_module()

    top_rows, generation_tokens, html, json_path = module.run_trace(
        model_name="demo/paris-smoke",
        prompt="Context: Paris is the capital of France. Berlin is the capital of Germany. Madrid is the capital of Spain. Question: What is the capital of France?",
        target="Paris",
        output_span="0:0",
        reasoning_span="",
        method="flashtrace",
        hops=1,
        top_k=3,
        device_map="auto",
        dtype="auto",
        chunk_tokens=16,
        sink_chunk_tokens=4,
        use_chat_template=False,
        work_dir=tmp_path,
    )

    assert top_rows[0][1:] == ["Paris", 1.0]
    assert generation_tokens == "0: 'Paris'"
    assert "Paris" in html
    assert Path(json_path).exists()


def test_default_model_uses_fast_paris_smoke_demo():
    module = load_live_app_module()

    assert module.DEFAULT_MODEL == "demo/paris-smoke"


def test_run_trace_validates_required_prompt(tmp_path):
    module = load_live_app_module()

    try:
        module.run_trace(
            model_name="fake-model",
            prompt="",
            target="Paris",
            output_span="0:0",
            reasoning_span="",
            method="flashtrace",
            hops=1,
            top_k=2,
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
        raise AssertionError("run_trace should reject an empty prompt")


def test_run_trace_from_ui_adapts_gradio_positional_args(tmp_path):
    module = load_live_app_module()

    def fake_loader(model_name, **kwargs):
        return "model", "tokenizer"

    outputs = module.run_trace_from_ui(
        "fake-model",
        "Context: Paris is in France.\nQuestion: Capital?",
        "Paris",
        "0:0",
        "",
        "flashtrace",
        1,
        2,
        "auto",
        "auto",
        16,
        4,
        False,
        loader=fake_loader,
        tracer_cls=FakeTracer,
        work_dir=tmp_path,
    )

    assert outputs[0][0] == [1, "Paris", 0.8]
    assert Path(outputs[3]).exists()


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


def test_generate_phase_smoke_returns_text_and_sections(tmp_path):
    module = load_live_app_module()

    text, sections, raw_rows, output_span_text, reasoning_span_text = module.run_generate_phase(
        model_name="demo/paris-smoke",
        prompt="What is the capital of France?",
        device_map="auto",
        dtype="auto",
        max_new_tokens=64,
    )

    assert "<answer>" in text and "Paris" in text
    assert sections["parser"] == "think_answer"
    assert ":" in output_span_text
    assert ":" in reasoning_span_text
    assert any("answer" in row[0] or "thinking" in row[0] for row in raw_rows)
