from __future__ import annotations

import subprocess
import sys
import threading
import time
import tomllib
import os
import inspect
from pathlib import Path

import pytest
import httpx

from flashtrace.result import TokenScore, TraceResult
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


@pytest.fixture(autouse=True)
def clear_live_service_cache():
    from demo.live import service

    service.clear_model_cache()
    yield
    service.clear_model_cache()


class ASGITestClient:
    def __init__(self, app):
        self.app = app

    def get(self, url: str):
        return self._run("GET", url)

    def post(self, url: str, *, json: dict):
        return self._run("POST", url, json=json)

    def _run(self, method: str, url: str, **kwargs):
        import asyncio

        async def request():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, url, **kwargs)

        return asyncio.run(request())


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


def test_tokenize_api_returns_prompt_render_model():
    from demo.live.server import create_app

    _model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        assert model_name == "tiny-qwen2"
        assert kwargs["device_map"] == "auto"
        assert kwargs["dtype"] == "auto"
        return "model", tokenizer

    client = ASGITestClient(create_app(loader=fake_loader))
    response = client.post(
        "/api/tokenize",
        json={"model": "tiny-qwen2", "prompt": "t10 t20", "chat_template": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["render_model"]["phase"] == "prompt"
    assert body["render_model"]["views"][0]["tokens"][0]["text"] == "t10"


def test_generate_api_returns_generated_text_default_span_and_status():
    from demo.live.server import create_app

    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        return model, tokenizer

    client = ASGITestClient(create_app(loader=fake_loader))
    response = client.post(
        "/api/generate",
        json={
            "model": "tiny-qwen2",
            "prompt": "t10 t20 t30 t40",
            "max_new_tokens": 4,
            "chat_template": False,
            "device_map": "auto",
            "dtype": "auto",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["render_model"]["phase"] == "generated"
    assert isinstance(body["generated_text"], str)
    assert body["target_span"]
    assert "Generated" in body["status"]


def test_trace_api_returns_traced_render_model_and_inline_json():
    from demo.live.server import create_app

    _model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        return "model", tokenizer

    client = ASGITestClient(create_app(loader=fake_loader, tracer_cls=FakeTracer))
    response = client.post(
        "/api/trace",
        json={
            "model": "fake-model",
            "prompt": "Context: Paris is in France.\nQuestion: Capital?",
            "generated_text": "Paris",
            "target_span": "0:0",
            "reasoning_span": "",
            "method": "flashtrace",
            "hops": 1,
            "chat_template": False,
            "device_map": "auto",
            "dtype": "auto",
            "chunk_tokens": 16,
            "sink_chunk_tokens": 4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["render_model"]["phase"] == "traced"
    assert body["trace_json"]["method"] == "flashtrace"
    assert body["trace_json"]["output_span"] == [0, 0]
    assert "Trace complete" in body["status"]


def test_trace_service_strips_single_trailing_eos_before_trace():
    from demo.live import service

    _model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        return "model", tokenizer

    service.run_trace_document_phase(
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
    )


def test_trace_service_passes_spans_to_tracer():
    from demo.live import service

    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        return model, tokenizer

    result = service.run_generate_document_phase(
        model_name="tiny-qwen2",
        prompt="t10 t20 t30 t40",
        device_map="auto",
        dtype="auto",
        max_new_tokens=4,
        use_chat_template=False,
        loader=fake_loader,
    )
    service.run_trace_document_phase(
        model_name="tiny-qwen2",
        prompt="t10 t20 t30 t40",
        generated_text=result["generated_text"],
        target_span=result["target_span"],
        reasoning_span=result["reasoning_span"],
        method="flashtrace",
        hops=1,
        device_map="auto",
        dtype="auto",
        chunk_tokens=16,
        sink_chunk_tokens=4,
        use_chat_template=False,
        loader=fake_loader,
        tracer_cls=RecordingTracer,
    )

    assert RecordingTracer.last_kwargs["output_span"] == service.parse_optional_span(
        result["target_span"]
    )
    assert RecordingTracer.last_kwargs["reasoning_span"] == service.parse_optional_span(
        result["reasoning_span"]
    )


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"model": "tiny", "prompt": ""}, "Prompt is required"),
        ({"model": "tiny", "prompt": "x" * 4001}, "Prompt must be at most"),
    ],
)
def test_tokenize_validation_errors(payload, message):
    from demo.live.server import create_app

    client = ASGITestClient(create_app(loader=lambda *_args, **_kwargs: ("model", "tokenizer")))
    response = client.post("/api/tokenize", json=payload)

    assert response.status_code == 400
    assert message in response.json()["error"]


@pytest.mark.parametrize("target_span", ["bad", "2:1", "4:4"])
def test_trace_validation_rejects_bad_span(target_span):
    from demo.live.server import create_app

    _model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    def fake_loader(model_name, **kwargs):
        return "model", tokenizer

    client = ASGITestClient(create_app(loader=fake_loader, tracer_cls=FakeTracer))
    response = client.post(
        "/api/trace",
        json={
            "model": "fake-model",
            "prompt": "Context: Paris is in France.\nQuestion: Capital?",
            "generated_text": "Paris",
            "target_span": target_span,
            "reasoning_span": "",
            "method": "flashtrace",
            "hops": 1,
            "chat_template": False,
            "device_map": "auto",
            "dtype": "auto",
            "chunk_tokens": 16,
            "sink_chunk_tokens": 4,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]


def test_inference_executor_serializes_concurrent_requests():
    from demo.live import server

    _model, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    active = 0
    max_active = 0
    lock = threading.Lock()

    def slow_loader(model_name, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return "model", tokenizer

    server.service.clear_model_cache()
    client = ASGITestClient(server.create_app(loader=slow_loader))
    responses = []

    def post_tokenize(index: int):
        responses.append(
            client.post(
                "/api/tokenize",
                json={
                    "model": f"tiny-{index}",
                    "prompt": "t10 t20",
                    "chat_template": False,
                },
            )
        )

    threads = [threading.Thread(target=post_tokenize, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert server.INFERENCE_EXECUTOR._max_workers == 1
    assert [response.status_code for response in responses] == [200, 200, 200]
    assert max_active == 1


def test_run_inference_uses_plain_run_in_executor():
    from demo.live import server

    source = inspect.getsource(server._run_inference)

    assert "await loop.run_in_executor(INFERENCE_EXECUTOR, partial(func, **kwargs))" in source
    assert "Event" not in source
    assert "asyncio.sleep" not in source
    assert ".cancel()" not in source


def test_static_root_serves_frontend():
    from demo.live.server import create_app

    response = ASGITestClient(create_app()).get("/")

    assert response.status_code == 200
    assert "FlashTrace" in response.text
    assert "app.js" in response.text


def test_static_frontend_tokenizes_automatically_and_morphs_buttons():
    repo_root = Path(__file__).resolve().parents[1]
    index = (repo_root / "demo" / "live" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    app_js = (repo_root / "demo" / "live" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "tokenize-button" not in index
    assert "promptInput.addEventListener(\"input\", tokenizePrompt)" in app_js
    assert "promptInput.addEventListener(\"change\", tokenizePrompt)" in app_js
    assert "updatePhaseButtons(state.phase)" in app_js
    assert "els.generateButton.hidden = phase === \"generated\"" in app_js
    assert "els.traceButton.hidden = phase !== \"generated\"" in app_js


def test_server_script_entrypoint_imports_local_package():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["FLASHTRACE_DEMO_DRY_RUN"] = "1"

    result = subprocess.run(
        [sys.executable, "demo/live/server.py"],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0
    assert "FlashTrace FastAPI demo dry run OK" in result.stdout


def test_demo_dependency_extra_and_space_requirements_are_declared():
    repo_root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    demo_extra = pyproject["project"]["optional-dependencies"]["demo"]
    assert any(item.startswith("fastapi") for item in demo_extra)
    assert any(item.startswith("uvicorn") for item in demo_extra)
    assert any(item.startswith("httpx") for item in demo_extra)
    assert not any(item.startswith("gradio") for item in demo_extra)

    requirements = (repo_root / "demo" / "live" / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "fastapi" in requirements
    assert "uvicorn" in requirements
    assert "httpx" in requirements
    assert "gradio" not in requirements
    assert "flashtrace" in requirements


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
