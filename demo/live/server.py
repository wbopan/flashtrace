from __future__ import annotations

import asyncio
import os
import sys
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal


def _bootstrap_local_flashtrace() -> None:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() and (parent / "flashtrace").is_dir():
            sys.path.insert(0, str(parent))
            return


_bootstrap_local_flashtrace()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from demo.live import service

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1)


class TokenizeRequest(BaseModel):
    model: str = Field(default=service.DEFAULT_MODEL)
    prompt: str
    device_map: str = Field(default_factory=lambda: os.environ.get("FLASHTRACE_DEMO_DEVICE_MAP", "auto"))
    dtype: str = "auto"


class GenerateRequest(BaseModel):
    model: str = Field(default=service.DEFAULT_MODEL)
    prompt: str
    max_new_tokens: int = 128
    device_map: str = Field(default_factory=lambda: os.environ.get("FLASHTRACE_DEMO_DEVICE_MAP", "auto"))
    dtype: str = "auto"


class TraceRequest(BaseModel):
    model: str = Field(default=service.DEFAULT_MODEL)
    prompt: str
    generated_text: str
    target_span: str
    reasoning_span: str = ""
    method: Literal["flashtrace", "ifr-span", "ifr-matrix"] = "flashtrace"
    hops: int = 1
    device_map: str = Field(default_factory=lambda: os.environ.get("FLASHTRACE_DEMO_DEVICE_MAP", "auto"))
    dtype: str = "auto"
    chunk_tokens: int = 128
    sink_chunk_tokens: int = 32


async def _run_inference(func: Callable[..., dict[str, Any]], **kwargs) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(INFERENCE_EXECUTOR, partial(func, **kwargs))


def _error(status_code: int, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": str(exc)})


def create_app(
    *,
    loader: Callable | None = None,
    tracer_cls: type | None = None,
) -> FastAPI:
    app = FastAPI(title="FlashTrace Live Demo")
    app.state.loader = loader
    app.state.tracer_cls = tracer_cls

    @app.get("/")
    async def root() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/static/{asset_path:path}")
    async def static_asset(asset_path: str) -> Response:
        path = (STATIC_DIR / asset_path).resolve()
        if STATIC_DIR.resolve() not in path.parents or not path.is_file():
            return _error(404, ValueError("Static asset not found."))
        media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Response(path.read_bytes(), media_type=media_type)

    @app.post("/api/tokenize")
    async def tokenize(payload: TokenizeRequest, request: Request):
        try:
            result = await _run_inference(
                service.run_prompt_document_phase,
                model_name=payload.model,
                prompt=payload.prompt,
                device_map=payload.device_map,
                dtype=payload.dtype,
                loader=request.app.state.loader,
            )
            return {"render_model": result["render_model"]}
        except ValueError as exc:
            return _error(400, exc)
        except Exception as exc:
            return _error(500, exc)

    @app.post("/api/generate")
    async def generate(payload: GenerateRequest, request: Request):
        try:
            return await _run_inference(
                service.run_generate_document_phase,
                model_name=payload.model,
                prompt=payload.prompt,
                device_map=payload.device_map,
                dtype=payload.dtype,
                max_new_tokens=payload.max_new_tokens,
                loader=request.app.state.loader,
            )
        except ValueError as exc:
            return _error(400, exc)
        except Exception as exc:
            return _error(500, exc)

    @app.post("/api/trace")
    async def trace(payload: TraceRequest, request: Request):
        try:
            return await _run_inference(
                service.run_trace_document_phase,
                model_name=payload.model,
                prompt=payload.prompt,
                generated_text=payload.generated_text,
                target_span=payload.target_span,
                reasoning_span=payload.reasoning_span,
                method=payload.method,
                hops=payload.hops,
                device_map=payload.device_map,
                dtype=payload.dtype,
                chunk_tokens=payload.chunk_tokens,
                sink_chunk_tokens=payload.sink_chunk_tokens,
                loader=request.app.state.loader,
                tracer_cls=request.app.state.tracer_cls,
            )
        except ValueError as exc:
            return _error(400, exc)
        except Exception as exc:
            return _error(500, exc)

    return app


app = create_app()


if __name__ == "__main__":
    if os.environ.get("FLASHTRACE_DEMO_DRY_RUN") == "1":
        print("FlashTrace FastAPI demo dry run OK")
        raise SystemExit(0)

    import uvicorn

    uvicorn.run(
        "demo.live.server:app",
        host=os.environ.get("FLASHTRACE_DEMO_HOST", "127.0.0.1"),
        port=int(os.environ.get("FLASHTRACE_DEMO_PORT", "7860")),
    )
