from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import mimetypes
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
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

from demo.live import gallery, service

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_GALLERY_DIR = APP_DIR / "gallery_store"
INFERENCE_EXECUTOR = ThreadPoolExecutor(max_workers=1)


class TokenizeRequest(BaseModel):
    model: str = Field(default=service.DEFAULT_MODEL)
    prompt: str
    device_map: str = Field(default_factory=lambda: os.environ.get("FLASHTRACE_DEMO_DEVICE_MAP", "auto"))
    dtype: str = "auto"


class GenerateRequest(BaseModel):
    model: str = Field(default=service.DEFAULT_MODEL)
    prompt: str
    max_new_tokens: int = 8096
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


class GallerySaveRequest(BaseModel):
    title: str
    model: str
    method: str
    hops: int
    prompt: str
    generated_text: str
    target_span: str
    reasoning_span: str = ""
    render_model: dict[str, Any]
    trace_json: dict[str, Any]


async def _run_inference(func: Callable[..., dict[str, Any]], **kwargs) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(INFERENCE_EXECUTOR, partial(func, **kwargs))


def _error(status_code: int, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": str(exc)})


def _is_oom(exc: Exception) -> bool:
    return (
        exc.__class__.__name__ == "OutOfMemoryError"
        or "out of memory" in str(exc).lower()
    )


_OOM_MESSAGE = (
    "GPU is out of memory — the card may be busy with another job. "
    "Please retry in a moment."
)


def _handle_inference_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, ValueError):
        return _error(400, exc)
    if _is_oom(exc):
        # Free whatever partially loaded so the next request starts clean.
        service.unload_model()
        return _error(503, ValueError(_OOM_MESSAGE))
    return _error(500, exc)


async def _idle_unloader(idle_timeout: float, check_interval: float) -> None:
    """Evict the model once it has been idle past the timeout."""
    loop = asyncio.get_running_loop()
    try:
        while True:
            await asyncio.sleep(check_interval)
            if service.should_unload(idle_timeout):
                await loop.run_in_executor(INFERENCE_EXECUTOR, service.unload_model)
    except asyncio.CancelledError:
        pass


def create_app(
    *,
    loader: Callable | None = None,
    tracer_cls: type | None = None,
    gallery_dir: Path | None = None,
    tokenizer_loader: Callable | None = None,
) -> FastAPI:
    idle_timeout = float(os.environ.get("FLASHTRACE_DEMO_IDLE_TIMEOUT", "600"))
    idle_check_interval = float(
        os.environ.get("FLASHTRACE_DEMO_IDLE_CHECK_INTERVAL", "60")
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        task = None
        if idle_timeout > 0 and idle_check_interval > 0:
            task = asyncio.create_task(
                _idle_unloader(idle_timeout, idle_check_interval)
            )
        try:
            yield
        finally:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    app = FastAPI(title="FlashTrace Live Demo", lifespan=lifespan)
    app.state.loader = loader
    app.state.tokenizer_loader = tokenizer_loader
    app.state.tracer_cls = tracer_cls
    app.state.idle_timeout = idle_timeout
    env_dir = os.environ.get("FLASHTRACE_DEMO_GALLERY_DIR")
    app.state.gallery_dir = Path(
        gallery_dir if gallery_dir is not None else (env_dir or DEFAULT_GALLERY_DIR)
    )

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
                tokenizer_loader=request.app.state.tokenizer_loader,
            )
            return {"render_model": result["render_model"]}
        except Exception as exc:
            return _handle_inference_error(exc)

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
        except Exception as exc:
            return _handle_inference_error(exc)

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
        except Exception as exc:
            return _handle_inference_error(exc)

    @app.get("/api/status")
    async def status(request: Request):
        return service.model_status(idle_timeout=request.app.state.idle_timeout)

    @app.post("/api/unload")
    async def unload():
        loop = asyncio.get_running_loop()
        unloaded = await loop.run_in_executor(INFERENCE_EXECUTOR, service.unload_model)
        return {"ok": True, "unloaded": bool(unloaded)}

    @app.get("/api/gallery")
    async def gallery_list(request: Request):
        return {"samples": gallery.list_samples(request.app.state.gallery_dir)}

    @app.post("/api/gallery")
    async def gallery_save(payload: GallerySaveRequest, request: Request):
        try:
            summary = gallery.save_sample(
                request.app.state.gallery_dir, payload.model_dump()
            )
            return {"sample": summary}
        except ValueError as exc:
            return _error(400, exc)

    @app.get("/api/gallery/{sample_id}")
    async def gallery_get(sample_id: str, request: Request):
        try:
            return {"sample": gallery.load_sample(request.app.state.gallery_dir, sample_id)}
        except (KeyError, ValueError):
            return _error(404, ValueError("Sample not found."))

    @app.delete("/api/gallery/{sample_id}")
    async def gallery_delete(sample_id: str, request: Request):
        try:
            gallery.delete_sample(request.app.state.gallery_dir, sample_id)
            return {"ok": True}
        except (KeyError, ValueError):
            return _error(404, ValueError("Sample not found."))

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
