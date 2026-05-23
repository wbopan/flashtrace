# Live Demo Gallery 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 给 `demo/live/` 加上「trace 后自动跳到 Aggregate」与「服务端共享的 Explore Gallery（带 Save、即时回放）」。

**Architecture:** 后端新增 `demo/live/gallery.py`（目录里逐条 JSON 文件的存储模块），`server.py` 暴露 4 个 `/api/gallery` 端点并把存储目录注入 `app.state.gallery_dir`。前端在 `app.js`/`index.html`/`styles.css` 加一个右侧 slide-over 抽屉，列出样本卡片、支持保存当前 trace 与点击即时回放（直接渲染存好的 `render_model`，零 GPU）。自动跳转通过让 `build_document_views` 在 traced phase 默认把 `active_view` 指向最后一个（Aggregate）视图实现。

**Tech Stack:** Python / FastAPI / Pydantic v2 / pytest + httpx ASGI / 原生 JS + CSS。

---

## 文件结构

- `demo/live/token_document.py`（改）— traced phase 默认 `active_view` 指向 Aggregate。
- `demo/live/gallery.py`（新）— 样本存储：`list_samples` / `load_sample` / `save_sample` / `delete_sample`，单一职责，无 web 依赖。
- `demo/live/server.py`（改）— `create_app` 增 `gallery_dir`；新增 `GallerySaveRequest` 与 4 个端点。
- `demo/live/static/index.html`（改）— Gallery / Save 按钮 + 抽屉 DOM。
- `demo/live/static/app.js`（改）— 抽屉开合、列表渲染、保存、回放、删除。
- `demo/live/static/styles.css`（改）— 抽屉与卡片样式。
- `.gitignore`（改）— 忽略 `demo/live/gallery_store/`。
- `tests/test_token_document.py`（改）— active_view 断言。
- `tests/test_live_gallery.py`（新）— gallery 模块单测。
- `tests/test_live_api.py`（改）— gallery API 往返 + 前端 wiring 字符串断言；`ASGITestClient` 加 `delete`。

测试命令统一前缀：`UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest`。

---

## Task 1: traced 文档默认跳到 Aggregate

**Files:**
- Modify: `demo/live/token_document.py:216-299`
- Test: `tests/test_token_document.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_token_document.py` 末尾追加：

```python
def test_build_document_views_traced_defaults_active_view_to_aggregate():
    from demo.live.token_document import build_document_views

    result = TraceResult(
        prompt_tokens=["t10", "t20"],
        generation_tokens=["t30"],
        scores=[0.2, 0.4],
        per_hop_scores=[[0.1, 0.9], [0.3, 0.7]],
        output_span=(0, 0),
        method="flashtrace",
    )

    model = build_document_views(phase="traced", result=result)

    # Views: [Select target, Hop 1, Hop 2, Aggregate]; default tab is Aggregate.
    assert model["active_view"] == len(model["views"]) - 1
    assert model["views"][model["active_view"]]["name"] == "Aggregate"


def test_build_document_views_non_traced_defaults_active_view_to_zero():
    from demo.live.token_document import build_document_views
    from demo.live.token_overlay import build_token_records

    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    prompt_records = build_token_records(
        text="t10 t20", tokenizer=tokenizer, section="prompt", role="user"
    )
    model = build_document_views(phase="prompt", prompt_records=prompt_records)
    assert model["active_view"] == 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_token_document.py::test_build_document_views_traced_defaults_active_view_to_aggregate -v`
Expected: FAIL（`active_view == 0`，不等于 `len(views)-1`）。

- [ ] **Step 3: 改实现**

把 `build_document_views` 的签名默认值从 `active_view: int = 0` 改为 `active_view: int | None = None`：

```python
def build_document_views(
    *,
    phase: Phase,
    prompt_records: Sequence[TokenRecord] | None = None,
    generation_records: Sequence[TokenRecord] | None = None,
    result: Any | None = None,
    target_span: tuple[int, int] | list[int] | None = None,
    answer_token_span: tuple[int, int] | list[int] | None = None,
    active_view: int | None = None,
) -> dict[str, Any]:
```

traced 分支的 return（原 `"active_view": int(active_view),`）改为：

```python
        views = [select_view, *hop_views, aggregate_view]
        resolved_active = int(active_view) if active_view is not None else len(views) - 1
        return {
            "phase": phase,
            "active_view": resolved_active,
            "views": views,
            "target_span": model_target,
        }
```

非 traced 分支的 return（原 `"active_view": int(active_view),`）改为：

```python
        "active_view": int(active_view) if active_view is not None else 0,
```

- [ ] **Step 4: 跑测试确认通过**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_token_document.py -v`
Expected: 全部 PASS（含两条新测试，原有 traced 测试不检查 active_view，不受影响）。

- [ ] **Step 5: 提交**

```bash
git add demo/live/token_document.py tests/test_token_document.py
git commit -m "feat(live): default traced view to Aggregate tab"
```

---

## Task 2: gallery 存储模块

**Files:**
- Create: `demo/live/gallery.py`
- Test: `tests/test_live_gallery.py`

- [ ] **Step 1: 写失败测试**

新建 `tests/test_live_gallery.py`：

```python
from __future__ import annotations

import pytest


def _record(title="Sample"):
    return {
        "title": title,
        "model": "tiny-qwen2",
        "method": "flashtrace",
        "hops": 1,
        "prompt": "Question: capital of France?",
        "generated_text": "Paris",
        "target_span": "0:0",
        "reasoning_span": "",
        "render_model": {"phase": "traced", "views": [], "active_view": 0},
        "trace_json": {"method": "flashtrace"},
    }


def test_save_then_list_then_load_roundtrip(tmp_path):
    from demo.live import gallery

    summary = gallery.save_sample(tmp_path, _record("First"))
    assert summary["title"] == "First"
    assert summary["id"]
    assert summary["created_at"]
    assert "render_model" not in summary
    assert summary["prompt_preview"].startswith("Question:")

    listed = gallery.list_samples(tmp_path)
    assert [item["id"] for item in listed] == [summary["id"]]

    full = gallery.load_sample(tmp_path, summary["id"])
    assert full["render_model"] == {"phase": "traced", "views": [], "active_view": 0}
    assert full["trace_json"] == {"method": "flashtrace"}


def test_list_sorted_newest_first(tmp_path):
    from demo.live import gallery

    a = gallery.save_sample(tmp_path, _record("A"))
    b = gallery.save_sample(tmp_path, _record("B"))
    ids = [item["id"] for item in gallery.list_samples(tmp_path)]
    assert set(ids) == {a["id"], b["id"]}
    # created_at descending; ids are unique per save.
    created = [item["created_at"] for item in gallery.list_samples(tmp_path)]
    assert created == sorted(created, reverse=True)


def test_delete_removes_sample(tmp_path):
    from demo.live import gallery

    summary = gallery.save_sample(tmp_path, _record())
    gallery.delete_sample(tmp_path, summary["id"])
    assert gallery.list_samples(tmp_path) == []
    with pytest.raises(KeyError):
        gallery.load_sample(tmp_path, summary["id"])


def test_list_on_missing_dir_returns_empty(tmp_path):
    from demo.live import gallery

    assert gallery.list_samples(tmp_path / "nope") == []


def test_save_missing_field_raises(tmp_path):
    from demo.live import gallery

    record = _record()
    del record["render_model"]
    with pytest.raises(ValueError):
        gallery.save_sample(tmp_path, record)


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", "x.json"])
def test_invalid_id_rejected(tmp_path, bad_id):
    from demo.live import gallery

    with pytest.raises(ValueError):
        gallery.load_sample(tmp_path, bad_id)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_live_gallery.py -v`
Expected: FAIL（`ModuleNotFoundError: demo.live.gallery`）。

- [ ] **Step 3: 写实现**

新建 `demo/live/gallery.py`：

```python
from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")

_REQUIRED_FIELDS = (
    "title",
    "model",
    "method",
    "hops",
    "prompt",
    "generated_text",
    "target_span",
    "reasoning_span",
    "render_model",
    "trace_json",
)

_SUMMARY_FIELDS = ("id", "title", "created_at", "model", "method", "hops", "target_span")
_PREVIEW_CHARS = 160


def _validate_id(sample_id: str) -> str:
    if not sample_id or not _ID_RE.match(sample_id):
        raise ValueError("Invalid sample id.")
    return sample_id


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    summary = {field: record.get(field) for field in _SUMMARY_FIELDS}
    summary["prompt_preview"] = (record.get("prompt") or "")[:_PREVIEW_CHARS]
    return summary


def list_samples(gallery_dir: Path) -> list[dict[str, Any]]:
    directory = Path(gallery_dir)
    if not directory.is_dir():
        return []
    summaries: list[dict[str, Any]] = []
    for path in directory.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summaries.append(_summary(record))
    summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return summaries


def load_sample(gallery_dir: Path, sample_id: str) -> dict[str, Any]:
    _validate_id(sample_id)
    path = Path(gallery_dir) / f"{sample_id}.json"
    if not path.is_file():
        raise KeyError(sample_id)
    return json.loads(path.read_text(encoding="utf-8"))


def save_sample(gallery_dir: Path, record: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in _REQUIRED_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}.")
    directory = Path(gallery_dir)
    directory.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    sample_id = f"{now.strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(4)}"
    stored = dict(record)
    stored["id"] = sample_id
    stored["created_at"] = now.isoformat()
    path = directory / f"{sample_id}.json"
    tmp = directory / f".{sample_id}.json.tmp"
    tmp.write_text(json.dumps(stored, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return _summary(stored)


def delete_sample(gallery_dir: Path, sample_id: str) -> None:
    _validate_id(sample_id)
    path = Path(gallery_dir) / f"{sample_id}.json"
    if not path.is_file():
        raise KeyError(sample_id)
    path.unlink()
```

- [ ] **Step 4: 跑测试确认通过**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_live_gallery.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add demo/live/gallery.py tests/test_live_gallery.py
git commit -m "feat(live): add gallery sample storage module"
```

---

## Task 3: gallery API 端点

**Files:**
- Modify: `demo/live/server.py:29-31`（常量）、`72-83`（create_app 签名/state）、`150-151`（端点尾部前插入）
- Modify: `.gitignore`
- Test: `tests/test_live_api.py`（顶部 `ASGITestClient` 加 `delete`；文件末尾加 gallery API 测试）

- [ ] **Step 1: 写失败测试**

先给 `tests/test_live_api.py` 的 `ASGITestClient` 类加一个 `delete` 方法（放在 `post` 方法之后、`_run` 之前）：

```python
    def delete(self, url: str):
        return self._run("DELETE", url)
```

然后在 `tests/test_live_api.py` 末尾追加：

```python
def _gallery_record():
    return {
        "title": "Capital of France",
        "model": "tiny-qwen2",
        "method": "flashtrace",
        "hops": 1,
        "prompt": "Question: capital of France?",
        "generated_text": "Paris",
        "target_span": "0:0",
        "reasoning_span": "",
        "render_model": {"phase": "traced", "views": [], "active_view": 0},
        "trace_json": {"method": "flashtrace"},
    }


def test_gallery_save_list_load_delete_roundtrip(tmp_path):
    from demo.live.server import create_app

    client = ASGITestClient(create_app(gallery_dir=tmp_path))

    save = client.post("/api/gallery", json=_gallery_record())
    assert save.status_code == 200
    sample_id = save.json()["sample"]["id"]
    assert sample_id
    assert "render_model" not in save.json()["sample"]

    listing = client.get("/api/gallery")
    assert listing.status_code == 200
    assert [item["id"] for item in listing.json()["samples"]] == [sample_id]

    full = client.get(f"/api/gallery/{sample_id}")
    assert full.status_code == 200
    assert full.json()["sample"]["render_model"] == {
        "phase": "traced",
        "views": [],
        "active_view": 0,
    }

    deleted = client.delete(f"/api/gallery/{sample_id}")
    assert deleted.status_code == 200
    assert client.get("/api/gallery").json()["samples"] == []


def test_gallery_get_missing_returns_404(tmp_path):
    from demo.live.server import create_app

    client = ASGITestClient(create_app(gallery_dir=tmp_path))
    response = client.get("/api/gallery/nonexistent")
    assert response.status_code == 404
    assert response.json()["error"]


def test_gallery_save_missing_field_returns_400(tmp_path):
    from demo.live.server import create_app

    client = ASGITestClient(create_app(gallery_dir=tmp_path))
    bad = _gallery_record()
    del bad["render_model"]
    response = client.post("/api/gallery", json=bad)
    # Pydantic rejects the missing required field with a 422.
    assert response.status_code == 422
```

- [ ] **Step 2: 跑测试确认失败**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_live_api.py -k gallery -v`
Expected: FAIL（`create_app()` 不接受 `gallery_dir`，且端点不存在）。

- [ ] **Step 3: 写实现**

在 `demo/live/server.py` 顶部常量区（`STATIC_DIR` 之后）加：

```python
DEFAULT_GALLERY_DIR = APP_DIR / "gallery_store"
```

并在文件已有 `from demo.live import service` 后加：

```python
from demo.live import gallery
```

加一个保存请求模型（放在 `TraceRequest` 之后）：

```python
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
```

`create_app` 签名与 state 改为：

```python
def create_app(
    *,
    loader: Callable | None = None,
    tracer_cls: type | None = None,
    gallery_dir: Path | None = None,
) -> FastAPI:
    app = FastAPI(title="FlashTrace Live Demo")
    app.state.loader = loader
    app.state.tracer_cls = tracer_cls
    env_dir = os.environ.get("FLASHTRACE_DEMO_GALLERY_DIR")
    app.state.gallery_dir = Path(
        gallery_dir if gallery_dir is not None else (env_dir or DEFAULT_GALLERY_DIR)
    )
```

在 `return app` 之前插入 4 个端点：

```python
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
```

在 `.gitignore` 的「FlashTrace generated artifacts」段落里加一行：

```
demo/live/gallery_store/
```

- [ ] **Step 4: 跑测试确认通过**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_live_api.py -k gallery -v`
Expected: 全部 PASS。

- [ ] **Step 5: 提交**

```bash
git add demo/live/server.py .gitignore tests/test_live_api.py
git commit -m "feat(live): add gallery REST endpoints"
```

---

## Task 4: 前端抽屉 + 保存 + 回放

**Files:**
- Modify: `demo/live/static/index.html:53-59`（actions）、`60-65`（body 末尾加抽屉）
- Modify: `demo/live/static/app.js`（state、updatePhaseButtons、新增函数、bind、test hook）
- Modify: `demo/live/static/styles.css`（文件末尾追加样式）
- Test: `tests/test_live_api.py`（新增 wiring 字符串断言测试）

- [ ] **Step 1: 写失败测试**

在 `tests/test_live_api.py` 末尾追加：

```python
def test_static_frontend_wires_gallery_and_save():
    repo_root = Path(__file__).resolve().parents[1]
    index = (repo_root / "demo" / "live" / "static" / "index.html").read_text(
        encoding="utf-8"
    )
    app_js = (repo_root / "demo" / "live" / "static" / "app.js").read_text(
        encoding="utf-8"
    )

    assert "gallery-button" in index
    assert "save-button" in index
    assert "gallery-drawer" in index
    assert "gallery-list" in index
    assert "gallery-title-input" in index

    assert "/api/gallery" in app_js
    assert "function openGallery" in app_js
    assert "function loadSample" in app_js
    assert "function confirmSave" in app_js
    # Save button only shows in the traced phase.
    assert 'state.phase !== "traced"' in app_js
```

- [ ] **Step 2: 跑测试确认失败**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_live_api.py::test_static_frontend_wires_gallery_and_save -v`
Expected: FAIL（字符串都不存在）。

- [ ] **Step 3a: 改 index.html**

把 `.actions` 块（含 download-link 那段）替换为：

```html
      <div class="actions">
        <button id="generate-button" type="button" class="primary">Generate</button>
        <button id="trace-button" type="button" class="primary">Trace</button>
        <button id="gallery-button" type="button">Gallery</button>
        <button id="save-button" type="button" hidden>Save to gallery</button>
      </div>

      <a id="download-link" class="download-link" hidden>Download trace JSON</a>
```

在 `</main>` 之后、`</body>` 之前插入抽屉 DOM：

```html
  <div id="gallery-drawer" class="gallery-drawer" hidden aria-label="Gallery">
    <div id="gallery-overlay" class="gallery-overlay"></div>
    <aside class="gallery-panel">
      <header class="gallery-header">
        <h2>Gallery</h2>
        <button id="gallery-close" type="button" class="gallery-close" aria-label="Close">&times;</button>
      </header>
      <div id="gallery-save-form" class="gallery-save-form" hidden>
        <input id="gallery-title-input" type="text" placeholder="Sample title" autocomplete="off">
        <div class="gallery-save-actions">
          <button id="gallery-save-confirm" type="button" class="primary">Save</button>
          <button id="gallery-save-cancel" type="button">Cancel</button>
        </div>
      </div>
      <div id="gallery-list" class="gallery-list"></div>
    </aside>
  </div>
```

- [ ] **Step 3b: 改 app.js**

在 `updatePhaseButtons` 函数体末尾（`}` 之前）加入 Save 按钮可见性：

```javascript
    if (els.saveButton) {
      els.saveButton.hidden = state.phase !== "traced";
    }
```

在 `updateDownload` 函数之后插入这些 gallery 函数：

```javascript
  function defaultTitle() {
    const prompt = $("prompt-input")?.value || "";
    const lines = prompt.split("\n").map((line) => line.trim()).filter(Boolean);
    const question = [...lines].reverse().find((line) => line.endsWith("?"));
    return (question || lines[0] || "Untitled").slice(0, 80);
  }

  function galleryCard(sample) {
    const card = document.createElement("div");
    card.className = "gallery-card";
    card.dataset.id = sample.id;
    const del = document.createElement("button");
    del.type = "button";
    del.className = "gallery-card-delete";
    del.textContent = "×";
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteSample(sample.id);
    });
    const title = document.createElement("div");
    title.className = "gallery-card-title";
    title.textContent = sample.title || "(untitled)";
    const meta = document.createElement("div");
    meta.className = "gallery-card-meta";
    meta.textContent = `${sample.model} · ${sample.method} · ${sample.hops} hop`;
    const preview = document.createElement("div");
    preview.className = "gallery-card-preview";
    preview.textContent = sample.prompt_preview || "";
    card.append(del, title, meta, preview);
    card.addEventListener("click", () => loadSample(sample.id));
    return card;
  }

  function renderGalleryList(samples) {
    const list = els.galleryList;
    if (!list) return;
    list.textContent = "";
    if (!samples.length) {
      const empty = document.createElement("p");
      empty.className = "gallery-empty";
      empty.textContent = "No saved samples yet.";
      list.appendChild(empty);
      return;
    }
    samples.forEach((sample) => list.appendChild(galleryCard(sample)));
  }

  async function loadGalleryList() {
    try {
      const response = await fetch("/api/gallery");
      const body = await response.json();
      renderGalleryList(body.samples || []);
    } catch (error) {
      setStatus(error.message);
    }
  }

  function openGallery() {
    if (els.galleryDrawer) els.galleryDrawer.hidden = false;
    loadGalleryList();
  }

  function closeGallery() {
    if (els.galleryDrawer) els.galleryDrawer.hidden = true;
    if (els.gallerySaveForm) els.gallerySaveForm.hidden = true;
  }

  function openSaveForm() {
    openGallery();
    if (els.gallerySaveForm) els.gallerySaveForm.hidden = false;
    if (els.galleryTitleInput) {
      els.galleryTitleInput.value = defaultTitle();
      els.galleryTitleInput.focus();
    }
  }

  async function confirmSave() {
    if (state.phase !== "traced" || !state.renderModel || !state.traceJson) {
      setStatus("Trace something before saving.");
      return;
    }
    const title = (els.galleryTitleInput?.value || "").trim() || defaultTitle();
    try {
      await postJSON("/api/gallery", {
        title,
        model: readText("model-input", "Qwen/Qwen3-4B-Thinking-2507"),
        method: readText("method-select", "flashtrace"),
        hops: readNumber("hops-input", 1),
        prompt: $("prompt-input")?.value ?? "",
        generated_text: state.generatedText,
        target_span: state.targetSpan,
        reasoning_span: state.reasoningSpan,
        render_model: state.renderModel,
        trace_json: state.traceJson,
      });
      if (els.gallerySaveForm) els.gallerySaveForm.hidden = true;
      setStatus("Saved to gallery.");
      loadGalleryList();
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function loadSample(id) {
    try {
      const response = await fetch(`/api/gallery/${id}`);
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || "Failed to load sample.");
      const sample = body.sample;
      if ($("model-input")) $("model-input").value = sample.model || "";
      if ($("prompt-input")) $("prompt-input").value = sample.prompt || "";
      if ($("method-select")) $("method-select").value = sample.method || "flashtrace";
      if ($("hops-input")) $("hops-input").value = String(sample.hops ?? 1);
      state.generatedText = sample.generated_text || "";
      state.reasoningSpan = sample.reasoning_span || "";
      state.traceJson = sample.trace_json || null;
      renderDocument(sample.render_model);
      if (sample.trace_json) updateDownload(sample.trace_json);
      closeGallery();
      setStatus(`Loaded "${sample.title}".`);
    } catch (error) {
      setStatus(error.message);
    }
  }

  async function deleteSample(id) {
    try {
      await fetch(`/api/gallery/${id}`, { method: "DELETE" });
      loadGalleryList();
    } catch (error) {
      setStatus(error.message);
    }
  }
```

在 `bind()` 里，已有的 `els.downloadLink = $("download-link");` 之后加：

```javascript
    els.galleryButton = $("gallery-button");
    els.saveButton = $("save-button");
    els.galleryDrawer = $("gallery-drawer");
    els.galleryList = $("gallery-list");
    els.gallerySaveForm = $("gallery-save-form");
    els.galleryTitleInput = $("gallery-title-input");
```

在 `bind()` 里已有的 `els.traceButton?.addEventListener("click", trace);` 之后加：

```javascript
    els.galleryButton?.addEventListener("click", openGallery);
    els.saveButton?.addEventListener("click", openSaveForm);
    $("gallery-close")?.addEventListener("click", closeGallery);
    $("gallery-overlay")?.addEventListener("click", closeGallery);
    $("gallery-save-confirm")?.addEventListener("click", confirmSave);
    $("gallery-save-cancel")?.addEventListener("click", () => {
      if (els.gallerySaveForm) els.gallerySaveForm.hidden = true;
    });
```

把 `window.flashtraceTest` 暴露对象扩展为：

```javascript
  window.flashtraceTest = {
    renderDocument,
    renderGalleryList,
    loadSample,
    getState: () => ({ ...state }),
    setState: (patch) => Object.assign(state, patch || {}),
  };
```

- [ ] **Step 3c: 改 styles.css**

在 `demo/live/static/styles.css` 末尾追加：

```css
.gallery-drawer {
  position: fixed;
  inset: 0;
  z-index: 50;
}

.gallery-overlay {
  position: absolute;
  inset: 0;
  background: rgba(15, 18, 24, 0.45);
}

.gallery-panel {
  position: absolute;
  top: 0;
  right: 0;
  width: min(420px, 92vw);
  height: 100%;
  background: #fff;
  box-shadow: -8px 0 24px rgba(0, 0, 0, 0.18);
  display: flex;
  flex-direction: column;
  padding: 16px;
  gap: 12px;
  overflow-y: auto;
}

.gallery-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.gallery-header h2 {
  margin: 0;
  font-size: 1.1rem;
}

.gallery-close {
  border: none;
  background: transparent;
  font-size: 1.4rem;
  line-height: 1;
  cursor: pointer;
}

.gallery-save-form {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 12px;
  border: 1px solid #e2e6ee;
  border-radius: 8px;
}

.gallery-save-form input {
  padding: 8px;
  border: 1px solid #cfd6e2;
  border-radius: 6px;
  font: inherit;
}

.gallery-save-actions {
  display: flex;
  gap: 8px;
}

.gallery-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.gallery-empty {
  color: #6b7280;
  font-size: 0.9rem;
}

.gallery-card {
  position: relative;
  border: 1px solid #e2e6ee;
  border-radius: 8px;
  padding: 12px 14px;
  cursor: pointer;
  transition: border-color 0.15s ease, box-shadow 0.15s ease;
}

.gallery-card:hover {
  border-color: #94a3b8;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
}

.gallery-card-title {
  font-weight: 600;
  margin-bottom: 4px;
  padding-right: 20px;
}

.gallery-card-meta {
  font-size: 0.78rem;
  color: #6b7280;
  margin-bottom: 6px;
}

.gallery-card-preview {
  font-size: 0.82rem;
  color: #374151;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.gallery-card-delete {
  position: absolute;
  top: 8px;
  right: 8px;
  border: none;
  background: transparent;
  font-size: 1.1rem;
  line-height: 1;
  color: #9ca3af;
  cursor: pointer;
}

.gallery-card-delete:hover {
  color: #ef4444;
}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_live_api.py::test_static_frontend_wires_gallery_and_save tests/test_live_api.py::test_static_frontend_tokenizes_automatically_and_morphs_buttons -v`
Expected: 两条都 PASS（原有 morphs 测试不受影响）。

- [ ] **Step 5: 提交**

```bash
git add demo/live/static/index.html demo/live/static/app.js demo/live/static/styles.css tests/test_live_api.py
git commit -m "feat(live): explore gallery drawer with save and instant replay"
```

---

## Task 5: 全量回归 + 手动验证

**Files:** 无新增改动（除非发现回归）。

- [ ] **Step 1: 跑 live demo 全量测试**

Run: `UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache uv run pytest tests/test_live_api.py tests/test_live_gallery.py tests/test_token_document.py -v`
Expected: 全部 PASS。

- [ ] **Step 2: 重启本地 server 验证（模型已缓存）**

复用已在跑的本地 server（端口 7860）或重启：
```bash
kill $(pgrep -f "demo/live/server.py") 2>/dev/null; sleep 1
UV_CACHE_DIR=/tmp/uv-cache MPLCONFIGDIR=/tmp/matplotlib-cache FLASHTRACE_DEMO_PORT=7860 FLASHTRACE_DEMO_HOST=127.0.0.1 nohup uv run python demo/live/server.py > /tmp/flashtrace-demo.log 2>&1 &
```
然后用浏览器（Chrome MCP / cloudflared 隧道）走一遍：Generate → 选 span → Trace（应自动停在 Aggregate）→ Save to gallery（填标题、确认）→ 打开 Gallery 抽屉看到卡片 → 点卡片即时回放（停在 Aggregate、无 GPU 等待）→ 删除卡片。

- [ ] **Step 3: 最终确认**

确认无回归、手动流程通过。无新增代码改动则不提交；若修了 bug，按 TDD 补测试后提交。

---

## 自检结论

- **Spec 覆盖**：自动跳转(Task 1)、存储(Task 2)、API(Task 3)、前端抽屉/Save/即时回放/删除(Task 4)、测试散落在各 Task + Task 5 回归 —— spec 各节均有对应 task。
- **Placeholder**：无 TODO/TBD，所有代码步骤含完整代码。
- **类型一致**：`gallery.{list_samples,load_sample,save_sample,delete_sample}` 签名在模块、server、测试三处一致；前端 `openGallery/loadSample/confirmSave/deleteSample/renderGalleryList` 命名贯穿 app.js 与 wiring 测试一致；record schema 字段在 spec、gallery 模块 `_REQUIRED_FIELDS`、`GallerySaveRequest`、前端 `confirmSave` POST body 四处一致。
- **决策细化**：spec 写「在 service 设 active_view」，计划改为在 `build_document_views` 设默认（render_model 自描述，gallery 回放天然带正确 active_view，且单测更直接）—— 行为与 spec 目标一致。
