# Live Demo：自动跳转 Aggregate + Explore Gallery 设计

日期：2026-05-23
范围：`demo/live/` FastAPI token-attribution live demo

## 背景

当前 traced 视图 tab 顺序为 `[Select target | Hop 1..N | Aggregate]`，默认停在最左的
Select target。需要两项改动：

1. trace 完成后自动跳到 Aggregate 视图。
2. 新增 Explore Gallery：服务端共享存储的预计算 trace 样本库，带 Save 按钮，点击样本即时
   回放（零 GPU）。

设计决策（已与用户确认）：
- 回放方式：**即时重渲染**，保存完整 traced `render_model` + `trace_json` + 元数据。
- 存储：**服务端共享文件**，所有访问者看到同一个画廊。
- 种子内容：**初始为空**，靠 Save 填充。
- UI：右侧 **slide-over 抽屉**，样本卡片列表。

## 功能 1：trace 后自动跳到 Aggregate

`service.run_trace_document_phase` 构建完 `document` 后，把
`document["active_view"]` 设为 Aggregate 的索引 `len(document["views"]) - 1`
（views 为 `[select, *hops, aggregate]`，Aggregate 恒为最后一个）。

前端 `renderDocument` 已读取 `model.active_view` 决定初始 tab，无需改动。Gallery 样本
保存时一并带上该 `active_view`，回放时停在 Aggregate，保持一致。

## 功能 2：Explore Gallery

### 存储层：`demo/live/gallery.py`（新模块）

- 样本存为一个目录里的逐条 JSON 文件。目录默认 `demo/live/gallery_store/`，由
  `app.state.gallery_dir` 注入（测试传 tmp 路径）。加入 `.gitignore`（运行时数据）。
- 目录在首次 Save 时惰性创建。
- 写入采用 temp 文件 + `os.replace` 原子落盘；列表通过扫目录得到。

样本 record schema：

```
{
  "id": str,                 # 时间戳 + 短随机后缀，URL-safe
  "title": str,
  "created_at": str,         # ISO8601 UTC
  "model": str,
  "method": str,
  "hops": int,
  "prompt": str,
  "generated_text": str,
  "target_span": str,        # "START:END"
  "reasoning_span": str,
  "render_model": {...},     # traced 文档视图，供即时回放
  "trace_json": {...}        # 完整 trace 结果，供下载
}
```

摘要（list 用，省略笨重字段）：`id / title / created_at / model / method / hops /
prompt_preview / target_span`。

模块函数：
- `list_samples(gallery_dir) -> list[dict]`：扫目录，返回按 `created_at` 倒序的摘要列表；
  目录不存在时返回 `[]`。
- `load_sample(gallery_dir, sample_id) -> dict`：读完整 record；找不到抛
  `KeyError`/`FileNotFoundError`。
- `save_sample(gallery_dir, record) -> dict`：校验必填字段、生成 `id`/`created_at`、
  原子写入，返回摘要。
- `delete_sample(gallery_dir, sample_id) -> None`：删除文件；找不到抛错。
- `sample_id` 必须经过校验（仅允许 `[A-Za-z0-9_-]`），防止路径穿越。

### API 层：`server.py`

- `GET /api/gallery` → `{"samples": [summary, ...]}`
- `POST /api/gallery`（body：`title` + `model/method/hops/prompt/generated_text/
  target_span/reasoning_span/render_model/trace_json`）→ 校验后写入，返回 `{"sample": summary}`。
- `GET /api/gallery/{id}` → `{"sample": full_record}`，找不到返回 404。
- `DELETE /api/gallery/{id}` → 204/200，找不到返回 404。
- 路径参数 `id` 非法或不存在均返回 404（不泄露文件系统细节）。
- `create_app` 增加可选 `gallery_dir` 参数，落到 `app.state.gallery_dir`，默认
  `APP_DIR / "gallery_store"`，可被 `FLASHTRACE_DEMO_GALLERY_DIR` 环境变量覆盖。

### 前端：`index.html` / `app.js` / `styles.css`

- 顶部常驻 **Gallery** 按钮 → 打开右侧 slide-over 抽屉。
- **Save to gallery** 按钮，仅 `phase === "traced"` 时显示；点击后在抽屉内弹出预填标题
  （取自 prompt 问句、可编辑）的输入框，确认即 `POST /api/gallery`，成功后刷新列表。
- 抽屉内样本卡片：标题、model/method/hops 徽章、prompt 预览片段、日期、删除 ×。
- 点击卡片 → `GET /api/gallery/{id}` → 把存好的 `render_model` 直接渲染进 workspace
  （零 GPU）；回填 prompt 文本域、model/method/hops 控件，以及 `state` 的
  `generatedText/targetSpan/reasoningSpan/traceJson`（这样仍可改 span 重 trace）；
  刷新 download 链接；关抽屉；停在 Aggregate。
- 删除 × → `DELETE /api/gallery/{id}` → 刷新列表。

### 测试

- `tests/test_live_api.py`：gallery 端点 save → list → load → delete 往返，使用注入的
  tmp `gallery_dir`；覆盖非法 id（404）、缺字段（400）、找不到（404）等错误路径。
- `tests/test_token_document.py`：断言 traced 文档的 `active_view` 指向最后一个视图
  （Aggregate）。

## 非目标（YAGNI）

- 不做样本编辑/重命名（删除后重存即可）。
- 不做分页、搜索、排序选项（按时间倒序即可）。
- 不做并发写锁（低流量，原子 replace 足够）。
- 不做存储容量上限。
