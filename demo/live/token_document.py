from __future__ import annotations

import json
from collections.abc import Sequence
from html import escape
from typing import Any, Literal

from demo.live.token_overlay import TokenRecord
from flashtrace.viz import _score_color

Phase = Literal["prompt", "generated", "traced"]
Region = Literal["prompt", "generation"]

TOKEN_DOCUMENT_ELEM_ID = "flashtrace-token-document"

TOKEN_DOCUMENT_CSS = """
#flashtrace-token-document {
  min-height: 260px;
}
"""

TOKEN_DOCUMENT_JS = r"""
() => {
  if (window.flashtraceTokenDocument && window.flashtraceTokenDocument.ready) {
    return;
  }

  const state = {
    root: null,
    pendingStart: null,
    committed: null,
  };

  function documentRoot() {
    const host = document.querySelector("#flashtrace-token-document");
    if (!host) return null;
    return host.querySelector(".ft-token-document") || host;
  }

  function parseCommitted(root) {
    const start = root ? root.dataset.targetStart : "";
    const end = root ? root.dataset.targetEnd : "";
    if (start === undefined || end === undefined || start === "" || end === "") {
      return null;
    }
    return [Number(start), Number(end)];
  }

  function readModel(root) {
    const blob = root ? root.querySelector("#flashtrace-token-document-data") : null;
    if (!blob) return null;
    try {
      return JSON.parse(blob.textContent || "{}");
    } catch (_) {
      return null;
    }
  }

  function mark(root) {
    if (!root) return;
    const committed = parseCommitted(root);
    root.querySelectorAll(".ft-token").forEach((token) => {
      token.classList.remove("is-target", "is-pending");
      const genIndex = Number(token.dataset.genIndex);
      if (!Number.isFinite(genIndex)) return;
      if (committed && genIndex >= committed[0] && genIndex <= committed[1]) {
        token.classList.add("is-target");
      }
      if (state.pendingStart !== null && genIndex === state.pendingStart) {
        token.classList.add("is-pending");
      }
    });
  }

  function attach() {
    const root = documentRoot();
    if (!root || root === state.root) return;
    state.root = root;
    const committed = parseCommitted(root);
    state.committed = committed;
    state.pendingStart = null;
    mark(root);
  }

  function clickToken(event) {
    attach();
    const root = state.root;
    const target = event.target.closest(".ft-token[data-selectable='true']");
    if (!root || !target || !root.contains(target)) return;
    const genIndex = Number(target.dataset.genIndex);
    if (!Number.isFinite(genIndex)) return;
    if (state.pendingStart === null) {
      state.pendingStart = genIndex;
      mark(root);
      return;
    }
    const start = Math.min(state.pendingStart, genIndex);
    const end = Math.max(state.pendingStart, genIndex);
    root.dataset.targetStart = String(start);
    root.dataset.targetEnd = String(end);
    state.pendingStart = null;
    state.committed = [start, end];
    mark(root);
  }

  function clickTab(event) {
    attach();
    const root = state.root;
    const tab = event.target.closest(".ft-tab[data-view-index]");
    if (!root || !tab || !root.contains(tab)) return;
    const viewIndex = Number(tab.dataset.viewIndex);
    if (!Number.isInteger(viewIndex)) return;
    const tabs = Array.from(root.querySelectorAll(".ft-tab[data-view-index]"));
    const views = Array.from(root.querySelectorAll(".ft-view"));
    if (viewIndex < 0 || viewIndex >= views.length) return;
    tabs.forEach((item) => item.classList.toggle("is-active", item === tab));
    views.forEach((view, index) => view.classList.toggle("is-active", index === viewIndex));
  }

  function clickDocument(event) {
    clickTab(event);
    clickToken(event);
  }

  function readSelection() {
    attach();
    const root = state.root;
    if (!root) return "";
    const committed = parseCommitted(root);
    if (!committed) return "";
    return `${committed[0]}:${committed[1]}`;
  }

  document.addEventListener("click", clickDocument);
  const observer = new MutationObserver(() => attach());
  observer.observe(document.body, { childList: true, subtree: true });
  attach();

  window.flashtraceTokenDocument = {
    ready: true,
    readSelection,
    readModel: () => readModel(state.root),
  };
}
"""

TRACE_SELECTION_JS = r"""
(currentSpan, ...rest) => {
  const api = window.flashtraceTokenDocument;
  const selected = api && typeof api.readSelection === "function" ? api.readSelection() : "";
  return [selected || currentSpan || "", ...rest];
}
"""


def _span_to_list(span: tuple[int, int] | list[int] | None) -> list[int] | None:
    if span is None:
        return None
    if len(span) != 2:
        return None
    return [int(span[0]), int(span[1])]


def _is_target(gen_index: int | None, target_span: list[int] | None) -> bool:
    if gen_index is None or target_span is None:
        return False
    return int(target_span[0]) <= int(gen_index) <= int(target_span[1])


def _record_token(
    record: TokenRecord,
    *,
    document_index: int,
    region: Region,
    gen_index: int | None,
    selectable: bool,
    score: float | None = None,
) -> dict[str, Any]:
    return {
        "i": int(document_index),
        "text": record.token_text,
        "kind": record.kind,
        "region": region,
        "gen_index": gen_index,
        "selectable": bool(selectable),
        "score": None if score is None else float(score),
    }


def _text_token(
    *,
    text: str,
    document_index: int,
    region: Region,
    gen_index: int | None,
    score: float | None,
) -> dict[str, Any]:
    return {
        "i": int(document_index),
        "text": str(text),
        "kind": "content",
        "region": region,
        "gen_index": gen_index,
        "selectable": False,
        "score": None if score is None else float(score),
    }


def _default_target_span(generation_records: Sequence[TokenRecord]) -> list[int] | None:
    selectable = [
        int(record.token_index)
        for record in generation_records
        if record.kind == "content" and bool(record.selectable)
    ]
    if not selectable:
        return None
    return [min(selectable), max(selectable)]


def _build_prompt_generated_view(
    *,
    name: str,
    prompt_records: Sequence[TokenRecord],
    generation_records: Sequence[TokenRecord],
    interactive: bool,
    target_span: list[int] | None,
) -> dict[str, Any]:
    tokens: list[dict[str, Any]] = []
    for record in prompt_records:
        tokens.append(
            _record_token(
                record,
                document_index=len(tokens),
                region="prompt",
                gen_index=None,
                selectable=False,
            )
        )
    for record in generation_records:
        selectable = bool(record.selectable and record.kind == "content")
        tokens.append(
            _record_token(
                record,
                document_index=len(tokens),
                region="generation",
                gen_index=int(record.token_index),
                selectable=selectable,
            )
        )
    return {"name": name, "interactive": bool(interactive), "tokens": tokens}


def _build_trace_view(
    *,
    name: str,
    prompt_tokens: Sequence[str],
    generation_tokens: Sequence[str],
    scores: Sequence[float],
) -> dict[str, Any]:
    tokens: list[dict[str, Any]] = []
    for index, token in enumerate(prompt_tokens):
        score = float(scores[index]) if index < len(scores) else 0.0
        tokens.append(
            _text_token(
                text=token,
                document_index=len(tokens),
                region="prompt",
                gen_index=None,
                score=score,
            )
        )
    for gen_index, token in enumerate(generation_tokens):
        tokens.append(
            _text_token(
                text=token,
                document_index=len(tokens),
                region="generation",
                gen_index=gen_index,
                score=None,
            )
        )
    return {"name": name, "interactive": False, "tokens": tokens}


def build_document_views(
    *,
    phase: Phase,
    prompt_records: Sequence[TokenRecord] | None = None,
    generation_records: Sequence[TokenRecord] | None = None,
    result: Any | None = None,
    target_span: tuple[int, int] | list[int] | None = None,
    active_view: int = 0,
) -> dict[str, Any]:
    if phase == "traced":
        if result is None:
            raise ValueError("result is required for traced phase")
        model_target = _span_to_list(result.output_span)
        views = [
            _build_trace_view(
                name="Aggregate",
                prompt_tokens=result.prompt_tokens,
                generation_tokens=result.generation_tokens,
                scores=result.scores,
            )
        ]
        for hop_index, hop_scores in enumerate(result.per_hop_scores or [], start=1):
            views.append(
                _build_trace_view(
                    name=f"Hop {hop_index}",
                    prompt_tokens=result.prompt_tokens,
                    generation_tokens=result.generation_tokens,
                    scores=hop_scores,
                )
            )
        return {
            "phase": phase,
            "active_view": int(active_view),
            "views": views,
            "target_span": model_target,
        }

    prompt = list(prompt_records or [])
    generation = list(generation_records or [])
    model_target = _span_to_list(target_span)
    if phase == "generated" and model_target is None:
        model_target = _default_target_span(generation)
    interactive = phase == "generated" and model_target is not None
    return {
        "phase": phase,
        "active_view": int(active_view),
        "views": [
            _build_prompt_generated_view(
                name="Document",
                prompt_records=prompt,
                generation_records=generation,
                interactive=interactive,
                target_span=model_target,
            )
        ],
        "target_span": model_target,
    }


def _render_tabs(model: dict[str, Any]) -> str:
    views = list(model.get("views", []))
    if len(views) <= 1:
        return ""
    active = int(model.get("active_view", 0))
    buttons = []
    for index, view in enumerate(views):
        cls = "ft-tab is-active" if index == active else "ft-tab"
        buttons.append(
            f"<button type=\"button\" class=\"{cls}\" data-view-index=\"{index}\">"
            f"{escape(str(view.get('name', f'View {index + 1}')))}</button>"
        )
    return f"<div class=\"ft-tabs\">{''.join(buttons)}</div>"


def _render_token(token: dict[str, Any], target_span: list[int] | None, max_score: float) -> str:
    gen_index = token.get("gen_index")
    selectable = bool(token.get("selectable"))
    classes = ["ft-token", f"kind-{token.get('kind')}", f"region-{token.get('region')}"]
    if selectable:
        classes.append("is-selectable")
    if _is_target(gen_index, target_span):
        classes.append("is-target")
    score = token.get("score")
    style = ""
    if score is not None:
        style = f" style=\"background:{escape(_score_color(float(score), max_score))}\""
    attrs = [
        f"class=\"{' '.join(classes)}\"",
        f"data-token-index=\"{int(token.get('i', 0))}\"",
        f"data-region=\"{escape(str(token.get('region')))}\"",
        f"data-selectable=\"{'true' if selectable else 'false'}\"",
    ]
    if gen_index is not None:
        attrs.append(f"data-gen-index=\"{int(gen_index)}\"")
    if score is not None:
        attrs.append(f"data-score=\"{float(score):.8f}\"")
    return f"<span {' '.join(attrs)}{style}>{escape(str(token.get('text', '')))}</span>"


def _render_view(view: dict[str, Any], target_span: list[int] | None, active: bool) -> str:
    tokens = list(view.get("tokens", []))
    scores = [abs(float(token["score"])) for token in tokens if token.get("score") is not None]
    max_score = max(scores, default=0.0)
    rendered = "".join(_render_token(token, target_span, max_score) for token in tokens)
    cls = "ft-view is-active" if active else "ft-view"
    return f"<section class=\"{cls}\"><div class=\"ft-token-stream\">{rendered}</div></section>"


def render_document_html(render_model: dict[str, Any]) -> str:
    model_json = json.dumps(render_model, ensure_ascii=False).replace("</", "<\\/")
    target_span = render_model.get("target_span")
    target_start = "" if target_span is None else str(int(target_span[0]))
    target_end = "" if target_span is None else str(int(target_span[1]))
    views = list(render_model.get("views", []))
    active_view = int(render_model.get("active_view", 0))
    view_html = "".join(
        _render_view(view, target_span, index == active_view)
        for index, view in enumerate(views)
    )
    return f"""
<style class="ft-token-document__style">
  .ft-token-document {{ font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172026; }}
  .ft-token-toolbar {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 0 0 10px; }}
  .ft-phase {{ font-size: 12px; font-weight: 650; color: #5d6874; text-transform: uppercase; letter-spacing: 0; }}
  .ft-tabs {{ display: inline-flex; gap: 4px; border: 1px solid #d4dde6; border-radius: 6px; padding: 3px; background: #f7f9fb; }}
  .ft-tab {{ border: 0; background: transparent; color: #44515f; font: inherit; font-size: 13px; padding: 4px 8px; border-radius: 4px; cursor: pointer; }}
  .ft-tab.is-active {{ background: #ffffff; color: #111820; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.12); }}
  .ft-token-stream {{ min-height: 220px; border: 1px solid #d7e0e8; border-radius: 8px; padding: 14px; background: #fbfcfd; line-height: 2.05; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; font-size: 13px; overflow-wrap: anywhere; }}
  .ft-token {{ display: inline; margin: 0 1px; padding: 2px 3px; border-radius: 4px; white-space: pre-wrap; transition: background 120ms ease, outline-color 120ms ease; }}
  .ft-token.region-prompt {{ color: #172026; }}
  .ft-token.region-generation {{ color: #0f3f5f; }}
  .ft-token.kind-whitespace {{ color: #7c8794; }}
  .ft-token.kind-special, .ft-token.kind-template, .ft-token.kind-control {{ color: #8b5a14; background: #fff3d6; }}
  .ft-token.is-selectable {{ cursor: pointer; }}
  .ft-token.is-selectable:hover {{ outline: 1px solid #4d8fb8; background: #e9f4fb; }}
  .ft-token.is-target {{ background: #ffe6a3; box-shadow: inset 0 -2px 0 #e1a900; }}
  .ft-token.is-pending {{ outline: 2px solid #216b9a; background: #dff0fb; }}
  .ft-view {{ display: none; }}
  .ft-view.is-active {{ display: block; }}
</style>
<div class="ft-token-document" data-phase="{escape(str(render_model.get('phase', 'prompt')))}" data-target-start="{target_start}" data-target-end="{target_end}">
  <div class="ft-token-toolbar">
    <div class="ft-phase">{escape(str(render_model.get('phase', 'prompt')))}</div>
    {_render_tabs(render_model)}
  </div>
  {view_html}
  <script type="application/json" id="flashtrace-token-document-data">{model_json}</script>
</div>
""".strip()
