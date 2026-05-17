(() => {
  "use strict";

  const state = {
    phase: "prompt",
    pendingStart: null,
    targetSpan: "",
    generatedText: "",
    reasoningSpan: "",
    renderModel: null,
    traceJson: null,
    downloadUrl: null,
  };

  const els = {};

  function $(id) {
    return document.getElementById(id);
  }

  function readNumber(id, fallback) {
    const value = Number($(id)?.value);
    return Number.isFinite(value) ? value : fallback;
  }

  function readText(id, fallback = "") {
    return ($(id)?.value ?? fallback).trim();
  }

  function basePayload() {
    return {
      model: readText("model-input", "Qwen/Qwen3-0.6B"),
      prompt: $("prompt-input")?.value ?? "",
      chat_template: Boolean($("chat-template-checkbox")?.checked),
      device_map: readText("device-map-input", "auto"),
      dtype: readText("dtype-select", "auto"),
    };
  }

  function setStatus(message) {
    if (els.statusLine) {
      els.statusLine.textContent = message;
    }
  }

  function setBusy(button, busy, label) {
    if (!button) return;
    button.disabled = busy;
    if (label) {
      button.textContent = busy ? `${label}...` : label;
    }
  }

  function updatePhaseButtons(phase) {
    if (els.generateButton) {
      els.generateButton.hidden = phase === "generated";
      els.generateButton.disabled = false;
      els.generateButton.textContent = "Generate";
    }
    if (els.traceButton) {
      els.traceButton.hidden = phase !== "generated";
      els.traceButton.disabled = !state.targetSpan;
      els.traceButton.textContent = "Trace";
    }
  }

  async function postJSON(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.error || `Request failed with ${response.status}`);
    }
    return body;
  }

  function spanToText(span) {
    return Array.isArray(span) && span.length === 2 ? `${span[0]}:${span[1]}` : "";
  }

  function parseSpan(text) {
    const match = /^(\d+):(\d+)$/.exec(text || "");
    return match ? [Number(match[1]), Number(match[2])] : null;
  }

  function isTarget(genIndex) {
    const span = parseSpan(state.targetSpan);
    return span && genIndex >= span[0] && genIndex <= span[1];
  }

  function markSelection(root) {
    root.querySelectorAll(".ft-token").forEach((token) => {
      token.classList.remove("is-target", "is-pending");
      const genIndex = Number(token.dataset.genIndex);
      if (!Number.isFinite(genIndex)) return;
      if (isTarget(genIndex)) {
        token.classList.add("is-target");
      }
      if (state.pendingStart !== null && genIndex === state.pendingStart) {
        token.classList.add("is-pending");
      }
    });
  }

  function makeToken(token) {
    const span = document.createElement("span");
    span.className = [
      "ft-token",
      `kind-${token.kind}`,
      `region-${token.region}`,
      token.selectable ? "is-selectable" : "",
    ].filter(Boolean).join(" ");
    span.dataset.tokenIndex = String(token.i ?? 0);
    span.dataset.region = String(token.region);
    span.dataset.selectable = token.selectable ? "true" : "false";
    if (token.gen_index !== null && token.gen_index !== undefined) {
      span.dataset.genIndex = String(token.gen_index);
    }
    if (token.score !== null && token.score !== undefined) {
      span.dataset.score = Number(token.score).toFixed(8);
    }
    if (token.color) {
      span.style.background = token.color;
    }
    span.textContent = token.text ?? "";
    return span;
  }

  function activateView(root, viewIndex) {
    root.querySelectorAll(".ft-tab").forEach((tab) => {
      tab.classList.toggle("is-active", Number(tab.dataset.viewIndex) === viewIndex);
    });
    root.querySelectorAll(".ft-view").forEach((view) => {
      view.classList.toggle("is-active", Number(view.dataset.viewIndex) === viewIndex);
    });
  }

  function renderTabs(root, model) {
    if (!model.views || model.views.length <= 1) return;
    const tabs = document.createElement("div");
    tabs.className = "ft-tabs";
    model.views.forEach((view, index) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = index === Number(model.active_view || 0) ? "ft-tab is-active" : "ft-tab";
      button.dataset.viewIndex = String(index);
      button.textContent = view.name || `View ${index + 1}`;
      button.addEventListener("click", () => activateView(root, index));
      tabs.appendChild(button);
    });
    root.querySelector(".ft-token-toolbar").appendChild(tabs);
  }

  function renderDocument(model) {
    state.renderModel = model;
    state.phase = model.phase || "prompt";
    if (model.target_span) {
      state.targetSpan = spanToText(model.target_span);
    } else if (state.phase !== "traced") {
      state.targetSpan = "";
    }
    state.pendingStart = null;
    updatePhaseButtons(state.phase);

    const host = els.tokenDocument || $("token-document");
    if (!host) return;
    host.textContent = "";

    const root = document.createElement("div");
    root.className = "ft-token-document";
    root.dataset.phase = state.phase;

    const toolbar = document.createElement("div");
    toolbar.className = "ft-token-toolbar";
    const phase = document.createElement("div");
    phase.className = "ft-phase";
    phase.textContent = state.phase;
    toolbar.appendChild(phase);
    root.appendChild(toolbar);
    renderTabs(root, model);

    (model.views || []).forEach((view, index) => {
      const section = document.createElement("section");
      section.className = index === Number(model.active_view || 0) ? "ft-view is-active" : "ft-view";
      section.dataset.viewIndex = String(index);
      const stream = document.createElement("div");
      stream.className = "ft-token-stream";
      (view.tokens || []).forEach((token) => stream.appendChild(makeToken(token)));
      section.appendChild(stream);
      root.appendChild(section);
    });

    root.addEventListener("click", (event) => {
      const token = event.target.closest(".ft-token[data-selectable='true']");
      if (!token || !root.contains(token)) return;
      const genIndex = Number(token.dataset.genIndex);
      if (!Number.isFinite(genIndex)) return;
      if (state.pendingStart === null) {
        state.pendingStart = genIndex;
        markSelection(root);
        return;
      }
      const start = Math.min(state.pendingStart, genIndex);
      const end = Math.max(state.pendingStart, genIndex);
      state.targetSpan = `${start}:${end}`;
      state.pendingStart = null;
      markSelection(root);
      setStatus(`Selected target span ${state.targetSpan}.`);
      if (els.traceButton) els.traceButton.disabled = false;
    });

    host.appendChild(root);
    markSelection(root);
  }

  function updateDownload(traceJson) {
    if (!els.downloadLink) return;
    if (state.downloadUrl) {
      URL.revokeObjectURL(state.downloadUrl);
    }
    const blob = new Blob([JSON.stringify(traceJson, null, 2)], { type: "application/json" });
    state.downloadUrl = URL.createObjectURL(blob);
    els.downloadLink.href = state.downloadUrl;
    els.downloadLink.download = "flashtrace-trace.json";
    els.downloadLink.hidden = false;
  }

  async function tokenizePrompt() {
    setStatus("Tokenizing prompt...");
    try {
      const body = await postJSON("/api/tokenize", basePayload());
      state.traceJson = null;
      if (els.downloadLink) els.downloadLink.hidden = true;
      renderDocument(body.render_model);
      setStatus("Prompt tokenized.");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(els.tokenizeButton, false, "Tokenize");
    }
  }

  async function generate() {
    setBusy(els.generateButton, true, "Generate");
    setStatus("Generating...");
    if (els.traceButton) els.traceButton.disabled = true;
    try {
      const body = await postJSON("/api/generate", {
        ...basePayload(),
        max_new_tokens: readNumber("max-new-tokens-input", 128),
      });
      state.generatedText = body.generated_text || "";
      state.targetSpan = body.target_span || "";
      state.reasoningSpan = body.reasoning_span || "";
      state.traceJson = null;
      if (els.downloadLink) els.downloadLink.hidden = true;
      renderDocument(body.render_model);
      if (els.traceButton) els.traceButton.disabled = !state.targetSpan;
      setStatus(body.status || "Generated.");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(els.generateButton, false, "Generate");
    }
  }

  async function trace() {
    setBusy(els.traceButton, true, "Trace");
    setStatus("Tracing...");
    try {
      const body = await postJSON("/api/trace", {
        ...basePayload(),
        generated_text: state.generatedText,
        target_span: state.targetSpan,
        reasoning_span: state.reasoningSpan,
        method: readText("method-select", "flashtrace"),
        hops: readNumber("hops-input", 1),
        chunk_tokens: readNumber("chunk-tokens-input", 128),
        sink_chunk_tokens: readNumber("sink-chunk-tokens-input", 32),
      });
      state.traceJson = body.trace_json;
      renderDocument(body.render_model);
      updateDownload(body.trace_json);
      setStatus(body.status || "Trace complete.");
    } catch (error) {
      setStatus(error.message);
    } finally {
      setBusy(els.traceButton, false, "Trace");
    }
  }

  function bind() {
    els.tokenDocument = $("token-document");
    els.statusLine = $("status-line");
    els.generateButton = $("generate-button");
    els.traceButton = $("trace-button");
    els.downloadLink = $("download-link");

    els.generateButton?.addEventListener("click", generate);
    els.traceButton?.addEventListener("click", trace);
    $("chat-template-checkbox")?.addEventListener("change", tokenizePrompt);
    const promptInput = $("prompt-input");
    if (promptInput) {
      promptInput.addEventListener("input", tokenizePrompt);
      promptInput.addEventListener("change", tokenizePrompt);
    }

    if (promptInput) {
      tokenizePrompt();
    }
  }

  window.flashtraceTest = {
    renderDocument,
    getState: () => ({ ...state }),
    setState: (patch) => Object.assign(state, patch || {}),
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind, { once: true });
  } else {
    bind();
  }
})();
