(() => {
  "use strict";

  const state = {
    phase: "prompt",
    dragAnchor: null,
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
      model: readText("model-input", "Qwen/Qwen3-4B-Thinking-2507"),
      prompt: $("prompt-input")?.value ?? "",
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

  function updatePhaseButtons() {
    // Generate and Trace are always available regardless of phase.
    if (els.generateButton) {
      els.generateButton.hidden = false;
      els.generateButton.disabled = false;
      els.generateButton.textContent = "Generate";
    }
    if (els.traceButton) {
      els.traceButton.hidden = false;
      els.traceButton.disabled = false;
      els.traceButton.textContent = "Trace";
    }
    if (els.saveButton) {
      els.saveButton.hidden = state.phase !== "traced";
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

  function activeViewIndex(root) {
    const activeView = root.querySelector(".ft-view.is-active");
    const index = Number(activeView?.dataset.viewIndex);
    return Number.isFinite(index) ? index : Number(state.renderModel?.active_view || 0);
  }

  function activeTargetSpan(root) {
    const model = state.renderModel || {};
    const view = (model.views || [])[activeViewIndex(root)] || {};
    // Interactive views (prompt/generated document, or the traced-phase
    // "Select target" tab) reflect the live, user-editable selection.
    if (view.interactive) {
      return parseSpan(state.targetSpan);
    }
    if (state.phase === "traced") {
      if (Array.isArray(view.target_span) && view.target_span.length === 2) {
        return view.target_span;
      }
      if (Array.isArray(model.target_span) && model.target_span.length === 2) {
        return model.target_span;
      }
      return null;
    }
    return parseSpan(state.targetSpan);
  }

  function isTarget(genIndex, span) {
    return span && genIndex >= span[0] && genIndex <= span[1];
  }

  function markSelection(root) {
    const span = activeTargetSpan(root);
    root.querySelectorAll(".ft-token").forEach((token) => {
      token.classList.remove("is-target");
      const genIndex = Number(token.dataset.genIndex);
      if (!Number.isFinite(genIndex)) return;
      if (state.phase === "traced" && !token.closest(".ft-view.is-active")) return;
      if (isTarget(genIndex, span)) {
        token.classList.add("is-target");
      }
    });
  }

  function buildLegend(view, phase) {
    if (phase === "prompt") return null;
    const legend = document.createElement("div");
    legend.className = "ft-legend";
    const addItem = (kind, label) => {
      const item = document.createElement("div");
      item.className = "ft-legend-item";
      const swatch = document.createElement("span");
      swatch.className = `ft-legend-swatch ft-legend-swatch-${kind}`;
      if (kind === "target") swatch.textContent = "Aa";
      const text = document.createElement("span");
      text.className = "ft-legend-label";
      text.textContent = label;
      item.append(swatch, text);
      legend.appendChild(item);
    };
    if (view.interactive) {
      addItem("target", "Target — drag to select");
    } else {
      addItem("input", "Input attribution");
      addItem("output", "Output attribution");
      addItem("target", "Target span");
    }
    return legend;
  }

  function makeToken(token) {
    const span = document.createElement("span");
    span.className = [
      "ft-token",
      `kind-${token.kind}`,
      `region-${token.region}`,
      token.selectable ? "is-selectable" : "",
      token.is_target ? "is-target" : "",
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
    markSelection(root);
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
    state.dragAnchor = null;
    updatePhaseButtons();

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
      const legend = buildLegend(view, state.phase);
      if (legend) stream.appendChild(legend);
      (view.tokens || []).forEach((token) => stream.appendChild(makeToken(token)));
      section.appendChild(stream);
      root.appendChild(section);
    });

    const tokenGenIndex = (node) => {
      const token = node && node.closest
        ? node.closest(".ft-token[data-selectable='true']")
        : null;
      if (!token || !root.contains(token)) return null;
      const genIndex = Number(token.dataset.genIndex);
      return Number.isFinite(genIndex) ? genIndex : null;
    };

    const applyDrag = (genIndex) => {
      if (state.dragAnchor === null || genIndex === null) return;
      const start = Math.min(state.dragAnchor, genIndex);
      const end = Math.max(state.dragAnchor, genIndex);
      state.targetSpan = `${start}:${end}`;
      markSelection(root);
    };

    root.addEventListener("mousedown", (event) => {
      const genIndex = tokenGenIndex(event.target);
      if (genIndex === null) return;
      // Suppress native text selection so dragging marks a target span.
      event.preventDefault();
      state.dragAnchor = genIndex;
      applyDrag(genIndex);
      const finishDrag = (upEvent) => {
        applyDrag(tokenGenIndex(upEvent.target));
        state.dragAnchor = null;
        if (state.targetSpan) {
          setStatus(`Selected target span ${state.targetSpan}.`);
        }
      };
      document.addEventListener("mouseup", finishDrag, { once: true });
    });

    root.addEventListener("mouseover", (event) => {
      if (state.dragAnchor === null) return;
      applyDrag(tokenGenIndex(event.target));
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
    try {
      const body = await postJSON("/api/generate", {
        ...basePayload(),
        max_new_tokens: readNumber("max-new-tokens-input", 8096),
      });
      state.generatedText = body.generated_text || "";
      state.targetSpan = body.target_span || "";
      state.reasoningSpan = body.reasoning_span || "";
      state.traceJson = null;
      if (els.downloadLink) els.downloadLink.hidden = true;
      renderDocument(body.render_model);
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
    els.galleryButton = $("gallery-button");
    els.saveButton = $("save-button");
    els.galleryDrawer = $("gallery-drawer");
    els.galleryList = $("gallery-list");
    els.gallerySaveForm = $("gallery-save-form");
    els.galleryTitleInput = $("gallery-title-input");

    els.generateButton?.addEventListener("click", generate);
    els.traceButton?.addEventListener("click", trace);
    els.galleryButton?.addEventListener("click", openGallery);
    els.saveButton?.addEventListener("click", openSaveForm);
    $("gallery-close")?.addEventListener("click", closeGallery);
    $("gallery-overlay")?.addEventListener("click", closeGallery);
    $("gallery-save-confirm")?.addEventListener("click", confirmSave);
    $("gallery-save-cancel")?.addEventListener("click", () => {
      if (els.gallerySaveForm) els.gallerySaveForm.hidden = true;
    });
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
    renderGalleryList,
    loadSample,
    getState: () => ({ ...state }),
    setState: (patch) => Object.assign(state, patch || {}),
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bind, { once: true });
  } else {
    bind();
  }
})();
