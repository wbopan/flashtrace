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

  // ---- Attribution weight visualisation (tooltip + cut-off slider) ----
  // Cut-off is a normalised [lo, hi] window shared across trace views; tokens
  // are recoloured client-side so long-tail outliers (e.g. "?") can be clipped.
  const cutoff = { lo: 0, hi: 1 };
  const EMPTY_COLOR = "rgba(245,245,245,0.75)";

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  // Mirror flashtrace.viz._score_color (prompt, warm) and
  // token_document._generation_color (generation, blue).
  function warmColor(ratio) {
    const g = Math.round(246 - 105 * ratio);
    const b = Math.round(226 - 170 * ratio);
    return `rgba(255,${g},${b},${(0.22 + 0.58 * ratio).toFixed(3)})`;
  }

  function blueColor(ratio) {
    const r = Math.round(226 - 158 * ratio);
    const g = Math.round(240 - 86 * ratio);
    return `rgba(${r},${g},255,${(0.22 + 0.58 * ratio).toFixed(3)})`;
  }

  function sectionMaxes(section) {
    if (section.__maxes) return section.__maxes;
    let prompt = 0;
    let gen = 0;
    section.querySelectorAll(".ft-token[data-score]").forEach((tok) => {
      const score = Math.abs(Number(tok.dataset.score));
      if (!Number.isFinite(score)) return;
      if (tok.dataset.region === "generation") gen = Math.max(gen, score);
      else prompt = Math.max(prompt, score);
    });
    section.__maxes = { prompt, gen };
    return section.__maxes;
  }

  function tokenNorm(tok, maxes) {
    const score = Number(tok.dataset.score);
    if (!Number.isFinite(score)) return null;
    const max = tok.dataset.region === "generation" ? maxes.gen : maxes.prompt;
    if (!(max > 0)) return null;
    return Math.abs(score) / max;
  }

  // Snapshot the scored tokens of a section so the bloom animation can repaint
  // each frame without re-querying the DOM.
  function collectScored(section) {
    const maxes = sectionMaxes(section);
    const out = [];
    section.querySelectorAll(".ft-token[data-score]").forEach((el) => {
      out.push({ el, region: el.dataset.region, norm: tokenNorm(el, maxes) });
    });
    return out;
  }

  function paint(entries) {
    const span = Math.max(1e-9, cutoff.hi - cutoff.lo);
    entries.forEach((entry) => {
      if (entry.norm === null) {
        entry.el.style.background = EMPTY_COLOR;
        return;
      }
      const ratio = clamp((entry.norm - cutoff.lo) / span, 0, 1);
      entry.el.style.background =
        entry.region === "generation" ? blueColor(ratio) : warmColor(ratio);
    });
  }

  function recolorSection(section) {
    if (section) paint(collectScored(section));
  }

  // ---- Hover tooltip ----
  let tooltipEl = null;

  function ensureTooltip() {
    if (!tooltipEl) {
      tooltipEl = document.createElement("div");
      tooltipEl.className = "ft-tooltip";
      tooltipEl.hidden = true;
      document.body.appendChild(tooltipEl);
    }
    return tooltipEl;
  }

  function showTooltip(tok) {
    const tip = ensureTooltip();
    const score = Number(tok.dataset.score);
    const section = tok.closest(".ft-view");
    const maxes = section ? sectionMaxes(section) : { prompt: 0, gen: 0 };
    const norm = section ? tokenNorm(tok, maxes) : null;
    const pct = norm === null ? "—" : `${(norm * 100).toFixed(1)}%`;
    tip.innerHTML =
      `<span class="ft-tooltip-score">${score.toPrecision(4)}</span>` +
      `<span class="ft-tooltip-meta">weight ${pct} of max</span>`;
    tip.hidden = false;
  }

  function moveTooltip(event) {
    if (!tooltipEl || tooltipEl.hidden) return;
    const pad = 14;
    let x = event.clientX + pad;
    let y = event.clientY + pad;
    const rect = tooltipEl.getBoundingClientRect();
    if (x + rect.width > window.innerWidth) x = event.clientX - rect.width - pad;
    if (y + rect.height > window.innerHeight) y = event.clientY - rect.height - pad;
    tooltipEl.style.left = `${x}px`;
    tooltipEl.style.top = `${y}px`;
  }

  function hideTooltip() {
    if (tooltipEl) tooltipEl.hidden = true;
  }

  function bindTooltip(root) {
    root.addEventListener("mouseover", (event) => {
      const tok = event.target.closest?.(".ft-token[data-score]");
      if (tok) showTooltip(tok);
    });
    root.addEventListener("mousemove", moveTooltip);
    root.addEventListener("mouseout", (event) => {
      const to = event.relatedTarget;
      if (!to || !to.closest?.(".ft-token[data-score]")) hideTooltip();
    });
  }

  // ---- Cut-off slider with weight-distribution histogram ----
  const HIST_BINS = 44;

  function histogramData(section) {
    const maxes = sectionMaxes(section);
    const counts = new Array(HIST_BINS).fill(0);
    section.querySelectorAll(".ft-token[data-score]").forEach((tok) => {
      const norm = tokenNorm(tok, maxes);
      if (norm === null) return;
      counts[Math.min(HIST_BINS - 1, Math.floor(norm * HIST_BINS))] += 1;
    });
    let peak = 1;
    counts.forEach((c) => {
      peak = Math.max(peak, Math.sqrt(c));
    });
    return { counts, peak };
  }

  function renderHistogram(canvas, data) {
    const width = canvas.clientWidth || 220;
    const height = canvas.clientHeight || 38;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const { counts, peak } = data;
    const barW = width / HIST_BINS;
    for (let i = 0; i < HIST_BINS; i += 1) {
      if (!counts[i]) continue;
      const h = (Math.sqrt(counts[i]) / peak) * (height - 2);
      const center = (i + 0.5) / HIST_BINS;
      const inRange = center >= cutoff.lo && center <= cutoff.hi;
      ctx.fillStyle = inRange ? "rgba(14,148,164,0.55)" : "rgba(20,60,80,0.18)";
      ctx.fillRect(i * barW + 0.5, height - h, Math.max(1, barW - 1), h);
    }
  }

  function setCutoffUI(root) {
    const e = root.__cutoffEls;
    if (!e) return;
    e.fill.style.left = `${cutoff.lo * 100}%`;
    e.fill.style.width = `${(cutoff.hi - cutoff.lo) * 100}%`;
    e.lo.value = String(cutoff.lo);
    e.hi.value = String(cutoff.hi);
    e.readout.textContent = `${Math.round(cutoff.lo * 100)}–${Math.round(cutoff.hi * 100)}%`;
  }

  function cancelCutoffAnim(root) {
    if (root.__cutoffAnim) {
      cancelAnimationFrame(root.__cutoffAnim);
      root.__cutoffAnim = null;
    }
  }

  // Bloom: open the cut-off window (hi: 0 -> 1, lo pinned at 0) over 10s. The
  // sweep advances by token *rank*, not by value or mass — at progress p the
  // window covers the first p of tokens ordered by ascending weight (i.e. hi
  // follows the empirical quantile of normalised weights). With a long tail the
  // many small-weight tokens are dense at the low end (slow start) and the few
  // large ones are sparse near 1 (fast finish), instead of stalling on the one
  // token that hoards most of the mass.
  function animateCutoff(root) {
    cancelCutoffAnim(root);
    const section = root.querySelector(".ft-view.is-active");
    if (!section || !section.querySelector(".ft-token[data-score]")) return;
    const entries = collectScored(section);
    const histData = histogramData(section);
    const els = root.__cutoffEls;
    cutoff.lo = 0;
    const frame = () => {
      setCutoffUI(root);
      paint(entries);
      if (els) renderHistogram(els.hist, histData);
    };

    // Normalised weights sorted ascending; hi is their empirical quantile.
    const norms = entries
      .filter((e) => e.norm !== null)
      .map((e) => e.norm)
      .sort((a, b) => a - b);
    const n = norms.length;
    const hiForProgress = (t) => {
      if (!n) return t;
      const x = t * n; // expected number of tokens covered
      if (x <= 0) return 0;
      if (x >= n) return norms[n - 1];
      const i = Math.floor(x);
      const lower = i > 0 ? norms[i - 1] : 0;
      return lower + (x - i) * (norms[i] - lower);
    };

    const reduce =
      window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      cutoff.hi = 1;
      frame();
      return;
    }
    const duration = 10000;
    const start = performance.now();
    const step = (now) => {
      const t = Math.min(1, (now - start) / duration);
      cutoff.hi = Math.max(0.001, Math.min(1, hiForProgress(t)));
      frame();
      if (t < 1) {
        root.__cutoffAnim = requestAnimationFrame(step);
      } else {
        cutoff.hi = 1;
        frame();
        root.__cutoffAnim = null;
      }
    };
    cutoff.hi = 0.001;
    frame();
    root.__cutoffAnim = requestAnimationFrame(step);
  }

  function buildCutoff(root) {
    const host = root.__cutoffHost || root.querySelector(".ft-token-toolbar");
    if (!host) return;
    const wrap = document.createElement("div");
    wrap.className = "ft-cutoff";
    wrap.innerHTML =
      '<span class="ft-cutoff-label">Cut-off</span>' +
      '<div class="ft-cutoff-track">' +
      '<canvas class="ft-cutoff-hist"></canvas>' +
      '<div class="ft-cutoff-fill"></div>' +
      '<input type="range" class="ft-cutoff-lo" min="0" max="1" step="0.005" value="0">' +
      '<input type="range" class="ft-cutoff-hi" min="0" max="1" step="0.005" value="1">' +
      "</div>" +
      '<span class="ft-cutoff-readout">0–100%</span>';
    host.appendChild(wrap);

    const lo = wrap.querySelector(".ft-cutoff-lo");
    const hi = wrap.querySelector(".ft-cutoff-hi");
    const fill = wrap.querySelector(".ft-cutoff-fill");
    const hist = wrap.querySelector(".ft-cutoff-hist");
    const readout = wrap.querySelector(".ft-cutoff-readout");
    root.__cutoffEls = { lo, hi, fill, hist, readout };
    root.__cutoffWrap = wrap;

    const refresh = () => {
      setCutoffUI(root);
      const section = root.querySelector(".ft-view.is-active");
      if (section) {
        renderHistogram(hist, histogramData(section));
        recolorSection(section);
      }
    };

    lo.addEventListener("input", () => {
      cancelCutoffAnim(root);
      cutoff.lo = Math.min(Number(lo.value), cutoff.hi - 0.01);
      refresh();
    });
    hi.addEventListener("input", () => {
      cancelCutoffAnim(root);
      cutoff.hi = Math.max(Number(hi.value), cutoff.lo + 0.01);
      refresh();
    });
  }

  function syncCutoffVisibility(root) {
    const wrap = root.__cutoffWrap;
    if (!wrap) return;
    const section = root.querySelector(".ft-view.is-active");
    const hasScores = !!section && !!section.querySelector(".ft-token[data-score]");
    wrap.hidden = !hasScores;
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
    syncCutoffVisibility(root);
    animateCutoff(root);
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

    bindTooltip(root);
    hideTooltip();

    host.appendChild(root);
    markSelection(root);

    if (state.phase === "traced") {
      buildCutoff(root);
      syncCutoffVisibility(root);
      animateCutoff(root);
    }
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

  function renderGalleryOptions(samples) {
    const select = els.gallerySelect;
    if (!select) return;
    const current = select.value;
    select.textContent = "";
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = samples.length ? "Load a saved sample…" : "No saved samples yet";
    select.appendChild(placeholder);
    samples.forEach((sample) => {
      const option = document.createElement("option");
      option.value = sample.id;
      const title = sample.title || "(untitled)";
      option.textContent = `${title} · ${sample.method} · ${sample.hops} hop`;
      select.appendChild(option);
    });
    select.disabled = !samples.length;
    if (current && Array.from(select.options).some((o) => o.value === current)) {
      select.value = current;
    }
  }

  async function loadGalleryList() {
    try {
      const response = await fetch("/api/gallery");
      const body = await response.json();
      renderGalleryOptions(body.samples || []);
    } catch (error) {
      setStatus(error.message);
    }
  }

  function openSaveForm() {
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
      setStatus("Saved to samples.");
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
      setStatus(`Loaded "${sample.title}".`);
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
    els.gallerySelect = $("gallery-select");
    els.saveButton = $("save-button");
    els.gallerySaveForm = $("gallery-save-form");
    els.galleryTitleInput = $("gallery-title-input");

    els.generateButton?.addEventListener("click", generate);
    els.traceButton?.addEventListener("click", trace);
    els.saveButton?.addEventListener("click", openSaveForm);
    els.gallerySelect?.addEventListener("change", (event) => {
      const id = event.target.value;
      if (id) loadSample(id);
    });
    $("gallery-save-confirm")?.addEventListener("click", confirmSave);
    $("gallery-save-cancel")?.addEventListener("click", () => {
      if (els.gallerySaveForm) els.gallerySaveForm.hidden = true;
    });
    const promptInput = $("prompt-input");
    if (promptInput) {
      promptInput.addEventListener("input", tokenizePrompt);
      promptInput.addEventListener("change", tokenizePrompt);
    }

    loadGalleryList();
    if (promptInput) {
      tokenizePrompt();
    }
  }

  window.flashtraceTest = {
    renderDocument,
    renderGalleryOptions,
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
