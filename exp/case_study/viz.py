"""HTML helpers for visualizing hop-wise IFR attributions."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

from html import escape


TOKEN_SCALE_QUANTILE = 0.995


def _robust_abs_max(scores: Sequence[float], *, quantile: float = TOKEN_SCALE_QUANTILE) -> float:
    """Return a robust abs max to avoid a single outlier washing out the colormap.

    Uses a high quantile (default: p99.5) over |scores|. Top outliers saturate.
    """

    abs_vals: List[float] = []
    for x in scores:
        try:
            v = float(x)
        except Exception:
            continue
        if math.isnan(v):
            continue
        abs_vals.append(abs(v))

    if not abs_vals:
        return 0.0

    abs_vals.sort()
    q = float(quantile)
    if q < 0.0:
        q = 0.0
    if q > 1.0:
        q = 1.0
    idx = int(q * (len(abs_vals) - 1))
    return float(abs_vals[idx])


def _color_for_score(score: float, max_score: float) -> str:
    if max_score <= 0:
        return "background-color: rgba(245,245,245,0.7);"
    ratio = min(1.0, score / (max_score + 1e-12))
    r = 255
    g = int(235 - 90 * ratio)
    b = int(220 - 160 * ratio)
    alpha = 0.25 + 0.55 * ratio
    return f"background-color: rgba({r}, {g}, {b}, {alpha});"


def _render_sentence_list(title: str, sentences: Sequence[str], scores: Sequence[float], max_score: float) -> str:
    rows: List[str] = []
    for sent, sc in zip(sentences, scores):
        style = _color_for_score(float(sc), max_score)
        rows.append(
            f'<div class="sent-row" style="{style}"><span class="score">{sc:.4f}</span>'
            f'<span class="text">{escape(sent)}</span></div>'
        )
    return f"""
    <div class="sent-block">
      <div class="sent-title">{escape(title)}</div>
      {''.join(rows)}
    </div>
    """


def _render_tokens(
    tokens: Sequence[str],
    scores: Sequence[float],
    max_score: float,
    roles: Sequence[str],
    *,
    score_transform: str = "positive",
) -> str:
    spans: List[str] = []
    if max_score <= 0:
        max_score = 1e-8
    for idx, tok in enumerate(tokens):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        if score_transform == "signed":
            style = _color_for_signed_score(score, max_score)
        else:
            style = _color_for_score(score, max_score)
        role = roles[idx] if idx < len(roles) else "gen"
        safe_tok = escape(tok)
        spans.append(
            f'<span class="tok {role}" title="idx={idx}, score={score:.6f}" style="{style}">{safe_tok}</span>'
        )
    return "".join(spans)


def _render_top_table(top_items: List[Dict[str, Any]]) -> str:
    if not top_items:
        return "<div class='top-table'><em>No attribution mass.</em></div>"

    header = "<div class='top-row top-header'><span>Rank</span><span>Idx</span><span>Score</span><span>Sentence</span></div>"
    body_rows = []
    for rank, item in enumerate(top_items, start=1):
        body_rows.append(
            f"<div class='top-row'><span>{rank}</span><span>{item['idx']}</span>"
            f"<span>{item['score']:.4f}</span><span>{escape(item['sentence'])}</span></div>"
        )
    return f"<div class='top-table'>{header}{''.join(body_rows)}</div>"


def render_case_html(
    case_meta: Dict[str, Any],
    token_view_trimmed: Dict[str, Any],
    *,
    context: Optional[Dict[str, Any]] = None,
    hops_sent: Optional[Sequence[Dict[str, Any]]] = None,
    token_view_raw: Optional[Dict[str, Any]] = None,
) -> str:
    has_sentence_view = bool(context) and bool(hops_sent)
    prompt_len = len((context or {}).get("prompt_sentences") or []) if has_sentence_view else 0
    gen_len = len((context or {}).get("generation_sentences") or []) if has_sentence_view else 0

    prompt_max = 0.0
    gen_max = 0.0
    if has_sentence_view:
        prompt_max = max(
            (
                max(h["sentence_scores_raw"][:prompt_len])
                for h in (hops_sent or [])
                if h.get("sentence_scores_raw") and h["sentence_scores_raw"][:prompt_len]
            ),
            default=0.0,
        )
        gen_max = max(
            (
                max(h["sentence_scores_raw"][prompt_len:])
                for h in (hops_sent or [])
                if h.get("sentence_scores_raw") and h["sentence_scores_raw"][prompt_len:]
            ),
            default=0.0,
        )

    hop_sections: List[str] = []
    trim_hops = token_view_trimmed.get("hops", [])
    hop_count = len(trim_hops)
    mode = case_meta.get("mode", "ft")
    ifr_view = case_meta.get("ifr_view", "aggregate")
    sink_span = case_meta.get("sink_span")
    score_transform = str(case_meta.get("score_transform") or "positive")

    def _panel_title(panel_idx: int) -> str:
        if mode in ("ft", "ft_attnlrp"):
            return f"Hop {panel_idx}"
        if mode == "attnlrp":
            return "AttnLRP (sink-span aggregate)"
        if ifr_view == "per_token" and isinstance(sink_span, (list, tuple)) and len(sink_span) == 2:
            try:
                base = int(sink_span[0])
            except Exception:
                base = 0
            return f"Sink token {base + panel_idx} (gen idx)"
        return "IFR (sink-span aggregate)"

    trim_roles = token_view_trimmed.get("roles", [])
    for hop_idx in range(hop_count):
        tok_entry_trim = trim_hops[hop_idx] if hop_idx < len(trim_hops) else {}
        tok_scores_trim = tok_entry_trim.get("token_scores") or []
        hop_total_mass = float(tok_entry_trim.get("total_mass", 0.0))
        tok_scale_trim = _robust_abs_max(tok_scores_trim)
        if tok_scale_trim <= 0:
            tok_scale_trim = float(tok_entry_trim.get("token_score_max") or 0.0)
        if tok_scale_trim <= 0:
            tok_scale_trim = 1e-8

        tok_raw_html = ""
        if token_view_raw is not None and hop_idx < len(token_view_raw.get("hops", [])):
            raw_entry = token_view_raw["hops"][hop_idx]
            tok_scores_raw = raw_entry.get("token_scores") or []
            tok_scale_raw = _robust_abs_max(tok_scores_raw)
            if tok_scale_raw <= 0:
                tok_scale_raw = float(raw_entry.get("token_score_max") or 0.0)
            if tok_scale_raw <= 0:
                tok_scale_raw = 1e-8
            tok_raw_html = f"""
                <div class="tokens-block">
                  <div class="tokens-title">{escape(token_view_raw.get("label", "Pre-trim token-level heatmap"))}</div>
                  <div class="tokens-row">
                  {_render_tokens(token_view_raw.get("tokens", []), tok_scores_raw, tok_scale_raw, token_view_raw.get("roles", []), score_transform=score_transform)}
                  </div>
                </div>
            """

        sentence_html = ""
        top_html = ""
        if has_sentence_view and hop_idx < len(hops_sent or []):
            hop = (hops_sent or [])[hop_idx]
            raw_scores = hop.get("sentence_scores_raw") or []
            prompt_scores = raw_scores[:prompt_len]
            gen_scores = raw_scores[prompt_len:]
            hop_total_mass = float(hop.get("total_mass", hop_total_mass))
            sentence_html = f"""
              <div class="columns">
                {_render_sentence_list('Prompt sentences', (context or {}).get('prompt_sentences') or [], prompt_scores, prompt_max)}
                {_render_sentence_list('Generation sentences', (context or {}).get('generation_sentences') or [], gen_scores, gen_max)}
              </div>
            """
            top_html = f"""
              <div class="top-wrap">
                <div class="section-label">Top sentences (all)</div>
                {_render_top_table(hop.get('top_sentences') or [])}
              </div>
            """

        hop_sections.append(
            f"""
            <div class="hop">
              <div class="hop-header">
                <div class="hop-title">{escape(_panel_title(hop_idx))}</div>
                <div class="hop-meta">total mass: {hop_total_mass:.6f} | scale(p{int(TOKEN_SCALE_QUANTILE*1000)/10:.1f} abs): {tok_scale_trim:.6g}</div>
              </div>
              {tok_raw_html}
              <div class="tokens-block">
                <div class="tokens-title">{escape(token_view_trimmed.get("label", "Post-trim token-level heatmap"))}</div>
                <div class="tokens-row">
                  {_render_tokens(token_view_trimmed.get("tokens", []), tok_scores_trim, tok_scale_trim, trim_roles, score_transform=score_transform)}
                </div>
              </div>
              {sentence_html}
              {top_html}
            </div>
            """
        )

    thinking_ratios = case_meta.get("thinking_ratios") or []
    ratios_str = ", ".join(f"{r:.4f}" for r in thinking_ratios) if thinking_ratios else "N/A"

    if mode == "ft":
        mode_label = "FT Multi-hop (IFR)"
    elif mode == "ifr":
        mode_label = "IFR Standard"
    elif mode == "attnlrp":
        mode_label = "AttnLRP"
    elif mode == "ft_attnlrp":
        mode_label = "FT Multi-hop (AttnLRP)"
    else:
        mode_label = str(mode)

    if mode in ("ft", "ft_attnlrp"):
        view_key = "Recursive hops"
        view_val = case_meta.get("n_hops")
    elif mode == "ifr":
        view_key = "IFR view"
        view_val = ifr_view
    elif mode == "attnlrp":
        view_key = "AttnLRP view"
        view_val = "ft_hop0_span_aggregate"
    else:
        view_key = "View"
        view_val = "N/A"

    transform_row = f"<div>Score transform: {escape(str(score_transform))}</div>" if score_transform else ""
    scale_row = f"<div>Token scale: per-panel p{int(TOKEN_SCALE_QUANTILE*1000)/10:.1f}(|score|)</div>"
    legend_row = "<div>Colors: red = +, blue = −</div>" if score_transform == "signed" else ""

    header = f"""
    <div class="header">
      <div>
        <div class="title">{escape(mode_label)} Case Study</div>
        <div class="subtitle">Dataset: {escape(str(case_meta.get('dataset')))} | index: {case_meta.get('index')}</div>
      </div>
      <div class="meta">
        <div>Sink span (gen idx): {escape(str(case_meta.get('sink_span')))}</div>
        <div>Thinking span (gen idx): {escape(str(case_meta.get('thinking_span')))}</div>
        <div>Panels: {hop_count}</div>
        <div>{escape(str(view_key))}: {escape(str(view_val))}</div>
        {transform_row}
        {scale_row}
        {legend_row}
        <div>Thinking ratios: {ratios_str}</div>
      </div>
    </div>
    """

    style = """
    <style>
      body { font-family: "Inter", "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 24px; background: #fcfcff; color: #1f2933; }
      .title { font-size: 24px; font-weight: 700; }
      .subtitle { font-size: 14px; color: #566; margin-top: 4px; }
      .header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; padding-bottom: 16px; border-bottom: 1px solid #e5e8ee; }
      .meta { font-size: 13px; color: #334; line-height: 1.6; }
      .hop { margin-top: 20px; padding: 16px; border: 1px solid #e5e8ee; border-radius: 10px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
      .hop-header { display: flex; justify-content: space-between; align-items: center; }
      .hop-title { font-weight: 600; font-size: 16px; }
      .hop-meta { font-size: 12px; color: #556; }
      .tokens-block { margin-top: 12px; border: 1px solid #eef1f6; border-radius: 8px; padding: 10px; background: #f9fbff; }
      .tokens-title { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #263; }
      .tokens-row { font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; line-height: 1.8; word-break: break-word; }
      .tok { display: inline; padding: 2px 1px; margin: 0 0px; border-radius: 3px; }
      .tok.prompt { border-bottom: 1px dashed #6b8fb8; }
      .tok.user { border-bottom: 1px dashed #4f72c7; }
      .tok.template { border-bottom: 1px dashed #9aa9c0; }
      .tok.think { border-bottom: 1px dashed #8ba86b; }
      .tok.output { border-bottom: 1px dashed #c78a6e; }
      .tok.gen { border-bottom: 1px dashed #999; }
      .tok:hover { outline: 1px solid #8899aa; }
      .columns { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 12px; }
      .sent-block { padding: 8px; border: 1px solid #eef1f6; border-radius: 8px; background: #f9fbff; }
      .sent-title { font-weight: 600; font-size: 13px; margin-bottom: 6px; color: #263; }
      .sent-row { padding: 6px 8px; border-radius: 6px; margin-bottom: 6px; display: flex; gap: 8px; align-items: flex-start; }
      .sent-row:last-child { margin-bottom: 0; }
      .sent-row .score { font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; color: #233; min-width: 60px; }
      .sent-row .text { flex: 1; font-size: 13px; }
      .top-wrap { margin-top: 10px; }
      .section-label { font-size: 13px; font-weight: 600; margin-bottom: 6px; color: #263; }
      .top-table { border: 1px solid #eef1f6; border-radius: 8px; background: #fff; }
      .top-row { display: grid; grid-template-columns: 50px 50px 80px 1fr; padding: 6px 8px; gap: 8px; font-size: 12px; }
      .top-header { background: #f3f6fb; font-weight: 700; color: #223; }
      .top-row:nth-child(odd):not(.top-header) { background: #fbfdff; }
    </style>
    """

    title = f"{mode_label} Case Study"
    html = f"""<!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{escape(title)}</title>
        {style}
      </head>
      <body>
        {header}
        {''.join(hop_sections)}
      </body>
    </html>"""
    return html


def _color_for_signed_score(score: float, max_abs: float) -> str:
    if max_abs <= 0:
        return "background-color: rgba(245,245,245,0.7);"
    ratio = min(1.0, abs(score) / (max_abs + 1e-12))
    alpha = 0.10 + 0.70 * ratio

    # Diverging palette: red for positive, blue for negative.
    if score < 0:
        r, g, b = 120, 170, 255
    else:
        r, g, b = 255, 120, 120
    return f"background-color: rgba({r}, {g}, {b}, {alpha});"


def _render_sentence_spans(title: str, sentences: Sequence[str], scores: Sequence[float]) -> str:
    max_abs = max((abs(float(x)) for x in scores), default=0.0)
    spans: List[str] = []
    for idx, sentence in enumerate(sentences):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        style = _color_for_signed_score(score, max_abs)
        spans.append(
            f'<span class="sent-span" title="idx={idx}, score={score:.6f}" style="{style}">{escape(sentence)}</span>'
        )
    return f"""
    <div class="sentmap">
      <div class="sentmap-title">{escape(title)}</div>
      <div class="sentmap-text">{''.join(spans)}</div>
    </div>
    """


def _render_token_spans(title: str, tokens: Sequence[str], scores: Sequence[float]) -> str:
    max_abs = max((abs(float(x)) for x in scores), default=0.0)
    spans: List[str] = []
    for idx, tok in enumerate(tokens):
        score = float(scores[idx]) if idx < len(scores) else 0.0
        style = _color_for_signed_score(score, max_abs)
        spans.append(
            f'<span class="tok-span" title="idx={idx}, score={score:.6f}" style="{style}">{escape(tok)}</span>'
        )
    return f"""
    <div class="tokmap">
      <div class="tokmap-title">{escape(title)}</div>
      <div class="tokmap-text">{''.join(spans)}</div>
    </div>
    """


def render_mas_sentence_html(
    case_meta: Dict[str, Any],
    *,
    prompt_sentences: Sequence[str],
    panels: Sequence[Dict[str, Any]],
    generation: Optional[str] = None,
) -> str:
    """Render MAS sentence-level diagnostics (attribution / pure ablation / guided marginal)."""

    method_label = case_meta.get("attr_method_label") or case_meta.get("attr_method") or "Unknown method"
    title = f"MAS Sentence Study ({method_label})"

    score_transform = case_meta.get("score_transform")
    legend_row = "<div>Colors: red = +, blue = −</div>" if score_transform == "signed" else ""

    base_score = case_meta.get("base_score")
    base_score_row = f"<div>Base score: {float(base_score):.6f}</div>" if isinstance(base_score, (int, float)) else ""

    gen_block = ""
    if isinstance(generation, str) and generation:
        gen_block = f"""
        <div class="text-block">
          <div class="text-title">Generation (scored)</div>
          <div class="text-body">{escape(generation)}</div>
        </div>
        """

    header = f"""
    <div class="header">
      <div>
        <div class="title">{escape(title)}</div>
        <div class="subtitle">Dataset: {escape(str(case_meta.get('dataset')))} | index: {case_meta.get('index')}</div>
      </div>
      <div class="meta">
        <div>Attribution method: {escape(str(case_meta.get('attr_method')))}</div>
        <div>Sink span (gen idx): {escape(str(case_meta.get('sink_span')))}</div>
        <div>Thinking span (gen idx): {escape(str(case_meta.get('thinking_span')))}</div>
        <div>Panels: {len(panels)}</div>
        <div>Score transform: {escape(str(score_transform))}</div>
        {legend_row}
        {base_score_row}
      </div>
    </div>
    """

    panel_sections: List[str] = []
    for panel in panels:
        label = panel.get("variant_label") or panel.get("panel_label") or panel.get("variant") or "Panel"
        metrics = panel.get("metrics") or {}
        metrics_str = " | ".join(
            f"{k}: {float(metrics[k]):.4f}" if isinstance(metrics.get(k), (int, float)) else f"{k}: {metrics.get(k)}"
            for k in ("RISE", "MAS", "RISE+AP")
            if k in metrics
        )

        attr_weights = panel.get("attr_weights") or []
        pure_deltas = panel.get("pure_sentence_deltas_raw") or []
        guided_deltas = panel.get("guided_sentence_deltas_raw") or panel.get("sentence_deltas_raw") or []
        rank_order = panel.get("sorted_attr_indices") or []
        rank_str = ", ".join(str(int(x)) for x in rank_order) if rank_order else "N/A"

        panel_sections.append(
            f"""
            <div class="panel">
              <div class="panel-header">
                <div class="panel-title">{escape(str(label))}</div>
                <div class="panel-meta">{escape(metrics_str)}</div>
              </div>

              {_render_sentence_spans("Method attribution (sentence weights)", prompt_sentences, attr_weights)}
              {_render_sentence_spans("Pure sentence ablation (base − score)", prompt_sentences, pure_deltas)}
              {_render_sentence_spans("Attribution-guided MAS marginal (path deltas)", prompt_sentences, guided_deltas)}

              <div class="panel-foot">Rank order: {escape(rank_str)}</div>
            </div>
            """
        )

    style = """
    <style>
      body { font-family: "Inter", "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 24px; background: #fcfcff; color: #1f2933; }
      .title { font-size: 24px; font-weight: 700; }
      .subtitle { font-size: 14px; color: #566; margin-top: 4px; }
      .header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; padding-bottom: 16px; border-bottom: 1px solid #e5e8ee; }
      .meta { font-size: 13px; color: #334; line-height: 1.6; }

      .text-block { margin-top: 16px; border: 1px solid #eef1f6; border-radius: 10px; padding: 12px; background: #fff; }
      .text-title { font-size: 13px; font-weight: 700; color: #263; margin-bottom: 8px; }
      .text-body { font-size: 13px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }

      .panel { margin-top: 18px; padding: 16px; border: 1px solid #e5e8ee; border-radius: 10px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
      .panel-header { display: flex; justify-content: space-between; align-items: center; }
      .panel-title { font-weight: 600; font-size: 16px; }
      .panel-meta { font-size: 12px; color: #556; }
      .panel-foot { margin-top: 8px; font-size: 12px; color: #556; }

      .sentmap { margin-top: 12px; border: 1px solid #eef1f6; border-radius: 8px; padding: 10px; background: #f9fbff; }
      .sentmap-title { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #263; }
      .sentmap-text { font-size: 13px; line-height: 1.8; white-space: pre-wrap; word-break: break-word; }
      .sent-span { display: inline; padding: 2px 2px; margin: 0 0px; border-radius: 4px; }
      .sent-span:hover { outline: 1px solid #8899aa; }
    </style>
    """

    html = f"""<!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{escape(title)}</title>
        {style}
      </head>
      <body>
        {header}
        {gen_block}
        {''.join(panel_sections)}
      </body>
    </html>"""
    return html


def render_mas_token_html(
    case_meta: Dict[str, Any],
    *,
    prompt_tokens: Sequence[str],
    panels: Sequence[Dict[str, Any]],
    generation: Optional[str] = None,
) -> str:
    """Render MAS token-level diagnostics (attribution weights + guided marginal deltas)."""

    method_label = case_meta.get("attr_method_label") or case_meta.get("attr_method") or "Unknown method"
    title = f"MAS Token Study ({method_label})"

    score_transform = case_meta.get("score_transform")
    legend_row = "<div>Colors: red = +, blue = −</div>" if score_transform == "signed" else ""

    base_score = case_meta.get("base_score")
    base_score_row = f"<div>Base score: {float(base_score):.6f}</div>" if isinstance(base_score, (int, float)) else ""

    gen_block = ""
    if isinstance(generation, str) and generation:
        gen_block = f"""
        <div class="text-block">
          <div class="text-title">Generation (scored)</div>
          <div class="text-body">{escape(generation)}</div>
        </div>
        """

    header = f"""
    <div class="header">
      <div>
        <div class="title">{escape(title)}</div>
        <div class="subtitle">Dataset: {escape(str(case_meta.get('dataset')))} | index: {case_meta.get('index')}</div>
      </div>
      <div class="meta">
        <div>Attribution method: {escape(str(case_meta.get('attr_method')))}</div>
        <div>Sink span (gen idx): {escape(str(case_meta.get('sink_span')))}</div>
        <div>Thinking span (gen idx): {escape(str(case_meta.get('thinking_span')))}</div>
        <div>Prompt tokens: {len(prompt_tokens)}</div>
        <div>Panels: {len(panels)}</div>
        <div>Score transform: {escape(str(score_transform))}</div>
        {legend_row}
        {base_score_row}
      </div>
    </div>
    """

    panel_sections: List[str] = []
    for panel in panels:
        label = panel.get("variant_label") or panel.get("panel_label") or panel.get("variant") or "Panel"
        metrics = panel.get("metrics") or {}
        metrics_str = " | ".join(
            f"{k}: {float(metrics[k]):.4f}" if isinstance(metrics.get(k), (int, float)) else f"{k}: {metrics.get(k)}"
            for k in ("RISE", "MAS", "RISE+AP")
            if k in metrics
        )

        attr_weights = panel.get("attr_weights") or []
        guided_deltas = panel.get("token_deltas_raw") or []
        rank_order = panel.get("sorted_attr_indices") or []
        rank_str = ", ".join(str(int(x)) for x in rank_order) if rank_order else "N/A"

        panel_sections.append(
            f"""
            <div class="panel">
              <div class="panel-header">
                <div class="panel-title">{escape(str(label))}</div>
                <div class="panel-meta">{escape(metrics_str)}</div>
              </div>

              {_render_token_spans("Method attribution (token weights)", prompt_tokens, attr_weights)}
              {_render_token_spans("Attribution-guided MAS marginal (path deltas)", prompt_tokens, guided_deltas)}

              <div class="panel-foot">Rank order: {escape(rank_str)}</div>
            </div>
            """
        )

    style = """
    <style>
      body { font-family: "Inter", "Helvetica Neue", Arial, sans-serif; margin: 0; padding: 24px; background: #fcfcff; color: #1f2933; }
      .title { font-size: 24px; font-weight: 700; }
      .subtitle { font-size: 14px; color: #566; margin-top: 4px; }
      .header { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; padding-bottom: 16px; border-bottom: 1px solid #e5e8ee; }
      .meta { font-size: 13px; color: #334; line-height: 1.6; }

      .text-block { margin-top: 16px; border: 1px solid #eef1f6; border-radius: 10px; padding: 12px; background: #fff; }
      .text-title { font-size: 13px; font-weight: 700; color: #263; margin-bottom: 8px; }
      .text-body { font-size: 13px; line-height: 1.7; white-space: pre-wrap; word-break: break-word; }

      .panel { margin-top: 18px; padding: 16px; border: 1px solid #e5e8ee; border-radius: 10px; background: #fff; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }
      .panel-header { display: flex; justify-content: space-between; align-items: center; }
      .panel-title { font-weight: 600; font-size: 16px; }
      .panel-meta { font-size: 12px; color: #556; }
      .panel-foot { margin-top: 8px; font-size: 12px; color: #556; }

      .tokmap { margin-top: 12px; border: 1px solid #eef1f6; border-radius: 8px; padding: 10px; background: #f9fbff; }
      .tokmap-title { font-size: 13px; font-weight: 600; margin-bottom: 8px; color: #263; }
      .tokmap-text { font-size: 13px; line-height: 1.8; white-space: pre-wrap; word-break: break-word; }
      .tok-span { display: inline; padding: 1px 1px; margin: 0 0px; border-radius: 3px; }
      .tok-span:hover { outline: 1px solid #8899aa; }
    </style>
    """

    html = f"""<!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>{escape(title)}</title>
        {style}
      </head>
      <body>
        {header}
        {gen_block}
        {''.join(panel_sections)}
      </body>
    </html>"""
    return html
