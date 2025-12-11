"""HTML helpers for visualizing hop-wise IFR attributions."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from html import escape


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


def _render_tokens(tokens: Sequence[str], scores: Sequence[float], max_score: float, roles: Sequence[str]) -> str:
    spans: List[str] = []
    if max_score <= 0:
        max_score = 1e-8
    for idx, tok in enumerate(tokens):
        score = float(scores[idx]) if idx < len(scores) else 0.0
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
    context: Dict[str, Any],
    hops: Sequence[Dict[str, Any]],
    tokens: Sequence[str],
    segments: Dict[str, Any],
) -> str:
    prompt_len = len(context["prompt_sentences"])
    gen_len = len(context["generation_sentences"])

    # shared scales for consistent coloring
    prompt_max = max(
        (max(h["sentence_scores_raw"][:prompt_len]) for h in hops if h["sentence_scores_raw"][:prompt_len]), default=0.0
    )
    gen_max = max(
        (max(h["sentence_scores_raw"][prompt_len:]) for h in hops if h["sentence_scores_raw"][prompt_len:]), default=0.0
    )
    token_global_max = max((h.get("token_score_max", 0.0) for h in hops), default=0.0)

    def roles_for_tokens() -> List[str]:
        roles = ["prompt" for _ in range(len(tokens))]
        prompt_len_tokens = segments.get("prompt_len", 0)
        for idx in range(prompt_len_tokens, len(tokens)):
            roles[idx] = "gen"
        thinking_span = segments.get("thinking_span")
        sink_span = segments.get("sink_span")
        if thinking_span is not None:
            start = prompt_len_tokens + int(thinking_span[0])
            end = prompt_len_tokens + int(thinking_span[1])
            for i in range(start, min(len(tokens), end + 1)):
                roles[i] = "think"
        if sink_span is not None:
            start = prompt_len_tokens + int(sink_span[0])
            end = prompt_len_tokens + int(sink_span[1])
            for i in range(start, min(len(tokens), end + 1)):
                roles[i] = "output"
        return roles

    token_roles = roles_for_tokens()

    hop_sections: List[str] = []
    for hop in hops:
        raw_scores = hop["sentence_scores_raw"]
        prompt_scores = raw_scores[:prompt_len]
        gen_scores = raw_scores[prompt_len:]
        tok_scores = hop.get("token_scores", [])

        hop_sections.append(
            f"""
            <div class="hop">
              <div class="hop-header">
                <div class="hop-title">Hop {hop['hop']}</div>
                <div class="hop-meta">total mass: {hop['total_mass']:.6f}</div>
              </div>
              <div class="tokens-block">
                <div class="tokens-title">Token-level heatmap (input + thinking + output)</div>
                <div class="tokens-row">
                  {_render_tokens(tokens, tok_scores, token_global_max, token_roles)}
                </div>
              </div>
              <div class="columns">
                {_render_sentence_list('Prompt sentences', context['prompt_sentences'], prompt_scores, prompt_max)}
                {_render_sentence_list('Generation sentences', context['generation_sentences'], gen_scores, gen_max)}
              </div>
              <div class="top-wrap">
                <div class="section-label">Top sentences (all)</div>
                {_render_top_table(hop['top_sentences'])}
              </div>
            </div>
            """
        )

    thinking_ratios = case_meta.get("thinking_ratios") or []
    ratios_str = ", ".join(f"{r:.4f}" for r in thinking_ratios) if thinking_ratios else "N/A"

    header = f"""
    <div class="header">
      <div>
        <div class="title">IFR Multi-hop Case Study</div>
        <div class="subtitle">Dataset: {escape(str(case_meta.get('dataset')))} | index: {case_meta.get('index')}</div>
      </div>
      <div class="meta">
        <div>Sink span (gen idx): {escape(str(case_meta.get('sink_span')))}</div>
        <div>Thinking span (gen idx): {escape(str(case_meta.get('thinking_span')))}</div>
        <div>Hops: {case_meta.get('n_hops')}</div>
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

    html = f"""<!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8" />
        <title>IFR Case Study</title>
        {style}
      </head>
      <body>
        {header}
        {''.join(hop_sections)}
      </body>
    </html>"""
    return html
