from __future__ import annotations

import pytest

from flashtrace import FlashTrace
from flashtrace.result import TraceResult
from tests.helpers import make_tiny_qwen2_model_and_tokenizer

from demo.live.token_overlay import build_token_records, build_token_records_from_ids


def test_build_document_views_defaults_to_selectable_generation_without_eos():
    from demo.live.token_document import build_document_views

    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    prompt_records = build_token_records(
        text="t10 t20",
        tokenizer=tokenizer,
        section="prompt",
        role="user",
    )
    generation_records, _decoded = build_token_records_from_ids(
        token_ids=[30, 40, tokenizer.eos_token_id],
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )

    model = build_document_views(
        phase="generated",
        prompt_records=prompt_records,
        generation_records=generation_records,
    )

    tokens = model["views"][0]["tokens"]
    generation = [token for token in tokens if token["region"] == "generation"]
    assert [token["text"] for token in generation] == ["t30", "t40", "t1"]
    assert [token["gen_index"] for token in generation] == [0, 1, 2]
    assert [token["selectable"] for token in generation] == [True, True, False]
    assert model["target_span"] == [0, 1]


def test_build_document_views_traced_uses_aggregate_only_when_hops_absent():
    from demo.live.token_document import build_document_views

    result = TraceResult(
        prompt_tokens=["t10", "t20"],
        generation_tokens=["t30", "t40"],
        scores=[0.25, 0.75],
        per_hop_scores=[],
        output_span=(1, 1),
        method="ifr-span",
    )

    model = build_document_views(phase="traced", result=result)

    assert [view["name"] for view in model["views"]] == ["Aggregate"]
    assert model["target_span"] == [1, 1]
    prompt_scores = [
        token["score"]
        for token in model["views"][0]["tokens"]
        if token["region"] == "prompt"
    ]
    assert prompt_scores == [0.25, 0.75]


def test_build_document_views_traced_adds_hop_views_when_scores_exist():
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

    assert [view["name"] for view in model["views"]] == ["Aggregate", "Hop 1", "Hop 2"]


def test_document_generation_tokens_match_trace_result_modulo_trailing_eos():
    from demo.live.token_document import build_document_views

    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
    )
    generated_ids = [60, 70]
    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prompt_records = build_token_records(
        text="t10 t20 t30",
        tokenizer=tokenizer,
        section="prompt",
        role="user",
    )
    generation_records, _decoded = build_token_records_from_ids(
        token_ids=generated_ids,
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )
    document = build_document_views(
        phase="generated",
        prompt_records=prompt_records,
        generation_records=generation_records,
    )

    result = FlashTrace(
        model,
        tokenizer,
        chunk_tokens=16,
        sink_chunk_tokens=4,
        recompute_attention=True,
    ).trace(
        prompt="t10 t20 t30",
        target=generated_text,
        output_span=(0, 1),
        method="ifr-span",
    )

    document_generation = [
        token["text"]
        for token in document["views"][0]["tokens"]
        if token["region"] == "generation"
    ]
    trace_generation = list(result.generation_tokens)
    if trace_generation and trace_generation[-1] == tokenizer.eos_token:
        trace_generation = trace_generation[:-1]
    assert document_generation == trace_generation


def test_traced_views_include_server_computed_colors_per_view():
    from demo.live.token_document import build_document_views
    from flashtrace.viz import _score_color

    result = TraceResult(
        prompt_tokens=["t10", "t20"],
        generation_tokens=["t30"],
        scores=[0.1, 0.4],
        per_hop_scores=[[0.8, 0.4]],
        output_span=(0, 0),
        method="flashtrace",
    )

    model = build_document_views(phase="traced", result=result)
    aggregate_prompt = [
        token for token in model["views"][0]["tokens"] if token["region"] == "prompt"
    ]
    hop_prompt = [
        token for token in model["views"][1]["tokens"] if token["region"] == "prompt"
    ]

    assert [token["color"] for token in aggregate_prompt] == [
        _score_color(0.1, 0.4),
        _score_color(0.4, 0.4),
    ]
    assert [token["color"] for token in hop_prompt] == [
        _score_color(0.8, 0.8),
        _score_color(0.4, 0.8),
    ]
    assert all(token["color"] is None for token in model["views"][0]["tokens"] if token["region"] == "generation")


def test_playwright_smoke_selection_and_tabs_best_effort():
    playwright = pytest.importorskip("playwright.sync_api")
    from demo.live.token_document import build_document_views

    _, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    prompt_records = build_token_records(
        text="t10",
        tokenizer=tokenizer,
        section="prompt",
        role="user",
    )
    generation_records, _decoded = build_token_records_from_ids(
        token_ids=[30, 40],
        tokenizer=tokenizer,
        section="answer",
        role="assistant",
    )
    generated_model = build_document_views(
        phase="generated",
        prompt_records=prompt_records,
        generation_records=generation_records,
    )

    result = TraceResult(
        prompt_tokens=["t10", "t20"],
        generation_tokens=["t30"],
        scores=[0.1, 0.2],
        per_hop_scores=[[0.8, 0.4]],
        output_span=(0, 0),
        method="flashtrace",
    )
    traced_model = build_document_views(phase="traced", result=result)

    with playwright.sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        repo_root = __import__("pathlib").Path(__file__).resolve().parents[1]
        app_js = (repo_root / "demo" / "live" / "static" / "app.js").read_text(
            encoding="utf-8"
        )
        page.set_content(
            """
            <div id="token-document"></div>
            <button id="generate-button"></button>
            <button id="trace-button"></button>
            <a id="download-link"></a>
            """
        )
        page.evaluate(app_js)
        page.evaluate("model => window.flashtraceTest.renderDocument(model)", generated_model)
        page.locator("[data-gen-index='0']").click()
        page.locator("[data-gen-index='1']").click()
        assert page.evaluate("window.flashtraceTest.getState().targetSpan") == "0:1"
        page.evaluate("model => window.flashtraceTest.renderDocument(model)", traced_model)
        page.locator(".ft-tab").nth(1).click()
        assert page.locator(".ft-tab.is-active").inner_text() == "Hop 1"
        assert page.locator(".ft-view.is-active").evaluate(
            "node => Array.from(node.querySelectorAll('.region-prompt')).map(t => t.dataset.score)"
        ) == ["0.80000000", "0.40000000"]
        browser.close()


def test_document_view_uses_record_display_text_for_token_text():
    from demo.live.token_document import build_document_views
    from demo.live.token_overlay import TokenRecord

    # token_text is the raw BPE piece; display_text is the faithful source
    # substring (real whitespace). The render model must show display_text.
    record = TokenRecord(
        section="prompt",
        token_index=0,
        token_id=5,
        token_text="Ċ",
        char_start=0,
        char_end=1,
        kind="content",
        selectable=False,
        role="user",
        display_text="\n",
    )

    model = build_document_views(phase="prompt", prompt_records=[record])

    assert model["views"][0]["tokens"][0]["text"] == "\n"
