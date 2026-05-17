from flashtrace import FlashTrace, TraceResult
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


def test_flashtrace_both_uses_output_hop0_then_all_generation_hops(monkeypatch):
    import types

    import torch

    import flashtrace.improved as improved
    from flashtrace.core import IFRAggregate

    calls = []

    def fake_compute_ifr_sentence_aggregate(
        *,
        sink_start,
        sink_end,
        sink_weights=None,
        params=None,
        **kwargs,
    ):
        calls.append(
            (
                int(sink_start),
                int(sink_end),
                None
                if sink_weights is None
                else tuple(float(x) for x in sink_weights.detach().cpu().flatten()),
            )
        )
        token_scores = torch.zeros(params.sequence_length, dtype=torch.float32)
        token_scores[int(sink_start) : int(sink_end) + 1] = 1.0
        return IFRAggregate(
            per_layer=[],
            token_importance_total=token_scores,
            head_importance_total=torch.zeros(1),
            ffn_importance_per_layer=torch.zeros(1),
            resid_ffn_importance_per_layer=torch.zeros(1),
        )

    monkeypatch.setattr(improved, "compute_ifr_sentence_aggregate", fake_compute_ifr_sentence_aggregate)

    engine = improved.LLMIFRAttributionBoth.__new__(improved.LLMIFRAttributionBoth)
    engine.tokenizer = types.SimpleNamespace(eos_token_id=99)
    engine.generation_ids = torch.tensor([[10, 11, 12, 99]])
    engine.generation_tokens = ["think0", "think1", "answer", "<eos>"]
    engine.prompt_tokens = ["p0", "p1", "p2", "p3"]
    engine.user_prompt_tokens = list(engine.prompt_tokens)
    engine.user_prompt_indices = [0, 1, 2, 3]
    engine.renorm_threshold_default = 0.0
    engine.recompute_attention = False

    engine._ensure_generation = lambda prompt, target: (
        torch.ones(1, 8, dtype=torch.long),
        torch.ones(1, 8, dtype=torch.long),
        4,
        4,
    )
    engine._capture_model_state = lambda *args, **kwargs: (
        {},
        None,
        types.SimpleNamespace(rotary_emb=None),
        [],
    )
    engine._build_ifr_params = lambda metadata, total_len: types.SimpleNamespace(
        sequence_length=total_len,
        model_dtype=torch.float32,
    )
    engine.extract_user_prompt_attributions = lambda input_tokens, matrix: matrix[:, :4]
    engine._project_vector = lambda vector: vector[:4]
    engine._finalize_result = lambda score_array, metadata=None: types.SimpleNamespace(
        score_array=score_array,
        metadata=metadata,
    )

    result = engine.calculate_ifr_multi_hop_both(
        "prompt",
        target="target",
        sink_span=(2, 2),
        thinking_span=(0, 1),
        n_hops=2,
    )

    assert calls[0] == (6, 6, None)
    assert calls[1][0:2] == (4, 6)
    assert calls[1][2] == (0.0, 0.0, 1.0)
    assert calls[2][0:2] == (4, 6)
    assert result.metadata["ifr"]["sink_span_absolute"] == (6, 6)
    assert result.metadata["ifr"]["all_gen_span_absolute"] == (4, 6)


def test_flashtrace_trace_returns_public_result():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)
    tracer = FlashTrace(model, tokenizer, chunk_tokens=16, sink_chunk_tokens=4, recompute_attention=True)

    result = tracer.trace(
        prompt="t10 t20 t30 t40",
        target="t60 t70 t80",
        output_span=(1, 2),
        reasoning_span=(0, 1),
        hops=1,
    )

    assert isinstance(result, TraceResult)
    assert result.method == "flashtrace"
    assert len(result.prompt_tokens) > 0
    assert len(result.scores) == len(result.prompt_tokens)
    assert result.output_span == (1, 2)
    assert result.reasoning_span == (0, 1)


def test_ifr_span_method_returns_public_result():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)
    tracer = FlashTrace(model, tokenizer, chunk_tokens=16, sink_chunk_tokens=4, recompute_attention=True)

    result = tracer.trace(
        prompt="t10 t20 t30 t40",
        target="t60 t70",
        output_span=(0, 1),
        method="ifr-span",
    )

    assert result.method == "ifr-span"
    assert len(result.scores) == len(result.prompt_tokens)


def test_flashtrace_default_raw_prompt_does_not_call_chat_template():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)

    def fail_apply_chat_template(*args, **kwargs):
        raise AssertionError("apply_chat_template should be opt-in")

    tokenizer.apply_chat_template = fail_apply_chat_template
    tracer = FlashTrace(model, tokenizer, chunk_tokens=16, sink_chunk_tokens=4, recompute_attention=True)

    result = tracer.trace(
        prompt="t3 t4 t5",
        target="t6 t7",
        output_span=(0, 1),
        method="ifr-span",
    )

    assert result.method == "ifr-span"
    assert result.prompt_tokens == ["t3", "t4", "t5"]


def test_flashtrace_target_without_eos_token():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)
    tokenizer.eos_token = None
    tokenizer.eos_token_id = None
    tracer = FlashTrace(model, tokenizer, chunk_tokens=16, sink_chunk_tokens=4, recompute_attention=True)

    result = tracer.trace(
        prompt="t10 t20 t30 t40",
        target="t60 t70",
        output_span=(0, 1),
        method="ifr-span",
    )

    assert result.method == "ifr-span"
    assert result.generation_tokens == ["t60", "t70"]
