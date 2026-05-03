from flashtrace import FlashTrace, TraceResult
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


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
