from __future__ import annotations

import torch

from flashtrace.attribution import LLMIFRAttribution
from flashtrace.core import extract_model_metadata
from flashtrace.qwen35 import gated_deltanet_effective_attention
from flashtrace.tracer import FlashTrace
from tests.helpers import make_tiny_qwen35_model_and_tokenizer


def test_extract_model_metadata_identifies_hybrid_layer_kinds():
    model, _ = make_tiny_qwen35_model_and_tokenizer()

    metadata = extract_model_metadata(model)

    assert len(metadata.layer_specs) == 8
    assert [spec.kind for spec in metadata.layer_specs] == [
        "linear",
        "linear",
        "linear",
        "full",
        "linear",
        "linear",
        "linear",
        "full",
    ]


def test_layer_specs_carry_per_layer_geometry():
    model, _ = make_tiny_qwen35_model_and_tokenizer()

    metadata = extract_model_metadata(model)

    full = metadata.layer_specs[3]
    assert (full.n_heads_q, full.n_kv_heads, full.head_dim, full.group_size) == (4, 2, 32, 2)

    linear = metadata.layer_specs[0]
    assert (linear.n_heads_q, linear.n_kv_heads, linear.head_dim, linear.group_size) == (
        8,
        8,
        16,
        1,
    )


def test_gated_deltanet_effective_attention_reconstructs_core_output():
    model, _ = make_tiny_qwen35_model_and_tokenizer()
    module = model.model.layers[0].linear_attn  # a linear_attention layer

    torch.manual_seed(0)
    seq_len = 7
    hidden = torch.randn(1, seq_len, model.config.hidden_size)

    attn, value, core_ref = gated_deltanet_effective_attention(module, hidden)

    n_v_heads = model.config.linear_num_value_heads
    assert attn.shape == (n_v_heads, seq_len, seq_len)
    assert value.shape[0] == seq_len and value.shape[1] == n_v_heads
    assert core_ref.shape == value.shape

    # The effective matrix must reconstruct the Gated-DeltaNet core output:
    # core_out[i, h, :] == sum_j attn[h, i, j] * value[j, h, :]
    recon = torch.einsum("hij,jhd->ihd", attn, value)
    assert torch.allclose(recon, core_ref, atol=1e-4, rtol=1e-3)


def test_gated_deltanet_effective_attention_is_causal():
    model, _ = make_tiny_qwen35_model_and_tokenizer()
    module = model.model.layers[0].linear_attn

    torch.manual_seed(1)
    seq_len = 6
    hidden = torch.randn(1, seq_len, model.config.hidden_size)

    attn, _value, _core = gated_deltanet_effective_attention(module, hidden)

    # No attention mass on future tokens (strictly upper triangle is zero).
    upper = torch.triu(attn, diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-5)


def test_ifr_all_positions_runs_on_hybrid_qwen35():
    model, tokenizer = make_tiny_qwen35_model_and_tokenizer()
    attr = LLMIFRAttribution(model, tokenizer)

    result = attr.calculate_ifr_for_all_positions("t10 t20 t30 t40", "t50 t60")

    matrix = result.attribution_matrix
    assert matrix.ndim == 2
    assert matrix.shape[0] == 2  # one row per generation token
    assert torch.isfinite(matrix).all()
    assert (matrix >= 0).all()


def test_flashtrace_method_runs_on_hybrid_qwen35():
    model, tokenizer = make_tiny_qwen35_model_and_tokenizer()
    tracer = FlashTrace(model, tokenizer)

    result = tracer.trace(
        prompt="t10 t20 t30 t40",
        target="t50 t60",
        output_span=(0, 0),
        method="flashtrace",
        hops=1,
    )

    assert len(result.scores) == len(result.prompt_tokens)
    assert len(result.prompt_tokens) > 0
    assert all(isinstance(s, float) for s in result.scores)
    assert all(s == s for s in result.scores)  # no NaNs


def test_flashtrace_smoke_generate_then_trace_on_hybrid_qwen35():
    """End-to-end smoke test: generate with Qwen3.5, then trace the answer."""
    model, tokenizer = make_tiny_qwen35_model_and_tokenizer()

    import torch as _torch

    prompt = "t10 t20 t30 t40 t50"
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    with _torch.inference_mode():
        generated = model.generate(input_ids=input_ids, max_new_tokens=4, do_sample=False)
    target = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=False)

    tracer = FlashTrace(model, tokenizer)
    result = tracer.trace(prompt=prompt, target=target, output_span=(0, 0), method="flashtrace")

    top = result.topk_inputs(3)
    assert len(top) == 3
    assert all(item.score == item.score for item in top)  # no NaNs
    assert result.scores  # non-empty attribution
