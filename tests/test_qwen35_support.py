from __future__ import annotations

import torch

from flashtrace.core import extract_model_metadata
from flashtrace.qwen35 import gated_deltanet_effective_attention
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
