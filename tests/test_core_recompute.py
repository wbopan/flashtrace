import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from flashtrace import core
from flashtrace.attribution import LLMIFRAttribution
from flashtrace.improved import LLMIFRAttributionBoth
from tests.helpers import (
    _make_tiny_word_tokenizer,
    make_tiny_qwen2_model_and_tokenizer,
)


def test_core_metadata_and_weight_pack():
    model, _ = make_tiny_qwen2_model_and_tokenizer()

    metadata = core.extract_model_metadata(model)
    weight_pack = core.build_weight_pack(metadata, next(model.parameters()).dtype)

    assert metadata.n_layers == 3
    assert metadata.n_heads_q == 4
    assert metadata.n_kv_heads == 2
    assert len(weight_pack) == 3
    assert torch.is_tensor(weight_pack[0]["v_w"])

def test_package_attribution_recompute_matches_stored_attention():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)
    prompt = "t10 t20 t30 t40"
    target = "t60 t70"

    stored = LLMIFRAttribution(model, tokenizer, recompute_attention=False).calculate_ifr_span(prompt, target)
    recomputed = LLMIFRAttribution(model, tokenizer, recompute_attention=True).calculate_ifr_span(prompt, target)

    diff = (stored.attribution_matrix - recomputed.attribution_matrix).abs().max().item()
    assert diff < 1e-5


def test_chunked_weighted_recompute_matches_stored_attention(monkeypatch):
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(
        n_layers=2,
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
    )
    prompt = "t10 t20 t30 t40"
    target = "t50 t60 t70 t80 t90 t100 t110 t120"
    kwargs = {
        "target": target,
        "sink_span": (6, 7),
        "thinking_span": (0, 5),
        "n_hops": 1,
    }

    stored = LLMIFRAttributionBoth(
        model,
        tokenizer,
        sink_chunk_tokens=2,
        recompute_attention=False,
    ).calculate_ifr_multi_hop_both(prompt, **kwargs)
    monkeypatch.setattr(core, "_MAX_AGGREGATE_ATTENTION_BYTES", 1)
    recomputed = LLMIFRAttributionBoth(
        model,
        tokenizer,
        sink_chunk_tokens=2,
        recompute_attention=True,
    ).calculate_ifr_multi_hop_both(prompt, **kwargs)

    torch.testing.assert_close(
        stored.attribution_matrix,
        recomputed.attribution_matrix,
        atol=1e-5,
        rtol=1e-5,
        equal_nan=True,
    )


def test_qwen3_recompute_matches_stored_attention_with_qk_norm():
    tokenizer = _make_tiny_word_tokenizer()
    config = Qwen3Config(
        vocab_size=500,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation="eager",
    )
    model.eval()
    prompt = "t10 t20 t30 t40"
    target = "t60 t70"

    stored = LLMIFRAttribution(
        model,
        tokenizer,
        recompute_attention=False,
    ).calculate_ifr_span(prompt, target)
    recomputed = LLMIFRAttribution(
        model,
        tokenizer,
        recompute_attention=True,
    ).calculate_ifr_span(prompt, target)

    torch.testing.assert_close(
        stored.attribution_matrix,
        recomputed.attribution_matrix,
        atol=1e-5,
        rtol=1e-5,
    )
