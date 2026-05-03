import torch

from flashtrace import core
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


def test_core_metadata_and_weight_pack():
    model, _ = make_tiny_qwen2_model_and_tokenizer()

    metadata = core.extract_model_metadata(model)
    weight_pack = core.build_weight_pack(metadata, next(model.parameters()).dtype)

    assert metadata.n_layers == 3
    assert metadata.n_heads_q == 4
    assert metadata.n_kv_heads == 2
    assert len(weight_pack) == 3
    assert torch.is_tensor(weight_pack[0]["v_w"])


from flashtrace.attribution import LLMIFRAttribution


def test_package_attribution_recompute_matches_stored_attention():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer(n_layers=2, d_model=32, n_heads=4, n_kv_heads=2)
    prompt = "t10 t20 t30 t40"
    target = "t60 t70"

    stored = LLMIFRAttribution(model, tokenizer, recompute_attention=False).calculate_ifr_span(prompt, target)
    recomputed = LLMIFRAttribution(model, tokenizer, recompute_attention=True).calculate_ifr_span(prompt, target)

    diff = (stored.attribution_matrix - recomputed.attribution_matrix).abs().max().item()
    assert diff < 1e-5
