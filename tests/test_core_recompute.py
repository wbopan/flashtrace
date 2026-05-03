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
