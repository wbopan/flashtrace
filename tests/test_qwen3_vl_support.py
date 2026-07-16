from __future__ import annotations

import torch

from flashtrace import FlashTrace
from flashtrace.core import extract_model_metadata
from tests.helpers import (
    make_tiny_qwen25_vl_model_and_processor,
    make_tiny_qwen3_vl_model_and_processor,
)


def test_extract_model_metadata_finds_qwen3_vl_language_decoder():
    model, _ = make_tiny_qwen3_vl_model_and_processor()

    metadata = extract_model_metadata(model)

    assert metadata.decoder is model.model.language_model
    assert metadata.layers is model.model.language_model.layers
    assert metadata.n_layers == 2
    assert metadata.d_model == 32
    assert metadata.n_heads_q == 4
    assert metadata.n_kv_heads == 2
    assert metadata.head_dim == 8


def test_qwen3_vl_trace_keeps_visual_tokens_and_forces_stored_attention(monkeypatch):
    model, processor = make_tiny_qwen3_vl_model_and_processor()

    def fail_recompute(*args, **kwargs):
        raise AssertionError("M-RoPE VLM attention must not use the 1-D recompute path")

    monkeypatch.setattr("flashtrace.core.recompute_layer_attention", fail_recompute)
    tracer = FlashTrace(
        model,
        processor,
        chunk_tokens=16,
        sink_chunk_tokens=4,
        recompute_attention=True,
    )

    result = tracer.trace(
        prompt="t20 t21",
        images=torch.zeros(3, 4, 4),
        target="t30 t31",
        output_span=(0, 1),
        reasoning_span=(0, 0),
        hops=1,
    )

    assert result.prompt_tokens == ["<|image_pad|>", "t20", "t21"]
    assert len(result.scores) == 3
    assert all(torch.isfinite(torch.tensor(result.scores)))

    multimodal = result.metadata["multimodal"]
    assert multimodal["attention_mode"] == "stored"
    assert multimodal["num_images"] == 1
    assert multimodal["image_grid_thw"] == [[1, 2, 2]]
    assert multimodal["spatial_merge_size"] == 2
    assert multimodal["visual_grid_thw"] == [[1, 1, 1]]
    assert multimodal["prompt_feature_indices_absolute"] == [2, 4, 5]
    assert multimodal["visual_token_spans_absolute"] == [(2, 2)]
    assert multimodal["visual_token_spans_prompt"] == [(0, 0)]


def test_qwen3_vl_generation_and_trace_smoke():
    model, processor = make_tiny_qwen3_vl_model_and_processor(seed=1)
    tracer = FlashTrace(
        model,
        processor,
        chunk_tokens=16,
        sink_chunk_tokens=4,
        generate_kwargs={"max_new_tokens": 1, "do_sample": False},
    )

    result = tracer.trace(
        prompt="t20 t21",
        images=torch.zeros(3, 4, 4),
        output_span=(0, 0),
        method="ifr-span",
    )

    assert len(result.generation_tokens) == 1
    assert result.prompt_tokens[0] == "<|image_pad|>"
    assert result.metadata["multimodal"]["attention_mode"] == "stored"


def test_qwen25_vl_fallback_trace_smoke():
    model, processor = make_tiny_qwen25_vl_model_and_processor()
    tracer = FlashTrace(
        model,
        processor,
        chunk_tokens=16,
        sink_chunk_tokens=4,
        recompute_attention=True,
    )

    result = tracer.trace(
        prompt="t20 t21",
        images=torch.zeros(3, 4, 4),
        target="t30 t31",
        output_span=(0, 1),
        method="ifr-span",
    )

    assert result.prompt_tokens == ["<|image_pad|>", "t20", "t21"]
    assert result.metadata["multimodal"]["visual_grid_thw"] == [[1, 1, 1]]
    assert result.metadata["multimodal"]["attention_mode"] == "stored"
