from __future__ import annotations

from tokenizers import AddedToken
from tokenizers import Tokenizer, models, pre_tokenizers
import torch
from transformers import AutoConfig, AutoModelForCausalLM, BatchFeature, PreTrainedTokenizerFast


def make_tiny_qwen2_model_and_tokenizer(
    *,
    n_layers: int = 3,
    d_model: int = 48,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    max_pos: int = 128,
):
    config = AutoConfig.for_model(
        "qwen2",
        vocab_size=500,
        hidden_size=d_model,
        intermediate_size=d_model * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        max_position_embeddings=max_pos,
        use_sliding_window=False,
        attn_implementation="eager",
    )
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    model.eval()

    vocab = {f"t{i}": i for i in range(498)}
    vocab["<|im_start|>"] = 498
    vocab["<|im_end|>"] = 499
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken("t1", single_word=True),
            "pad_token": AddedToken("t2", single_word=True),
            "additional_special_tokens": [
                AddedToken("<|im_start|>", single_word=False),
                AddedToken("<|im_end|>", single_word=False),
            ],
        }
    )
    tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|>\n"
        "{{ m['content'] }}\n"
        "<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>\n{% endif %}"
    )
    return model, tokenizer


def _make_tiny_word_tokenizer(vocab_size: int = 500):
    vocab = {f"t{i}": i for i in range(vocab_size - 2)}
    vocab["<|im_start|>"] = vocab_size - 2
    vocab["<|im_end|>"] = vocab_size - 1
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken("t1", single_word=True),
            "pad_token": AddedToken("t2", single_word=True),
            "additional_special_tokens": [
                AddedToken("<|im_start|>", single_word=False),
                AddedToken("<|im_end|>", single_word=False),
            ],
        }
    )
    tokenizer.chat_template = (
        "{% for m in messages %}<|im_start|>\n"
        "{{ m['content'] }}\n"
        "<|im_end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>\n{% endif %}"
    )
    return tokenizer


def make_tiny_qwen35_model_and_tokenizer(
    *,
    n_layers: int = 8,
    d_model: int = 64,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    head_dim: int = 32,
    linear_num_key_heads: int = 4,
    linear_num_value_heads: int = 8,
    linear_head_dim: int = 16,
    full_attention_interval: int = 4,
    max_pos: int = 256,
    seed: int = 0,
):
    """Build a tiny Qwen3.5 text-only causal LM with a hybrid layer stack.

    The layer pattern follows the real model: ``(full_attention_interval - 1)``
    Gated-DeltaNet linear-attention layers followed by one full-attention layer,
    repeated. With the defaults this yields 6 linear + 2 full attention layers.

    ``seed`` makes the random weight initialisation reproducible across runs.
    """

    import torch

    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)

    config = Qwen3_5TextConfig(
        vocab_size=500,
        hidden_size=d_model,
        intermediate_size=d_model * 2,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        linear_num_key_heads=linear_num_key_heads,
        linear_num_value_heads=linear_num_value_heads,
        linear_key_head_dim=linear_head_dim,
        linear_value_head_dim=linear_head_dim,
        linear_conv_kernel_dim=4,
        max_position_embeddings=max_pos,
        full_attention_interval=full_attention_interval,
    )
    # FlashTrace needs softmax attention weights from full-attention layers.
    model = Qwen3_5ForCausalLM._from_config(config, attn_implementation="eager")
    model.eval()

    tokenizer = _make_tiny_word_tokenizer(vocab_size=500)
    return model, tokenizer


class _TinyQwen3VLProcessor:
    """Processor-shaped fixture that emits a valid one-image Qwen3-VL batch."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        return_tensors=None,
    ):
        text = next(
            item["text"]
            for item in messages[0]["content"]
            if item.get("type") == "text"
        )
        user_ids = self.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids[0]
        # One merged visual token: grid 1x2x2 with spatial_merge_size=2.
        prefix = torch.tensor([58, 60, 61, 62], dtype=torch.long)
        suffix = torch.tensor([59, 58], dtype=torch.long)
        input_ids = torch.cat([prefix, user_ids, suffix]).unsqueeze(0)
        if not tokenize:
            return self.tokenizer.decode(
                input_ids[0], skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

        mm_token_type_ids = torch.zeros_like(input_ids)
        mm_token_type_ids[input_ids == 61] = 1
        return BatchFeature(
            data={
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "mm_token_type_ids": mm_token_type_ids,
                "pixel_values": torch.randn(4, 12),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
            },
            tensor_type="pt",
        )


def make_tiny_qwen3_vl_model_and_processor(*, seed: int = 0):
    """Build a tiny eager-attention Qwen3-VL model plus processor fixture."""

    import torch

    from transformers.models.qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    torch.manual_seed(seed)
    text_config = Qwen3VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        pad_token_id=2,
    )
    vision_config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
        out_hidden_size=32,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )
    config = Qwen3VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=61,
        video_token_id=63,
        vision_start_token_id=60,
        vision_end_token_id=62,
    )
    model = Qwen3VLForConditionalGeneration._from_config(
        config, attn_implementation="eager"
    )
    model.eval()

    vocab = {f"t{i}": i for i in range(60)}
    vocab.update(
        {
            "<|vision_start|>": 60,
            "<|image_pad|>": 61,
            "<|vision_end|>": 62,
            "<|video_pad|>": 63,
        }
    )
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken("t1", single_word=True),
            "pad_token": AddedToken("t2", single_word=True),
            "additional_special_tokens": [
                AddedToken("<|vision_start|>", single_word=False),
                AddedToken("<|image_pad|>", single_word=False),
                AddedToken("<|vision_end|>", single_word=False),
                AddedToken("<|video_pad|>", single_word=False),
            ],
        }
    )
    return model, _TinyQwen3VLProcessor(tokenizer)


def make_tiny_qwen25_vl_model_and_processor(*, seed: int = 0):
    """Build the Qwen2.5-VL fallback with the same tiny processor fixture."""

    from transformers.models.qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLTextConfig,
        Qwen2_5_VLVisionConfig,
    )

    torch.manual_seed(seed)
    text_config = Qwen2_5_VLTextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=2,
        use_sliding_window=False,
        rope_parameters={"rope_type": "default", "mrope_section": [1, 1, 2]},
    )
    vision_config = Qwen2_5_VLVisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
        window_size=4,
        out_hidden_size=32,
        fullatt_block_indexes=[0],
    )
    config = Qwen2_5_VLConfig(
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=61,
        video_token_id=63,
        vision_start_token_id=60,
        vision_end_token_id=62,
    )
    model = Qwen2_5_VLForConditionalGeneration._from_config(
        config, attn_implementation="eager"
    )
    model.eval()

    # Token IDs and visual geometry intentionally match the Qwen3-VL fixture.
    _, processor = make_tiny_qwen3_vl_model_and_processor(seed=seed)
    return model, processor
