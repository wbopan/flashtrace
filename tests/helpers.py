from __future__ import annotations

from tokenizers import AddedToken
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast


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

    backend = Tokenizer(models.WordLevel(vocab={f"t{i}": i for i in range(500)}, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken("t1", single_word=True),
            "pad_token": AddedToken("t2", single_word=True),
        }
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    return model, tokenizer


def _make_tiny_word_tokenizer(vocab_size: int = 500):
    backend = Tokenizer(
        models.WordLevel(vocab={f"t{i}": i for i in range(vocab_size)}, unk_token="t0")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
    tokenizer.add_special_tokens(
        {
            "eos_token": AddedToken("t1", single_word=True),
            "pad_token": AddedToken("t2", single_word=True),
        }
    )
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
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
):
    """Build a tiny Qwen3.5 text-only causal LM with a hybrid layer stack.

    The layer pattern follows the real model: ``(full_attention_interval - 1)``
    Gated-DeltaNet linear-attention layers followed by one full-attention layer,
    repeated. With the defaults this yields 6 linear + 2 full attention layers.
    """

    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

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
    model = Qwen3_5ForCausalLM(config)
    model.eval()

    tokenizer = _make_tiny_word_tokenizer(vocab_size=500)
    return model, tokenizer
