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
