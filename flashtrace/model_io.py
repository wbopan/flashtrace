from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_dtype(dtype: str | torch.dtype = "auto") -> str | torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    value = str(dtype).lower()
    if value == "auto":
        return "auto"
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[value]


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
    **model_kwargs: Any,
):
    """Load a Hugging Face causal LM and matching tokenizer."""

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    # FlashTrace needs attention weights; SDPA returns None for them.
    model_kwargs.setdefault("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=_resolve_dtype(dtype),
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    model.eval()
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_vlm_and_processor(
    model_name_or_path: str,
    *,
    device_map: str | dict[str, Any] | None = "auto",
    dtype: str | torch.dtype = "auto",
    trust_remote_code: bool = True,
    processor_kwargs: dict[str, Any] | None = None,
    **model_kwargs: Any,
):
    """Load an image-text model and processor for FlashTrace VLM attribution.

    Eager attention is intentional: Qwen3-VL's interleaved M-RoPE is not
    compatible with FlashTrace's 1-D RoPE attention recomputation path.
    """

    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM as AutoVLM
    except ImportError:  # compatibility with earlier Transformers 5.x releases
        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError as error:
            raise ImportError(
                "This Transformers version has no multimodal auto-model class; "
                "install Transformers 5.x with Qwen3-VL support."
            ) from error

    try:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **(processor_kwargs or {}),
        )
    except ImportError as error:
        raise ImportError(
            "Loading Qwen vision processors requires the FlashTrace VLM extra; "
            "install it with `pip install -e '.[vlm]'`."
        ) from error
    model_kwargs.setdefault("attn_implementation", "eager")
    if "dtype" not in model_kwargs and "torch_dtype" not in model_kwargs:
        model_kwargs["dtype"] = _resolve_dtype(dtype)
    model = AutoVLM.from_pretrained(
        model_name_or_path,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        **model_kwargs,
    )
    model.eval()
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, processor
