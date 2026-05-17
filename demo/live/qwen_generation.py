from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    token_ids: list[int]


def _chat_messages(prompt: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": prompt}]


def format_chat_prompt(
    prompt: str,
    tokenizer,
    *,
    tokenize: bool = False,
    return_tensors: str | None = None,
) -> str | Any:
    if getattr(tokenizer, "chat_template", None) is None:
        raise ValueError("model has no chat template")
    try:
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": bool(tokenize),
        }
        if return_tensors is not None:
            kwargs["return_tensors"] = return_tensors
        return tokenizer.apply_chat_template(_chat_messages(prompt), **kwargs)
    except ValueError as exc:
        raise ValueError("model has no chat template") from exc


def generate_with_qwen(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> GenerationOutput:
    device = next(model.parameters()).device
    input_ids = format_chat_prompt(
        prompt,
        tokenizer,
        tokenize=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(input_ids) and "input_ids" in input_ids:
        input_ids = input_ids["input_ids"]
    input_ids = input_ids.to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    prompt_len = input_ids.shape[1]
    new_token_ids = generated[0, prompt_len:].tolist()
    text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
    return GenerationOutput(text=text, token_ids=[int(t) for t in new_token_ids])
