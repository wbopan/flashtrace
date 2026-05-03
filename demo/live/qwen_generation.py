from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GenerationOutput:
    text: str
    token_ids: list[int]


def generate_with_qwen(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
) -> GenerationOutput:
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
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


def generate_smoke_response(*, prompt: str) -> GenerationOutput:
    text = (
        "<think>\n"
        "The user is asking about the capital of France. "
        "From the context, Paris is mapped to France.\n"
        "</think>\n"
        "<answer>\nParis\n</answer>"
    )
    return GenerationOutput(text=text, token_ids=[])
