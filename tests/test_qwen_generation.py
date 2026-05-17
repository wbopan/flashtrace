from __future__ import annotations

import torch

from demo.live.qwen_generation import GenerationOutput, generate_with_qwen
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


def test_generate_with_qwen_returns_text_and_token_ids():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    output = generate_with_qwen(
        model=model,
        tokenizer=tokenizer,
        prompt="t10 t20",
        max_new_tokens=4,
    )

    assert isinstance(output, GenerationOutput)
    assert isinstance(output.text, str)
    assert len(output.text) > 0
    assert isinstance(output.token_ids, list)
    assert len(output.token_ids) > 0
    assert all(isinstance(tid, int) for tid in output.token_ids)


def test_generate_with_qwen_uses_chat_templated_prompt_ids(monkeypatch):
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    recorded = {}

    def fake_generate(*, input_ids, **kwargs):
        recorded["input_ids"] = input_ids.detach().cpu().clone()
        generated_ids = torch.tensor([[30, 40]], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, generated_ids], dim=1)

    monkeypatch.setattr(model, "generate", fake_generate)

    output = generate_with_qwen(
        model=model,
        tokenizer=tokenizer,
        prompt="t10 t20",
        max_new_tokens=4,
    )

    expected = tokenizer.apply_chat_template(
        [{"role": "user", "content": "t10 t20"}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    )
    if not torch.is_tensor(expected) and "input_ids" in expected:
        expected = expected["input_ids"]
    assert recorded["input_ids"].tolist() == expected.tolist()
    assert output.token_ids == [30, 40]
    assert output.text == tokenizer.decode([30, 40], skip_special_tokens=False)


def test_generate_with_qwen_is_deterministic():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()

    torch.manual_seed(0)
    a = generate_with_qwen(model=model, tokenizer=tokenizer, prompt="t10 t20", max_new_tokens=4)
    torch.manual_seed(0)
    b = generate_with_qwen(model=model, tokenizer=tokenizer, prompt="t10 t20", max_new_tokens=4)

    assert a.token_ids == b.token_ids
    assert a.text == b.text
