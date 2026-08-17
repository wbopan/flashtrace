import json

import pytest
import torch

import llm_attr_eval

from exp.exp4.attribute_local import (
    METHODS,
    ResponseOnlyAttributionEvaluator,
    _attnlrp_gradient_checkpointing,
    _summary,
)
from tests.helpers import make_tiny_qwen2_model_and_tokenizer


def test_response_only_evaluator_matches_full_logits():
    model, tokenizer = make_tiny_qwen2_model_and_tokenizer()
    prompt_ids = torch.tensor([[10, 20, 30]])
    response_ids = torch.tensor([[40, 50]])
    full = llm_attr_eval.LLMAttributionEvaluator(model, tokenizer)
    response_only = ResponseOnlyAttributionEvaluator(model, tokenizer)

    with torch.no_grad():
        expected = full.compute_logprob_response_given_prompt(
            prompt_ids,
            response_ids,
        )
    actual = response_only.compute_logprob_response_given_prompt(
        prompt_ids,
        response_ids,
    )

    torch.testing.assert_close(actual, expected)


def test_attnlrp_checkpointing_preserves_gradients_and_model_state():
    model, _ = make_tiny_qwen2_model_and_tokenizer()
    input_ids = torch.tensor([[10, 20, 30, 40]])
    embedding_layer = model.get_input_embeddings()

    def input_gradient(*, checkpointed: bool) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        inputs = embedding_layer(input_ids).detach().clone().requires_grad_(True)
        if checkpointed:
            with _attnlrp_gradient_checkpointing(model):
                logits = model(
                    inputs_embeds=inputs,
                    use_cache=False,
                    logits_to_keep=torch.tensor([2]),
                ).logits
                logits.float().sum().backward()
        else:
            logits = model(
                inputs_embeds=inputs,
                use_cache=False,
                logits_to_keep=torch.tensor([2]),
            ).logits
            logits.float().sum().backward()
        return inputs.grad.detach().clone()

    assert not model.training
    assert not model.is_gradient_checkpointing
    expected = input_gradient(checkpointed=False)
    actual = input_gradient(checkpointed=True)

    assert not model.training
    assert not model.is_gradient_checkpointing
    torch.testing.assert_close(actual, expected)


def test_aider_attribution_summary_separates_execution_subset(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = []
    for index, passed in enumerate((True, False)):
        rows.append(
            {
                "example_id": str(index),
                "method": METHODS[0],
                "status": "success",
                "final_passed": passed,
                "result": {
                    "rise": 0.2 + index * 0.2,
                    "mas": 0.3 + index * 0.2,
                    "attribution_seconds": 0.4 + index,
                    "faithfulness_seconds": 0.6,
                    "total_seconds": 1.0 + index,
                },
            }
        )
    rows.append(
        {
            "example_id": "failed",
            "method": METHODS[0],
            "status": "error",
            "final_passed": False,
            "result": None,
        }
    )
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = _summary(path, protocol={}, expected_examples=3)
    method = summary["methods"][METHODS[0]]

    assert method["attempted"] == 3
    assert method["successful"] == 2
    assert method["all_valid"]["rise_mean"] == pytest.approx(0.3)
    assert method["execution_success"]["count"] == 1
    assert method["execution_success"]["rise_mean"] == 0.2
