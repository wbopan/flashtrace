import pytest

from evaluations.local_sglang import (
    LocalSGLangGenerator,
    _finish_reason,
    _loopback_base_url,
    _output_token_ids,
)


class _Tokenizer:
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False):
        del text, add_special_tokens
        return type("Encoded", (), {"input_ids": [1, 2]})()


def test_sglang_client_rejects_remote_endpoints():
    assert _loopback_base_url("http://127.0.0.1:8001/") == "http://127.0.0.1:8001"
    with pytest.raises(ValueError, match="non-loopback"):
        _loopback_base_url("https://example.com")


def test_sglang_token_ids_support_tuple_and_object_metadata():
    assert _output_token_ids(
        {"output_token_logprobs": [[-0.1, 3, "x"], [-0.2, 4, "y"]]}
    ) == [3, 4]
    assert _output_token_ids(
        {"output_token_logprobs": [{"logprob": -0.1, "token_id": 5}]}
    ) == [5]
    assert _output_token_ids({"output_token_logprobs": [[-0.1]]}) == []


def test_sglang_finish_reason_normalization():
    assert _finish_reason({"finish_reason": {"type": "stop"}}) == ("stop", True)
    assert _finish_reason({"finish_reason": {"type": "length"}}) == (
        "length",
        False,
    )


def test_sglang_sampling_uses_backend_seed_name(monkeypatch):
    client = LocalSGLangGenerator("http://127.0.0.1:8001", _Tokenizer())
    captured = {}

    def fake_request(path, *, payload=None):
        captured.update(path=path, payload=payload)
        return {"text": "ok", "meta_info": {"finish_reason": "stop"}}

    monkeypatch.setattr(client, "_request", fake_request)
    client.generate(
        "prompt",
        8,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        seed=17,
    )

    assert captured["path"] == "/generate"
    assert captured["payload"]["sampling_params"] == {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "max_new_tokens": 8,
        "skip_special_tokens": False,
        "no_stop_trim": True,
        "sampling_seed": 17,
    }


def test_sglang_conditional_scoring_returns_only_aggregates(monkeypatch):
    client = LocalSGLangGenerator("http://127.0.0.1:8001", _Tokenizer())
    captured = {}

    def fake_request(path, *, payload=None):
        captured.update(path=path, payload=payload)
        return {
            "text": "",
            "meta_info": {
                "input_token_logprobs": [
                    [None, 2, None],
                    [-0.2, 3, None],
                    [-0.4, 4, None],
                ]
            },
        }

    monkeypatch.setattr(client, "_request", fake_request)
    result = client.score_continuation_token_ids([1, 2], [3, 4])

    assert captured["path"] == "/generate"
    assert captured["payload"]["logprob_start_len"] == 1
    assert result["scored_token_count"] == 2
    assert result["sum_logprob"] == pytest.approx(-0.6)
    assert result["mean_logprob"] == pytest.approx(-0.3)
