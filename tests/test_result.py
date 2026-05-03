import json

from flashtrace.result import TokenScore, TraceResult


def make_result():
    return TraceResult(
        prompt_tokens=[" alpha", " beta", " gamma"],
        generation_tokens=[" answer"],
        scores=[0.2, 0.7, 0.1],
        per_hop_scores=[[0.1, 0.4, 0.0], [0.1, 0.3, 0.1]],
        thinking_ratios=[0.5, 0.2],
        output_span=(0, 0),
        reasoning_span=(0, 0),
        method="flashtrace",
        metadata={"model": "tiny"},
    )


def test_topk_inputs_sorted():
    result = make_result()

    top = result.topk_inputs(2)

    assert top == [
        TokenScore(index=1, token=" beta", score=0.7),
        TokenScore(index=0, token=" alpha", score=0.2),
    ]


def test_to_dict_is_json_serializable():
    result = make_result()

    payload = result.to_dict()

    assert payload["method"] == "flashtrace"
    assert payload["top_inputs"][0]["token"] == " beta"
    json.dumps(payload)


def test_to_dict_sanitizes_tensor_metadata():
    import torch

    result = TraceResult(
        prompt_tokens=[" alpha"],
        generation_tokens=[" answer"],
        scores=[1.0],
        metadata={"tensor": torch.tensor([1.0, 2.0]), "object": object()},
    )

    payload = result.to_dict()

    assert payload["metadata"]["tensor"] == [1.0, 2.0]
    assert isinstance(payload["metadata"]["object"], str)
    json.dumps(payload)


def test_json_and_html_export(tmp_path):
    result = make_result()
    json_path = tmp_path / "trace.json"
    html_path = tmp_path / "trace.html"

    result.to_json(json_path)
    result.to_html(html_path)

    assert json_path.read_text(encoding="utf-8").startswith("{")
    html = html_path.read_text(encoding="utf-8")
    assert "<html" in html
    assert " beta" in html
