from __future__ import annotations

from types import SimpleNamespace

from evaluations.qwen3_vl_quadrant_example import (
    _token_span_for_chars,
    make_quadrant_image,
    response_parts_and_spans,
    spatial_scores,
)


def test_quadrant_image_has_expected_shape_colors():
    image = make_quadrant_image(224)

    assert image.getpixel((52, 52))[0] > 200  # red circle
    assert image.getpixel((164, 52))[2] > 150  # blue square
    assert image.getpixel((52, 180))[1] > 100  # green triangle
    assert image.getpixel((164, 164))[0] > 200  # yellow star


def test_token_span_for_chars_uses_overlapping_tokens():
    offsets = [(0, 9), (9, 10), (10, 13), (13, 20)]

    assert _token_span_for_chars(offsets, 10, 20) == (2, 3)
    assert _token_span_for_chars(offsets, 20, 20) is None


def test_response_parts_and_spans_separates_reasoning_and_final_answer():
    class CharacterTokenizer:
        def __call__(self, text, **_kwargs):
            return {
                "input_ids": list(range(len(text))),
                "offset_mapping": [(index, index + 1) for index in range(len(text))],
            }

    response = "Reasoning: I inspected the upper-left.\nFinal answer: red circle"

    reasoning, answer, reasoning_span, final_span = response_parts_and_spans(
        CharacterTokenizer(), response
    )

    assert reasoning == "I inspected the upper-left."
    assert answer == "red circle"
    assert response[reasoning_span[0] : reasoning_span[1] + 1] == reasoning
    assert response[final_span[0] : final_span[1] + 1] == answer


def test_spatial_scores_recovers_top_left_quadrant():
    result = SimpleNamespace(
        scores=[4.0, 3.0, 1.0, 1.0, 2.0, 2.0, 0.5, 0.5],
        metadata={
            "multimodal": {
                "visual_grid_thw": [[1, 2, 4]],
                "visual_token_indices_prompt": list(range(8)),
            }
        },
    )

    grid, quadrants = spatial_scores(result)

    assert grid == [[4.0, 3.0, 1.0, 1.0], [2.0, 2.0, 0.5, 0.5]]
    assert max(quadrants, key=lambda name: quadrants[name]["score"]) == "top_left"
    assert sum(item["share"] for item in quadrants.values()) == 1.0
