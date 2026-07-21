from __future__ import annotations

import torch

from evaluations.agentdojo.pilot import (
    _aggregate,
    _char_span_to_token_indices,
    _find_injection_span,
    _localization_metrics,
    _random_metrics,
)


class _CharacterTokenizer:
    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        result = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return result


def test_injection_marker_span_maps_to_exact_tokens():
    text = "benign <INFORMATION>attack</INFORMATION> suffix"
    span = _find_injection_span(text)

    assert text[slice(*span)] == "<INFORMATION>attack</INFORMATION>"
    assert _char_span_to_token_indices(_CharacterTokenizer(), text, span) == tuple(
        range(span[0], span[1])
    )


def test_localization_recovery_rewards_gold_tokens_at_the_top():
    prompt_tokens = [f"t{index}" for index in range(20)]
    attribution = torch.zeros((1, 20), dtype=torch.float32)
    attribution[0, 4:8] = torch.tensor([4.0, 3.0, 2.0, 1.0])

    metrics = _localization_metrics(attribution, prompt_tokens, [4, 5, 6, 7])

    assert metrics["pointing_game"] is True
    assert metrics["recovery_at_20pct"] == 1.0
    assert metrics["injection_mass_fraction"] == 1.0


def test_random_recovery_uses_expected_top_fraction():
    metrics = _random_metrics([f"t{index}" for index in range(20)], [4, 5, 6, 7])

    assert metrics["recovery_at_5pct"] == 0.05
    assert metrics["recovery_at_10pct"] == 0.10
    assert metrics["recovery_at_20pct"] == 0.20


def test_aggregate_keeps_per_method_means_separate():
    rows = []
    for method, recovery in (("a", 0.2), ("b", 0.8)):
        rows.append(
            {
                "method": method,
                "pointing_game": recovery > 0.5,
                "recovery_at_5pct": recovery,
                "recovery_at_10pct": recovery,
                "recovery_at_20pct": recovery,
                "injection_mass_fraction": recovery,
                "runtime_seconds": recovery,
            }
        )

    summary = _aggregate(rows)

    assert summary["a"]["recovery_at_10pct_mean"] == 0.2
    assert summary["b"]["recovery_at_10pct_mean"] == 0.8
