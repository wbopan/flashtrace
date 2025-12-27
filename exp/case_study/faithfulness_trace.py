"""Faithfulness (MAS/RISE) trace utilities for exp/case_study.

This module is intentionally aligned with `llm_attr_eval.LLMAttributionEvaluator.faithfulness_test`,
but additionally returns the full trace arrays needed for visualization and supports providing
`user_prompt_indices` to avoid fragile subsequence matching.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, List

import numpy as np
import torch

import llm_attr_eval


def _auc(arr: np.ndarray) -> float:
    return float((arr.sum() - arr[0] / 2 - arr[-1] / 2) / max(1, (arr.shape[0] - 1)))


@torch.inference_mode()
def mas_trace(
    llm_evaluator: llm_attr_eval.LLMAttributionEvaluator,
    *,
    attribution: torch.Tensor,
    prompt: str,
    generation: str,
    user_prompt_indices: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Return a token-level faithfulness trace (RISE/MAS/RISE+AP) plus per-token deltas.

    attribution: [R, P] token attribution on prompt-side tokens only.
    prompt: raw prompt string.
    generation: target generation string; scored as generation + eos (if defined).
    user_prompt_indices: optional absolute positions of each prompt token inside formatted prompt ids.
    """

    if attribution.ndim != 2:
        raise ValueError("Expected 2D prompt-side attribution matrix [R, P].")

    pad_token_id = llm_evaluator._ensure_pad_token_id()

    user_prompt = " " + prompt
    formatted_prompt = llm_evaluator.format_prompt(user_prompt)
    formatted_ids = llm_evaluator.tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=False).input_ids

    prompt_ids = formatted_ids.to(llm_evaluator.device)
    prompt_ids_perturbed = prompt_ids.clone()

    eos = llm_evaluator.tokenizer.eos_token or ""
    generation_ids = llm_evaluator.tokenizer(
        generation + eos,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(llm_evaluator.device)

    attr_cpu = attribution.detach().cpu()
    w = attr_cpu.sum(0)
    sorted_attr_indices = torch.argsort(w, descending=True)
    attr_sum = float(w.sum().item())

    P = int(w.numel())

    prompt_positions: List[int]
    if user_prompt_indices is not None:
        prompt_positions = [int(x) for x in user_prompt_indices]
        if len(prompt_positions) != P:
            raise ValueError(
                "user_prompt_indices length does not match prompt-side attribution length: "
                f"indices P={len(prompt_positions)}, attr P={P}."
            )
        if P and max(prompt_positions) >= int(prompt_ids_perturbed.shape[1]):
            raise ValueError("user_prompt_indices contains an out-of-bounds index for formatted prompt ids.")
    else:
        user_ids = llm_evaluator.tokenizer(user_prompt, return_tensors="pt", add_special_tokens=False).input_ids
        user_start = llm_evaluator._find_subsequence_start(formatted_ids[0], user_ids[0])
        if user_start is None:
            raise RuntimeError("Failed to locate user prompt token span inside formatted chat prompt.")
        if int(user_ids.shape[1]) != P:
            raise ValueError(
                "Prompt-side attribution length does not match tokenized user prompt length: "
                f"attr P={P}, user_prompt P={int(user_ids.shape[1])}."
            )
        prompt_positions = [int(user_start) + j for j in range(P)]

    scores = np.zeros(P + 1, dtype=np.float64)
    density = np.zeros(P + 1, dtype=np.float64)

    scores[0] = (
        llm_evaluator.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
    )
    density[0] = 1.0

    if P == 0:
        return {
            "num_tokens": 0,
            "sorted_attr_indices": [],
            "scores_raw": scores.tolist(),
            "density": density.tolist(),
            "normalized_model_response": [1.0],
            "alignment_penalty": [0.0],
            "corrected_scores": [1.0],
            "token_deltas_raw": [],
            "attr_weights": [],
            "metrics": {"RISE": 0.0, "MAS": 0.0, "RISE+AP": 0.0},
        }

    if attr_sum <= 0:
        density = np.linspace(1.0, 0.0, P + 1)

    for step, idx_t in enumerate(sorted_attr_indices):
        idx = int(idx_t.item())
        abs_pos = int(prompt_positions[idx])
        prompt_ids_perturbed[0, abs_pos] = pad_token_id
        scores[step + 1] = (
            llm_evaluator.compute_logprob_response_given_prompt(prompt_ids_perturbed, generation_ids).sum().cpu().detach().item()
        )
        if attr_sum > 0:
            density[step + 1] = density[step] - (float(w[idx].item()) / attr_sum)

    min_normalized_pred = 1.0
    normalized_model_response = scores.copy()
    for i in range(len(scores)):
        normalized_pred = (normalized_model_response[i] - scores[-1]) / (abs(scores[0] - scores[-1]))
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        min_normalized_pred = min(min_normalized_pred, normalized_pred)
        normalized_model_response[i] = min_normalized_pred

    alignment_penalty = np.abs(normalized_model_response - density)
    corrected_scores = normalized_model_response + alignment_penalty
    corrected_scores = corrected_scores.clip(0.0, 1.0)
    corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))
    if np.isnan(corrected_scores).any():
        corrected_scores = np.linspace(1.0, 0.0, len(scores))

    rise = _auc(normalized_model_response)
    mas = _auc(corrected_scores)
    rise_ap = _auc(normalized_model_response + alignment_penalty)

    per_token_delta = np.zeros(P, dtype=np.float64)
    for step, idx_t in enumerate(sorted_attr_indices):
        idx = int(idx_t.item())
        per_token_delta[idx] = scores[step] - scores[step + 1]

    if attr_sum > 0:
        attr_weights = (w.numpy() / (attr_sum + 1e-12)).astype(np.float64)
    else:
        attr_weights = np.zeros(P, dtype=np.float64)

    return {
        "num_tokens": P,
        "sorted_attr_indices": [int(i.item()) for i in sorted_attr_indices],
        "scores_raw": scores.tolist(),
        "density": density.tolist(),
        "normalized_model_response": normalized_model_response.tolist(),
        "alignment_penalty": alignment_penalty.tolist(),
        "corrected_scores": corrected_scores.tolist(),
        "token_deltas_raw": per_token_delta.tolist(),
        "attr_weights": attr_weights.tolist(),
        "metrics": {"RISE": rise, "MAS": mas, "RISE+AP": rise_ap},
    }
