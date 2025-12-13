"""Helpers for IFR case studies (hop-wise aggregation + sanitization).

All utilities stay local to exp/case_study to avoid touching core eval code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from shared_utils import create_sentence_masks, create_sentences


@dataclass
class SentenceContext:
    """Token and sentence alignment used for aggregation and visualization."""

    prompt_sentences: List[str]
    generation_sentences: List[str]
    all_sentences: List[str]
    sentence_mask: torch.Tensor  # [num_all_sentences, num_all_tokens]


def _postprocess_scores(vec: torch.Tensor, transform: str) -> torch.Tensor:
    vec = torch.nan_to_num(vec.to(dtype=torch.float32), nan=0.0)
    if transform == "positive":
        return vec.clamp(min=0.0)
    if transform == "abs":
        return vec.abs()
    raise ValueError(f"Unsupported transform={transform!r}; expected 'positive' or 'abs'.")


def vector_stats(vec: torch.Tensor) -> Dict[str, float]:
    if vec.numel() == 0:
        return {"min": 0.0, "max": 0.0, "abs_max": 0.0, "mean": 0.0, "sum": 0.0}
    v = vec.detach().to(dtype=torch.float32)
    return {
        "min": float(v.min().item()),
        "max": float(v.max().item()),
        "abs_max": float(v.abs().max().item()),
        "mean": float(v.mean().item()),
        "sum": float(v.sum().item()),
    }


def tensor_to_list(x: Any) -> Any:
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, list):
        return [tensor_to_list(v) for v in x]
    if isinstance(x, dict):
        return {k: tensor_to_list(v) for k, v in x.items()}
    return x


def sanitize_ifr_meta(meta: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drop bulky raw objects and convert tensors to Python lists for JSON."""

    if meta is None:
        return None

    cleaned: Dict[str, Any] = {}
    for key, value in meta.items():
        if key == "raw":
            continue
        cleaned[key] = tensor_to_list(value)
    return cleaned


def build_sentence_context(tokenizer: Any, prompt_tokens: Sequence[str], generation_tokens: Sequence[str]) -> SentenceContext:
    """Create sentence segmentation and masks matching tokenized prompt/generation."""

    prompt_text = "".join(prompt_tokens)
    generation_text = "".join(generation_tokens)

    prompt_sentences = create_sentences(prompt_text, tokenizer)
    generation_sentences = create_sentences(generation_text, tokenizer)
    all_sentences = prompt_sentences + generation_sentences

    token_stream = list(prompt_tokens) + list(generation_tokens)
    mask = create_sentence_masks(token_stream, all_sentences)  # [sentences, tokens]

    return SentenceContext(
        prompt_sentences=prompt_sentences,
        generation_sentences=generation_sentences,
        all_sentences=all_sentences,
        sentence_mask=mask,
    )


def vector_to_sentence_scores(
    vector: Sequence[float],
    context: SentenceContext,
    *,
    transform: str = "positive",
) -> Tuple[List[float], List[float], float]:
    """Aggregate a token-level vector into per-sentence scores."""

    vec = _postprocess_scores(torch.as_tensor(vector, dtype=torch.float32), transform)

    mask = context.sentence_mask.to(dtype=torch.float32)
    scores = torch.matmul(mask, vec)
    total = float(scores.sum().item())
    if total > 0:
        norm = scores / (total + 1e-12)
    else:
        norm = torch.zeros_like(scores)

    return scores.tolist(), norm.tolist(), total


def topk_sentences(scores: Sequence[float], sentences: Sequence[str], k: int = 5) -> List[Tuple[int, float, str]]:
    paired = [(i, float(scores[i]), sentences[i]) for i in range(min(len(scores), len(sentences)))]
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired[:k]


def package_hops(
    hop_vectors: Iterable[Sequence[float]],
    context: SentenceContext,
    topk: int = 5,
    *,
    transform: str = "positive",
) -> List[Dict[str, Any]]:
    """Convert per-hop token vectors into token + sentence-level records."""

    packaged: List[Dict[str, Any]] = []
    for hop_idx, vec in enumerate(hop_vectors):
        vec_tensor_raw = torch.as_tensor(vec, dtype=torch.float32)
        vec_tensor = _postprocess_scores(vec_tensor_raw, transform)
        token_scores = vec_tensor.tolist()
        token_max = float(vec_tensor.max().item()) if vec_tensor.numel() > 0 else 0.0

        raw, norm, total = vector_to_sentence_scores(vec, context, transform=transform)
        packaged.append(
            {
                "hop": hop_idx,
                "token_scores": token_scores,
                "token_score_max": token_max,
                "token_stats_raw": vector_stats(torch.nan_to_num(vec_tensor_raw, nan=0.0)),
                "token_stats_post": vector_stats(vec_tensor),
                "sentence_scores_raw": raw,
                "sentence_scores_norm": norm,
                "total_mass": total,
                "top_sentences": [
                    {"idx": i, "score": s, "sentence": sent}
                    for i, s, sent in topk_sentences(raw, context.all_sentences, k=topk)
                ],
            }
        )
    return packaged


def package_token_hops(
    hop_vectors: Iterable[Sequence[float]],
    *,
    transform: str = "positive",
) -> List[Dict[str, Any]]:
    """Package per-hop token vectors without sentence aggregation."""

    packaged: List[Dict[str, Any]] = []
    for hop_idx, vec in enumerate(hop_vectors):
        vec_tensor_raw = torch.as_tensor(vec, dtype=torch.float32)
        vec_tensor = _postprocess_scores(vec_tensor_raw, transform)
        token_scores = vec_tensor.tolist()
        token_max = float(vec_tensor.max().item()) if vec_tensor.numel() > 0 else 0.0
        total = float(vec_tensor.sum().item())
        packaged.append(
            {
                "hop": hop_idx,
                "token_scores": token_scores,
                "token_score_max": token_max,
                "token_stats_raw": vector_stats(torch.nan_to_num(vec_tensor_raw, nan=0.0)),
                "token_stats_post": vector_stats(vec_tensor),
                "total_mass": total,
            }
        )
    return packaged
