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


def vector_to_sentence_scores(vector: Sequence[float], context: SentenceContext) -> Tuple[List[float], List[float], float]:
    """Aggregate a token-level vector into per-sentence scores."""

    vec = torch.as_tensor(vector, dtype=torch.float32)
    vec = torch.nan_to_num(vec, nan=0.0)
    vec = torch.clamp(vec, min=0.0)

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
) -> List[Dict[str, Any]]:
    """Convert per-hop token vectors into token + sentence-level records."""

    packaged: List[Dict[str, Any]] = []
    for hop_idx, vec in enumerate(hop_vectors):
        vec_tensor = torch.as_tensor(vec, dtype=torch.float32)
        vec_tensor = torch.nan_to_num(vec_tensor, nan=0.0).clamp(min=0.0)
        token_scores = vec_tensor.tolist()
        token_max = float(vec_tensor.max().item()) if vec_tensor.numel() > 0 else 0.0

        raw, norm, total = vector_to_sentence_scores(vec, context)
        packaged.append(
            {
                "hop": hop_idx,
                "token_scores": token_scores,
                "token_score_max": token_max,
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
