"""Helpers for IFR case studies (hop-wise aggregation + sanitization).

All utilities stay local to exp/case_study to avoid touching core eval code.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch


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


def package_token_hops(
    hop_vectors: Iterable[Sequence[float]],
) -> List[Dict[str, Any]]:
    """Package per-hop token vectors without sentence aggregation.

    hop_vectors are assumed to already match the experiment's configured
    postprocessing (e.g., FT-AttnLRP neg_handling/norm_mode).
    """

    packaged: List[Dict[str, Any]] = []
    for hop_idx, vec in enumerate(hop_vectors):
        vec_tensor = torch.nan_to_num(torch.as_tensor(vec, dtype=torch.float32), nan=0.0)
        token_scores = vec_tensor.tolist()
        token_max = float(vec_tensor.abs().max().item()) if vec_tensor.numel() > 0 else 0.0
        total = float(vec_tensor.sum().item())
        packaged.append(
            {
                "hop": hop_idx,
                "token_scores": token_scores,
                "token_score_max": token_max,
                "token_stats": vector_stats(vec_tensor),
                "total_mass": total,
            }
        )
    return packaged
