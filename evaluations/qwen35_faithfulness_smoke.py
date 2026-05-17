"""Faithfulness smoke evaluation for FlashTrace on Qwen3.5 (hybrid) models.

Compares, on the same samples and the same token grid:

* FlashTrace IFR attribution
* a coarse segment-level perturbation baseline

via a guided-deletion faithfulness AUC (lower AUC == more faithful: deleting the
top-ranked prompt tokens makes the generated answer's log-probability collapse
faster).

Used both as an importable library (unit-tested on a tiny synthetic model) and
as a script for the real Qwen3.5-9B vs Qwen3-8B comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from flashtrace.tracer import FlashTrace


# --------------------------------------------------------------------------
# Faithfulness primitives (token-grid consistent)
# --------------------------------------------------------------------------

def _answer_logprob(model, input_ids: torch.Tensor, prompt_len: int) -> float:
    """Total log-probability of the answer tokens ``input_ids[prompt_len:]``."""

    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    target = input_ids[0, prompt_len:]
    pred = logp[0, prompt_len - 1 : -1, :].gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return float(pred.sum().item())


def _auc(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float((arr.sum() - arr[0] / 2 - arr[-1] / 2) / max(1, arr.shape[0] - 1))


def _split_groups(indices: Sequence[int], k: int) -> List[List[int]]:
    k = max(1, min(int(k), len(indices)))
    base, rem = divmod(len(indices), k)
    groups, start = [], 0
    for i in range(k):
        size = base + (1 if i < rem else 0)
        groups.append(list(indices[start : start + size]))
        start += size
    return groups


def deletion_faithfulness_auc(
    model,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    ranking: Sequence[int],
    pad_id: int,
    *,
    k: int = 10,
) -> float:
    """Guided-deletion AUC of the normalised answer log-prob. Lower == better.

    ``ranking`` lists prompt-token indices ordered by descending attribution.
    """

    prompt_len = int(prompt_ids.shape[1])
    perturbed = prompt_ids.clone()
    scores = [_answer_logprob(model, torch.cat([perturbed, gen_ids], dim=1), prompt_len)]

    for group in _split_groups([int(i) for i in ranking], k):
        for idx in group:
            perturbed[0, idx] = pad_id
        scores.append(
            _answer_logprob(model, torch.cat([perturbed, gen_ids], dim=1), prompt_len)
        )

    arr = np.asarray(scores, dtype=np.float64)
    span = abs(arr[0] - arr[-1]) or 1.0
    norm = np.clip((arr - arr[-1]) / span, 0.0, 1.0)
    norm = np.minimum.accumulate(norm)  # enforce monotone non-increasing
    return _auc(norm)


def segment_perturbation_attribution(
    model,
    prompt_ids: torch.Tensor,
    gen_ids: torch.Tensor,
    pad_id: int,
    *,
    n_segments: int = 20,
) -> torch.Tensor:
    """Coarse perturbation baseline: ablate prompt segments, attribute the drop.

    Each segment is masked in turn; the resulting drop in answer log-prob is
    spread uniformly across the segment's tokens. This mirrors the project's
    sentence-level feature-ablation baseline.
    """

    prompt_len = int(prompt_ids.shape[1])
    base = _answer_logprob(model, torch.cat([prompt_ids, gen_ids], dim=1), prompt_len)

    scores = torch.zeros(prompt_len, dtype=torch.float32)
    for segment in _split_groups(list(range(prompt_len)), min(n_segments, prompt_len)):
        perturbed = prompt_ids.clone()
        for idx in segment:
            perturbed[0, idx] = pad_id
        ablated = _answer_logprob(model, torch.cat([perturbed, gen_ids], dim=1), prompt_len)
        drop = base - ablated
        for idx in segment:
            scores[idx] = drop / max(1, len(segment))
    return scores


# --------------------------------------------------------------------------
# Per-sample evaluation
# --------------------------------------------------------------------------

@dataclass
class SampleResult:
    prompt: str
    n_prompt_tokens: int
    n_gen_tokens: int
    flashtrace_auc: float
    perturbation_auc: float


def _ranking_from_scores(scores: Sequence[float], n: int) -> List[int]:
    vec = torch.zeros(n, dtype=torch.float32)
    m = min(n, len(scores))
    vec[:m] = torch.tensor([float(s) for s in scores[:m]])
    return torch.argsort(vec, descending=True).tolist()


def evaluate_sample(
    model,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 24,
    n_segments: int = 20,
    deletion_steps: int = 10,
) -> SampleResult:
    """Evaluate FlashTrace vs the perturbation baseline on one prompt."""

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids=prompt_ids, max_new_tokens=max_new_tokens, do_sample=False
        )
    gen_ids = generated[:, prompt_ids.shape[1] :]
    target = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
    prompt_len = int(prompt_ids.shape[1])
    gen_len = int(gen_ids.shape[1])

    tracer = FlashTrace(model, tokenizer)
    ft_result = tracer.trace(
        prompt=prompt,
        target=target,
        output_span=(0, max(0, gen_len - 1)),
        method="flashtrace",
    )
    ft_ranking = _ranking_from_scores(ft_result.scores, prompt_len)

    pert_scores = segment_perturbation_attribution(
        model, prompt_ids, gen_ids, pad_id, n_segments=n_segments
    )
    pert_ranking = torch.argsort(pert_scores, descending=True).tolist()

    ft_auc = deletion_faithfulness_auc(
        model, prompt_ids, gen_ids, ft_ranking, pad_id, k=deletion_steps
    )
    pert_auc = deletion_faithfulness_auc(
        model, prompt_ids, gen_ids, pert_ranking, pad_id, k=deletion_steps
    )

    return SampleResult(
        prompt=prompt,
        n_prompt_tokens=prompt_len,
        n_gen_tokens=gen_len,
        flashtrace_auc=ft_auc,
        perturbation_auc=pert_auc,
    )


DEFAULT_SAMPLES: List[str] = [
    "The Eiffel Tower is located in Paris, the capital of France. "
    "The tower was completed in 1889. Question: In which city is the Eiffel Tower?",
    "Marie Curie was a physicist and chemist who conducted pioneering research "
    "on radioactivity. She was born in Warsaw, Poland. "
    "Question: What was Marie Curie's field of research?",
    "A train leaves the station at 9 AM travelling at 60 km/h. "
    "Another train leaves the same station at 10 AM travelling at 90 km/h. "
    "Question: Which train is faster?",
]


def evaluate_model(model_name: str, samples: Sequence[str], **kwargs) -> List[SampleResult]:
    """Load ``model_name`` and evaluate every sample."""

    from flashtrace.model_io import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(model_name, dtype="bfloat16")
    results = [evaluate_sample(model, tokenizer, p, **kwargs) for p in samples]
    del model
    torch.cuda.empty_cache()
    return results


def _summary(results: Sequence[SampleResult]) -> dict:
    return {
        "flashtrace_auc": float(np.mean([r.flashtrace_auc for r in results])),
        "perturbation_auc": float(np.mean([r.perturbation_auc for r in results])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen35", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--qwen3", default="Qwen/Qwen3-8B")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args()

    print(f"== Evaluating Qwen3.5 ({args.qwen35}) ==")
    q35 = evaluate_model(args.qwen35, DEFAULT_SAMPLES, max_new_tokens=args.max_new_tokens)
    print(f"== Evaluating Qwen3  ({args.qwen3}) ==")
    q3 = evaluate_model(args.qwen3, DEFAULT_SAMPLES, max_new_tokens=args.max_new_tokens)

    s35, s3 = _summary(q35), _summary(q3)
    print("\n--- Faithfulness AUC (lower == more faithful) ---")
    for label, summ in [("Qwen3.5-9B", s35), ("Qwen3-8B", s3)]:
        print(
            f"  {label:12s}  FlashTrace={summ['flashtrace_auc']:.4f}  "
            f"Perturbation={summ['perturbation_auc']:.4f}"
        )

    ft35, pert35, ft3 = s35["flashtrace_auc"], s35["perturbation_auc"], s3["flashtrace_auc"]
    beats_baseline = ft35 <= pert35 + 1e-6
    close_to_qwen3 = abs(ft35 - ft3) <= 0.15
    print(f"\nFlashTrace(Qwen3.5) beats perturbation baseline: {beats_baseline}")
    print(f"FlashTrace(Qwen3.5) close to FlashTrace(Qwen3-8B): {close_to_qwen3}")
    if not (beats_baseline and close_to_qwen3):
        raise SystemExit("Qwen3.5 faithfulness smoke check FAILED")
    print("\nQwen3.5 faithfulness smoke check PASSED")


if __name__ == "__main__":
    main()
