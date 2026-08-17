# AgentDojo model-only natural-search pilot

## Decision

Reasoning reranking is now the preferred pilot strategy, but not yet a
population-level conclusion. On the same four action-target cases (one from
each suite), adaptive best-of-4 confirmed 0/4 cases and reasoning reranking
confirmed 1/4. Every token in every selected reranking completion was sampled
by the model; the successful target turn had a 411-token model-produced reasoning
bridge and zero reference-token writes.

The sample is too small for a stable coverage estimate. The evidence supports
an expanded matched pilot, not a full-inventory run or a claim that reranking
is universally better.

## Implemented primary strategies

`adaptive_best_of_n` samples complete agent trajectories and stops at the first
official-validator confirmation.

`reasoning_rerank` performs the following operation at a turn where the target
context is visible:

1. Independently sample `M=4` reasoning trajectories from the same prompt to a
   complete closing boundary.
2. Score the reference action as hidden conditional log-probability after each
   candidate. Scored tokens are not copied to the completion.
3. Select one complete candidate without splicing branches.
4. Sample the action naturally from the exact prompt-plus-reasoning token IDs.

Candidate reasoning used temperature 1.2 to obtain diversity; the final action
used the matched baseline settings (temperature 0.7, top-p 0.95, top-k 20).
This candidate-only temperature change is an explicit search-distribution
shift and is reported rather than treated as an ordinary natural sample.

## Matched four-case result

| Metric | Adaptive best-of-4 | Reasoning rerank |
|---|---:|---:|
| Validator-confirmed coverage | 0/4 (0%) | 1/4 (25%) |
| Selected completion model-token fraction | 100% | 100% |
| Reference writes | 0 | 0 |
| Mean generation requests / case | 9.0 | 11.5 |
| Mean sampled tokens / case | 2087.8 | 1568.0 |
| Mean scored tokens / case | 0 | 204.3 |
| Mean generation time / case | 124.7 s | 78.0 s |
| Mean distinct-reasoning fraction | n/a | 100% |
| Non-first reasoning selection rate | n/a | 80% |
| Mean conditional-log-prob gain over candidate 0 | n/a | 0.055 |
| Non-model bridge count | 0 | 0 |

The single successful reranking case used 19 generation/scoring requests and
2317 sampled tokens across its multi-turn run. Its target turn contained 480
completion tokens, including the 411-token bridge. Search costs are not the
length of the selected completion.

One case in each model-only strategy had no complete reasoning boundary at the
relevant turn. It is counted separately as a missing boundary, not as a
non-model bridge.

## Diagnostic controls

The same manifest was also run with full reference-writing controls. These
numbers are upper-bound diagnostics and are not eligible for the primary
FlashTrace sample set.

| Metric | Prefix control | Sparse-correction control |
|---|---:|---:|
| Validator-confirmed coverage | 3/4 (75%) | 3/4 (75%) |
| Cases with fully model-generated completion | 1/4 | 1/4 |
| Mean reference tokens at confirmation | 48.7 | 4.3 |
| Mean reference fraction at confirmation | 100% | 9.1% |
| Mean requests / successful case | 5.0 | 53.7 |

Sparse correction shows that a small number of next-token deviations can
control the result, but it remains a reference-writing intervention and is not
the main method.

## Failed variants and diagnosis

- Short natural rollouts produced no candidate-ranking signal on this pilot.
- Conditional scoring at reasoning temperature 0.7 also produced no ranking
  signal because all four reasoning candidates were identical.
- An earlier seed schedule additionally produced repeated candidates; it was
  replaced by hash-derived 32-bit seeds and candidate diversity is now audited.
- Raising only the reasoning-candidate temperature to 1.2 produced distinct
  candidates and a nonzero matched result while leaving final action sampling
  unchanged.

Particle/SMC search was not added in this round. It was the conditional next
step if coherent reasoning reranking failed to improve on best-of-N; the
corrected reranking pilot obtained a nonzero matched improvement. An expanded
pilot should determine whether that gain persists before adding a more complex
search method.

## Artifacts and next gate

The text-free aggregate comparison is
`results/natural_search_pilot_v4/comparison_with_controls.redacted.json`.
Raw trajectories remain audit artifacts and must not be rendered in routine
reports.

The next gate is an 8–16 case matched pilot with fixed settings and suite-level
stratification. Keep best-of-N as the fallback main strategy if reranking does
not retain a coverage advantage or if its per-success cost becomes unfavorable.
