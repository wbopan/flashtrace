# AgentDojo two-axis realization report

> Historical diagnostic-control report. Prefix and sparse reference-writing
> protocols are no longer primary realization strategies. See
> `NATURAL_SEARCH_REPORT.md` for the model-only search implementation and pilot.

## Implemented protocol

- Natural axis: cumulative rollout budgets, with one greedy attempt followed by
  seeded sampling attempts.
- Token axis: `1,2,4,8,16,32,64,full` target-action prefix tokens, stopping at
  the first official-validator-confirmed stage.
- Every token-budget stage deterministically replays the same conditionable
  natural prefix. The seed, decoding parameters, environment, and prefix are
  fixed; only the token budget changes.
- A token stage is skipped when no natural attempt reaches the target context
  with a complete reasoning boundary. No minimal reasoning fallback exists.
- Records include stage history, conditioning-source attempt, forced-token
  count and fraction, reasoning-token count, skip reason, and per-token
  model/reference provenance.
- Attribution rejects a record if any FlashTrace bridge token is not marked as
  model-produced or if the provenance mask does not align with tokenizer output.

## Validation

The AgentDojo test suite passes with 30 tests. Ruff and shell syntax checks also
pass.

A matched-model single-case ladder smoke test produced the following aggregate
result:

| Quantity | Result |
|---|---:|
| Natural greedy | not confirmed |
| Prefix budgets 1, 2, 4, 8, 16, 32 | not confirmed |
| First confirmed stage | 64 tokens |
| Full target-action length | 89 tokens |
| Reference-token fraction at first confirmation | 71.9% |
| Model-produced reasoning bridge | 352 tokens |
| Short-bridge count | 0 |
| Non-model bridge count | 0 |

This smoke test demonstrates a real intermediate point: confirmation occurred
before the full target action was supplied, while the entire reasoning bridge
remained model-produced.

A deliberately truncated four-case smoke test also verified the negative path:
when reasoning or target context is unavailable, token stages are skipped and
no fallback bridge is synthesized.

## Remaining evaluation

The single-case result validates mechanics, not a population-level lift. The
next run should use the fixed pilot script, report the complete stage funnel,
and keep natural and token-axis localization metrics stratified. Only after that
pilot should the full action-target inventory be scheduled.
