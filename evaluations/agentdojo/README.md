# AgentDojo prompt-injection localization pilot

This evaluation uses execution-grounded AgentDojo attack traces to test whether
token attribution recovers the injected span in an untrusted tool result that
causes a later malicious tool call.

The environment is isolated from the root project because AgentDojo has a large
API-provider dependency surface:

```bash
uv sync --project evaluations/agentdojo
uv run --project evaluations/agentdojo python evaluations/agentdojo/pilot.py \
  --source-root /path/to/agentdojo/runs/gpt-4o-2024-05-13
```

Omit `--source-root` to download the three official traces from the pinned
upstream commit. Use `--prepare-only` to verify hashes, render prompts, and
replay AgentDojo v1.2.2 validators without loading a white-box model.

The pilot compares:

- random expected recovery;
- IFR attribution of only the last target token;
- exhaustive IFR aggregated across the full malicious tool call;
- FlashTrace aggregated across the same full malicious tool call.

Recovery@k is the fraction of ground-truth injection tokens appearing among the
top k percent of attributed prompt tokens. AgentDojo utility and targeted attack
success are kept as separate execution metrics.

## Complete input and token-level exports

To regenerate the lossless per-sample audit reports after running the pilot:

```bash
HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo \
  python -m evaluations.agentdojo.explain \
  --model Qwen/Qwen3-4B-Thinking-2507 --device cuda:0
```

The exporter writes `results/pilot/explanations/INDEX.md` plus one Markdown,
JSON, and HTML heatmap file per case. Each self-contained case record includes:

- the exact structured AgentDojo trace messages and API tool schemas
  reconstructed from the pinned v1.2.2 suite;
- the exact Qwen chat-template prompt, token IDs, vocabulary tokens, source
  character slices, offsets, and stop-token ranking mask;
- the gold injected-tool-output character and token spans;
- the teacher-forced malicious tool-call target and appended EOS;
- span-aggregated pre-metric scores, kept-token-normalized scores, ranks, top-k
  bands, and semantic-region mass for IFR-last, exhaustive IFR-full, and
  FlashTrace-full;
- projected per-hop FlashTrace diagnostics and the complete pinned source trace.

The upstream GPT-4o tokenizer IDs are not available from AgentDojo. The reports
therefore keep the source agent's API-level logical input separate from the
exact Qwen token sequence used in the local white-box experiment.

## Autonomous end-to-end run

The teacher-forced pilot above measures localization conditioned on a recorded
malicious action. Use the E2E runner to instead let Qwen generate every
assistant turn and tool call, execute those calls in the injected AgentDojo
environment, and run the official utility and attack validators:

```bash
HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo \
  python -m evaluations.agentdojo.e2e \
  --model Qwen/Qwen3-4B-Thinking-2507 --device cuda:0 \
  --max-turns 10 --max-new-tokens 2048 --n-hops 3
```

The E2E protocol uses deterministic greedy decoding and Qwen's native tool-call
chat template. It attributes only an expected malicious action that Qwen
actually generated; safe or incomplete generations have no localization score.
The complete generated reasoning before that action is used as the FlashTrace
bridge span.

`results/e2e/qwen3_4b_thinking/RESULTS.md`, `summary.json`, and the
`*.redacted.json` files omit all prompt, injection, tool-result, and completion
text. `trajectories.jsonl` and the unsuffixed per-case JSON files are sensitive
raw audit artifacts and should not be rendered in routine reports.

## Model-only target realization

Greedy decoding follows only one trajectory and can leave too few generated
actions for a localization study. The primary protocols therefore search only
among tokens sampled by the model. Reference-token writes are diagnostic
controls and are never combined with the primary coverage rate.

The natural axis increases a cumulative sampling budget. Attempt one is greedy;
later attempts use the configured sampling parameters:

```bash
python -m evaluations.agentdojo.benchmark_local \
  --mode attacked --action-target-only \
  --natural-rollout-budgets 1,4,16 \
  --temperature 0.7 --top-p 0.95 --top-k 20 \
  --sglang-base-url http://127.0.0.1:8001 \
  --output-dir evaluations/agentdojo/results/natural_ladder
```

`adaptive_best_of_n` is the complete-trajectory baseline above. The more
targeted primary strategy samples complete reasoning candidates through the
closing boundary, scores the reference action only as hidden conditional
log-probability, chooses one coherent candidate, and then samples the final
action naturally from that exact token prefix:

```bash
python -m evaluations.agentdojo.benchmark_local \
  --mode attacked --action-target-only \
  --natural-rollout-budgets 1 \
  --reasoning-rerank-candidates 4 \
  --reasoning-score-method conditional_logprob \
  --reasoning-temperature 1.2 \
  --temperature 0.7 --top-p 0.95 --top-k 20 \
  --sglang-base-url http://127.0.0.1:8001 \
  --output-dir evaluations/agentdojo/results/reasoning_rerank
```

No scoring continuation is copied into the selected completion. The retained
reasoning is one complete, continuous model trajectory, and the final action is
an independent model continuation from that exact token prefix. Records audit
candidate diversity, selection index, score gain, candidate/final temperatures,
request and token cost, and a per-token provenance mask. A selected record is
eligible for FlashTrace only when every bridge token is marked `model`.

Two reference-writing controls remain available for diagnosis. Prefix forcing
writes the first `k` reference tokens; sparse correction writes a reference
token only at a next-token mismatch and immediately resumes model decoding:

```bash
python -m evaluations.agentdojo.benchmark_local \
  --mode attacked --action-target-only \
  --natural-rollout-budgets 1,4,16 \
  --prefix-forcing-token-budgets 1,2,4,8,16,32,64,full \
  --temperature 0.7 --top-p 0.95 --top-k 20 \
  --sglang-base-url http://127.0.0.1:8001 \
  --output-dir evaluations/agentdojo/results/prefix_control
```

```bash
python -m evaluations.agentdojo.benchmark_local \
  --mode attacked --action-target-only \
  --natural-rollout-budgets 1 \
  --sparse-correction-budgets 1,2,4,8,16,32,64,full \
  --temperature 0.7 --top-p 0.95 --top-k 20 \
  --sglang-base-url http://127.0.0.1:8001 \
  --output-dir evaluations/agentdojo/results/sparse_control
```

The former `after_injection` / minimal-reasoning path remains unavailable. A
control stage is skipped without a complete model reasoning boundary. Prefix,
sparse, and model-only results must be reported as separate strategies.

The complete v1.2.2 attacked inventory has 949 cases. `--action-target-only`
keeps the 609 cases with at least one concrete action target; the remaining
cases require a separate final-text attribution protocol. Reports expose both
the full-inventory denominator and the action-target denominator rather than
silently mixing them.

Generate an aggregate-only coverage report with no sample text:

```bash
python -m evaluations.agentdojo.diagnose_coverage \
  --generation-jsonl <trajectories.jsonl> \
  --output <coverage.redacted.json>
```

`run_agentdojo_natural_search_pilot.sh` runs matched best-of-N and reasoning
reranking pilots. The implementation audit and matched-model results are in
`NATURAL_SEARCH_REPORT.md`; `TWO_AXIS_REALIZATION_REPORT.md` is retained as the
historical diagnostic-control report.
