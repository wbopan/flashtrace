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
