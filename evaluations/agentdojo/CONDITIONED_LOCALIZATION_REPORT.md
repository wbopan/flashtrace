# AgentDojo conditioned-localization report

Date: 2026-08-11

## Result

The original single-trajectory run produced 30 localizable records. The new
conditioned pipeline produces 452 validator-confirmed records from the same
frozen run, a 15.1x increase.

| Funnel stage | Count | Full inventory | Action-target subset |
|---|---:|---:|---:|
| Full attacked inventory | 949 | 100.0% | — |
| Concrete action target | 609 | 64.2% | 100.0% |
| Original natural localizable output | 30 | 3.2% | 4.9% |
| Frozen prompt contains the annotated span | 459 | 48.4% | 75.4% |
| Conditioned suffix passes the official validator | 452 | 47.6% | 74.2% |
| Above, with an autonomous reasoning prefix | 426 | 44.9% | 70.0% |
| Above, using a minimal-prefix fallback | 26 | 2.7% | 4.3% |

There were no record-processing failures. Seven of the 459 candidate records
did not pass the official validator and are retained as ineligible audit rows.

## Protocol separation

The following measurements must remain separate:

1. Natural single-trajectory outcomes use the complete 949-case inventory.
2. Natural best-of-N reports both first-attempt and selected-trajectory rates.
3. Conditioned localization is a given-output analysis, not a natural outcome
   rate. It reports both 452/609 and 452/949.
4. Multi-hop analysis should use the 426 records with an autonomous reasoning
   prefix. The 26 minimal-prefix records are a direct-action diagnostic.

The 8-case best-of-4 pilot used 29 trajectories and did not add a success over
the first-attempt set (1/8 before and after selection). It is statistically too
small to estimate the full best-of-N curve, but it shows that rejection
sampling is substantially more expensive than conditioning for this model.

The 8-case hybrid smoke produced 6/8 validator-confirmed records, and all six
retained an autonomous reasoning prefix. A one-record attribution smoke passed
all configured methods and preserved the conditioned provenance in the
redacted export.

## Reproducible artifacts

- Aggregate coverage and validator replay:
  `results/conditioned/qwen36_27b_formal1046_conditioned/coverage.redacted.json`
- Conditioned trajectories (sensitive raw audit artifact):
  `results/conditioned/qwen36_27b_formal1046_conditioned/trajectories.jsonl`
- Safe trajectory audit:
  `results/conditioned/qwen36_27b_formal1046_conditioned/trajectories.redacted.jsonl`
- Attribution smoke:
  `results/conditioned/qwen36_27b_formal1046_conditioned/attribution_smoke1/`
- Natural best-of-4 pilot:
  `results/conditioned/qwen36_27b_natural_bestof4_pilot8/`
- Hybrid online pilot:
  `results/conditioned/qwen36_27b_hybrid_pilot8/`

Generate the conditioned audit set from an existing frozen run with:

```bash
python -m evaluations.agentdojo.analyze_conditioning \
  --generation-jsonl <frozen-trajectories.jsonl> \
  --output <coverage.redacted.json> \
  --replay-validators \
  --conditioned-jsonl <conditioned-trajectories.jsonl>
```

The local generation runner also supports online conditioned generation and
natural best-of-N; see `README.md` for the exact options and reporting rules.
