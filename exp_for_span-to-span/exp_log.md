# Experiment Log — 2026-08-12

## Goal

Prototype a mechanically grounded multi-hop reasoning circuit pipeline on top of FlashTrace.

Target pipeline:

FlashTrace span attribution
→ multi-hop circuit extraction
→ semantic redundancy removal
→ semantic consistency filtering
→ MCTS candidate expansion

The main design goal is to preserve attribution as the source of circuit structure while using lightweight semantic methods only to refine the extracted candidates.

---

# 1. FlashTrace Core: Span-to-Span Attribution

Extended the FlashTrace workflow from output-focused attribution to explicit span-to-span attribution across a reasoning trace.

For a generated trace:

Q, S1, S2, ..., S24, O

the experiment computes attribution for every causally valid earlier-span → later-span pair.

Matrix convention:

- row = target span
- column = source span
- matrix[target, source] = normalized attribution from source → target

Because the trace is autoregressive, only earlier → later relations are valid.

This produces an attribution-weighted causal reasoning graph rather than only measuring which reasoning spans directly affect the final output.

Main output:

`exp_for_span-to-span/results/case1_span_matrix.json`

Main script:

`exp_for_span-to-span/run_case1_span_matrix.py`

---

# 2. Multi-Hop Circuit Extraction

Built explicit Q→O reasoning circuits from the span-to-span attribution matrix.

For every target span, normalized incoming attribution is interpreted as a backward transition distribution.

Starting from O, attribution-conditioned flow is propagated backward toward Q.

The resulting graph is decomposed into explicit routes using greedy widest-path residual decomposition:

1. Find the widest remaining Q→O path.
2. Route flow is the bottleneck residual flow on that path.
3. Subtract that flow from all edges on the route.
4. Repeat until the Q-bound residual flow is exhausted.

This produces explicit candidate circuits rather than giving only a dense attribution DAG to MCTS.

### Case 1

- total decomposed routes: 300
- direct Q→O route: 1
- single-hop Q→Sj→O routes: 21
- multi-hop circuit candidates: 278
- multi-hop share of decomposed flow: ~0.436

Single-hop routes are valid direct-attribution effects, but the downstream circuit candidate set focuses on multi-hop routes because the target is recursive / long-horizon reasoning flow.

No attribution threshold, cumulative-flow elbow, or arbitrary top-K is applied during circuit generation.

The eventual top-K constraint should come from the MCTS maximum branching factor.

Main output:

`exp_for_span-to-span/results/case1_flow_routes.json`

Main script:

`exp_for_span-to-span/extract_candidate_routes.py`

---

# 3. Qualitative Circuit Check

The highest-flow circuits are qualitatively coherent.

Examples:

### M1
`Q → S1 → S6 → S7 → O`

Situation identification
→ validate feelings
→ acknowledge bullying and its effect on self-esteem

### M2
`Q → S4 → S5 → S6 → O`

Parents' conflicting stance
→ negative-thought / therapy barrier
→ emotional validation

### M3
`Q → S7 → S8 → O`

Bullying / self-esteem pain
→ possible isolation
→ emphasize that the user is not alone

### M4
`Q → S2 → S10 → S11 → O`

Repeated dieting
→ possible restriction/binge-cycle interpretation
→ weight loss is not the solution / self-worth is not appearance

### M5
`Q → S6 → S8 → S9 → S10 → O`

Validation
→ isolation/support
→ body-image issue
→ dieting concern

These should be interpreted as reasoning-flow / response-planning circuits rather than strict deductive logical proofs.

The important observation is that the strongest attribution-flow routes are not arbitrary span combinations; they correspond to interpretable multi-step reasoning trajectories.

---

# 4. Semantic Redundancy

## Motivation

The 278 extracted circuits contain routes that perform substantially overlapping reasoning.

Keeping all of these candidates would cause downstream MCTS to spend branches exploring duplicated reasoning trajectories.

Therefore the goal of semantic redundancy removal is:

> collapse redundant reasoning circuits into a smaller set of representative circuits while retaining the mechanically stronger representative.

"Merge" here means candidate consolidation.

Attribution flows are NOT summed.

For a redundant group, the highest-flow route remains as the representative and the other route is removed as an independent MCTS candidate.

---

## Initial semantic redundancy

Used:

`sentence-transformers/all-MiniLM-L6-v2`

Each complete ordered route is embedded and compared using cosine similarity.

Initial semantic-only filtering produced approximately:

278 routes
→ 94 representatives

However, inspection showed that routes sharing only a broad semantic topic could sometimes be merged despite having different reasoning structures.

---

## Adding mechanistic structure

Added a structural prerequisite:

Two routes must share at least one directed reasoning edge:

`Si → Sj`

before SentenceTransformer similarity can classify them as redundant.

With this rule:

278 routes
→ 123 representatives

155 candidate routes were removed as redundant.

Runtime remained low (~2 seconds).

---

## TP / FP Redundancy Audit

To verify whether semantic redundancy removal actually helps rather than simply reducing the number of routes, every removed pair was exported for inspection.

Audit artifact:

`results/case1_redundancy_tp_fp_audit.csv`

Definitions:

- TP: route was correctly removed as genuinely redundant
- FP: route contained materially different reasoning and should have remained

A conservative manual audit of all 155 removed pairs found:

- TP: 71
- FP: 84
- broad-rule precision: ~45.8%

This shows two things.

First, semantic redundancy removal is genuinely useful:

71 actual duplicated / substantially redundant candidate circuits were identified and could be removed from the downstream search space.

This can reduce duplicated MCTS exploration.

Second, the current broad criterion is too aggressive:

requiring only one shared directed edge plus high semantic similarity still removes too many distinct circuits.

Therefore the conclusion is NOT to remove semantic redundancy from the pipeline.

Instead:

> semantic redundancy is useful, but the hard-removal criterion should be high precision.

---

## High-precision structural result

The audit revealed a particularly clean subset.

When one route was an ordered structural subsequence of the other and the routes were semantically similar:

- audited pairs: 22
- TP: 22
- FP: 0

This suggests a conservative prototype criterion:

ordered structural containment
+
high route-level semantic similarity
→ remove weaker duplicate
→ retain higher-flow representative

This is preferable to aggressively collapsing all semantically similar routes.

The redundancy contribution can therefore be described as:

> Attribution-aware structural containment identifies candidate duplicate circuits, while semantic similarity verifies that the contained and extended routes represent the same reasoning trajectory.

---

# 5. Semantic Consistency

## Motivation

Redundancy compares different routes.

Consistency instead asks whether the reasoning relations inside an individual route are semantically compatible.

The relation to inspect is mechanically selected rather than chosen by a semantic model.

For each current reasoning span:

parent =
argmax attribution[current, previous_span]

over previous spans contained in the same route.

Therefore:

- attribution determines which dependency matters
- the semantic module only validates that relation

---

## NLI experiments

Several pretrained NLI formulations were tested.

### Direct NLI

Tested:

`NLI(parent, current)`

Problem:

normal reasoning transitions were frequently classified as contradictions.

For example:

body-image discussion
→ next discuss bullying

was sometimes classified as contradictory even though this is simply a topic transition.

---

### Dual-model agreement

Required two independent pretrained NLI models to agree on contradiction.

This reduced some false positives but did not solve the underlying problem.

Different NLI models often reproduced similar errors.

---

### Stronger NLI model

Tested:

`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`

The larger and more diverse NLI model reduced the number of rejected routes but still produced false contradictions on valid reasoning transitions.

---

### Bidirectional mutual contradiction

Tested a conservative condition:

NLI(parent, current) = contradiction
AND
NLI(current, parent) = contradiction
→ hard drop

This substantially reduced false-positive removals compared with direct NLI, but remaining rejected relations still included valid topic / reasoning transitions.

---

### Compatibility-vs-contradiction formulation

Also tested explicit hypotheses:

- the two reasoning steps are compatible
- the two reasoning steps contradict each other

This was not reliable because both entailment scores were often very low, causing small numerical differences to determine a hard route removal.

---

### Lightweight aspect/polarity experiment

Finally tested an NLI-free rule:

attribution-selected relation
→ shared semantic aspect
→ negation / opposite-polarity check
→ hard drop

This was extremely fast, but lexical polarity was not equivalent to proposition-level contradiction.

For example:

"parents do not want therapy"

and

"therapy resistance is a barrier"

contain different local negation patterns but are semantically consistent.

Therefore this rule also generated false positives.

---

# 6. Consistency Finding

The semantic consistency experiments exposed an important limitation:

> Generic sentence-level contradiction detection is not equivalent to reasoning-circuit consistency.

Reasoning traces contain:

- elaboration
- topic transitions
- planning transitions
- complementary recommendations
- speculative inference
- actual contradiction

Standard NLI and simple polarity rules do not reliably separate these cases.

Therefore consistency remains an open component of the prototype rather than forcing an unreliable hard-filter criterion.

The useful part already established is the attribution-guided formulation:

> mechanical attribution identifies the relation to validate, avoiding all-pairs semantic comparison.

Future work can replace the semantic validator without changing the mechanically grounded circuit extraction architecture.

---

# 7. Current Prototype Pipeline

Current working prototype:

1. FlashTrace span-to-span attribution
2. normalized causal attribution matrix
3. answer-conditioned flow propagation
4. explicit multi-hop circuit decomposition
5. attribution-flow ranking
6. semantic redundancy identification
7. redundancy TP/FP audit
8. attribution-guided semantic consistency experiments
9. MCTS-ready circuit candidates

The strongest extracted circuits are qualitatively coherent and are sufficient for demonstrating the circuit-generation component of the prototype.

---

# 8. MCTS Integration

No arbitrary global top-K is applied during circuit generation.

Instead, circuit selection should naturally connect to the MCTS search budget.

At an MCTS expansion:

candidate circuits
→ semantic refinement
→ sort by attribution flow
→ expand up to `max_branching_factor = K`

Therefore K is not an attribution threshold.

It is the configured MCTS branching budget.

This allows attribution flow to remain the mechanical ranking signal while preventing uncontrolled search branching.

---

# Main Experiment Files

### FlashTrace / span attribution

`exp_for_span-to-span/run_case1_span_matrix.py`

`results/case1_span_matrix.json`

### Circuit extraction

`exp_for_span-to-span/extract_candidate_routes.py`

`results/case1_flow_routes.json`

### Semantic refinement

`exp_for_span-to-span/filter_route_redundancy.py`

`results/case1_redundancy_filtered_routes.json`

`results/case1_redundancy_tp_fp_audit.csv`

---

# Takeaways

1. FlashTrace attribution can be extended into explicit span-to-span reasoning dependencies.

2. Attribution flow can be decomposed into interpretable multi-hop Q→O circuits.

3. The highest-flow circuits correspond to coherent reasoning / response-planning trajectories.

4. Semantic redundancy removal is useful because real duplicated reasoning circuits exist and removing them can reduce duplicated downstream search.

5. Semantic similarity alone is too aggressive; mechanistic structure is necessary for high-precision redundancy removal.

6. Manual TP/FP auditing identified a conservative structural-containment subset with substantially cleaner redundancy decisions.

7. Generic NLI contradiction and lexical polarity are not sufficiently reliable representations of reasoning-circuit consistency.

8. Attribution-guided semantic validation remains promising because attribution determines which relation should be checked.

9. MCTS branching factor provides a natural downstream top-K constraint without introducing an arbitrary attribution cutoff.