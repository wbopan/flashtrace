# Multimodal attribution method smoke results

Run date: 2026-07-16 (Asia/Hong_Kong)

All methods used the same ten frozen responses selected in
`SMOKE_RESULTS.md`: five VQA-X and five A-OKVQA examples. The model was
`Qwen/Qwen3-VL-8B-Instruct` at revision
`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`, in bfloat16 on one NVIDIA RTX
PRO 5000 72GB Blackwell. Images were resized to at most 448 pixels.

The FlashTrace/IFR implementation came from FlashTrace commit
`113335a2288bb1d387ea3bebc9f5a6bed620c3be`. The official TAM source was
pinned at commit `7c7df1d3df418bb9467ca68ff2f00ff1f26a6f4c`; the adapter changes only
Qwen3-VL token/grid discovery and response plumbing and calls the upstream
`TAM()` function unchanged.

## Protocol

Each method attributed the same output span (the reasoning and final answer,
excluding EOS) to visual tokens. Native visual maps were bilinearly resampled
to the 4x4 blur-region LOO grid. The comparison metrics are:

- Spearman: rank correlation with the 16 LOO region scores;
- Recall@25: fraction of positive LOO mass captured by the method's top four
  cells (random selection has expectation 0.25);
- Jaccard@25: overlap of the method and LOO top-four cell sets;
- Top hit: whether both methods select the same highest-scoring cell.

LOO is a perturbation sanity reference, not localization ground truth. Its
self-alignment values are therefore 1 by definition and should not be read as
a model-quality result.

## Results

### VQA-X (5 samples)

| Method | Success | Median time (s) | Spearman | Recall@25 | Jaccard@25 | Top hit | Peak VRAM (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual LOO reference | 5/5 | 2.687 | 1.000 | 1.000 | 1.000 | 5/5 | n/a |
| IFR-span | 5/5 | 0.278 | 0.013 | 0.308 | 0.152 | 2/5 | 16.951 |
| FlashTrace | 5/5 | 0.365 | 0.001 | 0.308 | 0.152 | 2/5 | 16.952 |
| TAM | 5/5 | 1.504 | 0.074 | 0.356 | 0.162 | 0/5 | 16.686 |

### A-OKVQA (5 samples)

| Method | Success | Median time (s) | Spearman | Recall@25 | Jaccard@25 | Top hit | Peak VRAM (GB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Visual LOO reference | 5/5 | 3.233 | 1.000 | 1.000 | 1.000 | 5/5 | n/a |
| IFR-span | 5/5 | 0.255 | 0.172 | 0.522 | 0.282 | 2/5 | 17.065 |
| FlashTrace | 5/5 | 0.351 | 0.179 | 0.522 | 0.282 | 2/5 | 17.066 |
| TAM | 5/5 | 1.722 | 0.079 | 0.389 | 0.253 | 1/5 | 16.701 |

Across all ten examples, median steady-state times were 0.265 seconds for
IFR-span, 0.365 seconds for FlashTrace, and 1.570 seconds for TAM. Maximum
incremental peak allocations were 0.704, 0.705, and 0.251 GB respectively.
The first IFR call took 70.4 seconds due to a one-time cold-start outlier, so
median rather than mean runtime is reported. Model loading is excluded.

TAM reproduced every frozen greedy response exactly. IFR-span and FlashTrace
produced non-identical raw maps on all ten examples, with maximum per-sample
4x4 score differences ranging from 0.0038 to 0.0321. Nevertheless their
top-four cells were identical on this smoke set. This is expected to be a weak
discriminator because the run deliberately used one output span and
`hops=1`; the paper-scale comparison must include multi-segment reasoning and
multi-hop FlashTrace.

A-OKVQA aligned more strongly with the LOO reference than VQA-X. Manual
inspection suggests the selected A-OKVQA questions more often name a concrete
visual object, whereas negative VQA-X questions such as whether it is raining
produce diffuse perturbation maps. With only five samples per dataset, these
numbers validate execution and metric plumbing, not statistical superiority.

## Status after this historical run

The tables above are the validated 2026-07-16 smoke result, not the final
baseline table. The implementation was subsequently expanded; no numbers for
the new methods should be inferred from the old run.

| Method | Current role | Implementation status |
| --- | --- | --- |
| Visual LOO | perturbation reference | implemented and validated above |
| IFR-tokenwise | paper-side IFR comparator | implemented via the per-output-token IFR matrix |
| Attention rollout | attention-only baseline | implemented on frozen target rows |
| Grad×Attention | gradient-weighted attention baseline | implemented with positive per-layer Grad×Attention rollout |
| Visual IG | input-gradient baseline | implemented on processed image patches with a zero baseline |
| FlashTrace | proposed method | implemented |
| TAM | vision-native external baseline | implemented and validated above |
| AttnLRP-Qwen3VL | relevance-propagation baseline | implemented on fused visual tokens plus DeepStack paths |
| IFR-span | no-recursion ablation | implemented, but not a main baseline |

The AttnLRP adapter deliberately stops at Qwen3-VL's fused visual-token
boundary. It applies AttnLRP inside the language decoder and includes every
DeepStack injection, but it does not propagate through the vision encoder.
This is a valid visual-token comparison and must be named explicitly rather
than described as end-to-end pixel-level AttnLRP.

CLP and REAGENT remain text-replacement perturbations; their clean visual
counterpart is covered by region LOO. EAGLE is not part of the selected
baseline set.

Machine-readable results for the historical four-method run are under
`data/multimodal_methods_final/results.jsonl`, with aggregates in
`data/multimodal_methods_final/summary.json` and 30 visual overlays under
`data/multimodal_methods_final/overlays/`.
