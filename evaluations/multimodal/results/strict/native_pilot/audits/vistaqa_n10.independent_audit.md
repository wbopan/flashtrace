# Independent audit: VistaQA native n=10 pilot

## Scope

I independently inspected all ten input images and the arrays referenced by
`evaluation.EVIDENCE_MASKS`, and compared whole model `OUTPUT` against the native
`REFERENCE_OUTPUT`. I did not treat function roles, dataset rationales, or error-tail
text as valid `THINKING`. The ten `.npy` masks were also verified pixel-for-pixel
against the union of the official COCO-RLE annotations.

The companion JSONL is deliberately **judgment-only**: it contains `sample_id` plus
human generation, THINKING, mask-quality, eligibility judgments, and audit notes. It
does not duplicate dataset inputs, reference answers, model outputs, or image fields.

## Main result

- Only 4/10 samples produced a valid, terminated THINKING plus a separate OUTPUT.
- The other 6/10 hit the 1024-token limit without `</think>` and produced no OUTPUT.
- Of the four completed outputs, 2 are semantically correct and stable, and 2 are
  incorrect. Human semantic correctness is therefore 2/4 among completed outputs,
  or 2/10 over the attempted pilot.
- The current automatic evaluator labels all four completed outputs incorrect. Two
  of those labels are false negatives caused by overly strict normalized-string
  equality:
  - `vistaqa-0656`: `slow down` fully answers the requested maneuver.
  - `vistaqa-0753`: naming the green object as farther fully answers the explicit
    red-versus-green comparison.
- The two genuine model errors are:
  - `vistaqa-0469`: plate OCR is wrong (`G8G 9998 B` vs `GBG 9938 B`), while `70`
    is correct.
  - `vistaqa-0999`: chooses the visible red cup instead of the native white bottle
    with orange lid.

## Per-sample findings

| Sample | Generation | Whole OUTPUT | Stability | THINKING | Official mask audit |
|---|---|---|---|---|---|
| 0356 | No terminator, 1024 tokens | Not produced | N/A | Strong answer-flipping loop | Red curve localized, but blue/green comparison evidence omitted |
| 0427 | No terminator, 1024 tokens | Not produced | N/A | Unfinished/backtracking | Tiny 81-pixel contact mask is tightly aligned with the asked relationship |
| 0439 | No terminator, 1024 tokens | Not produced | N/A | Restarts predator enumeration | Three answer organisms localized; required arrows/predators omitted |
| 0469 | Complete | Incorrect plate; `70` correct | Stable | Complete, no material loop | `70` mask good; plate mask is contaminated by a long unrelated road marking |
| 0569 | No terminator, 1024 tokens | Not produced | N/A | Strong hair-color loop | Hair localized; phone/row disambiguation evidence omitted |
| 0656 | Complete | **Semantically correct** | Stable | Complete, concise | Both warning and SLOW signs are appropriately masked |
| 0753 | Complete | **Semantically correct** | Stable | Complete but strongly looping (599 tokens) | Only green object masked; red object and arm base omitted |
| 0979 | No terminator, 1024 tokens | Not produced | N/A | Restarts counting procedure | All 13 target windows masked; balusters/excluded candidates omitted |
| 0999 | Complete | Incorrect: red cup | Stable | Complete, moderately repetitive | Native mask correctly targets the bottle |
| 1085 | No terminator, 1024 tokens | Not produced | N/A | Unfinished with reconfirmation | Four cards masked; last-row context is unmasked |

## Mask implications

The official masks are exact native annotations, but their semantics are not uniform:

1. Some are good direct evidence masks (`0427`, `0656`, `0999`).
2. Several localize only the answer entity, not the complete evidence needed for a
   uniqueness, comparison, or relational claim (`0356`, `0439`, `0569`, `0753`).
3. Counting masks usually cover positive targets but not the selection criterion or
   negative candidates (`0979`, `1085`).
4. `0469` contains a clear native annotation defect: the license-plate mask includes
   a long road/lane marking, so localization scores on this sample are misleading.

Consequently, these masks should be described as **answer/support regions**, not
universally as complete reasoning evidence. A clean main localization table should
either exclude `0469` and flag partial-evidence samples, or report results stratified
by mask semantics.

## THINKING behavior

The dominant pilot failure is generation control rather than answer scoring: 60% of
samples never close THINKING at 1024 tokens. Clear looping/restart behavior appears
in `0356`, `0569`, and `0979`; `0427`, `0439`, and `1085` are also unfinished but
the available tails provide less evidence about the full trajectory. Among completed
records, `0753` has a 599-token loop for a short comparison, and `0999` repeatedly
reconfirms a wrong distractor. `0469` and `0656` are the only cleanly completed,
non-looping generations.

## Recommendation

- Replace exact normalized-string equality with a semantic correctness rule suitable
  for whole OUTPUT; otherwise valid concise answers are systematically rejected.
- Before attribution, reduce the 1024-token non-termination rate through generation
  control or a larger audited cap. Do not silently accept truncated reasoning.
- Keep generation eligibility separate from localization-GT quality. With the current
  records, only `0656` and `0753` pass human semantic correctness + stability + image
  support, and `0753` still has incomplete comparison evidence in its native mask.
