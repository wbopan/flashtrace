# Formal-pipeline preview: Wiki-VISA n=20 / VizWiz-LF n=20

> **Preview only — not the frozen formal result.** These samples are disjoint from prior pilots and use the formal model, gates, eight-method panel, 64-region/10-step faithfulness budget, and 50,000-draw paired bootstrap. The formal targets remain Wiki-VISA n=120 and VizWiz-LF n=100.

## Completion and freeze

| dataset | candidates | strict eligible | frozen | attribution | faithfulness | localization GT |
|---|---:|---:|---:|---:|---:|---|
| Wiki-VISA | 240 | 44 | 20 | 160/160 | 160/160 | yes |
| VizWiz-LF | 40 | 28 | 20 | 160/160 | 160/160 | no, by design |

Wiki strata are balanced as frozen: first-page passage 7, later-page passage 7, non-passage 6. Both faithfulness panels have zero degenerate deletion or insertion curves.

VizWiz semantic grading covers all 20 samples: 13 fully correct, 3 partial, and 4 wrong. The deterministic 10% human audit is still unsigned (0/2); therefore the fully-correct sensitivity result below is LLM-judged and provisional.

## Spatial resolution disclosure

| dataset | method | native attribution grids | faithfulness layouts |
|---|---|---|---|
| Wiki-VISA | random | 32x32 (n=20) | 16x4 (n=20) |
| Wiki-VISA | center | 32x32 (n=20) | 16x4 (n=20) |
| Wiki-VISA | visual-loo | 4x4 (n=20) | 16x4 (n=20) |
| Wiki-VISA | ifr-span | 88x22 (n=20) | 16x4 (n=20) |
| Wiki-VISA | visual-ig | 88x22 (n=20) | 16x4 (n=20) |
| Wiki-VISA | attnlrp | 88x22 (n=20) | 16x4 (n=20) |
| Wiki-VISA | flashtrace | 88x22 (n=20) | 16x4 (n=20) |
| Wiki-VISA | flashtrace-all-gen | 88x22 (n=20) | 16x4 (n=20) |
| VizWiz-LF | random | 32x32 (n=20) | 7x9 (n=1), 8x8 (n=1), 9x7 (n=18) |
| VizWiz-LF | center | 32x32 (n=20) | 7x9 (n=1), 8x8 (n=1), 9x7 (n=18) |
| VizWiz-LF | visual-loo | 4x4 (n=20) | 7x9 (n=1), 8x8 (n=1), 9x7 (n=18) |
| VizWiz-LF | ifr-span | 14x14 (n=1), 17x13 (n=2), 26x19 (n=1), 38x51 (n=1), 40x30 (n=8), 51x38 (n=7) | 7x9 (n=1), 8x8 (n=1), 9x7 (n=18) |
| VizWiz-LF | visual-ig | 14x14 (n=1), 17x13 (n=2), 26x19 (n=1), 38x51 (n=1), 40x30 (n=8), 51x38 (n=7) | 7x9 (n=1), 8x8 (n=1), 9x7 (n=18) |
| VizWiz-LF | attnlrp | 14x14 (n=1), 17x13 (n=2), 26x19 (n=1), 38x51 (n=1), 40x30 (n=8), 51x38 (n=7) | 7x9 (n=1), 8x8 (n=1), 9x7 (n=18) |
| VizWiz-LF | flashtrace | 14x14 (n=1), 17x13 (n=2), 26x19 (n=1), 38x51 (n=1), 40x30 (n=8), 51x38 (n=7) | 7x9 (n=1), 8x8 (n=1), 9x7 (n=18) |
| VizWiz-LF | flashtrace-all-gen | 14x14 (n=1), 17x13 (n=2), 26x19 (n=1), 38x51 (n=1), 40x30 (n=8), 51x38 (n=7) | 7x9 (n=1), 8x8 (n=1), 9x7 (n=18) |

IFR-span, Visual IG, AttnLRP, FlashTrace, and FlashTrace all-generation share the same native visual-token grid within each image. Random/Center use 32x32 synthetic grids; Visual LOO is a coarse 4x4 perturbation reference. Every method uses the same approximately 64-region faithfulness layout for a given image, via nearest-patch resampling.

## Wiki-VISA localization

| method | Energy | Rank AUC | R@5 | R@20 |
|---|---:|---:|---:|---:|
| random | 0.0439 [0.0313, 0.0584] | 0.5156 [0.4938, 0.5410] | 0.0558 [0.0392, 0.0731] | 0.2106 [0.1776, 0.2471] |
| center | 0.0336 [0.0188, 0.0507] | 0.4929 [0.4029, 0.5856] | 0.0580 [0.0164, 0.1063] | 0.1774 [0.0757, 0.2882] |
| visual-loo | 0.1453 [0.0972, 0.2008] | 0.7363 [0.6606, 0.8073] | 0.2590 [0.1744, 0.3484] | 0.6593 [0.5284, 0.7839] |
| ifr-span | 0.2043 [0.1671, 0.2427] | 0.7547 [0.7024, 0.8030] | 0.2341 [0.1616, 0.3175] | 0.5404 [0.4549, 0.6269] |
| visual-ig | 0.0449 [0.0323, 0.0584] | 0.5171 [0.4891, 0.5459] | 0.0774 [0.0569, 0.1007] | 0.2758 [0.2285, 0.3266] |
| attnlrp | 0.3020 [0.2338, 0.3737] | 0.6484 [0.5988, 0.7012] | 0.3009 [0.2308, 0.3793] | 0.5400 [0.4697, 0.6100] |
| flashtrace | 0.2424 [0.2014, 0.2857] | 0.8240 [0.7833, 0.8633] | 0.3315 [0.2378, 0.4341] | 0.6573 [0.5690, 0.7458] |
| flashtrace-all-gen | 0.2305 [0.1913, 0.2713] | 0.8112 [0.7702, 0.8515] | 0.3043 [0.2157, 0.4004] | 0.6355 [0.5457, 0.7255] |

FlashTrace has the highest Rank AUC (0.8240) and R@5 (0.3315); AttnLRP has the highest Energy (0.3020); Visual LOO and FlashTrace are effectively tied at R@20 (0.6593 vs. 0.6573). With n=20, these are preview estimates rather than final rankings.

### Paired FlashTrace differences on primary endpoints

Positive values favor exact-span FlashTrace; parentheses are W/T/L.

| baseline | Energy Δ [95% CI] | Rank AUC Δ [95% CI] | R@5 Δ [95% CI] |
|---|---:|---:|---:|
| visual-loo | +0.0970 [+0.0345, +0.1630] (14/0/6) | +0.0878 [+0.0170, +0.1649] (15/0/5) | +0.0725 [-0.0623, +0.2088] (10/0/10) |
| ifr-span | +0.0381 [+0.0230, +0.0542] (18/0/2) | +0.0693 [+0.0437, +0.0987] (19/0/1) | +0.0974 [+0.0587, +0.1402] (17/2/1) |
| visual-ig | +0.1975 [+0.1529, +0.2454] (20/0/0) | +0.3070 [+0.2617, +0.3552] (20/0/0) | +0.2541 [+0.1668, +0.3494] (20/0/0) |
| attnlrp | -0.0596 [-0.0961, -0.0238] (6/0/14) | +0.1757 [+0.1359, +0.2151] (19/0/1) | +0.0306 [-0.0140, +0.0820] (10/1/9) |
| flashtrace-all-gen | +0.0119 [+0.0049, +0.0186] (18/0/2) | +0.0128 [+0.0065, +0.0195] (19/0/1) | +0.0272 [+0.0141, +0.0410] (13/4/3) |

## Frozen-response visual faithfulness

Deletion AUC and Visual-MAS are lower-is-better; insertion AUC is higher-is-better. Values are means with paired-sample bootstrap CIs.

### Wiki-VISA

| method | deletion AUC | insertion AUC | Visual-MAS |
|---|---:|---:|---:|
| random | 0.6013 [0.5358, 0.6697] | 0.6134 [0.5056, 0.7145] | 0.7159 [0.6395, 0.7908] |
| center | 0.6373 [0.5295, 0.7379] | 0.5344 [0.4148, 0.6522] | 0.7125 [0.6080, 0.8085] |
| visual-loo | 0.3395 [0.2289, 0.4610] | 0.8641 [0.8293, 0.8959] | 0.4438 [0.3002, 0.5947] |
| ifr-span | 0.4578 [0.3341, 0.5820] | 0.7594 [0.6817, 0.8276] | 0.5739 [0.4296, 0.7143] |
| visual-ig | 0.5337 [0.4319, 0.6339] | 0.6910 [0.6101, 0.7650] | 0.6193 [0.5022, 0.7313] |
| attnlrp | 0.5657 [0.4322, 0.6924] | 0.7137 [0.6057, 0.8079] | 0.6999 [0.5486, 0.8350] |
| flashtrace | 0.4175 [0.3019, 0.5360] | 0.7896 [0.7121, 0.8535] | 0.5406 [0.4058, 0.6759] |
| flashtrace-all-gen | 0.4388 [0.3220, 0.5585] | 0.7728 [0.6804, 0.8466] | 0.5646 [0.4258, 0.7008] |

### VizWiz-LF

| method | deletion AUC | insertion AUC | Visual-MAS |
|---|---:|---:|---:|
| random | 0.4259 [0.3356, 0.5118] | 0.5223 [0.4390, 0.6031] | 0.5795 [0.4853, 0.6765] |
| center | 0.4547 [0.3420, 0.5649] | 0.6839 [0.5946, 0.7584] | 0.5984 [0.4619, 0.7293] |
| visual-loo | 0.3783 [0.2887, 0.4672] | 0.6741 [0.5936, 0.7465] | 0.5511 [0.4389, 0.6644] |
| ifr-span | 0.4085 [0.3102, 0.5056] | 0.5573 [0.4727, 0.6370] | 0.5461 [0.4218, 0.6700] |
| visual-ig | 0.4175 [0.3195, 0.5138] | 0.4962 [0.4044, 0.5816] | 0.5630 [0.4335, 0.6878] |
| attnlrp | 0.4293 [0.3341, 0.5232] | 0.5467 [0.4528, 0.6364] | 0.5840 [0.4543, 0.7094] |
| flashtrace | 0.4032 [0.3082, 0.4985] | 0.5556 [0.4751, 0.6302] | 0.5425 [0.4212, 0.6647] |
| flashtrace-all-gen | 0.3989 [0.3018, 0.4953] | 0.5528 [0.4685, 0.6293] | 0.5302 [0.4059, 0.6557] |

### VizWiz-LF fully-correct sensitivity (LLM-judged n=13)

| method | deletion AUC | insertion AUC | Visual-MAS |
|---|---:|---:|---:|
| random | 0.4848 [0.3774, 0.5805] | 0.5458 [0.4414, 0.6400] | 0.6594 [0.5401, 0.7739] |
| center | 0.5170 [0.3885, 0.6430] | 0.6713 [0.5441, 0.7707] | 0.6653 [0.5106, 0.8058] |
| visual-loo | 0.4434 [0.3362, 0.5422] | 0.6942 [0.5800, 0.7904] | 0.6335 [0.4890, 0.7671] |
| ifr-span | 0.4522 [0.3340, 0.5635] | 0.5711 [0.4697, 0.6650] | 0.6117 [0.4597, 0.7559] |
| visual-ig | 0.4732 [0.3497, 0.5903] | 0.5209 [0.4027, 0.6282] | 0.6416 [0.4759, 0.7936] |
| attnlrp | 0.4490 [0.3225, 0.5710] | 0.5706 [0.4642, 0.6633] | 0.6149 [0.4444, 0.7757] |
| flashtrace | 0.4440 [0.3235, 0.5578] | 0.5808 [0.4814, 0.6729] | 0.5990 [0.4430, 0.7518] |
| flashtrace-all-gen | 0.4449 [0.3276, 0.5551] | 0.5734 [0.4681, 0.6692] | 0.5996 [0.4482, 0.7453] |

On Wiki-VISA, Visual LOO is strongest on all three faithfulness summaries, while FlashTrace is consistently stronger than IFR-span and exact-span is modestly stronger than all-generation. On VizWiz-LF there is no single overall winner: Visual LOO has the lowest deletion AUC, center has the highest insertion AUC, and all-generation has the lowest overall Visual-MAS. In the provisional fully-correct subset, Visual LOO narrowly leads deletion and insertion, while FlashTrace has the lowest Visual-MAS. The wide n=20/n=13 intervals argue against a final winner claim.

## Recursive mechanism and map diagnostics

| dataset | exact/all-gen cosine | recursive positive mass |
|---|---:|---:|
| Wiki-VISA | 0.9913 [0.9870, 0.9947] | 0.4540 [0.4369, 0.4725] |
| VizWiz-LF | 0.9964 [0.9954, 0.9973] | 0.3397 [0.3194, 0.3619] |

Exact-span and all-generation maps are highly aligned, but the paired Wiki localization analysis still favors exact-span FlashTrace over all-generation on Energy, Rank AUC, and R@5. Recursive visual mass is substantial (about 45% on Wiki and 34% on VizWiz). Signed baselines also show material negative-cell fractions, so their positive-only ordering sensitivity remains a required appendix result.

## Attribution resource profile

| dataset | method | seconds/sample | peak VRAM GiB |
|---|---|---:|---:|
| Wiki-VISA | visual-loo | 21.072 | 9.205 |
| Wiki-VISA | ifr-span | 1.587 | 13.018 |
| Wiki-VISA | visual-ig | 27.865 | 15.925 |
| Wiki-VISA | attnlrp | 1.608 | 44.291 |
| Wiki-VISA | flashtrace | 2.053 | 13.018 |
| Wiki-VISA | flashtrace-all-gen | 2.042 | 13.018 |
| VizWiz-LF | visual-loo | 13.137 | 5.245 |
| VizWiz-LF | ifr-span | 1.096 | 9.932 |
| VizWiz-LF | visual-ig | 17.824 | 9.226 |
| VizWiz-LF | attnlrp | 1.228 | 27.575 |
| VizWiz-LF | flashtrace | 1.504 | 9.932 |
| VizWiz-LF | flashtrace-all-gen | 1.496 | 9.932 |

## Pipeline findings before the full run

- VizWiz gate refresh rejected 9/40 explicit unanswerable responses that the earlier refusal pattern missed.

- Two long VizWiz AttnLRP cases originally exhausted device memory. Gradient checkpointing and target-row-only LM-head projection fixed them without changing the frozen target-logit objective; the final preview matrices are complete.

- No CLEVR-XAI rerun is part of this preview. A6 remains the existing offline legacy diagnostic appendix only.

## Interpretation boundary

This preview validates the execution path and exposes likely effect directions. It does not replace the formal n=120/n=100 freeze, does not justify paper-level significance claims, and does not satisfy the final independent human-audit requirement. The next execution step is to resume the full Wiki-VISA and VizWiz-LF queues with the corrected gates and memory-safe AttnLRP implementation.
