# A6: retained CLEVR-XAI and VISTAQA diagnostics

These are frozen protocol-validation diagnostics. They are not pooled with the formal Wiki-VISA or VizWiz-LF samples, and this analysis runs no new GPU inference.

## CLEVR-XAI dual-mask sensitivity

Paired samples: 20; bootstrap draws: 10000. The primary mask is Unique First-nonempty; Union is a sensitivity convention.

| method | Energy unique | Energy union | R@5 unique | R@5 union |
|---|---:|---:|---:|---:|
| random | 0.0868 | 0.1800 | 0.0478 | 0.0475 |
| center | 0.2030 | 0.4006 | 0.1342 | 0.1401 |
| visual-loo | 0.1756 | 0.2811 | 0.1813 | 0.1351 |
| ifr-span | 0.0424 | 0.0844 | 0.0099 | 0.0054 |
| attention-rollout | 0.0535 | 0.1089 | 0.0000 | 0.0001 |
| grad-attention | 0.0534 | 0.1016 | 0.0068 | 0.0039 |
| visual-ig | 0.3118 | 0.4962 | 0.2644 | 0.2033 |
| attnlrp | 0.1754 | 0.2455 | 0.1626 | 0.0968 |
| tam | 0.0683 | 0.1232 | 0.0256 | 0.0223 |
| flashtrace | 0.0573 | 0.1083 | 0.0144 | 0.0070 |
| flashtrace-all-gen | 0.0513 | 0.0983 | 0.0111 | 0.0060 |

The large convention-dependent shifts, especially for Center and Visual IG, show why CLEVR does not serve as the formal localization benchmark. Its centered synthetic objects can reward a spatial prior.

## CLEVR-XAI center-prior faithfulness counterexample

Budget: 64 regions and 10 steps on 20 paired samples.

| method | deletion AUC ↓ | insertion AUC ↑ | Visual-MAS ↓ |
|---|---:|---:|---:|
| center | 0.3620 | 0.8540 | 0.4536 |
| visual-ig | 0.4650 | 0.8505 | 0.6069 |
| attnlrp | 0.4469 | 0.8079 | 0.5858 |
| flashtrace | 0.4974 | 0.7463 | 0.6148 |

Center is strongest on all three retained faithfulness metrics in this synthetic diagnostic. We therefore treat centered-subject priors as an explicit baseline, not as evidence of causal grounding.

## VISTAQA failure-analysis pilot

The native manifest contains 10 samples, but only 4 lie in the common successful method intersection. These values are descriptive only.

| method | Energy | Rank AUC | R@5 | R@20 |
|---|---:|---:|---:|---:|
| random | 0.0031 | 0.4306 | 0.0000 | 0.1546 |
| center | 0.0027 | 0.5932 | 0.0365 | 0.2500 |
| visual-ig | 0.0155 | 0.7792 | 0.5789 | 0.5967 |
| attnlrp | 0.0507 | 0.6025 | 0.4971 | 0.5425 |
| flashtrace | 0.1023 | 0.9447 | 0.8334 | 0.9445 |
