# TPAMI Figures 4 and 5 reproduction

This package reproduces the two quantitative assets used as Figures 4 and 5
in the TPAMI manuscript:

- Figure 4: `cost_comparison.pdf`
- Figure 5: `cot_faithfulness.pdf`

From a fresh checkout:

```bash
bash paper_figures/fig4_5/reproduce.sh
```

The script installs the plotting dependencies in an isolated environment,
downloads the recovered Figure 5 source archive, verifies its SHA-256, extracts
only the required MoreHopQA and Variable Tracking records, and writes PDF and
PNG versions to `paper_figures/fig4_5/output/`.

## Provenance

Figure 5 is computed from processed per-sample attribution records in
`exp/proc_1/output/data.zip` at FlashTrace commit `075e7e4`. The archive
SHA-256 is
`44666abac156848c095fe141d913afc96370fd090ed0c2994df6dfa912c826ba`.
It contains 95 MoreHopQA samples per method and 100 samples per method for each
Variable Tracking subset used by the plot. These records were not digitized
from the published PDF.

Figure 4 retains the legacy paper values and their original limitations:

- generation-memory points at 1K and 2K for IFR, AttnLRP, and FlashTrace were
  interpolated after broken measurements;
- the Pareto speed values were hand-normalized;
- IG and IG-Attn faithfulness values in the Pareto panel were placeholders.

Those statuses are recorded in `data/efficiency.json`; they should not be
described as newly measured results.
