# Reproduce Figure 6

This self-contained package reproduces the manuscript figure captioned
"Qualitative comparison of visual attribution." It bundles only the two frozen
samples and four methods shown in the figure; no model, GPU, dataset download,
or full formal attribution archive is required.

From the repository root, run:

```bash
bash paper_figures/fig6/reproduce.sh
```

The figure is written to
`paper_figures/fig6/output/visual_examples.png`. Pass another path as the first
argument to change the destination:

```bash
bash paper_figures/fig6/reproduce.sh /tmp/visual_examples.png
```

The launcher uses `uv` when available. Otherwise it creates a local virtual
environment and installs the two pinned rendering dependencies. The renderer
verifies every bundled source file against its SHA-256 checksum before drawing.

## Frozen inputs

- VizWiz-LF `vizwiz-lf-261`: source image, four attribution grids, and
  full-image deletion AUC values.
- Wiki-VISA `wiki-visa-1128`: source page image, native evidence box, four
  attribution grids, and full-image evidence-rank AUC values.
- Methods: FlashTrace++, AttnLRP, Visual IG, and Visual LOO.

The displayed crops affect presentation only. All metrics shown in the figure
come from evaluation on the original full images.
