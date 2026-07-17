# Multimodal evaluation smoke test

This module puts VQA-X and A-OKVQA behind one protocol:

```text
image + question -> model-generated reasoning -> final answer
```

Human rationales are retained as metadata but are never supplied to the model.
The smoke runner generates a response and then computes a 4x4 visual
leave-one-region-out (LOO) map by blurring one image region at a time and
measuring the drop in mean log-probability of the frozen generated response.

The VQA-X/A-OKVQA suite below is the implementation that has actually been
downloaded and exercised so far.

## Data sources

- A-OKVQA v1.0 comes from the
  [official AllenAI release](https://github.com/allenai/aokvqa).
- The original Google Drive folder linked by the
  [VQA-X repository](https://github.com/Seth-Park/MultimodalExplanations) now
  returns 404. We use the structured VQA-X release from the authors of
  [NLX-GPT](https://github.com/fawazsammani/nlxgpt).
  It preserves the original questions, ten VQA answers, human explanations,
  COCO image IDs, and train/validation/test split. SHA-256 checksums are pinned
  in `prepare_data.py`.
- Images come directly from the COCO HTTP endpoint documented by the dataset.
  (Its HTTPS certificate currently has a hostname mismatch.) The smoke setup
  downloads only images referenced by the selected examples and records their
  SHA-256 hashes in the local manifest.

## Reproduce the 5 + 5 smoke run

From the repository root:

```bash
python -m evaluations.multimodal.prepare_data --samples 25
CUDA_VISIBLE_DEVICES=0 python -m evaluations.multimodal.run_smoke \
  --samples 5 \
  --correct-only \
  --candidate-limit 25 \
  --model Qwen/Qwen3-VL-8B-Instruct
```

The model revision defaults to the pinned commit in `run_smoke.py`. Outputs are
written beneath `data/multimodal_smoke/`:

- `results.jsonl`: complete per-example generations, scores and 4x4 LOO maps;
- `attempts.jsonl`: ordered correct-answer filtering decisions, including skips;
- `summary.json`: aggregate correctness and visual-sensitivity checks;
- `overlays/`: original images overlaid with red visual-LOO importance.

The whole benchmark can be configured by omitting `--samples` in custom code
and downloading the corresponding COCO image archives. The provided CLI keeps
the default at five because it is a smoke test, not the paper-scale run. The
`--correct-only` command mirrors the paper protocol: scan a deterministic
candidate prefix, retain the first five responses with VQA consensus accuracy
at least 0.6 and an explicit reasoning sentence of at least three words, and
only then run attribution.

## Compare attribution methods on the frozen responses

The method runner reuses the exact 5+5 images, prompts, and generated responses
from the smoke run. Its main baseline set is:

- visual LOO (the frozen-response perturbation reference);
- IFR-tokenwise;
- attention rollout;
- Grad×Attention;
- visual integrated gradients;
- FlashTrace;
- the official Token Activation Map (TAM) implementation;
- AttnLRP on Qwen3-VL's fused visual tokens.

`ifr-span` remains available through `--methods`, but is a FlashTrace
no-recursion ablation rather than a separate paper baseline. EAGLE and the
SLICO/search infrastructure proposed for it are intentionally outside the
current implementation scope.

The Qwen3-VL AttnLRP adapter applies the AttnLRP rules to the language decoder
and attributes both the initial fused image-token embeddings and every
DeepStack visual injection. It does not propagate relevance inside the vision
encoder. This boundary matches the visual-token comparison unit used by
FlashTrace, rollout, Grad×Attention, and TAM and must be stated in the paper.

TAM is kept beneath the ignored `data/` directory so its source remains pinned
without vendoring another project:

```bash
git clone https://github.com/xmed-lab/TAM data/external/TAM
python -m pip install scipy opencv-python-headless pymupdf

CUDA_VISIBLE_DEVICES=0 python -m evaluations.multimodal.run_methods \
  --reference-results data/multimodal_smoke_final/results.jsonl \
  --output-dir data/multimodal_methods_final \
  --tam-source data/external/TAM \
  --ig-steps 20
```

Do not install TAM's pinned Transformers version over the FlashTrace
environment: this adapter intentionally runs its official `TAM()` function
with the repository's Qwen3-VL-compatible Transformers version. Per-example
maps, timing, peak memory, LOO-alignment metrics, and overlays are written to
`data/multimodal_methods_final/`. See `METHOD_RESULTS.md` for the validated
5+5 run and interpretation.

The default method overlays are downsampled to 4x4 for direct comparison with
LOO. Render every stored visual-token score at its native rectangular grid
without rerunning the model:

```bash
python -m evaluations.multimodal.render_native \
  --reference-results data/multimodal_smoke_final/results.jsonl \
  --method-results data/multimodal_methods_final/results.jsonl
```

The renderer discovers every successful supported method in the result file
and produces native overlays plus one comparison sheet per sample. LOO remains
4x4 because only those 16 image regions were actually perturbed.
