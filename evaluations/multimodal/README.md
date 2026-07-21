# Multimodal evaluation smoke test

## Strict Wiki-VISA / CLEVR-XAI protocol

The paper-facing protocol keeps three artifacts separate:

1. `*.dataset.jsonl` contains only `I_IMAGE`, `I_QUESTION`, and evaluation
   metadata (`REFERENCE_OUTPUT`, boxes/masks, and optional functional program).
2. `*.model.jsonl` contains the model's own `THINKING`, whole `OUTPUT`, their
   inclusive token spans, and exact model/template identity. It contains no
   reference answer or dataset rationale.
3. `*.generation_eval.jsonl` joins the two by `sample_id` and records output
   correctness, repeated-greedy stability, exact generated/teacher-forced token
   identity, and the output-only log-probability drop under image blur.
4. `*.ablation.model.jsonl` separately stores deterministic generations on a
   globally blurred image and a uniform-gray image. The revised strict
   evaluation records only whether these ablations reproduce the original
   whole `OUTPUT`; ablation text never enters the primary model record.

Functional programs, human rationales, and GPT rationales are evaluation
metadata only and are never supplied to the model. The primary attribution
sink is always `OUTPUT_SPAN`. Primary `flashtrace` preserves the paper's
cumulative definition: direct `OUTPUT -> IMAGE` attribution is added to the
exact `OUTPUT -> THINKING -> IMAGE` recursive attribution, with each recursive
hop scaled by the cumulative fraction of attribution mass entering the
reasoning span. The public facade's broader THINKING+OUTPUT recursive span is
reported separately as
`flashtrace-all-gen`; it is an ablation, not the primary method. All methods
teacher-force the same complete frozen response.

A strict sample must satisfy all of the following: whole-output correctness;
two identical greedy generations; exact equality between generated token IDs
and decode/re-encoded teacher-forcing IDs; a positive output-only log-probability
drop under global blur; and failure of at least one deterministic blur/gray
ablation generation to reproduce the original whole output. No rationale or
functional program supplied by a dataset is accepted as `THINKING`.

The complete official data currently verified on disk is:

- Wiki-VISA official `test`: seven parquet shards and 3,000 rows;
- CLEVR-XAI v1.0 complex: 100,000 questions, 10,000 images, and all four
  official complex evidence-mask variants.

`strict_datasets.py` creates deterministic candidate manifests, and
`select_strict_subset.py` materializes the balanced strict-eligible subset only
after generation and ablation auditing. CLEVR is balanced across count,
existence, integer comparison, attribute comparison, and attribute query,
requires at least 12 functional-program steps, and uses a different image per
selected question. Its primary ground truth is Unique First-nonempty; Unique
and Union are sensitivity analyses. Wiki-VISA is balanced across first-page
passage, later-page passage, and non-passage, and preserves the native
980x3920 screenshots and absolute `xyxy` evidence boxes.
Formal Wiki-VISA runs use `max_pixels=2,007,040`; the earlier 1MP setting was
rejected after a controlled OCR resolution check.

The completed strict pilot, including final scores, paired bootstrap intervals,
frozen-response faithfulness, and two independent full visual audits, is in
[`results/strict/final/RESULTS.md`](results/strict/final/RESULTS.md). The final
paper-facing directories contain `wiki_visa_n18_2mp_methods_v2` and
`clevr_xai_complex_strict_n20_methods_v2`. The older
`clevr_xai_complex_n20_methods` run predates the strict generation-ablation
gate and exact-THINKING bridge fix; it is diagnostic history and must not be
used in paper tables.

The later native-dataset fit check for Wiki-VISA, VISTAQA, and VizWiz-LF is in
[`results/strict/native_pilot/RESULTS.md`](results/strict/native_pilot/RESULTS.md).
It uses whole-patch tie-aware metrics and the same cumulative
direct-plus-weighted-hops FlashTrace definition as the paper.

```bash
python -m evaluations.multimodal.verify_strict_data \
  --output evaluations/multimodal/results/strict/data_integrity.json

python -m evaluations.multimodal.strict_datasets \
  --dataset clevr-xai-complex \
  --output evaluations/multimodal/results/strict/clevr_xai_complex_seed17_n20.dataset.jsonl

python -m evaluations.multimodal.strict_datasets \
  --dataset wiki-visa \
  --output evaluations/multimodal/results/strict/wiki_visa_seed17_n20.dataset.jsonl
```

Generate real Qwen3-VL-Thinking records before attribution:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluations.multimodal.strict_generation \
  --dataset-manifest evaluations/multimodal/results/strict/clevr_xai_complex_seed17_n20.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/clevr_xai_complex.model.jsonl \
  --evaluation-output evaluations/multimodal/results/strict/clevr_xai_complex.generation_eval.jsonl \
  --model Qwen/Qwen3-VL-8B-Thinking
```

Then verify that the answer generation actually depends on visual input. This
command writes ablation model outputs separately and produces the revised
evaluation file used by subset selection:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluations.multimodal.strict_ablation_audit \
  --dataset-manifest <candidate.dataset.jsonl> \
  --model-output <candidate.model.jsonl> \
  --generation-evaluation <candidate.generation_eval.jsonl> \
  --ablation-model-output <candidate.ablation.model.jsonl> \
  --revised-evaluation-output <candidate.strict.generation_eval.jsonl>
```

Run FlashTrace and baselines only after the one-sample generation and
attribution smoke test succeeds:

```bash
CUDA_VISIBLE_DEVICES=0 python -m evaluations.multimodal.strict_attribution \
  --dataset-manifest evaluations/multimodal/results/strict/clevr_xai_complex_seed17_n20.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/clevr_xai_complex.model.jsonl \
  --generation-evaluation evaluations/multimodal/results/strict/clevr_xai_complex.generation_eval.jsonl \
  --output-dir evaluations/multimodal/results/strict/clevr_xai_complex_methods
```

The comparison table uses the intersection of successful sample IDs across
every requested method. Primary localization metrics are Recovery at 1%, 5%,
10%, and 20%, energy in evidence, pointing game, evidence-vs-background rank
AUC, and top-evidence-area IoU. CLEVR's Unique First-nonempty mask is the
official primary view and Union is reported as a reasoning-chain sensitivity
view. Spatial scoring operates on complete visual patches: native GT pixels
are assigned to their containing patch, and a cutoff-score tie receives its
expected credit under a uniform tie break. Heatmaps are expanded with nearest
neighbors for display only; bilinear smoothing and partial-patch top-q
selection are not used for metrics. `strict_visual_faithfulness.py`
additionally uses a common approximately
64-cell image partition for blur deletion/insertion. It fixes the complete
THINKING+OUTPUT token sequence and accumulates only OUTPUT_SPAN log probability
at every perturbation step.

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
