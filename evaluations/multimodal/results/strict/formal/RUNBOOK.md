# Formal multimodal v2 runbook

All commands run from the repository root on one A100 80GB GPU. Every
long-running command is resume-safe. The machine-readable contract is
`evaluations/multimodal/protocol.json`.

The checked-in stage runners execute the commands below with upstream
row-count/frozen-set guards:

```bash
bash evaluations/multimodal/run_formal_all.sh

# Or run individual stages:
bash evaluations/multimodal/run_formal_e1.sh
bash evaluations/multimodal/run_formal_e2.sh
bash evaluations/multimodal/run_formal_e3_e5.sh
```

Each runner may receive the PID of the preceding runner to form a single-GPU
queue. A failed guard stops the queue; it never relaxes a gate.
`run_formal_all.sh` is the preferred supervisor: it enters the fixed seed47
contingency only when E1 has completed the entire seed31 gate audit but cannot
freeze 40/40/40.

## E1: Wiki-VISA

Candidate generation:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m evaluations.multimodal.strict_generation \
  --dataset-manifest evaluations/multimodal/results/strict/formal/wiki_visa_candidates.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/wiki_visa_candidates.model.jsonl \
  --evaluation-output evaluations/multimodal/results/strict/formal/wiki_visa_candidates.generation_eval.jsonl \
  --resume \
  --skip-recorded-deterministic-errors
```

The dataset-aware generation budget is 1,024 new tokens.
The skip flag retains only two already-recorded protocol-deterministic
ValueErrors (unclosed THINKING at the frozen token cap and generated/teacher
token-identity mismatch). Transient failures such as CUDA OOM remain
retryable.

Generation ablation audit:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m evaluations.multimodal.strict_ablation_audit \
  --dataset-manifest evaluations/multimodal/results/strict/formal/wiki_visa_candidates.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/wiki_visa_candidates.model.jsonl \
  --generation-evaluation evaluations/multimodal/results/strict/formal/wiki_visa_candidates.generation_eval.jsonl \
  --ablation-model-output evaluations/multimodal/results/strict/formal/wiki_visa_candidates.ablation.model.jsonl \
  --revised-evaluation-output evaluations/multimodal/results/strict/formal/wiki_visa_candidates.strict.generation_eval.jsonl
```

Freeze the balanced n=120 subset:

```bash
.venv/bin/python -m evaluations.multimodal.select_strict_subset \
  --dataset-manifest evaluations/multimodal/results/strict/formal/wiki_visa_candidates.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/wiki_visa_candidates.model.jsonl \
  --generation-evaluation evaluations/multimodal/results/strict/formal/wiki_visa_candidates.strict.generation_eval.jsonl \
  --exclude-manifest evaluations/multimodal/results/strict/final/wiki_visa_n18_2mp.dataset.jsonl \
  --exclude-manifest evaluations/multimodal/results/strict/native_pilot/wiki_visa_n10.dataset.jsonl \
  --sample-size 120 --balance-key stratum --seed 17 \
  --output-dataset evaluations/multimodal/results/strict/formal/wiki_visa_n120.dataset.jsonl \
  --output-model evaluations/multimodal/results/strict/formal/wiki_visa_n120.model.jsonl \
  --output-evaluation evaluations/multimodal/results/strict/formal/wiki_visa_n120.generation_eval.jsonl \
  --funnel-output evaluations/multimodal/results/strict/formal/wiki_visa_funnel.json \
  --frozen-ids-output evaluations/multimodal/results/strict/formal/frozen_ids.json
```

If fewer than 40 strict-eligible samples exist in any stratum, generate an
additional disjoint candidate manifest with `--exclude-manifest`; never relax
a gate.

The initial observed pre-ablation yield triggered this rule. The frozen
extension is:

```bash
.venv/bin/python -m evaluations.multimodal.strict_datasets \
  --dataset wiki-visa --sample-size 600 --seed 31 \
  --exclude-manifest evaluations/multimodal/results/strict/formal/wiki_visa_candidates.dataset.jsonl \
  --output evaluations/multimodal/results/strict/formal/wiki_visa_candidates_extension_seed31_n600.dataset.jsonl
```

It contains 200 disjoint candidates per stratum. Run generation and ablation
to separate `extension_seed31_n600` outputs, then pass both bundles to
`select_strict_subset` using repeated path arguments.

If the completed seed31 pool still leaves the later-page stratum below 40
after ablation, the prebuilt seed47 contingency manifest contains 600
additional later-page-only candidates. It excludes both earlier candidate
pools and both Wiki pilot sets. Resume the frozen extension policy with:

```bash
bash evaluations/multimodal/run_formal_e1_later_extension.sh
```

The runner refuses to execute after `wiki_visa_n120.dataset.jsonl` exists.
It consumes the fixed seed47 order as nested 100-row prefixes, performs
generation and ablation for each completed prefix, and attempts the final
fixed-seed selection over all three bundles. It stops at the first prefix
that supplies 40/40/40; no attribution result is available to or used by
this stopping rule. If a quota is still short, it extends to the next prefix
without relaxing any gate.

## E2: VizWiz-LF

Use the equivalent generation and ablation commands with the
`vizwiz_lf_candidates` bundle. The dataset-aware generation budget is 2,048
new tokens. Freeze n=100 by fixed-seed sampling from the full strict-eligible
pool (omit `--balance-key`) and exclude
`results/strict/native_pilot/vizwiz_lf_n10.dataset.jsonl`. Question type and
OUTPUT-length tercile are recorded for stratified reporting but are not
selection gates.

After all 200 primary candidate responses are complete and the refusal gates
are refreshed, `run_formal_e2.sh` invokes
`reuse_preview_ablation_checkpoints`. It may seed complete blur/gray
ablations from the isolated 40-candidate preview only when the image,
question, primary response, token IDs, model revision, and ablation generation
configuration are identical. The source hash and exact reused IDs are saved
in `vizwiz_lf_candidates.preview_ablation_reuse.json`; the formal ablation
runner then resumes the remaining candidates. This reuses deterministic GPU
work only and does not reuse the preview selection or any preview estimate.

Prepare semantic correctness tasks after IDs are frozen:

```bash
.venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments prepare \
  --dataset-manifest evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.model.jsonl \
  --output evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.semantic_tasks.jsonl
```

After LLM judgments are present, materialize the deterministic image-linked
10% human audit packet and editable review template:

```bash
.venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments audit-packet \
  --dataset-manifest evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.model.jsonl \
  --judgments evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.semantic_judgments.llm.jsonl \
  --output-markdown evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.human_audit.md \
  --review-template evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.human_reviews.jsonl
```

Apply the completed human review rows, preserving the original LLM labels and
reasons as provenance:

```bash
.venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments apply-review \
  --judgments evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.semantic_judgments.llm.jsonl \
  --reviews evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.human_reviews.jsonl \
  --output evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.semantic_judgments.jsonl
```

Join the adjudicated judgments and enforce completeness:

```bash
.venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments join \
  --generation-evaluation evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.generation_eval.jsonl \
  --judgments evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.semantic_judgments.jsonl \
  --output-evaluation evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.reviewed.generation_eval.jsonl \
  --summary-output evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.semantic_summary.json \
  --require-complete
```

Prepare the separate, deterministic 10% caveat-only protocol audits after
both frozen sets exist. These reviews assess image dependence and THINKING
quality and must never change frozen IDs:

```bash
.venv/bin/python -m evaluations.multimodal.formal_manual_audit prepare \
  --dataset-manifest evaluations/multimodal/results/strict/formal/wiki_visa_n120.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/wiki_visa_n120.model.jsonl \
  --generation-evaluation evaluations/multimodal/results/strict/formal/wiki_visa_n120.generation_eval.jsonl \
  --ablation-model-output evaluations/multimodal/results/strict/formal/wiki_visa_candidates.ablation.model.jsonl \
  --ablation-model-output evaluations/multimodal/results/strict/formal/wiki_visa_candidates_extension_seed31_n600.ablation.model.jsonl \
  --output-markdown evaluations/multimodal/results/strict/formal/wiki_visa_n120.protocol_audit.md \
  --review-template evaluations/multimodal/results/strict/formal/wiki_visa_n120.protocol_reviews.jsonl

.venv/bin/python -m evaluations.multimodal.formal_manual_audit prepare \
  --dataset-manifest evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.model.jsonl \
  --generation-evaluation evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.generation_eval.jsonl \
  --ablation-model-output evaluations/multimodal/results/strict/formal/vizwiz_lf_candidates.ablation.model.jsonl \
  --output-markdown evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.protocol_audit.md \
  --review-template evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.protocol_reviews.jsonl
```

After a human completes every template row, validate and summarize:

```bash
.venv/bin/python -m evaluations.multimodal.formal_manual_audit summarize \
  --dataset-manifest evaluations/multimodal/results/strict/formal/wiki_visa_n120.dataset.jsonl \
  --reviews evaluations/multimodal/results/strict/formal/wiki_visa_n120.protocol_reviews.jsonl \
  --output evaluations/multimodal/results/strict/formal/wiki_visa_n120.protocol_audit_summary.json

.venv/bin/python -m evaluations.multimodal.formal_manual_audit summarize \
  --dataset-manifest evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.dataset.jsonl \
  --reviews evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.protocol_reviews.jsonl \
  --output evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.protocol_audit_summary.json
```

After both formal manifests are frozen, materialize the immutable input
integrity record once:

```bash
.venv/bin/python -m evaluations.multimodal.freeze_formal_input_hashes \
  --root . \
  --manifest evaluations/multimodal/results/strict/formal/wiki_visa_n120.dataset.jsonl \
  --manifest evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.dataset.jsonl \
  --output evaluations/multimodal/results/strict/formal/frozen_input_hashes.json
```

This records SHA-256 for both complete manifests and, for every frozen row,
the canonical dataset record, question, and original image. The final auditor
recomputes all 220 image hashes. A mismatch is a protocol violation; do not
regenerate the hash file to bless a changed input.

Freeze the matching generated responses once, before downstream attribution
is interpreted:

```bash
.venv/bin/python -m evaluations.multimodal.freeze_formal_response_hashes \
  --root . \
  --model-output evaluations/multimodal/results/strict/formal/wiki_visa_n120.model.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/vizwiz_lf_n100.model.jsonl \
  --output evaluations/multimodal/results/strict/formal/frozen_response_hashes.json
```

This separately hashes both complete model JSONLs and every canonical model
record, raw response, THINKING, OUTPUT, generated token sequence, and
teacher-forced token sequence. The final auditor recomputes all 220 response
hash bundles and requires the frozen revision and token identity. As with
input hashes, never regenerate this artifact to legitimize a changed response.

## E3: Wiki localization

Before loading the model, `run_formal_e3_e5.sh` calls
`reuse_preview_checkpoints` for both datasets. It seeds only successful n=20
records whose image, question, complete frozen response, generated and
teacher-forced token IDs, and resolved model revision exactly match the
formal bundle. The command records hashes and overlap accounting in
`*.preview_reuse.json`; mismatched rows are not reused. This is deterministic
checkpoint reuse, not pooling preview estimates: all formal summaries and
bootstraps are regenerated on the complete frozen intersection.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m evaluations.multimodal.strict_attribution \
  --dataset-manifest evaluations/multimodal/results/strict/formal/wiki_visa_n120.dataset.jsonl \
  --model-output evaluations/multimodal/results/strict/formal/wiki_visa_n120.model.jsonl \
  --generation-evaluation evaluations/multimodal/results/strict/formal/wiki_visa_n120.generation_eval.jsonl \
  --output-dir evaluations/multimodal/results/strict/formal/wiki_visa_n120_methods
```

The default method panel is the frozen eight-method panel. During an
incomplete run, each new sample/method pair is stored in the hidden atomic
journal beside `attribution_records.jsonl`; a resumed command overlays that
journal on the last canonical snapshot. On successful completion the journal
is compacted into the canonical JSONL and removed. Downstream analysis must
consume the canonical file only after the command exits successfully.

Before any downstream bootstrap, both the stage runner and finalizer invoke
`validate_paired_matrix`. It rejects duplicate, error, and extra rows and
requires the successful records to equal the exact frozen-ID by eight-method
Cartesian product; `common_samples == n` alone is not sufficient.

## E4/E5: faithfulness

Run attribution on VizWiz with `--allow-missing-evidence`, then:

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
  --dataset-manifest <formal-dataset.jsonl> \
  --model-output <formal-model.jsonl> \
  --attribution-dir <formal-attribution-dir> \
  --output-dir <formal-faithfulness-dir>
```

Defaults are 64 regions and ten steps. Every record saves signed-order curves,
positive-only-order sensitivity curves, raw region scores, and degenerate
curve flags. Faithfulness uses the same atomic pair-journal and final
compaction policy as attribution. Attribution and faithfulness summaries
record the exact processor budget (`min_pixels=200704`,
`max_pixels=2007040`) in addition to the frozen model revision and input
artifact paths.

Curve normalization uses the directional endpoint span. Deletion requires
`original_score - blurred_score > 1e-8`; insertion requires the identical
directional span expressed as `original_score - blurred_score > 1e-8`.
Non-positive or numerically zero spans are recorded as degenerate and receive
the neutral endpoint-preserving linear normalized curve. Raw deletion and
insertion log-probabilities remain unchanged and are always retained. The
machine-readable policy identifier is
`directional_endpoint_span_nonpositive_is_degenerate`.

After an already-complete faithfulness matrix, its canonical summary can be
rebuilt without loading the model. `--refresh-derived-metrics` additionally
recomputes only normalization, AUC, Visual-MAS, endpoint deltas, and degenerate
flags from the saved raw perturbation curves:

```bash
.venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
  --dataset-manifest <formal-dataset.jsonl> \
  --model-output <formal-model.jsonl> \
  --attribution-dir <formal-attribution-dir> \
  --output-dir <formal-faithfulness-dir> \
  --summary-only \
  --refresh-derived-metrics
```

This mode refuses an incomplete frozen-ID × eight-method matrix. The finalizer
uses it for both datasets before analysis so a resumed GPU process and a fresh
run produce the same summary metadata. It does not load the model or modify
the saved raw perturbation observations.

## A1-A5/A8 analyses

Localization paired bootstrap:

```bash
.venv/bin/python -m evaluations.multimodal.analyze_strict_results \
  --manifest evaluations/multimodal/results/strict/formal/wiki_visa_n120.dataset.jsonl \
  --attribution-dir evaluations/multimodal/results/strict/formal/wiki_visa_n120_methods \
  --output-json evaluations/multimodal/results/strict/formal/wiki_visa_n120_methods/analysis.json \
  --output-markdown evaluations/multimodal/results/strict/formal/wiki_visa_n120_methods/analysis.md \
  --draws 50000
```

Recursion, geometry, and sign diagnostics:

```bash
.venv/bin/python -m evaluations.multimodal.analyze_formal_diagnostics \
  --manifest <formal-dataset.jsonl> \
  --model-output <formal-model.jsonl> \
  --attribution-dir <formal-attribution-dir> \
  --output-json <formal-attribution-dir>/diagnostics.json \
  --output-markdown <formal-attribution-dir>/diagnostics.md
```

Faithfulness paired bootstrap and A8:

```bash
.venv/bin/python -m evaluations.multimodal.analyze_formal_faithfulness \
  --faithfulness-dir <formal-faithfulness-dir> \
  --generation-evaluation <reviewed-generation-evaluation.jsonl> \
  --model-output <formal-model.jsonl> \
  --output-json <formal-faithfulness-dir>/analysis.json \
  --output-markdown <formal-faithfulness-dir>/analysis.md
```

The retained A6 diagnostics are regenerated without model inference:

```bash
.venv/bin/python -m evaluations.multimodal.analyze_legacy_diagnostics \
  --root . \
  --output-json evaluations/multimodal/results/strict/formal/A6_LEGACY_DIAGNOSTICS.json \
  --output-markdown evaluations/multimodal/results/strict/formal/A6_LEGACY_DIAGNOSTICS.md
```

After semantic review is complete, rerun the VizWiz faithfulness analysis with
the reviewed generation evaluation so the fully-correct A8 subset is present.
Then render the formal report and the two paper table bodies:

```bash
.venv/bin/python -m evaluations.multimodal.render_formal_results \
  --formal-dir evaluations/multimodal/results/strict/formal \
  --output-markdown evaluations/multimodal/results/strict/formal/RESULTS.md \
  --localization-tex paper/generated/visual_localization_rows.tex \
  --faithfulness-tex paper/generated/visual_faithfulness_rows.tex \
  --appendix-tex paper/generated/visual_appendix_results.tex \
  --discussion-tex paper/generated/visual_results_discussion.tex
```

The complete CPU-only post-review sequence above is also encoded as one
resume-safe command. It validates all four E3--E5 paired matrices, rejects
blank human reviews, adjudicates the semantic labels, regenerates every 50k
analysis, renders the report and paper fragments, and exits non-zero unless
the final protocol audit is complete:

```bash
bash evaluations/multimodal/finalize_formal_results.sh
```

## Final protocol audit

The read-only audit remains non-zero while an artifact is absent and
distinguishes incomplete work from a protocol violation:

```bash
.venv/bin/python -m evaluations.multimodal.audit_formal_results \
  --root . \
  --output evaluations/multimodal/results/strict/formal/AUDIT.json
```

Do not publish the tables unless `AUDIT.json` reports `complete: true`, zero
protocol violations, and zero incomplete checks.
