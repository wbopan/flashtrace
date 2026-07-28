#!/usr/bin/env bash
set -euo pipefail

# CPU-only finalization after E3--E5 and all three human review templates are
# complete. Every command is deterministic and safe to rerun.

FLASH_FINAL_ROOT=$(git rev-parse --show-toplevel)
cd "$FLASH_FINAL_ROOT"
FLASH_FINAL_DIR=evaluations/multimodal/results/strict/formal

.venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
  --dataset-manifest "$FLASH_FINAL_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --model-output "$FLASH_FINAL_DIR/vizwiz_lf_n100.model.jsonl" \
  --attribution-dir "$FLASH_FINAL_DIR/vizwiz_lf_n100_methods" \
  --output-dir "$FLASH_FINAL_DIR/vizwiz_lf_n100_faithfulness" \
  --summary-only \
  --refresh-derived-metrics
.venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
  --dataset-manifest "$FLASH_FINAL_DIR/wiki_visa_n120.dataset.jsonl" \
  --model-output "$FLASH_FINAL_DIR/wiki_visa_n120.model.jsonl" \
  --attribution-dir "$FLASH_FINAL_DIR/wiki_visa_n120_methods" \
  --output-dir "$FLASH_FINAL_DIR/wiki_visa_n120_faithfulness" \
  --summary-only \
  --refresh-derived-metrics

if [[ $(jq -r .common_samples \
  "$FLASH_FINAL_DIR/wiki_visa_n120_methods/summary.json") -ne 120 ]]; then
  echo "Wiki attribution is not a complete paired n=120 matrix" >&2
  exit 3
fi
if [[ $(jq -r .common_samples \
  "$FLASH_FINAL_DIR/vizwiz_lf_n100_methods/summary.json") -ne 100 ]]; then
  echo "VizWiz attribution is not a complete paired n=100 matrix" >&2
  exit 4
fi
if [[ $(jq -r .common_samples \
  "$FLASH_FINAL_DIR/vizwiz_lf_n100_faithfulness/summary.json") -ne 100 ]]; then
  echo "VizWiz faithfulness is not a complete paired n=100 matrix" >&2
  exit 5
fi
if [[ $(jq -r .common_samples \
  "$FLASH_FINAL_DIR/wiki_visa_n120_faithfulness/summary.json") -ne 120 ]]; then
  echo "Wiki faithfulness is not a complete paired n=120 matrix" >&2
  exit 6
fi

.venv/bin/python -m evaluations.multimodal.validate_paired_matrix \
  --manifest "$FLASH_FINAL_DIR/wiki_visa_n120.dataset.jsonl" \
  --evaluation-dir "$FLASH_FINAL_DIR/wiki_visa_n120_methods" \
  --kind attribution --expected-samples 120
.venv/bin/python -m evaluations.multimodal.validate_paired_matrix \
  --manifest "$FLASH_FINAL_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --evaluation-dir "$FLASH_FINAL_DIR/vizwiz_lf_n100_methods" \
  --kind attribution --expected-samples 100
.venv/bin/python -m evaluations.multimodal.validate_paired_matrix \
  --manifest "$FLASH_FINAL_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --evaluation-dir "$FLASH_FINAL_DIR/vizwiz_lf_n100_faithfulness" \
  --kind faithfulness --expected-samples 100
.venv/bin/python -m evaluations.multimodal.validate_paired_matrix \
  --manifest "$FLASH_FINAL_DIR/wiki_visa_n120.dataset.jsonl" \
  --evaluation-dir "$FLASH_FINAL_DIR/wiki_visa_n120_faithfulness" \
  --kind faithfulness --expected-samples 120

# These validators fail before producing a publishable report if a human
# label, reviewer identity, or reason is absent.
.venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments apply-review \
  --judgments "$FLASH_FINAL_DIR/vizwiz_lf_n100.semantic_judgments.llm.jsonl" \
  --reviews "$FLASH_FINAL_DIR/vizwiz_lf_n100.human_reviews.jsonl" \
  --output "$FLASH_FINAL_DIR/vizwiz_lf_n100.semantic_judgments.jsonl"

.venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments join \
  --generation-evaluation "$FLASH_FINAL_DIR/vizwiz_lf_n100.generation_eval.jsonl" \
  --judgments "$FLASH_FINAL_DIR/vizwiz_lf_n100.semantic_judgments.jsonl" \
  --output-evaluation "$FLASH_FINAL_DIR/vizwiz_lf_n100.reviewed.generation_eval.jsonl" \
  --summary-output "$FLASH_FINAL_DIR/vizwiz_lf_n100.semantic_summary.json" \
  --require-complete

.venv/bin/python -m evaluations.multimodal.formal_manual_audit summarize \
  --dataset-manifest "$FLASH_FINAL_DIR/wiki_visa_n120.dataset.jsonl" \
  --reviews "$FLASH_FINAL_DIR/wiki_visa_n120.protocol_reviews.jsonl" \
  --output "$FLASH_FINAL_DIR/wiki_visa_n120.protocol_audit_summary.json"

.venv/bin/python -m evaluations.multimodal.formal_manual_audit summarize \
  --dataset-manifest "$FLASH_FINAL_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --reviews "$FLASH_FINAL_DIR/vizwiz_lf_n100.protocol_reviews.jsonl" \
  --output "$FLASH_FINAL_DIR/vizwiz_lf_n100.protocol_audit_summary.json"

.venv/bin/python -m evaluations.multimodal.analyze_strict_results \
  --manifest "$FLASH_FINAL_DIR/wiki_visa_n120.dataset.jsonl" \
  --attribution-dir "$FLASH_FINAL_DIR/wiki_visa_n120_methods" \
  --output-json "$FLASH_FINAL_DIR/wiki_visa_n120_methods/analysis.json" \
  --output-markdown "$FLASH_FINAL_DIR/wiki_visa_n120_methods/analysis.md" \
  --draws 50000

for FLASH_FINAL_PREFIX in wiki_visa_n120 vizwiz_lf_n100; do
  .venv/bin/python -m evaluations.multimodal.analyze_formal_diagnostics \
    --manifest "$FLASH_FINAL_DIR/$FLASH_FINAL_PREFIX.dataset.jsonl" \
    --model-output "$FLASH_FINAL_DIR/$FLASH_FINAL_PREFIX.model.jsonl" \
    --attribution-dir "$FLASH_FINAL_DIR/${FLASH_FINAL_PREFIX}_methods" \
    --output-json "$FLASH_FINAL_DIR/${FLASH_FINAL_PREFIX}_methods/diagnostics.json" \
    --output-markdown "$FLASH_FINAL_DIR/${FLASH_FINAL_PREFIX}_methods/diagnostics.md" \
    --draws 50000
done

.venv/bin/python -m evaluations.multimodal.analyze_formal_faithfulness \
  --faithfulness-dir "$FLASH_FINAL_DIR/vizwiz_lf_n100_faithfulness" \
  --generation-evaluation "$FLASH_FINAL_DIR/vizwiz_lf_n100.reviewed.generation_eval.jsonl" \
  --model-output "$FLASH_FINAL_DIR/vizwiz_lf_n100.model.jsonl" \
  --output-json "$FLASH_FINAL_DIR/vizwiz_lf_n100_faithfulness/analysis.json" \
  --output-markdown "$FLASH_FINAL_DIR/vizwiz_lf_n100_faithfulness/analysis.md" \
  --draws 50000

.venv/bin/python -m evaluations.multimodal.analyze_formal_faithfulness \
  --faithfulness-dir "$FLASH_FINAL_DIR/wiki_visa_n120_faithfulness" \
  --model-output "$FLASH_FINAL_DIR/wiki_visa_n120.model.jsonl" \
  --output-json "$FLASH_FINAL_DIR/wiki_visa_n120_faithfulness/analysis.json" \
  --output-markdown "$FLASH_FINAL_DIR/wiki_visa_n120_faithfulness/analysis.md" \
  --draws 50000

.venv/bin/python -m evaluations.multimodal.analyze_legacy_diagnostics \
  --root . \
  --output-json "$FLASH_FINAL_DIR/A6_LEGACY_DIAGNOSTICS.json" \
  --output-markdown "$FLASH_FINAL_DIR/A6_LEGACY_DIAGNOSTICS.md"

.venv/bin/python -m evaluations.multimodal.render_formal_results \
  --formal-dir "$FLASH_FINAL_DIR" \
  --output-markdown "$FLASH_FINAL_DIR/RESULTS.md" \
  --localization-tex paper/generated/visual_localization_rows.tex \
  --faithfulness-tex paper/generated/visual_faithfulness_rows.tex \
  --appendix-tex paper/generated/visual_appendix_results.tex \
  --discussion-tex paper/generated/visual_results_discussion.tex

.venv/bin/python -m evaluations.multimodal.audit_formal_results \
  --root . \
  --output "$FLASH_FINAL_DIR/AUDIT.json"
