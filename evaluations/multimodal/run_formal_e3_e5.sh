#!/usr/bin/env bash
set -euo pipefail

# Resume-safe E3--E5 runner for a single GPU. It can wait for the E2 runner,
# then validates both immutable frozen sets before computing any attribution.
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [e2-runner-pid]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  while kill -0 "$1" 2>/dev/null; do
    sleep 20
  done
fi

FLASH_E35_ROOT=$(git rev-parse --show-toplevel)
cd "$FLASH_E35_ROOT"
FLASH_E35_DIR=evaluations/multimodal/results/strict/formal

if [[ ! -f "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" ]] ||
  [[ $(wc -l < "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl") -ne 120 ]]; then
  echo "Wiki frozen set is absent or not n=120" >&2
  exit 3
fi
if [[ ! -f "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" ]] ||
  [[ $(wc -l < "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl") -ne 100 ]]; then
  echo "VizWiz frozen set is absent or not n=100" >&2
  exit 4
fi

for FLASH_E35_PREFIX in wiki_visa_n20 vizwiz_lf_n20; do
  if [[ "$FLASH_E35_PREFIX" == "wiki_visa_n20" ]]; then
    FLASH_E35_FORMAL_PREFIX=wiki_visa_n120
  else
    FLASH_E35_FORMAL_PREFIX=vizwiz_lf_n100
  fi
  .venv/bin/python -m evaluations.multimodal.reuse_preview_checkpoints \
    --formal-dataset "$FLASH_E35_DIR/$FLASH_E35_FORMAL_PREFIX.dataset.jsonl" \
    --formal-model "$FLASH_E35_DIR/$FLASH_E35_FORMAL_PREFIX.model.jsonl" \
    --preview-dataset "evaluations/multimodal/results/strict/formal_preview_n20/$FLASH_E35_PREFIX.dataset.jsonl" \
    --preview-model "evaluations/multimodal/results/strict/formal_preview_n20/$FLASH_E35_PREFIX.model.jsonl" \
    --preview-attribution-dir "evaluations/multimodal/results/strict/formal_preview_n20/${FLASH_E35_PREFIX}_methods" \
    --preview-faithfulness-dir "evaluations/multimodal/results/strict/formal_preview_n20/${FLASH_E35_PREFIX}_faithfulness" \
    --formal-attribution-dir "$FLASH_E35_DIR/${FLASH_E35_FORMAL_PREFIX}_methods" \
    --formal-faithfulness-dir "$FLASH_E35_DIR/${FLASH_E35_FORMAL_PREFIX}_faithfulness" \
    --summary-output "$FLASH_E35_DIR/${FLASH_E35_FORMAL_PREFIX}.preview_reuse.json"
done

.venv/bin/python -m evaluations.multimodal.strict_attribution \
  --dataset-manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
  --model-output "$FLASH_E35_DIR/wiki_visa_n120.model.jsonl" \
  --generation-evaluation "$FLASH_E35_DIR/wiki_visa_n120.generation_eval.jsonl" \
  --output-dir "$FLASH_E35_DIR/wiki_visa_n120_methods"
if [[ $(jq -r .common_samples "$FLASH_E35_DIR/wiki_visa_n120_methods/summary.json") -ne 120 ]]; then
  .venv/bin/python -m evaluations.multimodal.strict_attribution \
    --dataset-manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
    --model-output "$FLASH_E35_DIR/wiki_visa_n120.model.jsonl" \
    --generation-evaluation "$FLASH_E35_DIR/wiki_visa_n120.generation_eval.jsonl" \
    --output-dir "$FLASH_E35_DIR/wiki_visa_n120_methods"
fi
if [[ $(jq -r .common_samples "$FLASH_E35_DIR/wiki_visa_n120_methods/summary.json") -ne 120 ]]; then
  echo "Wiki attribution is not a complete paired n=120 matrix" >&2
  exit 5
fi
.venv/bin/python -m evaluations.multimodal.validate_paired_matrix \
  --manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
  --evaluation-dir "$FLASH_E35_DIR/wiki_visa_n120_methods" \
  --kind attribution --expected-samples 120

.venv/bin/python -m evaluations.multimodal.strict_attribution \
  --dataset-manifest "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --model-output "$FLASH_E35_DIR/vizwiz_lf_n100.model.jsonl" \
  --generation-evaluation "$FLASH_E35_DIR/vizwiz_lf_n100.generation_eval.jsonl" \
  --output-dir "$FLASH_E35_DIR/vizwiz_lf_n100_methods" \
  --allow-missing-evidence
if [[ $(jq -r .common_samples "$FLASH_E35_DIR/vizwiz_lf_n100_methods/summary.json") -ne 100 ]]; then
  .venv/bin/python -m evaluations.multimodal.strict_attribution \
    --dataset-manifest "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" \
    --model-output "$FLASH_E35_DIR/vizwiz_lf_n100.model.jsonl" \
    --generation-evaluation "$FLASH_E35_DIR/vizwiz_lf_n100.generation_eval.jsonl" \
    --output-dir "$FLASH_E35_DIR/vizwiz_lf_n100_methods" \
    --allow-missing-evidence
fi
if [[ $(jq -r .common_samples "$FLASH_E35_DIR/vizwiz_lf_n100_methods/summary.json") -ne 100 ]]; then
  echo "VizWiz attribution is not a complete paired n=100 matrix" >&2
  exit 6
fi
.venv/bin/python -m evaluations.multimodal.validate_paired_matrix \
  --manifest "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --evaluation-dir "$FLASH_E35_DIR/vizwiz_lf_n100_methods" \
  --kind attribution --expected-samples 100

.venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
  --dataset-manifest "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --model-output "$FLASH_E35_DIR/vizwiz_lf_n100.model.jsonl" \
  --attribution-dir "$FLASH_E35_DIR/vizwiz_lf_n100_methods" \
  --output-dir "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness"
if [[ $(jq -r .common_samples "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness/summary.json") -ne 100 ]]; then
  .venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
    --dataset-manifest "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" \
    --model-output "$FLASH_E35_DIR/vizwiz_lf_n100.model.jsonl" \
    --attribution-dir "$FLASH_E35_DIR/vizwiz_lf_n100_methods" \
    --output-dir "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness"
fi
if [[ $(jq -r .common_samples "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness/summary.json") -ne 100 ]]; then
  echo "VizWiz faithfulness is not a complete paired n=100 matrix" >&2
  exit 7
fi
.venv/bin/python -m evaluations.multimodal.validate_paired_matrix \
  --manifest "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --evaluation-dir "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness" \
  --kind faithfulness --expected-samples 100
.venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
  --dataset-manifest "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --model-output "$FLASH_E35_DIR/vizwiz_lf_n100.model.jsonl" \
  --attribution-dir "$FLASH_E35_DIR/vizwiz_lf_n100_methods" \
  --output-dir "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness" \
  --summary-only \
  --refresh-derived-metrics

.venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
  --dataset-manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
  --model-output "$FLASH_E35_DIR/wiki_visa_n120.model.jsonl" \
  --attribution-dir "$FLASH_E35_DIR/wiki_visa_n120_methods" \
  --output-dir "$FLASH_E35_DIR/wiki_visa_n120_faithfulness"
if [[ $(jq -r .common_samples "$FLASH_E35_DIR/wiki_visa_n120_faithfulness/summary.json") -ne 120 ]]; then
  .venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
    --dataset-manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
    --model-output "$FLASH_E35_DIR/wiki_visa_n120.model.jsonl" \
    --attribution-dir "$FLASH_E35_DIR/wiki_visa_n120_methods" \
    --output-dir "$FLASH_E35_DIR/wiki_visa_n120_faithfulness"
fi
if [[ $(jq -r .common_samples "$FLASH_E35_DIR/wiki_visa_n120_faithfulness/summary.json") -ne 120 ]]; then
  echo "Wiki faithfulness is not a complete paired n=120 matrix" >&2
  exit 8
fi
.venv/bin/python -m evaluations.multimodal.validate_paired_matrix \
  --manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
  --evaluation-dir "$FLASH_E35_DIR/wiki_visa_n120_faithfulness" \
  --kind faithfulness --expected-samples 120
.venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
  --dataset-manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
  --model-output "$FLASH_E35_DIR/wiki_visa_n120.model.jsonl" \
  --attribution-dir "$FLASH_E35_DIR/wiki_visa_n120_methods" \
  --output-dir "$FLASH_E35_DIR/wiki_visa_n120_faithfulness" \
  --summary-only \
  --refresh-derived-metrics

.venv/bin/python -m evaluations.multimodal.analyze_strict_results \
  --manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
  --attribution-dir "$FLASH_E35_DIR/wiki_visa_n120_methods" \
  --output-json "$FLASH_E35_DIR/wiki_visa_n120_methods/analysis.json" \
  --output-markdown "$FLASH_E35_DIR/wiki_visa_n120_methods/analysis.md" \
  --draws 50000

.venv/bin/python -m evaluations.multimodal.analyze_formal_diagnostics \
  --manifest "$FLASH_E35_DIR/wiki_visa_n120.dataset.jsonl" \
  --model-output "$FLASH_E35_DIR/wiki_visa_n120.model.jsonl" \
  --attribution-dir "$FLASH_E35_DIR/wiki_visa_n120_methods" \
  --output-json "$FLASH_E35_DIR/wiki_visa_n120_methods/diagnostics.json" \
  --output-markdown "$FLASH_E35_DIR/wiki_visa_n120_methods/diagnostics.md" \
  --draws 50000

.venv/bin/python -m evaluations.multimodal.analyze_formal_diagnostics \
  --manifest "$FLASH_E35_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --model-output "$FLASH_E35_DIR/vizwiz_lf_n100.model.jsonl" \
  --attribution-dir "$FLASH_E35_DIR/vizwiz_lf_n100_methods" \
  --output-json "$FLASH_E35_DIR/vizwiz_lf_n100_methods/diagnostics.json" \
  --output-markdown "$FLASH_E35_DIR/vizwiz_lf_n100_methods/diagnostics.md" \
  --draws 50000

.venv/bin/python -m evaluations.multimodal.analyze_formal_faithfulness \
  --faithfulness-dir "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness" \
  --model-output "$FLASH_E35_DIR/vizwiz_lf_n100.model.jsonl" \
  --output-json "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness/analysis.json" \
  --output-markdown "$FLASH_E35_DIR/vizwiz_lf_n100_faithfulness/analysis.md" \
  --draws 50000

.venv/bin/python -m evaluations.multimodal.analyze_formal_faithfulness \
  --faithfulness-dir "$FLASH_E35_DIR/wiki_visa_n120_faithfulness" \
  --model-output "$FLASH_E35_DIR/wiki_visa_n120.model.jsonl" \
  --output-json "$FLASH_E35_DIR/wiki_visa_n120_faithfulness/analysis.json" \
  --output-markdown "$FLASH_E35_DIR/wiki_visa_n120_faithfulness/analysis.md" \
  --draws 50000
