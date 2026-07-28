#!/usr/bin/env bash
set -euo pipefail

# End-to-end, pilot-disjoint n=20 preview using the frozen formal model,
# gates, eight-method panel, and 64-region/10-step faithfulness budget.
# Preview artifacts are isolated and never merged into formal frozen_ids.json.

if [[ $# -gt 1 ]]; then
  echo "usage: $0 [already-running-viz-generation-pid]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  while kill -0 "$1" 2>/dev/null; do
    sleep 10
  done
fi

FLASH_PREVIEW_ROOT=$(git rev-parse --show-toplevel)
cd "$FLASH_PREVIEW_ROOT"
FLASH_FORMAL_DIR=evaluations/multimodal/results/strict/formal
FLASH_PREVIEW_DIR=evaluations/multimodal/results/strict/formal_preview_n20
mkdir -p "$FLASH_PREVIEW_DIR"

.venv/bin/python -m evaluations.multimodal.select_strict_subset \
  --dataset-manifest "$FLASH_FORMAL_DIR/wiki_visa_candidates.dataset.jsonl" \
  --model-output "$FLASH_FORMAL_DIR/wiki_visa_candidates.model.jsonl" \
  --generation-evaluation "$FLASH_FORMAL_DIR/wiki_visa_candidates.strict.generation_eval.jsonl" \
  --exclude-manifest evaluations/multimodal/results/strict/final/wiki_visa_n18_2mp.dataset.jsonl \
  --exclude-manifest evaluations/multimodal/results/strict/native_pilot/wiki_visa_n10.dataset.jsonl \
  --sample-size 20 \
  --balance-key stratum \
  --seed 101 \
  --output-dataset "$FLASH_PREVIEW_DIR/wiki_visa_n20.dataset.jsonl" \
  --output-model "$FLASH_PREVIEW_DIR/wiki_visa_n20.model.jsonl" \
  --output-evaluation "$FLASH_PREVIEW_DIR/wiki_visa_n20.generation_eval.jsonl" \
  --funnel-output "$FLASH_PREVIEW_DIR/wiki_visa_n20_funnel.json"

.venv/bin/python -m evaluations.multimodal.select_preview_candidates \
  --source-manifest "$FLASH_FORMAL_DIR/vizwiz_lf_candidates.dataset.jsonl" \
  --exclude-manifest evaluations/multimodal/results/strict/native_pilot/vizwiz_lf_n10.dataset.jsonl \
  --sample-size 40 \
  --seed 101 \
  --output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.dataset.jsonl"

if [[ ! -f "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.generation_eval.jsonl" ]] ||
  [[ $(wc -l < "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.generation_eval.jsonl") -ne 40 ]]; then
  .venv/bin/python -m evaluations.multimodal.strict_generation \
    --dataset-manifest "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.dataset.jsonl" \
    --model-output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.model.jsonl" \
    --evaluation-output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.generation_eval.jsonl" \
    --resume \
    --skip-recorded-deterministic-errors
fi

if [[ $(wc -l < "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.generation_eval.jsonl") -ne 40 ]]; then
  echo "VizWiz preview candidate generation is incomplete" >&2
  exit 3
fi

.venv/bin/python -m evaluations.multimodal.refresh_generation_gates \
  --dataset-manifest "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.dataset.jsonl" \
  --model-output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.model.jsonl" \
  --evaluation-output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.generation_eval.jsonl"

.venv/bin/python -m evaluations.multimodal.strict_ablation_audit \
  --dataset-manifest "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.dataset.jsonl" \
  --model-output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.model.jsonl" \
  --generation-evaluation "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.generation_eval.jsonl" \
  --ablation-model-output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.ablation.model.jsonl" \
  --revised-evaluation-output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.strict.generation_eval.jsonl"

.venv/bin/python -m evaluations.multimodal.select_strict_subset \
  --dataset-manifest "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.dataset.jsonl" \
  --model-output "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.model.jsonl" \
  --generation-evaluation "$FLASH_PREVIEW_DIR/vizwiz_lf_candidates_n40.strict.generation_eval.jsonl" \
  --sample-size 20 \
  --seed 101 \
  --output-dataset "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.dataset.jsonl" \
  --output-model "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.model.jsonl" \
  --output-evaluation "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.generation_eval.jsonl" \
  --funnel-output "$FLASH_PREVIEW_DIR/vizwiz_lf_n20_funnel.json"

.venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments prepare \
  --dataset-manifest "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.dataset.jsonl" \
  --model-output "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.model.jsonl" \
  --output "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.semantic_tasks.jsonl"

run_attribution() {
  local prefix=$1
  shift
  .venv/bin/python -m evaluations.multimodal.strict_attribution \
    --dataset-manifest "$FLASH_PREVIEW_DIR/$prefix.dataset.jsonl" \
    --model-output "$FLASH_PREVIEW_DIR/$prefix.model.jsonl" \
    --generation-evaluation "$FLASH_PREVIEW_DIR/$prefix.generation_eval.jsonl" \
    --output-dir "$FLASH_PREVIEW_DIR/${prefix}_methods" \
    "$@"
  if [[ $(jq -r .common_samples "$FLASH_PREVIEW_DIR/${prefix}_methods/summary.json") -ne 20 ]]; then
    .venv/bin/python -m evaluations.multimodal.strict_attribution \
      --dataset-manifest "$FLASH_PREVIEW_DIR/$prefix.dataset.jsonl" \
      --model-output "$FLASH_PREVIEW_DIR/$prefix.model.jsonl" \
      --generation-evaluation "$FLASH_PREVIEW_DIR/$prefix.generation_eval.jsonl" \
      --output-dir "$FLASH_PREVIEW_DIR/${prefix}_methods" \
      "$@"
  fi
  if [[ $(jq -r .common_samples "$FLASH_PREVIEW_DIR/${prefix}_methods/summary.json") -ne 20 ]]; then
    echo "$prefix attribution is not a complete paired n=20 matrix" >&2
    exit 4
  fi
}

run_attribution wiki_visa_n20
run_attribution vizwiz_lf_n20 --allow-missing-evidence

run_faithfulness() {
  local prefix=$1
  .venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
    --dataset-manifest "$FLASH_PREVIEW_DIR/$prefix.dataset.jsonl" \
    --model-output "$FLASH_PREVIEW_DIR/$prefix.model.jsonl" \
    --attribution-dir "$FLASH_PREVIEW_DIR/${prefix}_methods" \
    --output-dir "$FLASH_PREVIEW_DIR/${prefix}_faithfulness" \
    --target-regions 64 \
    --steps 10
  if [[ $(jq -r .common_samples "$FLASH_PREVIEW_DIR/${prefix}_faithfulness/summary.json") -ne 20 ]]; then
    .venv/bin/python -m evaluations.multimodal.strict_visual_faithfulness \
      --dataset-manifest "$FLASH_PREVIEW_DIR/$prefix.dataset.jsonl" \
      --model-output "$FLASH_PREVIEW_DIR/$prefix.model.jsonl" \
      --attribution-dir "$FLASH_PREVIEW_DIR/${prefix}_methods" \
      --output-dir "$FLASH_PREVIEW_DIR/${prefix}_faithfulness" \
      --target-regions 64 \
      --steps 10
  fi
  if [[ $(jq -r .common_samples "$FLASH_PREVIEW_DIR/${prefix}_faithfulness/summary.json") -ne 20 ]]; then
    echo "$prefix faithfulness is not a complete paired n=20 matrix" >&2
    exit 5
  fi
}

run_faithfulness vizwiz_lf_n20
run_faithfulness wiki_visa_n20

.venv/bin/python -m evaluations.multimodal.analyze_strict_results \
  --manifest "$FLASH_PREVIEW_DIR/wiki_visa_n20.dataset.jsonl" \
  --attribution-dir "$FLASH_PREVIEW_DIR/wiki_visa_n20_methods" \
  --output-json "$FLASH_PREVIEW_DIR/wiki_visa_n20_methods/analysis.json" \
  --output-markdown "$FLASH_PREVIEW_DIR/wiki_visa_n20_methods/analysis.md" \
  --draws 50000

for prefix in wiki_visa_n20 vizwiz_lf_n20; do
  .venv/bin/python -m evaluations.multimodal.analyze_formal_diagnostics \
    --manifest "$FLASH_PREVIEW_DIR/$prefix.dataset.jsonl" \
    --model-output "$FLASH_PREVIEW_DIR/$prefix.model.jsonl" \
    --attribution-dir "$FLASH_PREVIEW_DIR/${prefix}_methods" \
    --output-json "$FLASH_PREVIEW_DIR/${prefix}_methods/diagnostics.json" \
    --output-markdown "$FLASH_PREVIEW_DIR/${prefix}_methods/diagnostics.md" \
    --draws 50000
  .venv/bin/python -m evaluations.multimodal.analyze_formal_faithfulness \
    --faithfulness-dir "$FLASH_PREVIEW_DIR/${prefix}_faithfulness" \
    --model-output "$FLASH_PREVIEW_DIR/$prefix.model.jsonl" \
    --output-json "$FLASH_PREVIEW_DIR/${prefix}_faithfulness/analysis.json" \
    --output-markdown "$FLASH_PREVIEW_DIR/${prefix}_faithfulness/analysis.md" \
    --draws 50000
done

if [[ -f "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.semantic_judgments.jsonl" ]]; then
  .venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments join \
    --generation-evaluation "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.generation_eval.jsonl" \
    --judgments "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.semantic_judgments.jsonl" \
    --output-evaluation "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.reviewed.generation_eval.jsonl" \
    --summary-output "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.semantic_summary.json"
  .venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments audit-packet \
    --dataset-manifest "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.dataset.jsonl" \
    --model-output "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.model.jsonl" \
    --judgments "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.semantic_judgments.jsonl" \
    --output-markdown "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.semantic_audit.md" \
    --review-template "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.semantic_audit_template.jsonl"
  .venv/bin/python -m evaluations.multimodal.analyze_formal_faithfulness \
    --faithfulness-dir "$FLASH_PREVIEW_DIR/vizwiz_lf_n20_faithfulness" \
    --generation-evaluation "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.reviewed.generation_eval.jsonl" \
    --model-output "$FLASH_PREVIEW_DIR/vizwiz_lf_n20.model.jsonl" \
    --output-json "$FLASH_PREVIEW_DIR/vizwiz_lf_n20_faithfulness/analysis.json" \
    --output-markdown "$FLASH_PREVIEW_DIR/vizwiz_lf_n20_faithfulness/analysis.md" \
    --draws 50000
  .venv/bin/python -m evaluations.multimodal.render_preview_results \
    --preview-dir "$FLASH_PREVIEW_DIR" \
    --output "$FLASH_PREVIEW_DIR/RESULTS.md"
fi
