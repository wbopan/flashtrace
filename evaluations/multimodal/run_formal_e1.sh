#!/usr/bin/env bash
set -euo pipefail

# Resume-safe E1 runner. When passed a PID, wait for that already-running
# initial generation process before validating its complete 240-row evaluation.
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [initial-generation-pid]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  while kill -0 "$1" 2>/dev/null; do
    sleep 20
  done
fi

FLASH_E1_ROOT=$(git rev-parse --show-toplevel)
cd "$FLASH_E1_ROOT"
FLASH_E1_DIR=evaluations/multimodal/results/strict/formal

if [[ $(wc -l < "$FLASH_E1_DIR/wiki_visa_candidates.generation_eval.jsonl") -ne 240 ]]; then
  echo "initial Wiki generation is incomplete; refusing to advance E1" >&2
  exit 3
fi

.venv/bin/python -m evaluations.multimodal.strict_ablation_audit \
  --dataset-manifest "$FLASH_E1_DIR/wiki_visa_candidates.dataset.jsonl" \
  --model-output "$FLASH_E1_DIR/wiki_visa_candidates.model.jsonl" \
  --generation-evaluation "$FLASH_E1_DIR/wiki_visa_candidates.generation_eval.jsonl" \
  --ablation-model-output "$FLASH_E1_DIR/wiki_visa_candidates.ablation.model.jsonl" \
  --revised-evaluation-output "$FLASH_E1_DIR/wiki_visa_candidates.strict.generation_eval.jsonl"

.venv/bin/python -m evaluations.multimodal.strict_generation \
  --dataset-manifest "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.dataset.jsonl" \
  --model-output "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.model.jsonl" \
  --evaluation-output "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.generation_eval.jsonl" \
  --resume \
  --skip-recorded-deterministic-errors

if [[ $(wc -l < "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.generation_eval.jsonl") -ne 600 ]]; then
  echo "Wiki extension generation is incomplete; refusing to freeze IDs" >&2
  exit 4
fi

.venv/bin/python -m evaluations.multimodal.strict_ablation_audit \
  --dataset-manifest "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.dataset.jsonl" \
  --model-output "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.model.jsonl" \
  --generation-evaluation "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.generation_eval.jsonl" \
  --ablation-model-output "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.ablation.model.jsonl" \
  --revised-evaluation-output "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.strict.generation_eval.jsonl"

.venv/bin/python -m evaluations.multimodal.select_strict_subset \
  --dataset-manifest "$FLASH_E1_DIR/wiki_visa_candidates.dataset.jsonl" \
  --dataset-manifest "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.dataset.jsonl" \
  --model-output "$FLASH_E1_DIR/wiki_visa_candidates.model.jsonl" \
  --model-output "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.model.jsonl" \
  --generation-evaluation "$FLASH_E1_DIR/wiki_visa_candidates.strict.generation_eval.jsonl" \
  --generation-evaluation "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.strict.generation_eval.jsonl" \
  --exclude-manifest evaluations/multimodal/results/strict/final/wiki_visa_n18_2mp.dataset.jsonl \
  --exclude-manifest evaluations/multimodal/results/strict/native_pilot/wiki_visa_n10.dataset.jsonl \
  --sample-size 120 \
  --balance-key stratum \
  --seed 17 \
  --output-dataset "$FLASH_E1_DIR/wiki_visa_n120.dataset.jsonl" \
  --output-model "$FLASH_E1_DIR/wiki_visa_n120.model.jsonl" \
  --output-evaluation "$FLASH_E1_DIR/wiki_visa_n120.generation_eval.jsonl" \
  --funnel-output "$FLASH_E1_DIR/wiki_visa_funnel.json" \
  --frozen-ids-output "$FLASH_E1_DIR/frozen_ids.json"

.venv/bin/python -m evaluations.multimodal.formal_manual_audit prepare \
  --dataset-manifest "$FLASH_E1_DIR/wiki_visa_n120.dataset.jsonl" \
  --model-output "$FLASH_E1_DIR/wiki_visa_n120.model.jsonl" \
  --generation-evaluation "$FLASH_E1_DIR/wiki_visa_n120.generation_eval.jsonl" \
  --ablation-model-output "$FLASH_E1_DIR/wiki_visa_candidates.ablation.model.jsonl" \
  --ablation-model-output "$FLASH_E1_DIR/wiki_visa_candidates_extension_seed31_n600.ablation.model.jsonl" \
  --output-markdown "$FLASH_E1_DIR/wiki_visa_n120.protocol_audit.md" \
  --review-template "$FLASH_E1_DIR/wiki_visa_n120.protocol_reviews.jsonl"
