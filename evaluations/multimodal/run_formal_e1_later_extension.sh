#!/usr/bin/env bash
set -euo pipefail

# Resume-safe contingency runner for the preregistered "extend, never relax"
# policy when the initial and seed31 Wiki pools cannot supply 40 later-page
# rows after every strict gate.

FLASH_E1X_ROOT=$(git rev-parse --show-toplevel)
cd "$FLASH_E1X_ROOT"
FLASH_E1X_DIR=evaluations/multimodal/results/strict/formal
FLASH_E1X_PREFIX=wiki_visa_candidates_later_extension_seed47_n600

if [[ -f "$FLASH_E1X_DIR/wiki_visa_n120.dataset.jsonl" ]]; then
  echo "Wiki n=120 is already frozen; refusing to change its candidate pool" >&2
  exit 3
fi
if [[ $(wc -l < "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.dataset.jsonl") -ne 600 ]]; then
  echo "fixed seed47 later-page manifest is absent or incomplete" >&2
  exit 4
fi
if [[ $(wc -l < "$FLASH_E1X_DIR/wiki_visa_candidates_extension_seed31_n600.strict.generation_eval.jsonl") -ne 600 ]]; then
  echo "seed31 Wiki extension has not completed ablation audit" >&2
  exit 5
fi

FLASH_E1X_CURRENT_ROWS=0
if [[ -f "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.generation_eval.jsonl" ]]; then
  FLASH_E1X_CURRENT_ROWS=$(wc -l < "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.generation_eval.jsonl")
fi
FLASH_E1X_FROZEN=0
FLASH_E1X_USED_PREFIX=
for FLASH_E1X_COUNT in 100 200 300 400 500 600; do
  if (( FLASH_E1X_COUNT < FLASH_E1X_CURRENT_ROWS )); then
    continue
  fi
  FLASH_E1X_PREFIX_MANIFEST="$FLASH_E1X_DIR/${FLASH_E1X_PREFIX%.dataset}_prefix_n${FLASH_E1X_COUNT}.dataset.jsonl"
  .venv/bin/python -m evaluations.multimodal.materialize_manifest_prefix \
    --source "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.dataset.jsonl" \
    --count "$FLASH_E1X_COUNT" \
    --output "$FLASH_E1X_PREFIX_MANIFEST"
  .venv/bin/python -m evaluations.multimodal.strict_generation \
    --dataset-manifest "$FLASH_E1X_PREFIX_MANIFEST" \
    --model-output "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.model.jsonl" \
    --evaluation-output "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.generation_eval.jsonl" \
    --resume \
    --skip-recorded-deterministic-errors
  if [[ $(wc -l < "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.generation_eval.jsonl") -ne "$FLASH_E1X_COUNT" ]]; then
    echo "seed47 prefix generation is incomplete at n=$FLASH_E1X_COUNT" >&2
    exit 6
  fi
  .venv/bin/python -m evaluations.multimodal.strict_ablation_audit \
    --dataset-manifest "$FLASH_E1X_PREFIX_MANIFEST" \
    --model-output "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.model.jsonl" \
    --generation-evaluation "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.generation_eval.jsonl" \
    --ablation-model-output "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.ablation.model.jsonl" \
    --revised-evaluation-output "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.strict.generation_eval.jsonl"
  if .venv/bin/python -m evaluations.multimodal.select_strict_subset \
    --dataset-manifest "$FLASH_E1X_DIR/wiki_visa_candidates.dataset.jsonl" \
    --dataset-manifest "$FLASH_E1X_DIR/wiki_visa_candidates_extension_seed31_n600.dataset.jsonl" \
    --dataset-manifest "$FLASH_E1X_PREFIX_MANIFEST" \
    --model-output "$FLASH_E1X_DIR/wiki_visa_candidates.model.jsonl" \
    --model-output "$FLASH_E1X_DIR/wiki_visa_candidates_extension_seed31_n600.model.jsonl" \
    --model-output "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.model.jsonl" \
    --generation-evaluation "$FLASH_E1X_DIR/wiki_visa_candidates.strict.generation_eval.jsonl" \
    --generation-evaluation "$FLASH_E1X_DIR/wiki_visa_candidates_extension_seed31_n600.strict.generation_eval.jsonl" \
    --generation-evaluation "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.strict.generation_eval.jsonl" \
    --exclude-manifest evaluations/multimodal/results/strict/final/wiki_visa_n18_2mp.dataset.jsonl \
    --exclude-manifest evaluations/multimodal/results/strict/native_pilot/wiki_visa_n10.dataset.jsonl \
    --sample-size 120 \
    --balance-key stratum \
    --seed 17 \
    --output-dataset "$FLASH_E1X_DIR/wiki_visa_n120.dataset.jsonl" \
    --output-model "$FLASH_E1X_DIR/wiki_visa_n120.model.jsonl" \
    --output-evaluation "$FLASH_E1X_DIR/wiki_visa_n120.generation_eval.jsonl" \
    --funnel-output "$FLASH_E1X_DIR/wiki_visa_funnel.json" \
    --frozen-ids-output "$FLASH_E1X_DIR/frozen_ids.json"; then
    FLASH_E1X_FROZEN=1
    FLASH_E1X_USED_PREFIX="$FLASH_E1X_PREFIX_MANIFEST"
    break
  fi
done
if [[ "$FLASH_E1X_FROZEN" -ne 1 ]]; then
  echo "seed47 fixed extension exhausted 600 rows without a balanced Wiki n=120 freeze" >&2
  exit 7
fi

.venv/bin/python -m evaluations.multimodal.formal_manual_audit prepare \
  --dataset-manifest "$FLASH_E1X_DIR/wiki_visa_n120.dataset.jsonl" \
  --model-output "$FLASH_E1X_DIR/wiki_visa_n120.model.jsonl" \
  --generation-evaluation "$FLASH_E1X_DIR/wiki_visa_n120.generation_eval.jsonl" \
  --ablation-model-output "$FLASH_E1X_DIR/wiki_visa_candidates.ablation.model.jsonl" \
  --ablation-model-output "$FLASH_E1X_DIR/wiki_visa_candidates_extension_seed31_n600.ablation.model.jsonl" \
  --ablation-model-output "$FLASH_E1X_DIR/$FLASH_E1X_PREFIX.ablation.model.jsonl" \
  --output-markdown "$FLASH_E1X_DIR/wiki_visa_n120.protocol_audit.md" \
  --review-template "$FLASH_E1X_DIR/wiki_visa_n120.protocol_reviews.jsonl"
