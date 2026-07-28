#!/usr/bin/env bash
set -euo pipefail

# Resume-safe E2 runner. When passed the E1 runner PID, it waits for E1 and
# verifies that the Wiki frozen set exists before taking over the single GPU.
if [[ $# -gt 1 ]]; then
  echo "usage: $0 [e1-runner-pid]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  while kill -0 "$1" 2>/dev/null; do
    sleep 20
  done
fi

FLASH_E2_ROOT=$(git rev-parse --show-toplevel)
cd "$FLASH_E2_ROOT"
FLASH_E2_DIR=evaluations/multimodal/results/strict/formal

if [[ ! -f "$FLASH_E2_DIR/wiki_visa_n120.dataset.jsonl" ]] ||
  [[ $(wc -l < "$FLASH_E2_DIR/wiki_visa_n120.dataset.jsonl") -ne 120 ]]; then
  echo "E1 did not freeze Wiki n=120; refusing to start E2" >&2
  exit 3
fi

# The candidate generation may be opportunistically prefilled while E1 is
# finishing.  Serialize writers so the supervisor can safely take over the
# same resume checkpoint without racing an already-running prefill process.
FLASH_E2_GENERATED=0
if [[ -f "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl" ]]; then
  FLASH_E2_GENERATED=$(wc -l < "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl")
fi
if [[ "$FLASH_E2_GENERATED" -ne 200 ]]; then
  exec 9>"$FLASH_E2_DIR/.vizwiz_lf_generation.lock"
  flock 9
  # Recheck after acquiring the lock: an opportunistic prefill may have
  # completed while this runner was waiting.
  FLASH_E2_GENERATED=0
  if [[ -f "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl" ]]; then
    FLASH_E2_GENERATED=$(wc -l < "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl")
  fi
  if [[ "$FLASH_E2_GENERATED" -ne 200 ]]; then
    .venv/bin/python -m evaluations.multimodal.strict_generation \
      --dataset-manifest "$FLASH_E2_DIR/vizwiz_lf_candidates.dataset.jsonl" \
      --model-output "$FLASH_E2_DIR/vizwiz_lf_candidates.model.jsonl" \
      --evaluation-output "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl" \
      --resume \
      --skip-recorded-deterministic-errors
  fi
  flock -u 9
fi

if [[ $(wc -l < "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl") -ne 200 ]]; then
  echo "VizWiz generation is incomplete; refusing to advance E2" >&2
  exit 4
fi

.venv/bin/python -m evaluations.multimodal.refresh_generation_gates \
  --dataset-manifest "$FLASH_E2_DIR/vizwiz_lf_candidates.dataset.jsonl" \
  --model-output "$FLASH_E2_DIR/vizwiz_lf_candidates.model.jsonl" \
  --evaluation-output "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl"

.venv/bin/python -m evaluations.multimodal.reuse_preview_ablation_checkpoints \
  --formal-dataset "$FLASH_E2_DIR/vizwiz_lf_candidates.dataset.jsonl" \
  --formal-model "$FLASH_E2_DIR/vizwiz_lf_candidates.model.jsonl" \
  --formal-generation-evaluation "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl" \
  --preview-dataset evaluations/multimodal/results/strict/formal_preview_n20/vizwiz_lf_candidates_n40.dataset.jsonl \
  --preview-model evaluations/multimodal/results/strict/formal_preview_n20/vizwiz_lf_candidates_n40.model.jsonl \
  --preview-ablation-model evaluations/multimodal/results/strict/formal_preview_n20/vizwiz_lf_candidates_n40.ablation.model.jsonl \
  --formal-ablation-model "$FLASH_E2_DIR/vizwiz_lf_candidates.ablation.model.jsonl" \
  --summary-output "$FLASH_E2_DIR/vizwiz_lf_candidates.preview_ablation_reuse.json"

.venv/bin/python -m evaluations.multimodal.strict_ablation_audit \
  --dataset-manifest "$FLASH_E2_DIR/vizwiz_lf_candidates.dataset.jsonl" \
  --model-output "$FLASH_E2_DIR/vizwiz_lf_candidates.model.jsonl" \
  --generation-evaluation "$FLASH_E2_DIR/vizwiz_lf_candidates.generation_eval.jsonl" \
  --ablation-model-output "$FLASH_E2_DIR/vizwiz_lf_candidates.ablation.model.jsonl" \
  --revised-evaluation-output "$FLASH_E2_DIR/vizwiz_lf_candidates.strict.generation_eval.jsonl"

.venv/bin/python -m evaluations.multimodal.select_strict_subset \
  --dataset-manifest "$FLASH_E2_DIR/vizwiz_lf_candidates.dataset.jsonl" \
  --model-output "$FLASH_E2_DIR/vizwiz_lf_candidates.model.jsonl" \
  --generation-evaluation "$FLASH_E2_DIR/vizwiz_lf_candidates.strict.generation_eval.jsonl" \
  --exclude-manifest evaluations/multimodal/results/strict/native_pilot/vizwiz_lf_n10.dataset.jsonl \
  --sample-size 100 \
  --seed 17 \
  --output-dataset "$FLASH_E2_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --output-model "$FLASH_E2_DIR/vizwiz_lf_n100.model.jsonl" \
  --output-evaluation "$FLASH_E2_DIR/vizwiz_lf_n100.generation_eval.jsonl" \
  --funnel-output "$FLASH_E2_DIR/vizwiz_lf_funnel.json" \
  --frozen-ids-output "$FLASH_E2_DIR/frozen_ids.json"

.venv/bin/python -m evaluations.multimodal.vizwiz_semantic_judgments prepare \
  --dataset-manifest "$FLASH_E2_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --model-output "$FLASH_E2_DIR/vizwiz_lf_n100.model.jsonl" \
  --output "$FLASH_E2_DIR/vizwiz_lf_n100.semantic_tasks.jsonl"

.venv/bin/python -m evaluations.multimodal.formal_manual_audit prepare \
  --dataset-manifest "$FLASH_E2_DIR/vizwiz_lf_n100.dataset.jsonl" \
  --model-output "$FLASH_E2_DIR/vizwiz_lf_n100.model.jsonl" \
  --generation-evaluation "$FLASH_E2_DIR/vizwiz_lf_n100.generation_eval.jsonl" \
  --ablation-model-output "$FLASH_E2_DIR/vizwiz_lf_candidates.ablation.model.jsonl" \
  --output-markdown "$FLASH_E2_DIR/vizwiz_lf_n100.protocol_audit.md" \
  --review-template "$FLASH_E2_DIR/vizwiz_lf_n100.protocol_reviews.jsonl"
