#!/usr/bin/env bash
set -euo pipefail

# Resume-safe single-GPU supervisor for E1 -> optional Wiki extension -> E2 ->
# E3-E5. A non-freeze E1 exit may enter the fixed seed47 contingency only
# after the complete seed31 generation and ablation artifacts exist.

FLASH_ALL_ROOT=$(git rev-parse --show-toplevel)
cd "$FLASH_ALL_ROOT"
FLASH_ALL_DIR=evaluations/multimodal/results/strict/formal

FLASH_ALL_E1_STATUS=0
bash evaluations/multimodal/run_formal_e1.sh || FLASH_ALL_E1_STATUS=$?
if [[ ! -f "$FLASH_ALL_DIR/wiki_visa_n120.dataset.jsonl" ]] ||
  [[ $(wc -l < "$FLASH_ALL_DIR/wiki_visa_n120.dataset.jsonl" 2>/dev/null || true) -ne 120 ]]; then
  if [[ "$FLASH_ALL_E1_STATUS" -eq 0 ]]; then
    echo "E1 exited successfully without freezing Wiki n=120" >&2
    exit 3
  fi
  if [[ ! -f "$FLASH_ALL_DIR/wiki_visa_candidates_extension_seed31_n600.strict.generation_eval.jsonl" ]] ||
    [[ $(wc -l < "$FLASH_ALL_DIR/wiki_visa_candidates_extension_seed31_n600.strict.generation_eval.jsonl" 2>/dev/null || true) -ne 600 ]]; then
    echo "E1 failed before the seed31 gate audit completed; refusing contingency" >&2
    exit "$FLASH_ALL_E1_STATUS"
  fi
  bash evaluations/multimodal/run_formal_e1_later_extension.sh
fi

bash evaluations/multimodal/run_formal_e2.sh
bash evaluations/multimodal/run_formal_e3_e5.sh
