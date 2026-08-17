#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  printf 'Usage: %s GENERATION_JSONL OUTPUT_DIR\n' "$0" >&2
  exit 2
fi

readonly REPO_ROOT="$(git rev-parse --show-toplevel)"
readonly GENERATION_JSONL="$1"
readonly OUTPUT_DIR="$2"
readonly ATTRIBUTION_MODEL="${AGENTDOJO_ATTRIBUTION_MODEL:-Qwen/Qwen3-8B}"
readonly ATTRIBUTION_REVISION="${AGENTDOJO_ATTRIBUTION_REVISION:-b968826d9c46dd6066d109eabc6255188de91218}"
readonly DEVICE="${AGENTDOJO_ATTRIBUTION_DEVICE:-cuda:0}"
readonly N_HOPS="${AGENTDOJO_ATTRIBUTION_HOPS:-1}"
readonly ATTACKED_INVENTORY="${AGENTDOJO_ATTACKED_INVENTORY:-949}"

cd "${REPO_ROOT}"

if [[ ! -f "${GENERATION_JSONL}" ]]; then
  printf 'Missing generation manifest: %s\n' "${GENERATION_JSONL}" >&2
  exit 1
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
HF_HUB_OFFLINE=1 \
  uv run --project evaluations/agentdojo --offline \
  python -m evaluations.agentdojo.attribute_full \
  --generation-jsonl "${GENERATION_JSONL}" \
  --model "${ATTRIBUTION_MODEL}" \
  --revision "${ATTRIBUTION_REVISION}" \
  --device "${DEVICE}" \
  --n-hops "${N_HOPS}" \
  --attacked-inventory "${ATTACKED_INVENTORY}" \
  --require-model-only \
  --output-dir "${OUTPUT_DIR}"
