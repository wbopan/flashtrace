#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DOJO_GENERATION="${ROOT}/evaluations/agentdojo/results/full/qwen36_27b_formal1046/trajectories.jsonl"
readonly AIDER_GENERATION="${ROOT}/exp/exp4/output/local_generation/qwen36_27b_formal225/trajectories.jsonl"
readonly DOJO_ATTR="${ROOT}/evaluations/agentdojo/results/full/qwen36_27b_attribution_qwen3_8b"
readonly AIDER_ATTR="${ROOT}/exp/exp4/output/local_attribution/qwen36_27b_generation_qwen3_8b"
readonly AIDER_COMPLETE_METRICS="${AIDER_ATTR}/metrics.complete.jsonl"

cd "${ROOT}"
uv run --project evaluations/agentdojo --offline \
  python -m evaluations.agentdojo.sanitize_redacted \
  --raw "${DOJO_GENERATION}" \
  --output "${ROOT}/evaluations/agentdojo/results/full/qwen36_27b_formal1046/trajectories.redacted.jsonl"

for path in "${DOJO_GENERATION}" "${AIDER_GENERATION}"; do
  if [[ ! -f "${path}" ]]; then
    printf 'Missing formal generation artifact: %s\n' "${path}" >&2
    exit 1
  fi
done

[[ "$(wc -l <"${DOJO_GENERATION}")" -eq 1046 ]]
[[ "$(wc -l <"${AIDER_GENERATION}")" -eq 225 ]]

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo --offline \
  python -m evaluations.agentdojo.attribute_full \
  --generation-jsonl "${DOJO_GENERATION}" \
  --num-examples 1 \
  --output-dir \
    evaluations/agentdojo/results/full/qwen36_27b_attribution_recompute_smoke1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo --offline \
  python -m exp.exp4.attribute_local \
  --generation-jsonl "${AIDER_GENERATION}" \
  --num-examples 1 \
  --output-dir exp/exp4/output/local_attribution/qwen36_27b_smoke1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo --offline \
  python -m evaluations.agentdojo.attribute_full \
  --generation-jsonl "${DOJO_GENERATION}" \
  --output-dir "${DOJO_ATTR}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo --offline \
  python -m exp.exp4.attribute_local \
  --generation-jsonl "${AIDER_GENERATION}" \
  --output-dir "${AIDER_ATTR}"

AIDER_METRICS="${AIDER_ATTR}/metrics.jsonl"
if [[ -f "${AIDER_COMPLETE_METRICS}" ]]; then
  AIDER_METRICS="${AIDER_COMPLETE_METRICS}"
fi

uv run --project evaluations/agentdojo --offline \
  python -m evaluations.analyze_agent_results \
  --aider-generation "${AIDER_GENERATION}" \
  --aider-attribution "${AIDER_METRICS}" \
  --agentdojo-generation "${DOJO_GENERATION}" \
  --agentdojo-attribution "${DOJO_ATTR}/attribution.redacted.jsonl" \
  --generator-label Qwen3.6-27B \
  --bootstrap-draws 50000 \
  --output-dir evaluations/results/agent_final \
  --paper-generated-dir paper/generated

uv run --project evaluations/agentdojo --offline \
  python -m evaluations.render_agent_qualitative \
  --generation-redacted \
    evaluations/agentdojo/results/full/qwen36_27b_formal1046/trajectories.redacted.jsonl \
  --attribution-redacted "${DOJO_ATTR}/attribution.redacted.jsonl" \
  --output-json evaluations/results/agent_final/qualitative.redacted.json \
  --output-tex paper/generated/agent_qualitative_rows.tex
