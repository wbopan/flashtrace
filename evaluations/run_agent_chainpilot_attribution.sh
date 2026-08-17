#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly DOJO_DIR="${ROOT}/evaluations/agentdojo/results/full/qwen36_27b_chainpilot16"
readonly AIDER_DIR="${ROOT}/exp/exp4/output/local_generation/qwen36_27b_chainpilot18"
readonly DOJO_GENERATION="${DOJO_DIR}/trajectories.jsonl"
readonly DOJO_LOCALIZATION_GENERATION="${ROOT}/evaluations/agentdojo/results/full/qwen36_27b_formal1046/trajectories.jsonl"
readonly DOJO_LOCALIZATION_REDACTED="${ROOT}/evaluations/agentdojo/results/full/qwen36_27b_formal1046/trajectories.redacted.jsonl"
readonly AIDER_GENERATION="${AIDER_DIR}/trajectories.jsonl"
readonly DOJO_ATTR="${ROOT}/evaluations/agentdojo/results/full/qwen36_27b_localization_chainpilot10_attribution_qwen3_8b"
readonly AIDER_ATTR="${ROOT}/exp/exp4/output/local_attribution/qwen36_27b_chainpilot18_qwen3_8b"

[[ "$(wc -l <"${DOJO_GENERATION}")" -eq 16 ]]
[[ "$(wc -l <"${AIDER_GENERATION}")" -eq 18 ]]

cd "${ROOT}"
uv run --project evaluations/agentdojo --offline \
  python -m evaluations.agentdojo.sanitize_redacted \
  --raw "${DOJO_GENERATION}" \
  --output "${DOJO_DIR}/trajectories.redacted.jsonl"

if [[ -f "${DOJO_ATTR}/attribution.jsonl" ]] \
  && [[ "$(wc -l <"${DOJO_ATTR}/attribution.jsonl")" -eq 10 ]]; then
  echo "AgentDojo chain pilot attribution already complete; skipping."
else
  HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo --offline \
    python -m evaluations.agentdojo.attribute_full \
    --generation-jsonl "${DOJO_LOCALIZATION_GENERATION}" \
    --num-examples 10 \
    --attacked-inventory 286 \
    --output-dir "${DOJO_ATTR}"
fi

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo --offline \
  python -m exp.exp4.attribute_local \
  --generation-jsonl "${AIDER_GENERATION}" \
  --output-dir "${AIDER_ATTR}"

uv run --project evaluations/agentdojo --offline \
  python -m evaluations.analyze_agent_results \
  --aider-generation "${AIDER_GENERATION}" \
  --aider-attribution "${AIDER_ATTR}/metrics.jsonl" \
  --agentdojo-generation "${DOJO_GENERATION}" \
  --agentdojo-localization-generation "${DOJO_LOCALIZATION_GENERATION}" \
  --agentdojo-attribution "${DOJO_ATTR}/attribution.redacted.jsonl" \
  --agentdojo-attacked-inventory 286 \
  --generator-label Qwen3.6-27B \
  --bootstrap-draws 50000 \
  --output-dir evaluations/results/agent_chainpilot \
  --paper-generated-dir evaluations/results/agent_chainpilot/paper_generated

uv run --project evaluations/agentdojo --offline \
  python -m evaluations.render_agent_qualitative \
  --generation-redacted "${DOJO_LOCALIZATION_REDACTED}" \
  --attribution-redacted "${DOJO_ATTR}/attribution.redacted.jsonl" \
  --output-json evaluations/results/agent_chainpilot/qualitative.redacted.json \
  --output-tex evaluations/results/agent_chainpilot/paper_generated/agent_qualitative_rows.tex
