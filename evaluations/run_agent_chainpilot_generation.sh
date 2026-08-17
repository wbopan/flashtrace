#!/usr/bin/env bash
set -euo pipefail

readonly ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PORT="${SGLANG_PORT:-8001}"
readonly BASE_URL="http://127.0.0.1:${PORT}"
readonly MODEL="Qwen/Qwen3.6-27B"
readonly REVISION="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
readonly DOJO_ROOT="${ROOT}/evaluations/agentdojo/results/full/qwen36_27b_chainpilot16"
readonly AIDER_ROOT="${ROOT}/exp/exp4/output/local_generation/qwen36_27b_chainpilot18"
readonly LOG="${DOJO_ROOT}/sglang_server.log"

mkdir -p "${DOJO_ROOT}" "${AIDER_ROOT}"

server_pid=""
cleanup() {
  if [[ -n "${server_pid}" ]]; then
    docker stop flashtrace-qwen36-sglang >/dev/null 2>&1 || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

(
  cd "${ROOT}"
  SGLANG_PORT="${PORT}" evaluations/launch_qwen36_sglang.sh
) >"${LOG}" 2>&1 &
server_pid="$!"

for _ in $(seq 1 360); do
  if curl --fail --silent --show-error "${BASE_URL}/health" >/dev/null; then
    break
  fi
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    tail -200 "${LOG}" >&2
    exit 1
  fi
  sleep 10
done
curl --fail --silent --show-error "${BASE_URL}/health" >/dev/null

cd "${ROOT}"
HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo --offline \
  python -m evaluations.agentdojo.benchmark_local \
  --model "${MODEL}" \
  --revision "${REVISION}" \
  --mode both \
  --pilot-per-suite 2 \
  --max-turns 10 \
  --max-new-tokens 4096 \
  --sglang-base-url "${BASE_URL}" \
  --output-dir "${DOJO_ROOT}"

HF_HUB_OFFLINE=1 uv run --project evaluations/agentdojo --offline \
  python -m exp.exp4.run_local \
  --model "${MODEL}" \
  --revision "${REVISION}" \
  --num-tasks 18 \
  --max-new-tokens 8192 \
  --sglang-base-url "${BASE_URL}" \
  --output-dir "${AIDER_ROOT}"

[[ "$(wc -l <"${DOJO_ROOT}/trajectories.jsonl")" -eq 16 ]]
[[ "$(wc -l <"${AIDER_ROOT}/trajectories.jsonl")" -eq 18 ]]
