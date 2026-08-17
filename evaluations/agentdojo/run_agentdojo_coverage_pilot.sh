#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(git rev-parse --show-toplevel)"
readonly OUTPUT_ROOT="${AGENTDOJO_OUTPUT_DIR:-${REPO_ROOT}/evaluations/agentdojo/results/realization_pilot}"
readonly LOCAL_ENDPOINT="${SGLANG_BASE_URL:-http://127.0.0.1:8001}"
readonly PILOT_PER_SUITE="${AGENTDOJO_PILOT_PER_SUITE:-2}"
readonly NATURAL_BUDGETS="${AGENTDOJO_NATURAL_BUDGETS:-1,4,16}"
readonly PREFIX_BUDGETS="${AGENTDOJO_PREFIX_BUDGETS:-1,2,4,8,16,32,64,full}"
readonly SPARSE_BUDGETS="${AGENTDOJO_SPARSE_BUDGETS:-1,2,4,8,16,32,64,full}"
readonly MAX_TURNS="${AGENTDOJO_MAX_TURNS:-10}"
readonly MAX_NEW_TOKENS="${AGENTDOJO_MAX_NEW_TOKENS:-2048}"
readonly SUITES="${AGENTDOJO_SUITES:-banking,slack,travel,workspace}"
readonly MODEL_ID="${AGENTDOJO_MODEL_ID:-Qwen/Qwen3.6-27B}"
readonly MODEL_REVISION="${AGENTDOJO_MODEL_REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"

cd "${REPO_ROOT}"

run_protocol() {
  local method_flag="$1"
  local budgets="$2"
  local output_dir="$3"
  uv run --project evaluations/agentdojo \
    python -m evaluations.agentdojo.benchmark_local \
    --model "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --mode attacked \
    --suites "${SUITES}" \
    --action-target-only \
    --pilot-per-suite "${PILOT_PER_SUITE}" \
    --natural-rollout-budgets "${NATURAL_BUDGETS}" \
    "${method_flag}" "${budgets}" \
    --max-turns "${MAX_TURNS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.7 \
    --top-p 0.95 \
    --top-k 20 \
    --sglang-base-url "${LOCAL_ENDPOINT}" \
    --output-dir "${output_dir}"

  uv run --project evaluations/agentdojo \
    python -m evaluations.agentdojo.diagnose_coverage \
    --generation-jsonl "${output_dir}/trajectories.jsonl" \
    --output "${output_dir}/coverage.redacted.json"
}

run_protocol \
  --prefix-forcing-token-budgets \
  "${PREFIX_BUDGETS}" \
  "${OUTPUT_ROOT}/prefix"
run_protocol \
  --sparse-correction-budgets \
  "${SPARSE_BUDGETS}" \
  "${OUTPUT_ROOT}/sparse"

uv run --project evaluations/agentdojo \
  python -m evaluations.agentdojo.compare_realization_pilots \
  --prefix-report "${OUTPUT_ROOT}/prefix/coverage.redacted.json" \
  --sparse-report "${OUTPUT_ROOT}/sparse/coverage.redacted.json" \
  --output "${OUTPUT_ROOT}/comparison.redacted.json"
