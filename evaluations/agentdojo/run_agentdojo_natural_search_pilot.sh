#!/usr/bin/env bash
set -euo pipefail

readonly REPO_ROOT="$(git rev-parse --show-toplevel)"
readonly OUTPUT_ROOT="${AGENTDOJO_OUTPUT_DIR:-${REPO_ROOT}/evaluations/agentdojo/results/natural_search_pilot}"
readonly LOCAL_ENDPOINT="${SGLANG_BASE_URL:-http://127.0.0.1:8001}"
readonly PILOT_PER_SUITE="${AGENTDOJO_PILOT_PER_SUITE:-1}"
readonly BEST_OF_BUDGETS="${AGENTDOJO_BEST_OF_BUDGETS:-1,4}"
readonly RERANK_CANDIDATES="${AGENTDOJO_RERANK_CANDIDATES:-4}"
readonly SCORE_ROLLOUTS="${AGENTDOJO_SCORE_ROLLOUTS:-1}"
readonly SCORE_METHOD="${AGENTDOJO_SCORE_METHOD:-conditional_logprob}"
readonly REASONING_TEMPERATURE="${AGENTDOJO_REASONING_TEMPERATURE:-1.2}"
readonly SCORE_MAX_TOKENS="${AGENTDOJO_SCORE_MAX_TOKENS:-192}"
readonly MAX_TURNS="${AGENTDOJO_MAX_TURNS:-10}"
readonly MAX_NEW_TOKENS="${AGENTDOJO_MAX_NEW_TOKENS:-2048}"
readonly SUITES="${AGENTDOJO_SUITES:-banking,slack,travel,workspace}"
readonly MODEL_ID="${AGENTDOJO_MODEL_ID:-Qwen/Qwen3.6-27B}"
readonly MODEL_REVISION="${AGENTDOJO_MODEL_REVISION:-6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}"

cd "${REPO_ROOT}"

run_protocol() {
  local output_dir="$1"
  shift
  uv run --project evaluations/agentdojo \
    python -m evaluations.agentdojo.benchmark_local \
    --model "${MODEL_ID}" \
    --revision "${MODEL_REVISION}" \
    --mode attacked \
    --suites "${SUITES}" \
    --action-target-only \
    --pilot-per-suite "${PILOT_PER_SUITE}" \
    --max-turns "${MAX_TURNS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --temperature 0.7 \
    --top-p 0.95 \
    --top-k 20 \
    --sglang-base-url "${LOCAL_ENDPOINT}" \
    --output-dir "${output_dir}" \
    "$@"

  uv run --project evaluations/agentdojo \
    python -m evaluations.agentdojo.diagnose_coverage \
    --generation-jsonl "${output_dir}/trajectories.jsonl" \
    --output "${output_dir}/coverage.redacted.json" >/dev/null
}

run_protocol \
  "${OUTPUT_ROOT}/best_of" \
  --natural-rollout-budgets "${BEST_OF_BUDGETS}"

run_protocol \
  "${OUTPUT_ROOT}/reasoning_rerank" \
  --natural-rollout-budgets 1 \
  --reasoning-rerank-candidates "${RERANK_CANDIDATES}" \
  --reasoning-score-method "${SCORE_METHOD}" \
  --reasoning-temperature "${REASONING_TEMPERATURE}" \
  --reasoning-score-rollouts "${SCORE_ROLLOUTS}" \
  --reasoning-score-max-new-tokens "${SCORE_MAX_TOKENS}"

uv run --project evaluations/agentdojo \
  python -m evaluations.agentdojo.compare_search_pilots \
  --best-of "${OUTPUT_ROOT}/best_of/trajectories.jsonl" \
  --reasoning-rerank "${OUTPUT_ROOT}/reasoning_rerank/trajectories.jsonl" \
  --output "${OUTPUT_ROOT}/comparison.redacted.json"
