#!/usr/bin/env bash
set -euo pipefail

readonly IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang@sha256:7515a1e626d22a8582ad4478d3d68b294d2b484c78a023f8b61c6867454bbf4a}"
readonly REVISION="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
readonly MODEL_ROOT="${HF_HUB_CACHE:-${HOME}/.cache/huggingface/hub}"
readonly MODEL_DIR="${MODEL_ROOT}/models--Qwen--Qwen3.6-27B/snapshots/${REVISION}"
readonly CONTAINER_MODEL_DIR="/model-hub/models--Qwen--Qwen3.6-27B/snapshots/${REVISION}"
readonly PORT="${SGLANG_PORT:-8001}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  printf 'Pinned Qwen3.6 snapshot is missing: %s\n' "${MODEL_DIR}" >&2
  exit 1
fi

exec docker run --rm \
  --name flashtrace-qwen36-sglang \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --volume "${MODEL_ROOT}:/model-hub:ro" \
  "${IMAGE}" \
  python3 -m sglang.launch_server \
  --model-path "${CONTAINER_MODEL_DIR}" \
  --served-model-name Qwen/Qwen3.6-27B \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --context-length 131072 \
  --mem-fraction-static 0.8 \
  --max-running-requests 1 \
  --schedule-policy fcfs \
  --random-seed 20260728 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --attention-backend triton \
  --sampling-backend flashinfer \
  --disable-overlap-schedule \
  --disable-cuda-graph
