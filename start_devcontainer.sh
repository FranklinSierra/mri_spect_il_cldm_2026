#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-il-cldm-dev}"
IMAGE_NAME="${IMAGE_NAME:-il-cldm-dev}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Bind-mount host data into repo data/ (override via HOST_DATA_DIR or .env)
HOST_DATA_DIR="${HOST_DATA_DIR:-$REPO_DIR/data}"

if [[ -f "$REPO_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_DIR/.env"
  set +a
  HOST_DATA_DIR="${HOST_DATA_DIR:-$REPO_DIR/data}"
fi

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container '$CONTAINER_NAME' is already running."
  exit 0
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  docker start "$CONTAINER_NAME"
  echo "Started existing container '$CONTAINER_NAME'."
  exit 0
fi

mkdir -p "$HOST_DATA_DIR"

PARENT_DIR="$(cd "$REPO_DIR/.." && pwd)"
# Sibling repo if the host move already happened; otherwise the nested copy.
resolve_sibling() {
  local name="$1"
  if [[ -d "$PARENT_DIR/$name/.git" ]]; then
    echo "$PARENT_DIR/$name"
  else
    echo "$REPO_DIR/$name"
  fi
}
COCOLIT_HOST="$(resolve_sibling CoCoLIT)"
FST_HOST="$(resolve_sibling IL-FST)"

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  -e ILCLDM_EPOCHS="${ILCLDM_EPOCHS:-1000}" \
  -e ILCLDM_NOISE_STEPS="${ILCLDM_NOISE_STEPS:-1000}" \
  -e ILCLDM_ROOT=/workspace/IL-CLDM \
  -v "$REPO_DIR:/workspace/IL-CLDM" \
  -v "$COCOLIT_HOST:/workspace/CoCoLIT" \
  -v "$FST_HOST:/workspace/IL-FST" \
  -v "$HOST_DATA_DIR:/workspace/IL-CLDM/data" \
  -v "$PARENT_DIR/pasta-pd-ref:/workspace/pasta-pd-ref" \
  -w /workspace/IL-CLDM \
  "$IMAGE_NAME" \
  sleep infinity

echo "Started container '$CONTAINER_NAME'."
echo "Repo: /workspace/IL-CLDM"
echo "Data mount: $HOST_DATA_DIR -> /workspace/IL-CLDM/data"
echo "Attach from VS Code Remote Explorer > Dev Containers."
