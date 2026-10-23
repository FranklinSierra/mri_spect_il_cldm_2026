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
  elif [[ -d "$REPO_DIR/$name" ]]; then
    echo "$REPO_DIR/$name"
  else
    echo ""
  fi
}
COCOLIT_HOST="$(resolve_sibling CoCoLIT)"
FST_HOST="$(resolve_sibling IL-FST)"
PASTA_HOST=""
if [[ -d "$PARENT_DIR/pasta-pd-ref" ]]; then
  PASTA_HOST="$PARENT_DIR/pasta-pd-ref"
fi

MOUNT_ARGS=(
  -v "$REPO_DIR:/workspace/IL-CLDM"
  -v "$HOST_DATA_DIR:/workspace/IL-CLDM/data"
)
if [[ -n "$COCOLIT_HOST" ]]; then
  MOUNT_ARGS+=(-v "$COCOLIT_HOST:/workspace/CoCoLIT")
fi
if [[ -n "$FST_HOST" ]]; then
  MOUNT_ARGS+=(-v "$FST_HOST:/workspace/IL-FST")
fi
if [[ -n "$PASTA_HOST" ]]; then
  MOUNT_ARGS+=(-v "$PASTA_HOST:/workspace/pasta-pd-ref")
fi

ENV_ARGS=(
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID
  -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
  -e "ILCLDM_EPOCHS=${ILCLDM_EPOCHS:-1000}"
  -e "ILCLDM_NOISE_STEPS=${ILCLDM_NOISE_STEPS:-1000}"
  -e ILCLDM_ROOT=/workspace/IL-CLDM
)
# Forward optional ILCLDM_* from .env / environment into the container.
while IFS= read -r var; do
  [[ -n "${!var:-}" ]] || continue
  ENV_ARGS+=(-e "${var}=${!var}")
done < <(compgen -v ILCLDM_ || true)

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  "${ENV_ARGS[@]}" \
  "${MOUNT_ARGS[@]}" \
  -w /workspace/IL-CLDM \
  "$IMAGE_NAME" \
  sleep infinity

echo "Started container '$CONTAINER_NAME'."
echo "Repo: /workspace/IL-CLDM"
echo "Data mount: $HOST_DATA_DIR -> /workspace/IL-CLDM/data"
[[ -n "$COCOLIT_HOST" ]] || echo "Note: CoCoLIT mount skipped (not found)."
[[ -n "$FST_HOST" ]] || echo "Note: IL-FST mount skipped (not found)."
[[ -n "$PASTA_HOST" ]] || echo "Note: pasta-pd-ref mount skipped (not found)."
echo "Attach from VS Code Remote Explorer > Dev Containers."
