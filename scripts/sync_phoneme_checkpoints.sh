#!/usr/bin/env bash
# Periodically rsync phoneme-VAE checkpoints from the active vast.ai
# instance back to the local machine.
#
# Queries `lfm cloud status` to find the running instance's SSH
# endpoint, then rsyncs the remote output dir.  Runs in a loop until
# killed.
#
# Usage:
#   ./scripts/sync_phoneme_checkpoints.sh [interval_seconds] [local_dir]
#   Defaults: 600 seconds (10 min), data/cloud_results/phoneme_v1

set -euo pipefail

INTERVAL="${1:-600}"
LOCAL_DIR="${2:-data/cloud_results/phoneme_v1}"
REMOTE_DIR="/workspace/lfm/data/models/phoneme_v1"
SSH_KEY="${HOME}/.ssh/id_ed25519"

mkdir -p "$LOCAL_DIR"

export $(grep VAST_API_KEY .secrets)

get_ssh_endpoint() {
  # Parse the first RUNNING vast.ai instance from `lfm cloud status`.
  # Output line format (abridged):
  #   <id>  RTX 3090  running  <host>:<port>  <name>
  poetry run lfm cloud status --provider vastai 2>/dev/null \
    | awk '/running/ && /RTX 3090/ {print $(NF-1); exit}'
}

while true; do
  ENDPOINT=$(get_ssh_endpoint || true)
  if [[ -z "${ENDPOINT:-}" ]]; then
    echo "[$(date +%H:%M:%S)] no running RTX 3090 instance; will retry in ${INTERVAL}s"
  else
    HOST="${ENDPOINT%%:*}"
    PORT="${ENDPOINT##*:}"
    echo "[$(date +%H:%M:%S)] syncing from root@${HOST}:${PORT}${REMOTE_DIR} → ${LOCAL_DIR}"
    rsync -avz --partial \
      -e "ssh -o StrictHostKeyChecking=no -p ${PORT} -i ${SSH_KEY}" \
      --include='*.pt' --include='*.json' --include='*.yaml' \
      --include='*.log' --include='*.parquet' \
      --include='*/' --exclude='*' \
      "root@${HOST}:${REMOTE_DIR}/" "${LOCAL_DIR}/" \
      2>&1 | grep -E '^[A-Za-z].*\.(pt|json|log|parquet)$|sent|received' || true
  fi
  sleep "${INTERVAL}"
done
