#!/bin/bash
# Kill ALL GPU processes and free VRAM completely.
# Usage: bash scripts/kill_gpu.sh
#
# This script:
# 1. Gets PIDs from nvidia-smi (the only reliable source)
# 2. Kills them with SIGKILL
# 3. Waits for CUDA to release memory
# 4. Verifies GPU memory is free
# 5. Exits non-zero if cleanup failed

set -e

echo "=== GPU Cleanup ==="

# Get all PIDs using the GPU
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')

if [ -z "$PIDS" ]; then
    echo "No GPU processes found."
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB')
    echo "GPU memory: ${MEM} MiB"
    exit 0
fi

echo "Killing GPU processes: $PIDS"
for PID in $PIDS; do
    kill -9 "$PID" 2>/dev/null && echo "  killed $PID" || echo "  $PID already dead"
done

# Wait for CUDA to release (up to 30s)
echo "Waiting for VRAM to free..."
for i in $(seq 1 30); do
    sleep 1
    REMAINING=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB')
    if [ "$REMAINING" -eq 0 ] && [ "$MEM" -lt 100 ]; then
        echo "GPU clean: ${MEM} MiB used, 0 processes."
        exit 0
    fi
done

# If still not free, try harder
echo "VRAM not freed after 30s. Attempting nvidia-smi reset..."
nvidia-smi --gpu-reset 2>/dev/null || true

sleep 2
MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr -d ' MiB')
REMAINING=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)

if [ "$MEM" -lt 100 ]; then
    echo "GPU clean after reset: ${MEM} MiB used."
    exit 0
fi

echo "WARNING: GPU still holding ${MEM} MiB with ${REMAINING} processes."
echo "May need instance restart to fully clear."
exit 1
