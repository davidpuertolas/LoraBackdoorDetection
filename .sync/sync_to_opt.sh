#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO_DIR="${LOCAL_REPO_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-opt}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/vskate/LoraBackdoorDetection}"
SYNC_RESULTS="${SYNC_RESULTS:-0}"
SYNC_ANALYSIS="${SYNC_ANALYSIS:-0}"
SYNC_OUTPUTS="${SYNC_OUTPUTS:-0}"

RSYNC_ARGS=(
  -az
  --delete
  --info=progress2
  --exclude=.git/
  --exclude=.venv/
  --exclude=venv/
  --exclude=__pycache__/
  --exclude=.pytest_cache/
  --exclude=*.pyc
  --exclude=.DS_Store
  --exclude=offload_cache/
)

if [ "$SYNC_RESULTS" != "1" ]; then
  RSYNC_ARGS+=(
    --exclude=results/
    --exclude=resultsFinal/
  )
fi

if [ "$SYNC_ANALYSIS" != "1" ]; then
  RSYNC_ARGS+=(--exclude=projection_analysis/)
fi

if [ "$SYNC_OUTPUTS" != "1" ]; then
  RSYNC_ARGS+=(
    --exclude=output_qwen/
    --exclude=output_llama/
    --exclude=output_gemma/
  )
fi

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_REPO_DIR'"
rsync "${RSYNC_ARGS[@]}" "$LOCAL_REPO_DIR/" "$REMOTE_HOST:$REMOTE_REPO_DIR/"

echo "Sync complete:"
echo "  Local:  $LOCAL_REPO_DIR"
echo "  Remote: $REMOTE_HOST:$REMOTE_REPO_DIR"
