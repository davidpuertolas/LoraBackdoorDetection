#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-opt}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-/home/vskate/LoraBackdoorDetection}"
HF_TOKEN_VALUE="${HF_TOKEN:-}"

if [ -z "$HF_TOKEN_VALUE" ]; then
  echo "HF_TOKEN is not set. Export it before running this script." >&2
  exit 1
fi

REMOTE_REPO_DIR_ESCAPED="$(printf "%q" "$REMOTE_REPO_DIR")"
HF_TOKEN_ESCAPED="$(printf "%q" "$HF_TOKEN_VALUE")"

ssh "$REMOTE_HOST" "export REMOTE_REPO_DIR=$REMOTE_REPO_DIR_ESCAPED HF_TOKEN=$HF_TOKEN_ESCAPED; bash -lc '
set -euo pipefail

if [ -x \"\$HOME/miniconda3/envs/env/bin/python\" ]; then
  PYTHON_BIN=\"\$HOME/miniconda3/envs/env/bin/python\"
elif [ -x \"\$HOME/miniconda3/bin/python\" ]; then
  PYTHON_BIN=\"\$HOME/miniconda3/bin/python\"
else
  PYTHON_BIN=\"python3\"
fi

cd \"\$REMOTE_REPO_DIR\"

if [ ! -d venv ]; then
  \"\$PYTHON_BIN\" -m venv venv
fi

source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
printf \"HF_TOKEN=%s\n\" \"\$HF_TOKEN\" > .env

echo \"Python: \$(python --version)\"
echo \"Pip: \$(pip --version)\"
echo \"GPU snapshot:\"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader
'"

echo "Remote environment bootstrapped at $REMOTE_HOST:$REMOTE_REPO_DIR"
