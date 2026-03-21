#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/sync_to_opt.sh"
"$SCRIPT_DIR/bootstrap_opt.sh"

echo "Remote deploy is ready."
