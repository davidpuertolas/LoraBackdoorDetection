#!/usr/bin/env bash
set -euo pipefail
cd /home/weshi/LoraBackdoorDetection
while ! ssh opt test -f /home/vskate/LoraBackdoorDetection/qwen_generation_done.flag; do
  sleep 120
done
rsync -az --delete opt:/home/vskate/LoraBackdoorDetection/output_qwen/poison/ output_qwen/poison/
rsync -az --delete opt:/home/vskate/LoraBackdoorDetection/output_qwen/test/ output_qwen/test/
touch qwen_sync_back_done.flag
