#!/usr/bin/env bash
set -euo pipefail

model="$(python3 - <<'PY'
import config
print(config.MODEL)
PY
)"

run_id="$(date +%s)"
run_dir="runs/run_${run_id}"
mkdir -p "${run_dir}"
mkdir -p "${run_dir}/metrics"

python3 evaluation/calibrate_detector.py --run_dir "${run_dir}"
python3 evaluation/evaluate_test_set.py --run_dir "${run_dir}"
python3 proj_dependency_check.py \
  --models "${model}" \
  --output_dir "${run_dir}/metrics" \
  --artifact_output_dir "${run_dir}" \
  --flat_output
