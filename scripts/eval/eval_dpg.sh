#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

DIST_EVAL=./eval/gen/dpg/dist_eval.sh

OUTPUT_PATH="./benchmarks/dpg_rvr_refined"

RESOLUTION=${RESOLUTION:-1024}

echo "eval -> output_path=${OUTPUT_PATH}"
bash "$DIST_EVAL" \
  "$OUTPUT_PATH" \
  "$RESOLUTION"
