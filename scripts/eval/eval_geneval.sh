#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

GPUS=8

OUTPUT_PATH="./benchmarks/geneval_rvr_refined"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-3600}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$(( (RANDOM % 1000) + 29000 ))}

echo "eval -> output_path=${OUTPUT_PATH}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=$GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  ./eval/gen/geneval/evaluation/evaluate_images_mp.py \
  "$OUTPUT_PATH/images" \
  --outfile "$OUTPUT_PATH/results.jsonl" \
  --model-path ./eval/gen/geneval/model

python ./eval/gen/geneval/evaluation/summary_scores.py "$OUTPUT_PATH/results.jsonl"
