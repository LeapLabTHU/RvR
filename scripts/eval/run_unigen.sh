#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

GPUS=8

model_path="./models/RvR-7B-MoT"

INPUT_BASE="./benchmarks/unigen_bagel"
# INPUT_BASE="./benchmarks/unigen_rvr_t2i"
OUTPUT_PATH="./benchmarks/unigen_rvr_refined"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-3600}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$(( (RANDOM % 1000) + 29000 ))}

mkdir -p "$OUTPUT_PATH"

echo "generate -> output_path=${OUTPUT_PATH} model_path=${model_path}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=$GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  ./eval/gen/regen_images_mp_unigen.py \
  --output_dir "$OUTPUT_PATH" \
  --prompt_dir ./eval/gen/unigen/test_prompts_en.csv \
  --input_base_dir "$INPUT_BASE" \
  --batch_size 1 \
  --num_images 4 \
  --resolution 1024 \
  --max_latent_size 64 \
  --model-path "$model_path" \
  --seed 42
