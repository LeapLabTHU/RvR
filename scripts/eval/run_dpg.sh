#!/usr/bin/env bash
set -e
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
cd "$PROJECT_ROOT"

GPUS=8

model_path="./models/RvR-7B-MoT"

INPUT_BASE="./benchmarks/dpg_bagel"
# INPUT_BASE="./benchmarks/dpg_rvr_t2i"
OUTPUT_PATH="./benchmarks/dpg_rvr_refined"

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_TIMEOUT=${TORCH_DISTRIBUTED_TIMEOUT:-3600}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-$(( (RANDOM % 1000) + 29000 ))}

echo "generate -> output_path=${OUTPUT_PATH} seed=${SEED} model_path=${model_path}"
torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --nproc_per_node=$GPUS \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  ./eval/gen/regen_images_mp_dpg.py \
  --output_dir "$OUTPUT_PATH" \
  --prompts_file ./eval/gen/dpg/prompts.json \
  --input_base_dir "$INPUT_BASE" \
  --model-path "$model_path" \
  --num_images 4 --resolution 1024 \
  --cfg_text_scale 4.0 --cfg_img_scale 2.0 \
  --cfg_interval 0.0 --cfg_renorm_min 0.0 \
  --timestep_shift 3.0 --num_timesteps 50 \
  --seed 0
