#!/bin/bash

NODES=(
    "XXX.XXX.XXX.XXX"
    "XXX.XXX.XXX.XXX"
)

MASTER_IP=${NODES[0]}
NNODES=${#NODES[@]}
NPROC_PER_NODE=8
WORLD_SIZE=$((NPROC_PER_NODE * NNODES))


export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_MODE="offline"

# Project root is derived from this script's location so that running
# `bash scripts/train.sh` from anywhere still resolves to absolute paths
# (required by pdsh `cd $WORK_DIR` on remote nodes).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXP_DIR="$WORK_DIR/exp"

# Single source of truth for the dataset root. All embedded paths
# inside the parquets / parquet_info.json / jsonl are stored RELATIVE to this
# root, so moving the data only requires updating this one variable.
export DATA_ROOT="$WORK_DIR/toy_data"
export BAGEL_MODEL_PATH="$WORK_DIR/models/BAGEL-7B-MoT"

CONFIG_FILE="./data/configs/rvr.yaml"
EXP_NAME="rvr"
RUNID=1

LOG_SAVE_DIR="${WORK_DIR}/${EXP_NAME}_run${RUNID}/logs"
CHECKPOINT_SAVE_DIR="${EXP_DIR}/${EXP_NAME}_run${RUNID}/checkpoints"


export PDSH_RCMD_TYPE=ssh
export PDSH_SSH_ARGS_APPEND="-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5"
REMOTE_USER="${REMOTE_USER:-$USER}"


echo "Launching RvR multi-node training"
echo "Total nodes: $NNODES"
echo "GPUs per node: $NPROC_PER_NODE"
echo "World size: $WORLD_SIZE"
echo "Master node: $MASTER_IP"

for i in $(seq 0 $((NNODES - 1))); do
    node_ip=${NODES[$i]}
    node_rank=$i

    echo "Launching node $node_ip (NODE_RANK=$node_rank)..."

    pdsh -R ssh -l $REMOTE_USER -w $node_ip "
        eval \"\$(conda shell.bash hook)\" && conda activate rvr &&
        cd $WORK_DIR &&
        echo \"node: $node_ip, NODE_RANK: $node_rank\" &&

        export WANDB_API_KEY=$WANDB_API_KEY &&
        export WANDB_MODE=$WANDB_MODE &&
        export DATA_ROOT=$DATA_ROOT &&
        BAGEL_MODEL_PATH=$BAGEL_MODEL_PATH &&

        torchrun \
          --nnodes=$NNODES \
          --node_rank=$node_rank \
          --nproc_per_node=$NPROC_PER_NODE \
          --master_addr=$MASTER_IP \
          --master_port=12701 \
          train/pretrain_unified_navit.py \
          --visual_gen True \
          --visual_und True \
          --freeze_vit False \
          --freeze_llm False \
          --dataset_config_file $CONFIG_FILE \
          --model_path $BAGEL_MODEL_PATH \
          --layer_module Qwen2MoTDecoderLayer \
          --max_latent_size 64 \
          --resume-from $BAGEL_MODEL_PATH \
          --finetune_from_hf True \
          --auto_resume True \
          --resume-model-only True \
          --finetune-from-ema True \
          --log_every 1 \
          --lr 1e-4 \
          --ema 0.9999 \
          --warmup_steps 2500 \
          --num_worker 2 \
          --save_every 1000 \
          --expected_num_tokens 40000 \
          --max_num_tokens_per_sample 40000 \
          --max_num_tokens 45000 \
          --prefer_buffer_before 16384 \
          --max_buffer_size 50 \
          --num_shard $WORLD_SIZE \
          --wandb_name $EXP_NAME \
          --wandb_runid $RUNID \
          --results_dir $LOG_SAVE_DIR \
          --checkpoint_dir $CHECKPOINT_SAVE_DIR \
          --ce_weight 0.25 \
          --mse_weight 1.0 \
          --timestep_shift 4.0
    " &

    # Stagger node launches slightly to avoid startup races.
    sleep 1
done

wait
echo "All nodes finished."

