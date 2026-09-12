#!/bin/bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motus
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
# G1-D pick-kettle-and-cup (Unitree G1 Dex1, LeRobot v3.0) post-training
# — 16-dim qpos joint-position action, T-shape stitched cameras.
#
# Usage:
#   bash scripts/train_ADS_MotusV2_g1d.sh
#
# Multi-node / multi-GPU:
#   WORLD_SIZE=<nodes> VC_WORKER_NUM=<nodes> VC_TASK_INDEX=<rank> \
#     MASTER_ADDR=<addr> MASTER_PORT=<port> NPROC_PER_NODE=<gpus> \
#     bash scripts/train_ADS_MotusV2_g1d.sh
#
# Smoke test (few steps, no checkpoint save):
#   bash scripts/train_ADS_MotusV2_g1d.sh --max_steps 20
set -euo pipefail

TASK="eai_pick_place_wan_vlm_mask"
CONFIG_FILE="configs/eai_pick_place_wan_vlm_mask.yaml"

export OUTPUT_DIR="outputs/motus-${TASK}"

if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "Folder '$OUTPUT_DIR' created"
else
    echo "Folder '$OUTPUT_DIR' already exists"
fi

NNODES=${WORLD_SIZE:-${VC_WORKER_NUM:-1}}
NODE_RANK=${VC_TASK_INDEX:-0}
MASTER_ADDR=${MASTER_ADDR:-${DOLPHIN_MASTER_IP:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-23456}
# NPROC_PER_NODE=${NPROC_PER_NODE:-8}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

echo "===== distributed config ====="
echo "HOSTNAME=$(hostname)"
echo "NNODES=$NNODES"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-}"
echo "=============================="

LOG_FILE="${OUTPUT_DIR}/console_$(date '+%Y%m%d_%H%M%S').log"

# Pass through extra CLI args (e.g. --max_steps 20 for smoke test)
EXTRA_ARGS="$*"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train/train_wan_vlm_mask.py \
    --deepspeed configs/zero1.json \
    --config ${CONFIG_FILE} \
    --run_name ${TASK} \
    --report_to tensorboard \
    ${EXTRA_ARGS} \
    2>&1 | tee "$LOG_FILE"