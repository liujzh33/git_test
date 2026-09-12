#!/bin/bash
# EAI pick_place G1-D Motus training (4 GPU: 0,1,2,3)
# Data: pick_place_100_motus (T-shape, 16-dim)
# Pretrain: checkpoints/88

set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate motus

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "============== EAI pick_place G1-D Motus Training (4 GPU) =============="
echo "Python: $(which python)"
echo "GPU: 0,1,2,3"
echo "Data: /home/ma-user/work/wx1513998/data/EAI_DATA/2026-09-11-153200/pick_place_100_motus"
echo "Pretrain: /home/ma-user/work/wx1513998/checkpoints/88"

cd /home/ma-user/work/wx1513998/motus_guass_best_2

mkdir -p /home/ma-user/work/wx1513998/motus_eai_pick_place/checkpoints
mkdir -p /home/ma-user/work/wx1513998/motus_eai_pick_place/tensorboard

export OMP_NUM_THREADS=2
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29100 \
    train/train_wan_vlm_mask_reweight_progress.py \
    --deepspeed configs/zero1.json \
    --config /home/ma-user/work/wx1513998/tmp/eai_pick_place_g1d_motus.yaml \
    --report_to tensorboard \
    2>&1 | tee /home/ma-user/work/wx1513998/motus_eai_pick_place_train.log

EXIT_CODE=${PIPESTATUS[0]}
echo "Training exit code: $EXIT_CODE"
echo "============== Training End =============="
exit $EXIT_CODE
