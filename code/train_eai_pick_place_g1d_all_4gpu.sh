#!/bin/bash
# EAI pick_place G1-D Motus training (4 GPU: 0,1,2,3)
# Data: 3 datasets combined (264 episodes)
#   - pick_place_100_motus   (100 episodes)
#   - pick_place_9_16_motus  (117 episodes)
#   - pick_place_916_motus   (47 episodes)
# Pretrain: checkpoints/88

set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate motus

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "============== EAI pick_place G1-D ALL Motus Training (4 GPU) =============="
echo "Python: $(which python)"
echo "GPU: 4,5,6,7"
echo "Data: data/all/ (pick_place_100_motus + pick_place_9_16_motus + pick_place_916_motus, 264 episodes)"
echo "Pretrain: /home/ma-user/work/wx1513998/checkpoints/88"

cd /home/ma-user/work/wx1513998/motus_guass_best_2

mkdir -p /home/ma-user/work/wx1513998/motus_eai_pick_place_all/checkpoints
mkdir -p /home/ma-user/work/wx1513998/motus_eai_pick_place_all/tensorboard

export OMP_NUM_THREADS=2
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=lo
export TOKENIZERS_PARALLELISM=false

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29101 \
    train/train_wan_vlm_mask_reweight_progress.py \
    --deepspeed configs/zero1.json \
    --config /home/ma-user/work/wx1513998/tmp/eai_pick_place_g1d_all.yaml \
    --report_to tensorboard \
    2>&1 | tee /home/ma-user/work/wx1513998/motus_eai_pick_place_all_train.log

EXIT_CODE=${PIPESTATUS[0]}
echo "Training exit code: $EXIT_CODE"
echo "============== Training End =============="
exit $EXIT_CODE
