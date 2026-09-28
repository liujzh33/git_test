#!/bin/bash
# Launch: robotwin_wan_vlm_mask_stage2_franka_dobot_dim16_place.yaml
# Data: franka_data_dobot_format / place_objects_into_the_box_rc_0524 (274 eps, T-shape 320x360)
# Init weights: Wan2.2-TI2V-5B + Qwen3-VL-2B-Instruct + checkpoints/88
#   (action_expert input_encoder/decoder SKIPPED -> random init, retrain)
# GPUs: 0-3, port 29104

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /home/ma-user/work/wx1513998/env/motus

cd /home/ma-user/work/wx1513998/Motus_initial_franka

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port=29104 \
    train/train_wan_vlm_mask_franka.py \
    --deepspeed configs/zero1.json \
    --config configs/robotwin_wan_vlm_mask_stage2_franka_dobot_dim16_place.yaml \
    --report_to tensorboard
