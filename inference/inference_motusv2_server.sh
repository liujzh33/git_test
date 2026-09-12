#!/bin/bash
# MotusV2 inference server launcher — runs on the 5090 GPU server.
#
# This script starts the ZMQ inference server that loads the MotusV2 model
# and serves predict_action_chunk requests from the robot-side client.
#
# Usage:
#   bash inference_motusv2_server.sh
#
# Before running:
#   1. Edit the PATHS section below to match your server directory layout.
#   2. Ensure conda environment "inference_motus" is activated
#      (or adjust PYTHON below).
#   3. Ensure the 5090 server's firewall allows inbound TCP on SERVER_PORT.

set -Eeuo pipefail

# ==================== ROOT DIRS ====================
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(dirname "$SCRIPTS_DIR")"
export MOTUS_POLICY_DIR="${MOTUS_POLICY_DIR:-$BUNDLE_ROOT/policy/MotusWanVlmDirectMaskMotion}"
# ===================================================

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MOTUS_COMPILE=0

# ==================== PATHS (modify these) ====================
# --- Model weights ---
CHECKPOINT_PATH="/home/ma-user/work/wx1513998/checkpoints/eai_pick_place_ewam/eai_pick_place_wan_vlm_mask/eai_pick_place_wan_vlm_mask/checkpoint_step_20000/pytorch_model/mp_rank_00_model_states.pt"
WAN_PATH="/home/ma-user/work/wx1513998/pretrained_models/Wan2.2-TI2V-5B"
VLM_PATH="/home/ma-user/work/wx1513998/pretrained_models/Qwen3-VL-2B-Instruct"
T5_CACHE_DIR="/home/ma-user/work/wx1513998/data/EAI_DATA/2026-09-11-153200/pick_place_100_ewam/t5_cache"

# --- Config (ships with the bundle, edit paths inside it too) ---
CONFIG_PATH="$BUNDLE_ROOT/configs/eai_pick_place_wan_vlm_mask.yaml"

# --- Task instruction ---
INSTRUCTION="Pick up the marker and the plush toy and place them into the right basket, then pick up the empty water bottle and the draft paper and place them into the left basket"
# ==============================================================

# ==================== SERVER OPTIONS ====================
HOST="0.0.0.0"
PORT=5555

# Denoising steps: 5 = fast (~254ms), 10 = higher quality (~461ms)
NUM_INFERENCE_STEPS=10
# =========================================================

# --- Activate conda ---
source /root/miniconda3/etc/profile.d/conda.sh
conda activate motus

echo "[server] Starting MotusV2 inference server"
echo "[server] Checkpoint: $CHECKPOINT_PATH"
echo "[server] Config:     $CONFIG_PATH"
echo "[server] Listen:     tcp://$HOST:$PORT"
echo "[server] Steps:      $NUM_INFERENCE_STEPS"

python "$SCRIPTS_DIR"/motus_inference_server.py \
    --checkpoint_path=$CHECKPOINT_PATH \
    --wan_path=$WAN_PATH \
    --vlm_path=$VLM_PATH \
    --config_path=$CONFIG_PATH \
    --t5_cache_dir=$T5_CACHE_DIR \
    --host=$HOST \
    --port=$PORT \
    --num_inference_steps=$NUM_INFERENCE_STEPS \
    --instruction="$INSTRUCTION"
