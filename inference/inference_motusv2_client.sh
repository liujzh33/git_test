#!/bin/bash
# MotusV2 inference client launcher — runs on the robot-side machine (no GPU).
#
# This script starts the ZMQ client that captures images, sends them to the
# remote GPU server for inference, receives action chunks, and executes them
# on the G1 robot at 30Hz.
#
# Usage:
#   bash inference_motusv2_client.sh
#
# Before running:
#   1. Edit the PATHS section below.
#   2. Ensure the inference server is already running on the 5090 GPU server.
#   3. Ensure unitree_lerobot is installed and the robot is powered on.
#   4. Ensure UNITREE_DDSINTERFACE is set to the correct network interface.

set -Eeuo pipefail

# ==================== ROOT DIRS ====================
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="$(dirname "$SCRIPTS_DIR")"
# unitree_lerobot repo (hardware interfaces) — external, robot-specific.
LEROBOT_ROOT="${LEROBOT_ROOT:-/home/robo/codes/unitree_lerobot/}"
# ===================================================

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$LEROBOT_ROOT"
export PYTHONPATH="$LEROBOT_ROOT":"$SCRIPTS_DIR"

# Unitree DDS network interface (find with `ip link`)
export UNITREE_DDSINTERFACE="${UNITREE_DDSINTERFACE:-eth0}"

# ==================== PATHS (modify these) ====================
# --- GPU server address ---
SERVER_HOST="10.0.0.50"
SERVER_PORT=5555

# --- Config (only needs common + dataset.stitch_mode fields) ---
CONFIG_PATH="$BUNDLE_ROOT/configs/eai_pick_place_wan_vlm_mask.yaml"

# --- Task instruction ---
INSTRUCTION="pick and place the item"

# --- Dataset root for initial pose (empty = use robot's current pose) ---
DATASET_ROOT=""
# ==============================================================

# ==================== INFERENCE OPTIONS ====================
# Denoising steps must match the server setting
NUM_INFERENCE_STEPS=10

# Action interpolation: 0 = auto (uses config's global_downsample_rate)
ACTION_INTERP_FACTOR=0

# Async double-buffer (recommended for continuous motion)
ASYNC_INFERENCE=1
ASYNC_TRIGGER=32
# ============================================================

# ==================== ROBOT OPTIONS ====================
ARM="G1_29"
EE="dex1"
FREQUENCY=30
IMAGE_HOST="192.168.123.164"
# ========================================================

# --- Detect conda ---
if command -v conda &>/dev/null; then
    if conda env list 2>/dev/null | grep -q "inference_motus"; then
        eval "$(conda shell.bash hook)"
        conda activate inference_motus
    fi
fi

EXTRA_FLAGS=""
if [ "$ASYNC_INFERENCE" = "1" ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --async_inference"
fi

echo "[client] Starting MotusV2 remote inference client"
echo "[client] Server:    tcp://$SERVER_HOST:$SERVER_PORT"
echo "[client] Image host: $IMAGE_HOST"
echo "[client] Config:    $CONFIG_PATH"
echo "[client] Async:     $ASYNC_INFERENCE (trigger=$ASYNC_TRIGGER)"

python "$SCRIPTS_DIR"/motus_inference_client.py \
    --arm=$ARM \
    --ee=$EE \
    --frequency=$FREQUENCY \
    --image_host=$IMAGE_HOST \
    --server_host=$SERVER_HOST \
    --server_port=$SERVER_PORT \
    --config_path=$CONFIG_PATH \
    --root=$DATASET_ROOT \
    --instruction="$INSTRUCTION" \
    --action_interp_factor=$ACTION_INTERP_FACTOR \
    --async_trigger=$ASYNC_TRIGGER \
    $EXTRA_FLAGS
