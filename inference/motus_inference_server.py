#!/usr/bin/env python3
"""MotusV2 inference server — runs on the 5090 GPU server.

Listens on a ZMQ REP socket, receives (first_frame, state, instruction) from
the robot-side client, runs MotusV2 inference, and returns the action chunk.

Usage:
    python motus_inference_server.py \
        --checkpoint_path /home/robo/pretrained_models/ckpts/eai_pick_place/mp_rank_00_model_states.pt \
        --wan_path /home/robo/pretrained_models/Wan2.2-TI2V-5B \
        --vlm_path /home/robo/pretrained_models/Qwen3-VL-2B-Instruct \
        --config_path /home/robo/InferenceMotusV2/configs/eai_pick_place_wan_vlm_mask.yaml \
        --t5_cache_dir /home/robo/pretrained_models/t5_cache_eai_pick_place \
        --host 0.0.0.0 --port 5555
"""

import argparse
import logging
import os
import sys
import time
import pickle

import numpy as np
import torch
import zmq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
logger = logging.getLogger(__name__)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_ROOT = os.path.dirname(SCRIPTS_DIR)
POLICY_DIR = os.path.join(BUNDLE_ROOT, "policy", "MotusWanVlmDirectMaskMotion")
MODELS_DIR = os.path.join(POLICY_DIR, "models")
BAK_DIR = os.path.join(POLICY_DIR, "bak")
UTILS_DIR = os.path.join(POLICY_DIR, "utils")

for _p in [BUNDLE_ROOT, POLICY_DIR, MODELS_DIR, BAK_DIR, UTILS_DIR,
           os.path.join(BUNDLE_ROOT, "models"), os.path.join(BUNDLE_ROOT, "bak"),
           os.path.join(BUNDLE_ROOT, "utils")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("MOTUS_POLICY_DIR", POLICY_DIR)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MOTUS_COMPILE", "0")


def main():
    parser = argparse.ArgumentParser(description="MotusV2 inference server")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to mp_rank_00_model_states.pt or its parent directory")
    parser.add_argument("--wan_path", type=str, required=True,
                        help="Path to Wan2.2-TI2V-5B directory")
    parser.add_argument("--vlm_path", type=str, required=True,
                        help="Path to Qwen3-VL-2B-Instruct directory")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to inference YAML config")
    parser.add_argument("--t5_cache_dir", type=str, default="",
                        help="Path to pre-populated T5 cache directory")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address (default: all interfaces)")
    parser.add_argument("--port", type=int, default=5555,
                        help="ZMQ REP port (default: 5555)")
    parser.add_argument("--num_inference_steps", type=int, default=0,
                        help="Override denoising steps (0 = use config default)")
    parser.add_argument("--instruction", type=str, default="",
                        help="Default instruction if client doesn't send one")
    args = parser.parse_args()

    from motusv2_adapter.adapter import MotusV2PolicyAdapter

    logger.info("Loading MotusV2 model (this takes ~30s)...")
    t0 = time.perf_counter()

    policy = MotusV2PolicyAdapter(
        checkpoint_path=args.checkpoint_path,
        wan_path=args.wan_path,
        vlm_path=args.vlm_path,
        config_path=args.config_path,
        t5_cache_dir=args.t5_cache_dir,
        device="cuda",
        arm_ik=None,
        arm_dof=14,
        ee_dof=1,
        instruction=args.instruction,
        num_inference_steps=args.num_inference_steps,
        action_interp_factor=1,
        rtc_enabled=False,
        rtc_mode="inpaint",
    )
    policy.set_instruction(args.instruction)
    load_time = time.perf_counter() - t0
    logger.info("Model loaded in %.1fs. chunk_size=%d, action_mode=%s, stitch_mode=%s",
                load_time, policy._action_chunk_size, policy.action_mode,
                policy._stitch_mode)

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, 0)
    socket.bind(f"tcp://{args.host}:{args.port}")
    logger.info("Server listening on tcp://%s:%d", args.host, args.port)

    request_count = 0
    try:
        while True:
            try:
                msg = socket.recv()
                req = pickle.loads(msg)
            except Exception as e:
                logger.error("Failed to parse request: %s", e)
                socket.send(pickle.dumps({"status": "error", "message": str(e)}))
                continue

            first_frame_np = req["first_frame"]
            state_np = req["state"]
            instruction = req.get("instruction", args.instruction)
            rtc_prev_np = req.get("rtc_prev", None)
            rtc_inference_delay = req.get("rtc_inference_delay", None)

            if instruction != policy.current_instruction:
                policy.set_instruction(instruction)

            t0 = time.perf_counter()

            first_frame = torch.from_numpy(first_frame_np).permute(2, 0, 1).unsqueeze(0).float().cuda()
            state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).cuda()

            rtc_prev = None
            if rtc_prev_np is not None:
                rtc_prev = np.asarray(rtc_prev_np, dtype=np.float32)

            try:
                actions_np = policy.predict_action_chunk(
                    first_frame,
                    state_tensor,
                    prev_chunk_left_over=rtc_prev,
                    rtc_inference_delay=rtc_inference_delay,
                )
                predict_ms = (time.perf_counter() - t0) * 1000
                request_count += 1
                if request_count % 10 == 0:
                    logger.info("Request #%d: predict=%.1fms chunk=%s",
                                request_count, predict_ms, actions_np.shape)
                resp = {
                    "status": "ok",
                    "actions": actions_np,
                    "predict_ms": predict_ms,
                }
            except Exception as e:
                logger.error("Inference failed: %s", e, exc_info=True)
                resp = {"status": "error", "message": str(e)}

            socket.send(pickle.dumps(resp))

    except KeyboardInterrupt:
        logger.info("Server interrupted")
    finally:
        socket.close()
        ctx.term()
        logger.info("Server stopped (total requests: %d)", request_count)


if __name__ == "__main__":
    main()
