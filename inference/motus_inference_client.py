#!/usr/bin/env python3
"""MotusV2 inference client — runs on the robot-side machine (no local GPU).

Captures images from the robot cameras, reads joint states, sends preprocessed
inputs to the remote GPU server, receives action chunks, and executes them on
the robot at 30Hz. Uses async double-buffering for continuous motion.

Usage:
    python motus_inference_client.py \
        --server_host 10.0.0.50 --server_port 5555 \
        --image_host 192.168.123.164 \
        --config_path /home/robo/configs/eai_pick_place_wan_vlm_mask.yaml \
        --instruction "pick and place" \
        --async_inference --async_trigger 32
"""

import argparse
import logging
import os
import sys
import time
import pickle
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import yaml
import zmq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [client] %(message)s")
logger = logging.getLogger(__name__)

_REORDER_FROM_RAW = [0, 1, 2, 3, 4, 5, 6, 14, 7, 8, 9, 10, 11, 12, 13, 15]


@dataclass
class ClientConfig:
    arm: str = "G1_29"
    ee: str = "dex1"
    frequency: float = 30.0
    image_host: str = "192.168.123.164"
    instruction: str = ""
    server_host: str = "127.0.0.1"
    server_port: int = 5555
    config_path: str = ""
    root: str = ""
    action_interp_factor: int = 0
    async_inference: bool = False
    async_trigger: int = 32
    num_inference_steps: int = 0


class MotusV2ClientAdapter:
    """Lightweight client-side adapter — no model loading, no GPU.

    Handles image stitching, state building, action interpolation, and action
    conversion. The heavy inference is delegated to the remote server.
    """

    def __init__(self, config_path: str, instruction: str = "",
                 action_interp_factor: int = 0):
        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)

        common = self.config_dict["common"]
        self._video_height = int(common["video_height"])
        self._video_width = int(common["video_width"])
        self._action_chunk_size = int(common["num_video_frames"]) * int(common["video_action_freq_ratio"])
        self._global_downsample_rate = int(common.get("global_downsample_rate", 1))

        ds_cfg = self.config_dict.get("dataset", {}) or {}
        self._stitch_mode = str(ds_cfg.get("stitch_mode", "aspect"))
        self.action_mode = str(ds_cfg.get("action_mode", "qpos"))

        if action_interp_factor > 0:
            self.action_interp_factor = action_interp_factor
        else:
            self.action_interp_factor = max(1, self._global_downsample_rate)

        self.current_instruction = instruction
        self.action_queue = []
        self.current_state = None
        self._first_frame = None
        self._cond_state_abs = None
        self._pending_cond_state_abs = None

        logger.info("ClientAdapter: chunk=%d, interp=%d, stitch=%s, action_mode=%s",
                     self._action_chunk_size, self.action_interp_factor,
                     self._stitch_mode, self.action_mode)

    def set_instruction(self, instruction: str):
        self.current_instruction = instruction

    @staticmethod
    def _to_arm_interleaved(vec: np.ndarray) -> np.ndarray:
        return vec[_REORDER_FROM_RAW]

    @staticmethod
    def _from_arm_interleaved(vec: np.ndarray) -> np.ndarray:
        out = np.empty(16, dtype=vec.dtype)
        out[0:7] = vec[0:7]
        out[14] = vec[7]
        out[7:14] = vec[8:15]
        out[15] = vec[15]
        return out

    def _t_shape_geometry(self, top_hw, bl_hw, br_hw):
        th, tw = self._video_height, self._video_width
        split_w = tw // 2
        right_w = tw - split_w

        def _natural_h(hw, target_w):
            h, w = hw
            return max(1, int(round(h * target_w / w)))

        bot_h = max(_natural_h(bl_hw, split_w), _natural_h(br_hw, right_w))
        bot_h = min(bot_h, th - 1)
        top_h = min(_natural_h(top_hw, tw), th - bot_h)
        natural_h = top_h + bot_h
        pad_top = (th - natural_h) // 2
        return {
            "top_h": top_h, "bot_h": bot_h,
            "split_w": split_w, "right_w": right_w,
            "pad_top": pad_top, "pad_bottom": th - natural_h - pad_top,
        }

    def _resize_with_padding(self, img, target_hw):
        """Resize image to fit target_hw preserving aspect ratio, pad with black."""
        import cv2
        th, tw = target_hw
        h, w = img.shape[:2]
        scale = min(tw / w, th / h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, img.shape[2] if img.ndim == 3 else 1), dtype=img.dtype)
        y0 = (th - new_h) // 2
        x0 = (tw - new_w) // 2
        if img.ndim == 3:
            canvas[y0:y0 + new_h, x0:x0 + new_w, :] = resized
        else:
            canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
        return canvas

    def build_stitched_image(self, head_img, left_wrist_img, right_wrist_img):
        th, tw = self._video_height, self._video_width

        if self._stitch_mode == "aspect":
            geo = self._t_shape_geometry(
                head_img.shape[:2], left_wrist_img.shape[:2], right_wrist_img.shape[:2])
            top_h, bot_h = geo["top_h"], geo["bot_h"]
            split_w, right_w = geo["split_w"], geo["right_w"]
            y0 = geo["pad_top"]

            canvas = np.zeros((th, tw, 3), dtype=np.uint8)
            canvas[y0:y0 + top_h, :, :] = self._resize_with_padding(head_img, (top_h, tw))
            yb = y0 + top_h
            canvas[yb:yb + bot_h, :split_w, :] = self._resize_with_padding(left_wrist_img, (bot_h, split_w))
            canvas[yb:yb + bot_h, split_w:, :] = self._resize_with_padding(right_wrist_img, (bot_h, right_w))
            return canvas.astype(np.float32) / 255.0

        # legacy double resize
        top_full = self._resize_with_padding(head_img, (th, tw))
        bl_full = self._resize_with_padding(left_wrist_img, (th, tw))
        br_full = self._resize_with_padding(right_wrist_img, (th, tw))

        top_h = th // 2
        bottom_h = th - top_h
        split_w = tw // 2
        right_w = tw - split_w

        def _to_sub_region(full_img, sub_h, sub_w):
            if full_img.shape[0] == sub_h and full_img.shape[1] == sub_w:
                return full_img.astype(np.float32) / 255.0
            arr_uint8 = (full_img * 255.0).clip(0, 255).astype(np.uint8) if full_img.dtype != np.uint8 else full_img
            resized = self._resize_with_padding(arr_uint8, (sub_h, sub_w))
            return resized.astype(np.float32) / 255.0

        top_r = _to_sub_region(top_full, top_h, tw)
        bl_r = _to_sub_region(bl_full, bottom_h, split_w)
        br_r = _to_sub_region(br_full, bottom_h, right_w)

        stitched = np.zeros((th, tw, 3), dtype=np.float32)
        stitched[:top_h, :] = top_r
        stitched[top_h:, :split_w] = bl_r
        stitched[top_h:, split_w:] = br_r
        return stitched

    def observation_to_model_input(self, observation, current_arm_q, ee_shared_mem):
        """Convert observation to (first_frame_np, state_np) for server.

        Returns numpy arrays (not CUDA tensors) since the client has no GPU.
        """
        import cv2

        def _to_uint8_hwc(img):
            if hasattr(img, 'cpu'):
                img = img.cpu().numpy()
            if img is not None and img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            return img

        head_img = _to_uint8_hwc(observation.get("observation.images.cam_left_high"))
        if head_img is None:
            raise ValueError("Missing cam_left_high image")

        left_wrist = _to_uint8_hwc(observation.get("observation.images.cam_left_wrist"))
        right_wrist = _to_uint8_hwc(observation.get("observation.images.cam_right_wrist"))
        if left_wrist is None or right_wrist is None:
            raise ValueError("Missing wrist camera images")

        stitched = self.build_stitched_image(head_img, left_wrist, right_wrist)
        first_frame_np = (stitched * 255).astype(np.uint8)

        left_grip_val = 0.0
        right_grip_val = 0.0
        if ee_shared_mem:
            from multiprocessing.sharedctypes import SynchronizedArray
            if isinstance(ee_shared_mem.get("left"), SynchronizedArray):
                pass
            elif hasattr(ee_shared_mem.get("left"), "value"):
                left_grip_val = float(ee_shared_mem["left"].value)
                right_grip_val = float(ee_shared_mem["right"].value)

        if self.action_mode == "qpos":
            raw_state = np.zeros(16, dtype=np.float32)
            raw_state[0:7] = current_arm_q[0:7]
            raw_state[7:14] = current_arm_q[7:14]
            raw_state[14] = left_grip_val
            raw_state[15] = right_grip_val
            state = self._to_arm_interleaved(raw_state)
        else:
            raise NotImplementedError("eef/eef_delta modes require FK on GPU side; "
                                      "use the monolithic eval_g1_motus.py for those.")

        return first_frame_np, state.astype(np.float32)

    def interpolate_chunk(self, actions):
        f = self.action_interp_factor
        if f <= 1:
            return actions
        T, A = actions.shape
        N = T * f
        xp = np.arange(T, dtype=np.float64)
        xq = np.linspace(0.0, T - 1, N)
        out = np.empty((N, A), dtype=actions.dtype)
        for a in range(A):
            out[:, a] = np.interp(xq, xp, actions[:, a])
        return out

    def build_exec_queue(self, raw_actions):
        exec_actions = self.interpolate_chunk(raw_actions)
        return [(exec_actions[i], None) for i in range(exec_actions.shape[0])]

    def qpos_action_to_g1_action(self, qpos_action_16):
        raw = self._from_arm_interleaved(qpos_action_16)
        arm_action = np.concatenate([raw[0:7], raw[7:14]])
        left_grip = raw[14]
        right_grip = raw[15]
        return arm_action, left_grip, right_grip

    def step(self, current_arm_q):
        if not self.action_queue:
            return None
        entry = self.action_queue.pop(0)
        if isinstance(entry, tuple):
            entry, cond = entry
        if self.action_mode == "qpos":
            return self.qpos_action_to_g1_action(entry)
        return self.qpos_action_to_g1_action(entry)

    @property
    def has_cached_actions(self):
        return len(self.action_queue) > 0


class RemoteInferenceClient:
    """ZMQ client that sends inference requests to the remote GPU server."""

    def __init__(self, server_host: str, server_port: int, timeout_ms: int = 10000):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.connect(f"tcp://{server_host}:{server_port}")
        self.server_addr = f"{server_host}:{server_port}"
        logger.info("Connected to inference server at tcp://%s", self.server_addr)

    def predict(self, first_frame_np, state_np, instruction,
                rtc_prev=None, rtc_inference_delay=None):
        req = {
            "first_frame": first_frame_np,
            "state": state_np,
            "instruction": instruction,
            "rtc_prev": rtc_prev,
            "rtc_inference_delay": rtc_inference_delay,
        }
        self.socket.send(pickle.dumps(req))
        resp = pickle.loads(self.socket.recv())
        if resp["status"] != "ok":
            raise RuntimeError(f"Server error: {resp.get('message', 'unknown')}")
        return resp["actions"], resp.get("predict_ms", 0.0)

    def close(self):
        self.socket.close()
        self.ctx.term()


def _write_gripper(ee_shared_mem, left_grip, right_grip):
    if not ee_shared_mem:
        return
    from multiprocessing.sharedctypes import SynchronizedArray
    from unitree_lerobot.eval_robot.utils.utils import to_list
    if isinstance(ee_shared_mem.get("left"), SynchronizedArray):
        ee_shared_mem["left"][:] = to_list(np.array([left_grip]))
        ee_shared_mem["right"][:] = to_list(np.array([right_grip]))
    elif hasattr(ee_shared_mem.get("left"), "value"):
        ee_shared_mem["left"].value = float(left_grip)
        ee_shared_mem["right"].value = float(right_grip)


def run_eval_async(cfg: ClientConfig):
    """Async double-buffer eval loop with remote inference."""
    from unitree_lerobot.eval_robot.make_robot import (
        setup_image_client, setup_robot_interface, process_images_and_observations,
    )

    logger.info("[async] Config: %s", cfg)

    image_info = setup_image_client(cfg)
    if isinstance(image_info, tuple):
        image_client, camera_config = image_info
        image_info = {"image_client": image_client, "camera_config": camera_config}

    robot_interface = setup_robot_interface(cfg)
    arm_ctrl = robot_interface["arm_ctrl"]
    arm_ik = robot_interface["arm_ik"]
    ee_shared_mem = robot_interface["ee_shared_mem"]
    img_client = image_info["image_client"]
    camera_config = image_info["camera_config"]

    adapter = MotusV2ClientAdapter(cfg.config_path, cfg.instruction, cfg.action_interp_factor)
    adapter.set_instruction(cfg.instruction)
    remote = RemoteInferenceClient(cfg.server_host, cfg.server_port)

    if cfg.root:
        _load_and_move_to_initial_pose(cfg.root, arm_ctrl, arm_ik, ee_shared_mem)
    else:
        init_arm_q = arm_ctrl.get_current_dual_arm_q()
        tau = arm_ik.solve_tau(init_arm_q)
        arm_ctrl.ctrl_dual_arm(init_arm_q, tau)
        time.sleep(1.0)
        logger.info("Robot initialized to current pose")

    input("Enter 's' to start evaluation: ")
    logger.info("[async] Starting at %.1f Hz, instruction: %s", cfg.frequency, cfg.instruction)

    interp_f = max(1, adapter.action_interp_factor)
    trigger = max(1, cfg.async_trigger)
    lock = threading.Lock()
    shared = {
        "busy": False,
        "req_frame": None, "req_state": None,
        "result_actions": None, "result_predict_ms": 0.0,
        "stop": False,
    }

    def worker():
        while True:
            with lock:
                if shared["stop"]:
                    return
                go = shared["busy"] and shared["req_frame"] is not None
                frame = shared["req_frame"]
                st = shared["req_state"]
            if not go:
                time.sleep(0.002)
                continue
            try:
                actions, predict_ms = remote.predict(
                    frame, st, cfg.instruction)
            except Exception as e:
                logger.error("[async worker] predict error: %s", e)
                actions, predict_ms = None, 0.0
            with lock:
                shared["result_actions"] = actions
                shared["result_predict_ms"] = predict_ms
                shared["req_frame"] = None
                shared["req_state"] = None
                shared["busy"] = False

    th = threading.Thread(target=worker, daemon=True)
    th.start()

    # Bootstrap: first (blocking) chunk
    observation, current_arm_q = process_images_and_observations(img_client, camera_config, arm_ctrl)
    first_frame_np, state_np = adapter.observation_to_model_input(observation, current_arm_q, ee_shared_mem)
    t0 = time.perf_counter()
    raw, predict_ms = remote.predict(first_frame_np, state_np, cfg.instruction)
    logger.info("[bootstrap] predict=%.1fms chunk=%s", predict_ms, raw.shape)
    adapter.action_queue = adapter.build_exec_queue(raw)
    exec_count = 0
    pending_mark = None
    idx = 0

    try:
        while True:
            loop_start = time.perf_counter()
            current_arm_q = arm_ctrl.get_current_dual_arm_q()

            # 1) consume one action
            result = adapter.step(current_arm_q)
            if result is not None:
                arm_action, left_grip, right_grip = result
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)
                _write_gripper(ee_shared_mem, left_grip, right_grip)
                exec_count += 1

            # 2) splice ready result
            with lock:
                ready_actions = shared["result_actions"]
                ready_ms = shared["result_predict_ms"]
                if ready_actions is not None:
                    shared["result_actions"] = None
            if ready_actions is not None:
                adapter.action_queue = adapter.build_exec_queue(ready_actions)
                pending_mark = None
                logger.info("[splice] predict=%.1fms queue=%d", ready_ms, len(adapter.action_queue))

            # 3) trigger next inference
            with lock:
                worker_idle = not shared["busy"]
            if worker_idle and pending_mark is None and len(adapter.action_queue) <= trigger:
                observation, snap_arm_q = process_images_and_observations(img_client, camera_config, arm_ctrl)
                snap_frame, snap_state = adapter.observation_to_model_input(
                    observation, snap_arm_q, ee_shared_mem)
                pending_mark = exec_count
                with lock:
                    shared["req_frame"] = snap_frame
                    shared["req_state"] = snap_state
                    shared["busy"] = True

            idx += 1
            if idx % 30 == 0:
                logger.info("[%d|async] queue=%d busy=%s",
                            idx, len(adapter.action_queue), shared["busy"])
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("[async] Error: %s", e, exc_info=True)
    finally:
        with lock:
            shared["stop"] = True
        th.join(timeout=2.0)
        remote.close()
        if image_info:
            image_info["image_client"].close()
        logger.info("Evaluation ended (async)")


def run_eval_serial(cfg: ClientConfig):
    """Serial eval loop — predict one chunk, execute fully, repeat."""
    from unitree_lerobot.eval_robot.make_robot import (
        setup_image_client, setup_robot_interface, process_images_and_observations,
    )

    logger.info("Config: %s", cfg)

    image_info = setup_image_client(cfg)
    if isinstance(image_info, tuple):
        image_client, camera_config = image_info
        image_info = {"image_client": image_client, "camera_config": camera_config}

    robot_interface = setup_robot_interface(cfg)
    arm_ctrl = robot_interface["arm_ctrl"]
    arm_ik = robot_interface["arm_ik"]
    ee_shared_mem = robot_interface["ee_shared_mem"]
    img_client = image_info["image_client"]
    camera_config = image_info["camera_config"]

    adapter = MotusV2ClientAdapter(cfg.config_path, cfg.instruction, cfg.action_interp_factor)
    adapter.set_instruction(cfg.instruction)
    remote = RemoteInferenceClient(cfg.server_host, cfg.server_port)

    if cfg.root:
        _load_and_move_to_initial_pose(cfg.root, arm_ctrl, arm_ik, ee_shared_mem)
    else:
        init_arm_q = arm_ctrl.get_current_dual_arm_q()
        tau = arm_ik.solve_tau(init_arm_q)
        arm_ctrl.ctrl_dual_arm(init_arm_q, tau)
        time.sleep(1.0)
        logger.info("Robot initialized to current pose")

    input("Enter 's' to start evaluation: ")
    logger.info("Starting at %.1f Hz, instruction: %s", cfg.frequency, cfg.instruction)

    idx = 0
    try:
        while True:
            loop_start = time.perf_counter()

            if adapter.has_cached_actions:
                current_arm_q = arm_ctrl.get_current_dual_arm_q()
                result = adapter.step(current_arm_q)
                if result is not None:
                    arm_action, left_grip, right_grip = result
                    tau = arm_ik.solve_tau(arm_action)
                    arm_ctrl.ctrl_dual_arm(arm_action, tau)
                    _write_gripper(ee_shared_mem, left_grip, right_grip)
            else:
                t0 = time.perf_counter()
                observation, current_arm_q = process_images_and_observations(
                    img_client, camera_config, arm_ctrl)
                t_image = time.perf_counter() - t0

                t0 = time.perf_counter()
                first_frame_np, state_np = adapter.observation_to_model_input(
                    observation, current_arm_q, ee_shared_mem)
                t_obs = time.perf_counter() - t0

                t0 = time.perf_counter()
                actions_np, predict_ms = remote.predict(
                    first_frame_np, state_np, cfg.instruction)
                t_total = time.perf_counter() - t0

                adapter.action_queue = adapter.build_exec_queue(actions_np)
                logger.info("[%d|predict] image=%.1fms obs=%.1fms server=%.1fms chunk=%s queue=%d",
                            idx, t_image * 1000, t_obs * 1000, t_total * 1000,
                            actions_np.shape, len(adapter.action_queue))

                result = adapter.step(current_arm_q)
                if result is not None:
                    arm_action, left_grip, right_grip = result
                    tau = arm_ik.solve_tau(arm_action)
                    arm_ctrl.ctrl_dual_arm(arm_action, tau)
                    _write_gripper(ee_shared_mem, left_grip, right_grip)

            idx += 1
            time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start)))

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Error: %s", e, exc_info=True)
    finally:
        remote.close()
        if image_info:
            image_info["image_client"].close()
        logger.info("Evaluation ended")


def _load_and_move_to_initial_pose(root, arm_ctrl, arm_ik, ee_shared_mem, steps=50):
    import pyarrow.parquet as pq
    data_dir = os.path.join(root, "data", "chunk-000")
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])
    for f in files:
        t = pq.read_table(os.path.join(data_dir, f))
        states = t.column("observation.state").to_pylist()
        ep_indices = t.column("episode_index").to_pylist()
        for i, ep in enumerate(ep_indices):
            if ep == 0:
                state = np.array(states[i], dtype=np.float32)
                arm_q = state[:14].copy()
                left_grip = float(state[14])
                right_grip = float(state[15])
                logger.info("Loaded initial pose from dataset episode 0: arm_q=%s", arm_q)
                current_arm_q = arm_ctrl.get_current_dual_arm_q()
                for j in range(steps):
                    alpha = (j + 1) / steps
                    interp_q = current_arm_q * (1 - alpha) + arm_q * alpha
                    tau = arm_ik.solve_tau(interp_q)
                    arm_ctrl.ctrl_dual_arm(interp_q, tau)
                    _write_gripper(ee_shared_mem, left_grip, right_grip)
                    time.sleep(0.02)
                tau = arm_ik.solve_tau(arm_q)
                arm_ctrl.ctrl_dual_arm(arm_q, tau)
                _write_gripper(ee_shared_mem, left_grip, right_grip)
                time.sleep(0.5)
                logger.info("Robot moved to dataset initial pose")
                return
    raise FileNotFoundError(f"No episode 0 found in {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MotusV2 remote inference client")
    parser.add_argument("--arm", type=str, default="G1_29")
    parser.add_argument("--ee", type=str, default="dex1")
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--image_host", type=str, default="192.168.123.164")
    parser.add_argument("--instruction", type=str, required=True)
    parser.add_argument("--server_host", type=str, required=True,
                        help="GPU server IP address")
    parser.add_argument("--server_port", type=int, default=5555)
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to inference YAML config")
    parser.add_argument("--root", type=str, default="",
                        help="LeRobot dataset root for initial pose (empty = use current pose)")
    parser.add_argument("--action_interp_factor", type=int, default=0)
    parser.add_argument("--async_inference", action="store_true", default=False)
    parser.add_argument("--async_trigger", type=int, default=32)
    parser.add_argument("--num_inference_steps", type=int, default=0)
    parser.add_argument("--motion", action="store_true", default=False)

    args = parser.parse_args()
    cfg = ClientConfig(
        arm=args.arm, ee=args.ee, frequency=args.frequency,
        image_host=args.image_host, instruction=args.instruction,
        server_host=args.server_host, server_port=args.server_port,
        config_path=args.config_path, root=args.root,
        action_interp_factor=args.action_interp_factor,
        async_inference=args.async_inference,
        async_trigger=args.async_trigger,
        num_inference_steps=args.num_inference_steps,
    )

    if cfg.async_inference:
        run_eval_async(cfg)
    else:
        run_eval_serial(cfg)
