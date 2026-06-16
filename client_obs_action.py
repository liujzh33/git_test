#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
import time

import websockets


def make_obs(seq: int) -> dict:
    """
    模拟真实机器人 obs。
    后面可以替换成真实机械臂状态、相机图像、夹爪状态等。
    """
    qpos = [random.uniform(-1.0, 1.0) for _ in range(6)]

    obs = {
        "robot_id": "robot_001",
        "seq": seq,

        # 机械臂关节状态
        "qpos": qpos,
        "qvel": [random.uniform(-0.05, 0.05) for _ in range(6)],

        # 目标关节位置，这里只是模拟
        "target_qpos": [0.0, 0.2, -0.1, 0.3, 0.0, 0.1],

        # 末端位姿，模拟数据
        "eef_pose": {
            "x": random.uniform(0.2, 0.5),
            "y": random.uniform(-0.2, 0.2),
            "z": random.uniform(0.1, 0.4),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 0.0,
        },

        # 夹爪状态
        "gripper_width": random.uniform(0.0, 0.08),
        "gripper_cmd": 0.0,

        # 相机信息，目前只传元数据
        "camera": {
            "front_rgb_shape": [480, 640, 3],
            "wrist_rgb_shape": [240, 320, 3],
            "note": "当前只传元数据；真实图像可改成 JPEG bytes/base64",
        },

        "obs_ts": time.time(),
    }

    return obs


async def check_websocket(host: str, port: int, timeout: float):
    """
    用真正的 WebSocket 握手检查端口是否可用。
    不再使用普通 TCP 空连接，因此不会触发 server 的 opening handshake failed。
    """
    uri = f"ws://{host}:{port}"
    print(f"[*] Checking WebSocket {uri} ...")

    try:
        async with websockets.connect(
            uri,
            ping_interval=20,
            ping_timeout=20,
            open_timeout=timeout,
        ) as ws:
            await ws.send(json.dumps({
                "type": "ping",
                "client_ts": time.time(),
            }, ensure_ascii=False))

            raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            reply = json.loads(raw)

            if reply.get("type") == "pong":
                print(f"[+] WebSocket port is reachable: {uri}")
                return True

            print(f"[!] Connected but unexpected reply: {reply}")
            return False

    except Exception as e:
        print(f"[!] WebSocket check failed: {e}")
        return False


async def run_client(host: str, port: int, hz: float, timeout: float):
    uri = f"ws://{host}:{port}"
    interval = 1.0 / hz

    print(f"[*] Connecting to {uri}")

    async with websockets.connect(
        uri,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=timeout,
    ) as ws:
        print("[+] WebSocket connected")

        seq = 0

        while True:
            client_send_ts = time.time()

            obs = make_obs(seq)

            msg = {
                "type": "obs",
                "seq": seq,
                "client_send_ts": client_send_ts,
                "obs": obs,
            }

            await ws.send(json.dumps(msg, ensure_ascii=False))

            raw = await ws.recv()
            client_recv_ts = time.time()

            try:
                reply = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[!] Invalid JSON from server: {raw}")
                continue

            if reply.get("type") == "action":
                rtt_ms = (client_recv_ts - client_send_ts) * 1000

                print(
                    f"[action] seq={reply.get('seq')} "
                    f"rtt_ms={rtt_ms:.2f}ms "
                    f"infer_ms={reply.get('infer_ms')} "
                    f"action={reply.get('action')}"
                )

                # 这里未来接真实机械臂执行动作
                # execute_action(reply["action"])

            else:
                print(f"[server] {reply}")

            seq += 1
            await asyncio.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Robot obs-action WebSocket client")
    parser.add_argument("host", help="server host")
    parser.add_argument("--port", type=int, default=8000, help="server port")
    parser.add_argument("--hz", type=float, default=10.0, help="obs send frequency")
    parser.add_argument("--timeout", type=float, default=10.0, help="connect timeout")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="only check WebSocket connectivity, then exit",
    )

    args = parser.parse_args()

    if args.check_only:
        ok = asyncio.run(check_websocket(args.host, args.port, args.timeout))
        raise SystemExit(0 if ok else 1)

    asyncio.run(run_client(args.host, args.port, args.hz, args.timeout))


if __name__ == "__main__":
    main()