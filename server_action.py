#!/usr/bin/env python3
import argparse
import asyncio
import json
import logging
import time
from typing import Any, Dict

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed


def make_action(obs: Dict[str, Any]) -> Dict[str, Any]:
    """
    模拟服务器端推理：
    输入 obs，输出 action。
    后面你可以把这里替换成真实模型推理。
    """
    qpos = obs.get("qpos", [0.0] * 6)
    target_qpos = obs.get("target_qpos", [0.0] * len(qpos))

    joint_delta = []

    for q, target in zip(qpos, target_qpos):
        delta = 0.5 * (target - q)

        # 限幅，模拟真实机械臂动作安全限制
        delta = max(min(delta, 0.2), -0.2)
        joint_delta.append(delta)

    return {
        "joint_delta": joint_delta,
        "gripper": obs.get("gripper_cmd", 0.0),
    }


async def handle_client(ws):
    addr = ws.remote_address
    print(f"[+] client connected: {addr}", flush=True)

    try:
        async for raw in ws:
            server_recv_ts = time.time()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await ws.send(json.dumps({
                    "type": "error",
                    "message": "invalid json",
                    "server_ts": time.time(),
                }, ensure_ascii=False))
                continue

            msg_type = msg.get("type")

            # WebSocket 层面的连通性检测
            if msg_type == "ping":
                await ws.send(json.dumps({
                    "type": "pong",
                    "server_ts": time.time(),
                }, ensure_ascii=False))
                continue

            if msg_type != "obs":
                await ws.send(json.dumps({
                    "type": "error",
                    "message": "expected message type: obs",
                    "server_ts": time.time(),
                    "received_type": msg_type,
                }, ensure_ascii=False))
                continue

            seq = msg.get("seq", -1)
            obs = msg.get("obs", {})
            client_send_ts = msg.get("client_send_ts")

            infer_start = time.time()

            # 模拟真实推理耗时
            await asyncio.sleep(0.02)

            # 这里未来替换成真实策略模型
            action = make_action(obs)

            infer_end = time.time()

            reply = {
                "type": "action",
                "seq": seq,
                "action": action,
                "client_send_ts": client_send_ts,
                "server_recv_ts": server_recv_ts,
                "server_send_ts": time.time(),
                "infer_ms": round((infer_end - infer_start) * 1000, 3),
            }

            await ws.send(json.dumps(reply, ensure_ascii=False))

            print(
                f"[obs] seq={seq} "
                f"infer_ms={reply['infer_ms']} "
                f"action={action}",
                flush=True,
            )

    except ConnectionClosed as e:
        print(
            f"[-] client disconnected: {addr}, code={e.code}, reason={e.reason}",
            flush=True,
        )
    except Exception as e:
        print(f"[!] server error from {addr}: {e}", flush=True)


async def main(host: str, port: int):
    async with serve(
        handle_client,
        host,
        port,
        ping_interval=20,
        ping_timeout=20,
    ):
        print(f"[*] WebSocket action server listening on ws://{host}:{port}", flush=True)
        print("[*] Waiting for obs from client...", flush=True)
        await asyncio.Future()


if __name__ == "__main__":
    # 屏蔽 websockets 对普通 TCP 空连接产生的 traceback 日志
    logging.getLogger("websockets.server").setLevel(logging.CRITICAL)
    logging.getLogger("websockets").setLevel(logging.CRITICAL)

    parser = argparse.ArgumentParser(description="WebSocket obs-action server")
    parser.add_argument("--host", default="127.0.0.1", help="listen host")
    parser.add_argument("--port", type=int, default=8000, help="listen port")
    args = parser.parse_args()

    asyncio.run(main(args.host, args.port))