#!/usr/bin/env python3
"""
Standalone WebSocket client for MotusWanVlmDirectMask Dobot inference.

This file intentionally does not import client.py, server_vlm_mask.py,
websocket_client_policy.py, or websocket_policy_server.py. It sends observations
as binary MessagePack frames, including NumPy image/state arrays without Base64.
"""

import argparse
import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import msgpack
import numpy as np
import torch
import websockets.sync.client
from PIL import Image


log = logging.getLogger("dobot_websocket_vlm_client")


def _pack_default(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes(),
        }
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unsupported type for MessagePack: {type(obj)!r}")


def pack_message(data: Any) -> bytes:
    return msgpack.packb(data, default=_pack_default, use_bin_type=True)


def _object_hook(obj: Dict[Any, Any]) -> Any:
    if obj.get("__ndarray__"):
        array = np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"]))
        return array.reshape(obj["shape"]).copy()
    return obj


def unpack_message(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, object_hook=_object_hook)


class DobotWebSocketVLMClient:
    """Small policy-style client for the standalone Dobot WebSocket server."""

    def __init__(
        self,
        host: str = "localhost",
        port: Optional[int] = 6790,
        reconnect_interval_s: float = 5.0,
    ) -> None:
        if host.startswith("ws://") or host.startswith("wss://"):
            self.uri = host
        else:
            self.uri = f"ws://{host}"
        if port is not None and ":" not in self.uri.removeprefix("ws://").removeprefix("wss://"):
            self.uri += f":{port}"

        self.reconnect_interval_s = reconnect_interval_s
        self._ws, self._metadata = self._connect_with_retry()

    def _connect_with_retry(self) -> Tuple[websockets.sync.client.ClientConnection, Dict[str, Any]]:
        log.info("Waiting for WebSocket server at %s ...", self.uri)
        while True:
            try:
                ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None)
                metadata_raw = ws.recv()
                if isinstance(metadata_raw, str):
                    raise RuntimeError(f"Expected binary metadata, got text:\n{metadata_raw}")
                metadata = unpack_message(metadata_raw)
                log.info("Connected to server metadata: %s", metadata)
                return ws, metadata
            except ConnectionRefusedError:
                log.info("Server is not ready, retrying in %.1fs", self.reconnect_interval_s)
                time.sleep(self.reconnect_interval_s)

    def get_server_metadata(self) -> Dict[str, Any]:
        return self._metadata

    def infer(self, obs: Dict[str, Any], timeout_s: Optional[float] = None) -> Dict[str, Any]:
        del timeout_s  # websockets.sync does not expose per-recv timeout in the same shape.
        self._ws.send(pack_message(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return unpack_message(response)

    def close(self) -> None:
        self._ws.close()


def read_image_as_uint8(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def create_random_image(width: int = 384, height: int = 320) -> np.ndarray:
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json_float_list(path: str) -> List[float]:
    data = read_json(path)
    if not isinstance(data, list):
        raise ValueError("state json must be a one-dimensional list, e.g. [0.1, 0.2, ...]")
    return [float(x) for x in data]


def parse_csv_to_float_list(csv_str: str) -> List[float]:
    values = [x.strip() for x in csv_str.split(",") if x.strip()]
    return [float(x) for x in values]


def save_frame_grid_if_present(result: Dict[str, Any], output_path: Optional[str]) -> None:
    frame_grid_b64 = result.get("frame_grid_image")
    if not frame_grid_b64 or not output_path:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(frame_grid_b64))


def build_images_payload(args: argparse.Namespace) -> Optional[Any]:
    if args.images:
        return [read_image_as_uint8(path) for path in args.images]

    if args.top_image or args.left_wrist_image or args.right_wrist_image:
        images: Dict[str, np.ndarray] = {}
        if args.top_image:
            images["top"] = read_image_as_uint8(args.top_image)
        if args.left_wrist_image:
            images["left_wrist"] = read_image_as_uint8(args.left_wrist_image)
        if args.right_wrist_image:
            images["right_wrist"] = read_image_as_uint8(args.right_wrist_image)
        return images

    if args.image:
        return [read_image_as_uint8(args.image)]

    if args.use_random_images:
        return {
            "top": create_random_image(),
            "left_wrist": create_random_image(),
            "right_wrist": create_random_image(),
        }

    return None


def load_t5_embeddings_payload(path: Optional[str]) -> Any:
    if not path:
        return None
    loaded = torch.load(path, map_location="cpu")
    if isinstance(loaded, torch.Tensor):
        return loaded.detach().cpu().numpy()
    if isinstance(loaded, list):
        return [
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
            for item in loaded
        ]
    return loaded


def build_observation(args: argparse.Namespace) -> Dict[str, Any]:
    state = None
    if args.state_json:
        state = read_json_float_list(args.state_json)
    elif args.state_csv:
        state = parse_csv_to_float_list(args.state_csv)

    obs: Dict[str, Any] = {
        "instruction": args.instruction,
        "auto_find_t5_embeddings": not args.disable_auto_find_t5_embeddings,
        "return_frame_grid": args.return_frame_grid,
    }

    images = build_images_payload(args)
    if images is not None:
        obs["images"] = images
    elif args.image_path:
        obs["image_path"] = args.image_path

    if state is not None:
        obs["state"] = np.asarray(state, dtype=np.float32)
    if args.t5_embeddings_path:
        if args.send_t5_embeddings:
            obs["t5_embeddings"] = load_t5_embeddings_payload(args.t5_embeddings_path)
        else:
            obs["t5_embeddings_path"] = args.t5_embeddings_path
    if args.t5_embeddings_dir:
        obs["t5_embeddings_dir"] = args.t5_embeddings_dir
    if args.num_inference_steps is not None:
        obs["num_inference_steps"] = args.num_inference_steps

    return obs


def test_connectivity(client: DobotWebSocketVLMClient) -> bool:
    metadata = client.get_server_metadata()
    print("Connected to WebSocket server")
    print(f"  Device: {metadata.get('device')}")
    print(f"  Model loaded: {metadata.get('model_loaded')}")
    print(f"  Checkpoint: {metadata.get('checkpoint_path')}")
    common = metadata.get("common", {})
    print(f"  State dim: {common.get('state_dim')}")
    print(f"  Action dim: {common.get('action_dim')}")
    return bool(metadata.get("model_loaded"))


def test_inference(client: DobotWebSocketVLMClient, obs: Dict[str, Any], frame_grid_output: Optional[str]) -> bool:
    start_time = time.time()
    result = client.infer(obs)
    elapsed_ms = (time.time() - start_time) * 1000.0

    print("Inference completed")
    print(f"  Client round-trip: {elapsed_ms:.2f}ms")
    print(f"  Server processing: {result.get('processing_time_ms', 0):.2f}ms")
    print(f"  Action shape: {result.get('action_shape')}")
    print(f"  Predicted frames shape: {result.get('predicted_frames_shape')}")
    if "server_timing" in result:
        print(f"  Server timing: {result['server_timing']}")

    actions = np.asarray(result.get("predicted_actions", []), dtype=np.float32)
    if actions.size:
        print(f"  First action: {actions[0].tolist()}")

    save_frame_grid_if_present(result, frame_grid_output)
    if frame_grid_output and result.get("frame_grid_image"):
        print(f"  Saved frame grid to: {frame_grid_output}")
    return True


def benchmark_inference(client: DobotWebSocketVLMClient, obs: Dict[str, Any], num_requests: int) -> bool:
    times = []
    for idx in range(num_requests):
        start_time = time.time()
        client.infer(obs)
        elapsed_ms = (time.time() - start_time) * 1000.0
        times.append(elapsed_ms)
        print(f"  Request {idx + 1}/{num_requests}: {elapsed_ms:.2f}ms")

    if not times:
        return False
    avg_time = float(np.mean(times))
    print("Benchmark results")
    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {float(np.min(times)):.2f}ms")
    print(f"  Max time: {float(np.max(times)):.2f}ms")
    print(f"  Std deviation: {float(np.std(times)):.2f}ms")
    print(f"  Throughput: {1000.0 / avg_time:.2f} requests/second")
    return True


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Dobot Motus WebSocket client")
    parser.add_argument("--host", default="localhost", help="Server host or ws:// URI")
    parser.add_argument("--port", type=int, default=6790, help="Server port")
    parser.add_argument(
        "--test",
        choices=["connectivity", "inference", "benchmark", "all"],
        default="inference",
        help="Client action to run",
    )
    parser.add_argument("--instruction", type=str, default="cook vegetable", help="Task instruction")
    parser.add_argument("--image", type=str, default=None, help="Single image file sent as images[0]")
    parser.add_argument("--image_path", type=str, default=None, help="Server-local image path, mainly for debugging")
    parser.add_argument("--top_image", type=str, default=None, help="Top camera image")
    parser.add_argument("--left_wrist_image", type=str, default=None, help="Left wrist camera image")
    parser.add_argument("--right_wrist_image", type=str, default=None, help="Right wrist camera image")
    parser.add_argument("--images", nargs="+", default=None, help="Ordered image paths: top left_wrist right_wrist")
    parser.add_argument("--use_random_images", action="store_true", help="Send three random uint8 images")
    parser.add_argument("--state_csv", type=str, default=None, help="Comma-separated state vector")
    parser.add_argument("--state_json", type=str, default=None, help="Path to a JSON state vector")
    parser.add_argument("--t5_embeddings_path", type=str, default=None, help="T5 embedding .pt path")
    parser.add_argument(
        "--send_t5_embeddings",
        action="store_true",
        help="Send the T5 embedding tensor bytes instead of a server-local path",
    )
    parser.add_argument("--t5_embeddings_dir", type=str, default=None, help="Directory for server-side auto T5 lookup")
    parser.add_argument(
        "--disable_auto_find_t5_embeddings",
        action="store_true",
        help="Disable server-side automatic T5 lookup by instruction",
    )
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Override inference step count")
    parser.add_argument("--return_frame_grid", action="store_true", help="Ask server to return a PNG frame grid")
    parser.add_argument("--frame_grid_output", type=str, default=None, help="Where to save returned frame grid PNG")
    parser.add_argument("--benchmark_requests", type=int, default=10, help="Number of benchmark requests")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = build_argparser().parse_args()

    client = DobotWebSocketVLMClient(host=args.host, port=args.port)
    obs = build_observation(args)

    tests_passed = 0
    total_tests = 0

    try:
        if args.test in ("connectivity", "all"):
            total_tests += 1
            if test_connectivity(client):
                tests_passed += 1

        if args.test in ("inference", "all"):
            total_tests += 1
            if test_inference(client, obs, args.frame_grid_output):
                tests_passed += 1

        if args.test in ("benchmark", "all"):
            total_tests += 1
            if benchmark_inference(client, obs, args.benchmark_requests):
                tests_passed += 1
    finally:
        client.close()

    print(f"Tests passed: {tests_passed}/{total_tests}")
    return 0 if tests_passed == total_tests else 1


if __name__ == "__main__":
    raise SystemExit(main())
