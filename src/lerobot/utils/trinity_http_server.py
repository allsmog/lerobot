# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Serve one pretrained LIBERO policy to an external scheduler: one HTTP POST per action chunk.

An external GPU scheduler (Trinity) owns the control loop. It sends the raw LIBERO observation (both camera frames as
captured, end-effector position and quaternion, gripper joint positions) and the task text. This server runs the same
steps ``lerobot-eval`` runs for LIBERO (``preprocess_observation``, the LIBERO environment processor, the policy's
saved preprocessor), predicts one chunk, and applies the saved postprocessor and the environment postprocessor to every
action. The reply is the chunk as float32 rows of 7 values, ready for the LIBERO environment.

Hold points for the scheduler are the same as the remote-inference server's: one before any model work, the prefix
hooks and the per-denoising-step hook. All are no-ops unless ``TSG_CB`` is set (see :mod:`lerobot.utils.trinity_gate`).

Wire format, both directions: a 4-byte little-endian header length, a UTF-8 JSON header, then the raw array bytes the
header describes (``{"name", "dtype", "shape", "offset", "nbytes"}``). The request header carries ``request_id``,
``task`` and ``captured_ns``; the reply echoes ``request_id`` and ``captured_ns`` and adds ``error`` (null on success).

    python -m lerobot.utils.trinity_http_server --policy-path lerobot/smolvla_libero --port 7460
"""

from __future__ import annotations

import argparse
import http.server
import json
import struct
import sys
import threading
import traceback
from typing import Any

import numpy as np

REQUEST_ARRAYS = ("agentview_image", "robot0_eye_in_hand_image", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos")


def pack(header: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    """Encode a header and named arrays in the wire format."""
    entries, blobs, offset = [], [], 0
    for name, value in arrays.items():
        data = np.ascontiguousarray(value)
        raw = data.tobytes()
        entries.append({"name": name, "dtype": data.dtype.str, "shape": list(data.shape), "offset": offset, "nbytes": len(raw)})
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps({**header, "arrays": entries}).encode()
    return struct.pack("<I", len(head)) + head + b"".join(blobs)


def unpack(payload: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Decode the wire format into its header and arrays (copies, safe to keep)."""
    if len(payload) < 4:
        raise ValueError("payload shorter than its header length")
    (size,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4 : 4 + size].decode())
    body = payload[4 + size :]
    arrays = {}
    for entry in header.pop("arrays", []):
        start, end = entry["offset"], entry["offset"] + entry["nbytes"]
        if end > len(body):
            raise ValueError(f"array {entry['name']!r} runs past the payload")
        arrays[entry["name"]] = np.frombuffer(body[start:end], dtype=np.dtype(entry["dtype"])).reshape(entry["shape"]).copy()
    return header, arrays


class LiberoChunkPolicy:
    """One pretrained policy with its saved processors and the LIBERO environment processors."""

    def __init__(self, policy_path: str, device: str = "cuda", task_suite: str = "libero_object", dtype: str | None = None,
                 rename_map: dict[str, str] | None = None, overrides: dict[str, Any] | None = None):
        import torch

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.trinity_gate import install_prefix_boundaries

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = policy_path
        cfg.device = device
        if dtype is not None and hasattr(cfg, "dtype"):
            cfg.dtype = dtype
        for key, value in (overrides or {}).items():   # as lerobot-eval --policy.<key>=<value>
            if not hasattr(cfg, key):
                raise ValueError(f"policy config has no field {key!r}")
            setattr(cfg, key, value)
        self.env_cfg = LiberoEnv(task=task_suite)
        rename_map = dict(rename_map or {})
        self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg, rename_map=rename_map)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": str(self.policy.config.device)},
                                    "rename_observations_processor": {"rename_map": rename_map}})
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(env_cfg=self.env_cfg, policy_cfg=cfg)
        self.device = self.policy.config.device
        self.prefix_hooks = install_prefix_boundaries(self.policy, self.device)

    def chunk(self, task: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION
        from lerobot.utils.trinity_gate import boundary

        torch = self.torch
        missing = [name for name in REQUEST_ARRAYS if name not in arrays]
        if missing:
            raise ValueError(f"request lacks {missing}")
        observation = {
            "pixels": {"image": arrays["agentview_image"][None], "image2": arrays["robot0_eye_in_hand_image"][None]},
            "robot_state": {
                "eef": {"pos": arrays["robot0_eef_pos"][None], "quat": arrays["robot0_eef_quat"][None], "mat": np.eye(3)[None]},
                "gripper": {"qpos": arrays["robot0_gripper_qpos"][None], "qvel": np.zeros((1, 2))},
                "joints": {"pos": np.zeros((1, 7)), "vel": np.zeros((1, 7))},
            },
        }
        boundary(self.device)
        observation = preprocess_observation(observation)
        observation["task"] = [task]
        observation = self.env_preprocessor(observation)
        observation = self.preprocessor(observation)
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(observation)
        steps = int(getattr(self.policy.config, "n_action_steps", chunk.shape[1]))
        rows = []
        for t in range(min(steps, chunk.shape[1])):
            action = self.postprocessor(chunk[:, t])
            action = self.env_postprocessor({ACTION: action})[ACTION]
            rows.append(action.to("cpu").float().numpy().reshape(-1))
        out = np.stack(rows).astype(np.float32)
        if out.ndim != 2 or out.shape[1] < 7 or not np.isfinite(out).all():
            raise ValueError(f"policy returned an invalid chunk of shape {out.shape}")
        return out[:, :7]


def make_handler(model: LiberoChunkPolicy, lock: threading.Lock):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # one line per request is noise at 10 Hz
            return

        def do_POST(self) -> None:
            header: dict[str, Any] = {}
            try:
                payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                header, arrays = unpack(payload)
                with lock:
                    actions = model.chunk(str(header.get("task", "")), arrays)
                body = pack({"request_id": header.get("request_id"), "captured_ns": header.get("captured_ns"), "error": None},
                            {"actions": actions})
            except Exception as error:  # noqa: BLE001 - the error travels to the caller
                traceback.print_exc()
                body = pack({"request_id": header.get("request_id"), "captured_ns": header.get("captured_ns"),
                             "error": f"{type(error).__name__}: {error}"[:500]}, {})
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--dtype", default=None, help="policy dtype override, for example bfloat16")
    parser.add_argument("--rename-map", default="{}", help="JSON map of observation keys, as lerobot-eval --rename_map")
    parser.add_argument("--policy-overrides", default="{}", help='JSON policy config fields, as lerobot-eval --policy.<field>')
    options = parser.parse_args(argv)
    model = LiberoChunkPolicy(options.policy_path, options.device, options.task_suite, options.dtype,
                              json.loads(options.rename_map), json.loads(options.policy_overrides))
    server = http.server.ThreadingHTTPServer((options.host, options.port), make_handler(model, threading.Lock()))
    print(f"prefix hold points: {model.prefix_hooks}", flush=True)
    print("ready.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
