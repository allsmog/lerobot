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
import os
import re
import struct
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lerobot.utils.trinity_residency_shadow import ParameterShadow

import numpy as np

REQUEST_ARRAYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)


def pack(header: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
    """Encode a header and named arrays in the wire format."""
    entries, blobs, offset = [], [], 0
    for name, value in arrays.items():
        data = np.ascontiguousarray(value)
        raw = data.tobytes()
        entries.append(
            {
                "name": name,
                "dtype": data.dtype.str,
                "shape": list(data.shape),
                "offset": offset,
                "nbytes": len(raw),
            }
        )
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
        arrays[entry["name"]] = (
            np.frombuffer(body[start:end], dtype=np.dtype(entry["dtype"])).reshape(entry["shape"]).copy()
        )
    return header, arrays


class LiberoChunkPolicy:
    """One pretrained policy with its saved processors and the LIBERO environment processors."""

    def __init__(
        self,
        policy_path: str,
        device: str = "cuda",
        task_suite: str = "libero_object",
        dtype: str | None = None,
        rename_map: dict[str, str] | None = None,
        overrides: dict[str, Any] | None = None,
        *,
        allow_residency: bool = False,
    ) -> None:
        import torch

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.trinity_gate import install_prefix_boundaries

        self.torch = torch
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = Path(policy_path)
        cfg.device = device
        if dtype is not None and hasattr(cfg, "dtype"):
            cfg.dtype = dtype
        for key, value in (overrides or {}).items():  # as lerobot-eval --policy.<key>=<value>
            if not hasattr(cfg, key):
                raise ValueError(f"policy config has no field {key!r}")
            setattr(cfg, key, value)
        self.residency_enabled = allow_residency
        self.residency_state = "resident"
        self.residency_epoch = 0
        self._residency_shadow: ParameterShadow | None = None
        if allow_residency:
            if getattr(cfg, "type", None) != "pi05":
                raise ValueError("policy residency currently supports the qualified pi05 path only")
            if torch.device(cfg.device).type != "cuda":
                raise ValueError("policy residency requires CUDA")
            if any(
                getattr(cfg, key, False)
                for key in ("compile_model", "use_visual_memory", "use_proprioceptive_memory")
            ):
                raise ValueError("policy residency requires compilation and temporal memory disabled")
        self.env_cfg = LiberoEnv(task=task_suite)
        rename_map = dict(rename_map or {})
        self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg, rename_map=rename_map)
        self.policy.eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=policy_path,
            preprocessor_overrides={
                "device_processor": {"device": str(self.policy.config.device)},
                "rename_observations_processor": {"rename_map": rename_map},
            },
        )
        self.env_preprocessor, self.env_postprocessor = make_env_pre_post_processors(
            env_cfg=self.env_cfg, policy_cfg=cfg
        )
        self.device = self.policy.config.device
        self.prefix_hooks = install_prefix_boundaries(self.policy, self.device)

    def prepare_residency_shadow(self, host_reserve_bytes: int) -> dict[str, Any]:
        """Retain CPU parameter copies once, under the same inference lock.

        Qualification currently covers the inference-only pi0.5 path. Mutable
        buffers keep the normal migration path; changed parameter versions are
        refreshed before parking. Reserve admission precedes CPU allocation.
        """
        if not self.residency_enabled or self.residency_state != "resident":
            raise RuntimeError("prepare the parameter cache only on an idle resident policy")
        if getattr(self.policy.config, "type", None) != "pi05":
            raise ValueError("parameter-cache residency currently supports pi05 only")
        if self._residency_shadow is None:
            from lerobot.utils.trinity_residency_shadow import ParameterShadow

            try:
                self._residency_shadow = ParameterShadow(
                    self.policy, self.torch, self.device, host_reserve_bytes
                )
            except BaseException:
                self.residency_state = "fault"
                raise
        return self._residency_shadow.metadata()

    def residency(self, action: str) -> dict[str, Any]:
        """Move an idle policy under the same lock used for inference.

        The HTTP handler owns serialization. A transfer error leaves a faulted
        policy which refuses inference. Values, processors and RNG are preserved;
        this operation never resets inference state or substitutes CPU inference.
        """
        if not self.residency_enabled:
            raise ValueError("policy residency is disabled")
        if action not in ("status", "park", "resume"):
            raise ValueError("residency action must be status, park or resume")
        target = {"park": "parked", "resume": "resident"}.get(action)
        elapsed = 0.0
        if target is not None and target != self.residency_state:
            if self.residency_state == "fault":
                raise RuntimeError("policy residency is faulted")
            started = time.monotonic()
            self.residency_epoch += 1
            self.residency_state = "transferring"
            try:
                self.torch.cuda.synchronize(self.device)
                shadow = getattr(self, "_residency_shadow", None)
                if shadow is not None and target == "parked":
                    shadow.park()
                elif shadow is not None:
                    self.policy.to(self.device, non_blocking=True)
                else:
                    self.policy.to("cpu" if target == "parked" else self.device)
                if target == "parked":
                    self.torch.cuda.empty_cache()
                self.torch.cuda.synchronize(self.device)
                if shadow is not None and target == "resident":
                    shadow.resumed()
                self.residency_state = target
            except BaseException:
                self.residency_state = "fault"
                raise
            elapsed = time.monotonic() - started
        free, total = self.torch.cuda.mem_get_info(self.device)
        result = {
            "state": self.residency_state,
            "epoch": self.residency_epoch,
            "device": str(self.device),
            "transition_s": elapsed,
            "memory": {
                "free_bytes": int(free),
                "total_bytes": int(total),
                "allocated_bytes": int(self.torch.cuda.memory_allocated(self.device)),
                "reserved_bytes": int(self.torch.cuda.memory_reserved(self.device)),
            },
        }
        shadow = getattr(self, "_residency_shadow", None)
        if shadow is not None:
            result["parameter_shadow"] = shadow.metadata()
        return result

    def chunk(self, task: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
        if self.residency_state != "resident":
            raise RuntimeError("policy is not resident: " + self.residency_state)
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION
        from lerobot.utils.trinity_gate import boundary

        torch = self.torch
        missing = [name for name in REQUEST_ARRAYS if name not in arrays]
        if missing:
            raise ValueError(f"request lacks {missing}")
        observation: dict[str, Any] = {
            "pixels": {
                "image": arrays["agentview_image"][None],
                "image2": arrays["robot0_eye_in_hand_image"][None],
            },
            "robot_state": {
                "eef": {
                    "pos": arrays["robot0_eef_pos"][None],
                    "quat": arrays["robot0_eef_quat"][None],
                    "mat": np.eye(3)[None],
                },
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
            chunk: Any = self.policy.predict_action_chunk(observation)
        action_count: Any = getattr(self.policy.config, "n_action_steps", chunk.shape[1])
        steps = int(action_count)
        rows = []
        for t in range(min(steps, chunk.shape[1])):
            action = self.postprocessor(chunk[:, t])
            action = self.env_postprocessor({ACTION: action})[ACTION]
            rows.append(action.to("cpu").float().numpy().reshape(-1))
        out = np.stack(rows).astype(np.float32)
        if out.ndim != 2 or out.shape[1] < 7 or not np.isfinite(out).all():
            raise ValueError(f"policy returned an invalid chunk of shape {out.shape}")
        return out[:, :7]


def checked_owner(owner: str | None) -> str:
    if not isinstance(owner, str) or re.fullmatch(r"[0-9a-f]{32}", owner) is None:
        raise ValueError("residency requires a 32-character lowercase hex owner nonce")
    return owner


def make_handler(
    model: LiberoChunkPolicy, lock: threading.Lock, owner: str | None = None
) -> type[http.server.BaseHTTPRequestHandler]:
    owned = model.residency_enabled
    if owned:
        owner = checked_owner(owner)

    class Handler(http.server.BaseHTTPRequestHandler):
        response_generation: int | None = None

        def log_message(self, *args: Any) -> None:
            return

        def identity(self) -> None:
            if owned:
                if self.headers.get("X-Trinity-Owner") != owner:
                    raise RuntimeError("request belongs to another server owner")
                self.response_generation = model.residency_epoch + 1
                if self.headers.get("X-Trinity-Generation") != str(self.response_generation):
                    raise RuntimeError("request residency generation is stale")

        def reply(self, body: bytes, code: int, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if owned and self.response_generation is not None:
                assert owner is not None, "owned handler validated its server nonce"
                self.send_header("X-Trinity-Owner", owner)
                self.send_header("X-Trinity-Generation", str(self.response_generation))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            self.response_generation = None
            if self.path == "/trinity/residency":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < length <= 65536:
                        raise ValueError("residency command length must be 1 through 65536 bytes")
                    command = json.loads(self.rfile.read(length))
                    if not isinstance(command, dict):
                        raise ValueError("residency command must be an object")
                    with lock:
                        self.identity()
                        if command["action"] == "prepare_shadow":
                            result = model.prepare_residency_shadow(command["host_reserve_bytes"])
                        else:
                            result = model.residency(command["action"])
                        generation = model.residency_epoch + 1
                        self.response_generation = generation if owned else None
                        receipt = {"ok": True, "residency": result}
                        if owned:
                            receipt.update(
                                owner_nonce=owner,
                                generation=generation,
                                state=result.get("state", model.residency_state),
                            )
                    body = json.dumps(receipt, allow_nan=False).encode()
                    code = 200
                except Exception as error:
                    body = json.dumps(
                        {"ok": False, "error": f"{type(error).__name__}: {error}"[:500]}
                    ).encode()
                    code = 409
                self.reply(body, code, "application/json")
                return
            header: dict[str, Any] = {}
            try:
                payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                header, arrays = unpack(payload)
                with lock:
                    self.identity()
                    actions = model.chunk(str(header.get("task", "")), arrays)
                    if owned:
                        self.response_generation = model.residency_epoch + 1
                body = pack(
                    {
                        "request_id": header.get("request_id"),
                        "captured_ns": header.get("captured_ns"),
                        "error": None,
                    },
                    {"actions": actions},
                )
                code = 200
            except Exception as error:
                if not owned:
                    traceback.print_exc()
                body = pack(
                    {
                        "request_id": header.get("request_id"),
                        "captured_ns": header.get("captured_ns"),
                        "error": f"{type(error).__name__}: {error}"[:500],
                    },
                    {},
                )
                code = 409 if owned else 200
            self.reply(body, code, "application/octet-stream")

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-suite", default="libero_object")
    parser.add_argument("--dtype", default=None, help="policy dtype override, for example bfloat16")
    parser.add_argument(
        "--rename-map", default="{}", help="JSON map of observation keys, as lerobot-eval --rename_map"
    )
    parser.add_argument(
        "--policy-overrides", default="{}", help="JSON policy config fields, as lerobot-eval --policy.<field>"
    )
    parser.add_argument("--residency", action="store_true", help="allow request-safe CPU/GPU park and resume")
    options = parser.parse_args(argv)
    owner = checked_owner(os.environ.get("TRINITY_LEROBOT_OWNER_NONCE")) if options.residency else None
    model = LiberoChunkPolicy(
        options.policy_path,
        options.device,
        options.task_suite,
        options.dtype,
        json.loads(options.rename_map),
        json.loads(options.policy_overrides),
        allow_residency=options.residency,
    )
    server = http.server.ThreadingHTTPServer(
        (options.host, options.port), make_handler(model, threading.Lock(), owner)
    )
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
