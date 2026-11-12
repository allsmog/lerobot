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
    from lerobot.utils.trinity_vocab_placement import CpuVocabularyPlacement

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

    supports_policy_noise = True

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
        residency_retain_mib: int = 0,
        batch_action_postprocess: bool = False,
        residency_cpu_vocabulary: bool = False,
    ) -> None:
        if not isinstance(residency_cpu_vocabulary, bool):
            raise ValueError("CPU vocabulary placement must be a boolean")
        if residency_cpu_vocabulary and not allow_residency:
            raise ValueError("CPU vocabulary placement requires owned residency")
        self.residency_cpu_vocabulary = residency_cpu_vocabulary
        self._cpu_vocabulary: CpuVocabularyPlacement | None = None
        if not isinstance(batch_action_postprocess, bool):
            raise ValueError("batch action postprocessing must be a boolean")
        self.batch_action_postprocess = batch_action_postprocess
        if (
            isinstance(residency_retain_mib, bool)
            or not isinstance(residency_retain_mib, int)
            or residency_retain_mib < 0
        ):
            raise ValueError("residency retention must be a nonnegative integer MiB count")
        if residency_retain_mib and not allow_residency:
            raise ValueError("residency retention requires owned residency")
        self.residency_retain_bytes = residency_retain_mib * 1024**2
        import torch

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.utils.trinity_gate import install_prefix_boundaries

        self.torch = torch
        self.last_inference_metadata: dict[str, Any] | None = None
        self.last_policy_chunk: np.ndarray | None = None
        cfg = PreTrainedConfig.from_pretrained(policy_path)
        cfg.pretrained_path = Path(policy_path)
        cfg.device = device
        if dtype is not None and hasattr(cfg, "dtype"):
            precision = getattr(torch, dtype, None)
            if not isinstance(precision, torch.dtype):
                raise ValueError(f"invalid policy dtype: {dtype!r}")
            cfg.dtype = precision
        for key, value in (overrides or {}).items():  # as lerobot-eval --policy.<key>=<value>
            if not hasattr(cfg, key):
                raise ValueError(f"policy config has no field {key!r}")
            setattr(cfg, key, value)
        if batch_action_postprocess and (
            getattr(cfg, "type", None) != "pi05" or getattr(cfg, "use_relative_actions", False)
        ):
            raise ValueError("batch action postprocessing requires stateless pi05 actions")
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
        if self.batch_action_postprocess:
            from lerobot.utils.trinity_action_postprocess import check_batch_postprocess

            check_batch_postprocess(cfg, self.postprocessor, self.env_postprocessor)
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
                shadow_options: dict[str, Any] = {}
                if getattr(self, "residency_retain_bytes", 0):
                    shadow_options["park_retain_bytes"] = self.residency_retain_bytes
                if getattr(self, "residency_cpu_vocabulary", False):
                    from lerobot.utils.trinity_vocab_placement import prepare_pi05_cpu_vocabulary

                    self._cpu_vocabulary = prepare_pi05_cpu_vocabulary(
                        self.policy, self.torch, self.device, host_reserve_bytes, self._vocabulary_fault
                    )
                    shadow_options["resident_cpu_names"] = self._cpu_vocabulary.cpu_names
                self._residency_shadow = ParameterShadow(
                    self.policy,
                    self.torch,
                    self.device,
                    host_reserve_bytes,
                    **shadow_options,
                )
            except BaseException:
                self.residency_state = "fault"
                raise
        return self._residency_shadow.metadata()

    def _vocabulary_fault(self) -> None:
        self.residency_state = "fault"

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
                if getattr(self, "residency_cpu_vocabulary", False) and shadow is None:
                    raise RuntimeError(
                        "prepare the owned CPU vocabulary shadow before a residency transition"
                    )
                if shadow is not None and target == "parked":
                    shadow.park()
                elif shadow is not None:
                    if getattr(self, "residency_cpu_vocabulary", False):
                        shadow.restore()
                    else:
                        self.policy.to(self.device, non_blocking=True)
                else:
                    self.policy.to("cpu" if target == "parked" else self.device)
                if target == "parked":
                    self.torch.cuda.empty_cache()
                self.torch.cuda.synchronize(self.device)
                if shadow is not None and target == "resident":
                    shadow.resumed()
                placement = getattr(self, "_cpu_vocabulary", None)
                if placement is not None:
                    placement.check(parked=target == "parked")
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
        if getattr(self, "residency_cpu_vocabulary", False):
            placement = getattr(self, "_cpu_vocabulary", None)
            result["cpu_vocabulary"] = {"prepared": False} if placement is None else placement.metadata()
        return result

    def chunk(self, task: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
        if self.residency_state != "resident":
            raise RuntimeError("policy is not resident: " + self.residency_state)
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION
        from lerobot.utils.trinity_gate import boundary
        from lerobot.utils.trinity_policy_noise import (
            SCHEMA,
            array_sha256,
            checked_noise,
            observation_sha256,
        )

        torch = self.torch
        self.last_inference_metadata = None
        self.last_policy_chunk = None
        placement = getattr(self, "_cpu_vocabulary", None)
        if placement is not None:
            placement.check()
        if getattr(self, "batch_action_postprocess", False):
            from lerobot.utils.trinity_action_postprocess import check_batch_postprocess

            check_batch_postprocess(self.policy.config, self.postprocessor, self.env_postprocessor)
        missing = [name for name in REQUEST_ARRAYS if name not in arrays]
        if missing:
            raise ValueError(f"request lacks {missing}")
        noise = checked_noise(self.policy.config, arrays)
        noise_fingerprint = input_fingerprint = None
        predict_options: dict[str, Any] = {}
        predicted = None
        observe_padded = (
            noise is not None and getattr(self.policy, "supports_action_chunk_observer", False) is True
        )
        if noise is not None:
            noise_fingerprint = array_sha256(noise)
            input_fingerprint = observation_sha256(task, arrays, REQUEST_ARRAYS)
            predict_options["noise"] = torch.from_numpy(noise).to(device=self.device, dtype=torch.float32)
            if observe_padded:

                def capture_prediction(value: Any) -> None:
                    nonlocal predicted
                    predicted = value.detach().to("cpu").float().numpy().copy()

                predict_options["action_chunk_observer"] = capture_prediction
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
            chunk: Any = self.policy.predict_action_chunk(observation, **predict_options)
        if noise is not None:
            if observe_padded and predicted is None:
                raise ValueError("policy did not expose its padded predictions")
            if predicted is None:
                predicted = chunk.detach().to("cpu").float().numpy().copy()
            if predicted.ndim != 3 or predicted.shape[0] != 1 or not np.isfinite(predicted).all():
                raise ValueError("policy returned invalid full predictions for controlled noise")
        action_count: Any = getattr(self.policy.config, "n_action_steps", chunk.shape[1])
        steps = int(action_count)
        if getattr(self, "batch_action_postprocess", False):
            count = min(steps, chunk.shape[1])
            if count <= 0 or chunk.ndim != 3 or chunk.shape[0] != 1:
                raise ValueError("batch action postprocessing requires a nonempty single-policy chunk")
            action = self.postprocessor(chunk[:, :count])
            action = self.env_postprocessor({ACTION: action})[ACTION]
            if action.ndim != 3 or action.shape[:2] != (1, count):
                raise ValueError("batch action postprocessing changed the chunk shape")
            out = action.to("cpu").float().numpy().reshape(count, -1).astype(np.float32)
        else:
            rows = []
            for t in range(min(steps, chunk.shape[1])):
                action = self.postprocessor(chunk[:, t])
                action = self.env_postprocessor({ACTION: action})[ACTION]
                rows.append(action.to("cpu").float().numpy().reshape(-1))
            out = np.stack(rows).astype(np.float32)
        if out.ndim != 2 or out.shape[1] < 7 or not np.isfinite(out).all():
            raise ValueError(f"policy returned an invalid chunk of shape {out.shape}")
        actions = out[:, :7]
        if noise is not None and predicted is not None:
            self.last_policy_chunk = np.ascontiguousarray(predicted[0])
            self.last_inference_metadata = {
                "schema": SCHEMA,
                "noise_shape": list(noise.shape),
                "noise_dtype": noise.dtype.str,
                "noise_sha256": noise_fingerprint,
                "raw_observation_sha256": input_fingerprint,
                "predicted_shape": list(self.last_policy_chunk.shape),
                "prediction_space": "policy_before_postprocessing",
                "predicted_sha256": array_sha256(self.last_policy_chunk),
                "actions_shape": list(actions.shape),
                "actions_space": "libero_environment_after_postprocessing",
                "actions_sha256": array_sha256(actions),
                "policy_config": {
                    "type": self.policy.config.type,
                    "chunk_size": self.policy.config.chunk_size,
                    "max_action_dim": self.policy.config.max_action_dim,
                    "n_action_steps": steps,
                    "dtype": str(getattr(self.policy.config, "dtype", None)),
                },
            }
        return actions


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
                    if getattr(model, "residency_cpu_vocabulary", False):
                        traceback.print_exc()
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
                provenance = predictions = None
                with lock:
                    self.identity()
                    if "policy_noise" in arrays and not getattr(model, "supports_policy_noise", False):
                        raise ValueError("policy does not support explicit noise")
                    actions = model.chunk(str(header.get("task", "")), arrays)
                    if "policy_noise" in arrays:
                        provenance = model.last_inference_metadata
                        predictions = model.last_policy_chunk
                        if provenance is None or predictions is None:
                            raise ValueError("policy did not return controlled-noise provenance")
                    if owned:
                        self.response_generation = model.residency_epoch + 1
                reply_arrays: dict[str, np.ndarray] = {"actions": actions}
                reply_header: dict[str, Any] = {
                    "request_id": header.get("request_id"),
                    "captured_ns": header.get("captured_ns"),
                    "error": None,
                }
                if provenance is not None and predictions is not None:
                    reply_header["policy_provenance"] = provenance
                    reply_arrays["predicted_actions"] = predictions
                body = pack(
                    reply_header,
                    reply_arrays,
                )
                code = 200
            except Exception as error:
                if not owned or getattr(model, "residency_cpu_vocabulary", False):
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
    parser.add_argument(
        "--residency-cpu-vocabulary",
        action="store_true",
        help="keep the qualified pi05 vocabulary tables on CPU; requires --residency",
    )
    parser.add_argument(
        "--batch-action-postprocess",
        action="store_true",
        help="batch the qualified stateless pi05 output processors and CPU copy",
    )
    parser.add_argument(
        "--residency-retain-mib",
        type=int,
        default=0,
        help="maximum parameter storage to retain while parked; requires --residency",
    )
    options = parser.parse_args(argv)
    if options.residency_retain_mib < 0 or (options.residency_retain_mib and not options.residency):
        parser.error("--residency-retain-mib needs a nonnegative value and --residency when nonzero")
    if options.residency_cpu_vocabulary and not options.residency:
        parser.error("--residency-cpu-vocabulary requires --residency")
    owner = checked_owner(os.environ.get("TRINITY_LEROBOT_OWNER_NONCE")) if options.residency else None
    model = LiberoChunkPolicy(
        options.policy_path,
        options.device,
        options.task_suite,
        options.dtype,
        json.loads(options.rename_map),
        json.loads(options.policy_overrides),
        allow_residency=options.residency,
        **({"residency_retain_mib": options.residency_retain_mib} if options.residency_retain_mib else {}),
        **({"batch_action_postprocess": True} if options.batch_action_postprocess else {}),
        **({"residency_cpu_vocabulary": True} if options.residency_cpu_vocabulary else {}),
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
