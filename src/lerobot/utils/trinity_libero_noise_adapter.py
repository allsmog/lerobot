# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Standalone NumPy adapter for controlled LIBERO noise and lossless provenance.

Load this file as a scheduler adapter. It needs no installed LeRobot or Torch.
The caller supplies semantic capture context and closes the episode outside the
control interval, after outstanding workers have stopped. This adapter never
relabels request count or wall time as capture identity.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
from pathlib import Path
from typing import Any

import numpy as np

CONTEXT_SCHEMA = "trinity/policy-noise-context@1"
FIELD_ALGORITHM = "sha256-canonical-json/numpy-philox-normal-f32@1"
RAW_KEYS = (
    "agentview_image",
    "robot0_eye_in_hand_image",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
)
CONTEXT_KEYS = {"schema", "seed", "task_id", "state_id", "episode", "phase", "tick", "model"}


def array_hash(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_context(value: Any) -> dict[str, Any]:
    """Reject incomplete contexts and accidental arm/time/request-based keys."""
    if not isinstance(value, dict) or set(value) != CONTEXT_KEYS:
        raise ValueError("controlled noise requires the exact semantic capture context")
    if value["schema"] != CONTEXT_SCHEMA or value["model"] != "pi05":
        raise ValueError("unsupported policy noise context schema or model")
    for key in ("seed", "task_id", "state_id", "episode", "tick"):
        n = value[key]
        if isinstance(n, bool) or not isinstance(n, int) or n > 2**63 - 1:
            raise ValueError(f"invalid context {key}")
        if n < (-1 if key == "tick" else 0):
            raise ValueError(f"invalid context {key}")
    phase, tick = value["phase"], value["tick"]
    if phase not in ("initialization", "prefill", "control"):
        raise ValueError("invalid capture phase")
    if (phase == "prefill" and tick != -1) or (phase != "prefill" and tick < 0):
        raise ValueError("capture tick does not match its phase")
    return dict(value)


def policy_noise(context: dict[str, Any], chunk: int, width: int) -> np.ndarray:
    """Evaluate one exogenous field point without consuming a shared RNG."""
    key = {"algorithm": FIELD_ALGORITHM, "context": canonical_context(context)}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True, separators=(",", ":")).encode()).digest()
    generator = np.random.Generator(np.random.Philox(int.from_bytes(digest, "little")))
    return generator.standard_normal((1, chunk, width), dtype=np.float32)


def pack(header: dict[str, Any], arrays: dict[str, np.ndarray]) -> bytes:
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
    if len(payload) < 4:
        raise ValueError("reply is shorter than its header length")
    (size,) = struct.unpack("<I", payload[:4])
    if size > len(payload) - 4:
        raise ValueError("reply header runs past its payload")
    header = json.loads(payload[4 : 4 + size].decode())
    body = payload[4 + size :]
    arrays = {}
    for entry in header.pop("arrays", []):
        start, count = entry["offset"], entry["nbytes"]
        if start < 0 or count < 0 or start + count > len(body) or entry["name"] in arrays:
            raise ValueError("invalid or duplicate reply array")
        dtype = np.dtype(entry["dtype"])
        if dtype.hasobject:
            raise ValueError("object reply arrays are unsupported")
        arrays[entry["name"]] = np.frombuffer(body[start : start + count], dtype=dtype).reshape(
            entry["shape"]
        )
    return header, arrays


def observation_hash(task: str, arrays: dict[str, np.ndarray]) -> str:
    values = [(name, np.ascontiguousarray(arrays[name])) for name in RAW_KEYS]
    header = {
        "task": task,
        "arrays": [
            {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)} for name, value in values
        ],
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for _, value in values:
        digest.update(value.tobytes())
    return digest.hexdigest()


class ControlledLeRobotLiberoAdapter:
    """Verify the exact noise/observation/full-output chain for every reply."""

    supports_semantic_noise = True

    def __init__(self, options: dict[str, Any]):
        self.chunk = self._positive(options.get("chunk_size", 50), "chunk_size")
        self.width = self._positive(options.get("max_action_dim", 32), "max_action_dim")
        self.steps = self._positive(options.get("n_action_steps", 50), "n_action_steps")
        self.max_records = self._positive(options.get("max_records", 256), "max_records")
        self.max_bytes = self._positive(options.get("max_bytes", 64 * 1024**2), "max_bytes")
        if self.steps > self.chunk or self.width < 7:
            raise ValueError("incompatible policy chunk dimensions")
        self.task = ""
        self._records: dict[int, dict[str, Any]] = {}
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _positive(value: Any, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def episode(self, prompt: str) -> None:
        with self._lock:
            if self._records:
                raise RuntimeError("close the previous episode before changing its task")
            self.task = str(prompt)

    def encode(self, obs: dict[str, Any], request_id: int) -> bytes:
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 1:
            raise ValueError("controlled request ID must be a positive integer")
        context = canonical_context(obs.get("policy_noise_context"))
        raw = obs.get("raw")
        if raw is None:
            raise ValueError("controlled adapter needs raw LIBERO observations")
        noise = policy_noise(context, self.chunk, self.width)
        arrays = {key: raw[key] for key in RAW_KEYS}
        task = self.task or str(obs.get("prompt", ""))
        captured = obs.get("captured_ns")
        if isinstance(captured, bool) or not isinstance(captured, int) or captured < 0:
            raise ValueError("controlled adapter needs an actual capture timestamp")
        wire = pack(
            {
                "request_id": int(request_id),
                "task": task,
                "captured_ns": captured,
                "policy_noise_context": context,
            },
            {**arrays, "policy_noise": noise},
        )
        with self._lock:
            if request_id in self._records:
                raise ValueError("duplicate controlled request ID")
            if len(self._records) >= self.max_records or self._bytes + len(wire) > self.max_bytes:
                raise RuntimeError("controlled provenance allocation exhausted")
            self._records[request_id] = {
                "request_id": request_id,
                "context": context,
                "captured_ns": captured,
                "noise_sha256": array_hash(noise),
                "observation_sha256": observation_hash(task, arrays),
                "request": wire,
                "reply": None,
            }
            self._bytes += len(wire)
        return wire

    def decode(self, reply: bytes, request_id: int) -> np.ndarray:
        with self._lock:
            record = self._records.get(request_id)
            if record is None or record["reply"] is not None:
                raise ValueError("unknown or duplicate controlled reply")
            if self._bytes + len(reply) > self.max_bytes:
                raise RuntimeError("controlled provenance allocation exhausted")
            # Preserve even a rejected reply before checking its claims.
            record["reply"] = bytes(reply)
            self._bytes += len(reply)
            header, arrays = unpack(reply)
            if header.get("error"):
                raise ValueError(f"policy server: {header['error']}")
            if header.get("request_id") != request_id or header.get("captured_ns") != record["captured_ns"]:
                raise ValueError("reply does not match the actual request capture")
            metadata = header.get("policy_provenance", {})
            predicted, actions = arrays.get("predicted_actions"), arrays.get("actions")
            for value, shape in [(predicted, (self.chunk, self.width)), (actions, (self.steps, 7))]:
                if (
                    value is None
                    or value.dtype != np.float32
                    or value.shape != shape
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("invalid controlled prediction or environment actions")
            assert predicted is not None and actions is not None
            expected = {
                "schema": "trinity/policy-noise-provenance@1",
                "noise_shape": [1, self.chunk, self.width],
                "noise_dtype": np.dtype(np.float32).str,
                "noise_sha256": record["noise_sha256"],
                "raw_observation_sha256": record["observation_sha256"],
                "predicted_shape": [self.chunk, self.width],
                "prediction_space": "policy_before_postprocessing",
                "predicted_sha256": array_hash(predicted),
                "actions_shape": [self.steps, 7],
                "actions_space": "libero_environment_after_postprocessing",
                "actions_sha256": array_hash(actions),
            }
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError("controlled reply provenance does not match its request and arrays")
            config = metadata.get("policy_config", {})
            if any(
                config.get(key) != value
                for key, value in {
                    "type": "pi05",
                    "chunk_size": self.chunk,
                    "max_action_dim": self.width,
                    "n_action_steps": self.steps,
                }.items()
            ):
                raise ValueError("reply policy configuration changed")
            record["verified"] = True
            return np.array(actions, copy=True)

    def close_episode(self, output: Path) -> None:
        """Flush bounded lossless evidence after the caller has stopped workers."""
        with self._lock:
            directory = Path(output) / "policy-provenance"
            directory.mkdir(exist_ok=False)
            rows = []
            for request_id, record in self._records.items():
                row = {key: value for key, value in record.items() if key not in ("request", "reply")}
                for kind in ("request", "reply"):
                    wire = record[kind]
                    if wire is not None:
                        name = f"{request_id}.{kind}.bin"
                        (directory / name).write_bytes(wire)
                        row[kind] = {
                            "path": name,
                            "sha256": hashlib.sha256(wire).hexdigest(),
                            "bytes": len(wire),
                        }
                rows.append(row)
            (directory / "manifest.json").write_text(
                json.dumps(
                    {
                        "algorithm": FIELD_ALGORITHM,
                        "numpy_version": np.__version__,
                        "bytes": self._bytes,
                        "records": rows,
                    },
                    indent=2,
                )
                + "\n"
            )
            self._records.clear()
            self._bytes = 0
