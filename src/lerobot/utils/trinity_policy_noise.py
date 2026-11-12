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

"""Validate explicit policy noise and fingerprint actual HTTP input content."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

NOISE_ARRAY = "policy_noise"
SCHEMA = "trinity/policy-noise-provenance@1"


def checked_noise(config: Any, arrays: dict[str, np.ndarray]) -> np.ndarray | None:
    """Accept only the qualified stateless pi0.5 tensor interface; leave defaults alone."""
    if NOISE_ARRAY not in arrays:
        return None
    if getattr(config, "type", None) != "pi05":
        raise ValueError("explicit policy noise currently requires pi05")
    if getattr(config, "use_visual_memory", False) or getattr(config, "use_proprioceptive_memory", False):
        raise ValueError("controlled HTTP noise requires stateless pi05 inference")
    chunk = getattr(config, "chunk_size", None)
    width = getattr(config, "max_action_dim", None)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (chunk, width)):
        raise ValueError("policy noise requires positive configured chunk and action dimensions")
    noise = arrays[NOISE_ARRAY]
    if not isinstance(noise, np.ndarray) or noise.dtype != np.dtype(np.float32):
        raise ValueError("policy_noise must be a float32 array")
    if noise.shape != (1, chunk, width) or not np.isfinite(noise).all():
        raise ValueError(f"policy_noise must be finite with shape {(1, chunk, width)}")
    return np.array(noise, dtype=np.float32, order="C", copy=True)


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def observation_sha256(task: str, arrays: dict[str, np.ndarray], names: tuple[str, ...]) -> str:
    """Bind raw model inputs; capture timestamps and policy noise are separate identities."""
    inputs = [(name, np.ascontiguousarray(arrays[name])) for name in names]
    header = {
        "task": task,
        "arrays": [
            {"name": name, "dtype": value.dtype.str, "shape": list(value.shape)} for name, value in inputs
        ],
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
    for _, value in inputs:
        digest.update(value.tobytes())
    return digest.hexdigest()
