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
"""Optional hold points for an external GPU scheduler (Trinity segment gate).

A *boundary* is a point where this process has no queued GPU work. When the
``TSG_CB`` environment variable names a Trinity TSG1 control block, each boundary
synchronizes the device and lets the external scheduler hold this process there.

With ``TSG_CB`` unset (the default) :func:`boundary` returns immediately, imports
nothing and never synchronizes, so behavior is unchanged.

The gate implementation is not part of lerobot. ``TSG_GATE_IMPL`` names where it
lives: a package directory (the Literate AI generated ``lerobot-segment-gate``
component's ``lerobot_segment_gate`` package) or a single Python file. It must expose
``gate_boundary(synchronize=None)``, which builds one
process-wide gate from ``TSG_CB``, ``TSG_POLL_US`` and ``TSG_MAX_WAIT_MS`` and runs a
boundary, calling ``synchronize`` before it reads the hold flag.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

_lock = threading.Lock()
_impl: Any = None
_impl_key: tuple[str, str] | None = None


def gate_enabled() -> bool:
    """True when a control block is configured for this process."""
    return bool(os.environ.get("TSG_CB"))


def _device_synchronizer(device: torch.device | str | None) -> Callable[[], None]:
    dev = torch.device(device) if device is not None else None
    if dev is not None and dev.type == "cuda":
        return lambda: torch.cuda.synchronize(dev)
    if dev is not None and dev.type == "mps":
        return torch.mps.synchronize
    return lambda: None


_IMPL_MODULE = "lerobot_tsg_gate_impl"


def _load_impl(impl_path: str) -> Any:
    path = Path(impl_path)
    if path.is_dir():
        # A package: register it so its relative imports resolve.
        spec = importlib.util.spec_from_file_location(
            _IMPL_MODULE, path / "__init__.py", submodule_search_locations=[str(path)]
        )
    else:
        spec = importlib.util.spec_from_file_location(_IMPL_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"TSG_GATE_IMPL is not a loadable Python package or file: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_IMPL_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_IMPL_MODULE, None)
        raise
    if not callable(getattr(module, "gate_boundary", None)):
        raise ImportError(f"TSG_GATE_IMPL does not define gate_boundary(synchronize): {impl_path}")
    return module


def _current_impl() -> Any:
    """Load the gate implementation once per (control block, implementation) pair."""
    global _impl, _impl_key
    cb_path = os.environ.get("TSG_CB", "")
    impl_path = os.environ.get("TSG_GATE_IMPL", "")
    if not impl_path:
        raise RuntimeError("TSG_CB is set but TSG_GATE_IMPL does not name the gate implementation")
    key = (cb_path, impl_path)
    with _lock:
        if _impl is None or _impl_key != key:
            _impl = _load_impl(impl_path)
            _impl_key = key
        return _impl


def boundary(device: torch.device | str | None = None) -> dict | None:
    """Run one hold point. No-op unless ``TSG_CB`` is set.

    Synchronizes ``device`` first, so the external scheduler only ever holds this
    process when it has no queued GPU work.
    """
    if not gate_enabled():
        return None
    return _current_impl().gate_boundary(_device_synchronizer(device))


def reset_for_tests() -> None:
    """Drop the cached implementation so tests can change the environment."""
    global _impl, _impl_key
    with _lock:
        _impl = None
        _impl_key = None
        for name in [n for n in sys.modules if n == _IMPL_MODULE or n.startswith(_IMPL_MODULE + ".")]:
            del sys.modules[name]
