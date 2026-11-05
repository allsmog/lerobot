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
"""One control tick for one robot driven by several inference engines.

Each engine keeps lerobot's own per-engine sequence (permission check, pull,
interpolate, permission check) from :mod:`lerobot.inference.dispatch`. An external
*arbiter* (the Literate AI generated ``trinity.multi-engine-arbiter`` component) then
decides which engine's value reaches each motor feature. This module owns no policy
and no hardware: the caller supplies the engines, the clock and the ``send`` callable.

The arbiter must provide ``Arbiter(deployment)`` with ``tick(now_ns, offers)`` and
``summary()``, as described by its interface document.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.utils.action_interpolator import ActionInterpolator

from .base import InferenceEngine
from .dispatch import next_interpolated_action

_ARBITER_MODULE = "lerobot_multi_engine_arbiter_impl"


@dataclass
class EngineSlot:
    """One engine on the robot, with its own interpolator and action feature order."""

    engine_id: str
    engine: InferenceEngine
    interpolator: ActionInterpolator
    action_keys: list[str]
    produced_ns: int | None = None
    last_values: dict[str, float] | None = None


@dataclass
class TickResult:
    """What one tick decided and sent."""

    decision: dict[str, Any]
    sent: dict[str, float] | None
    contributing: list[str] = field(default_factory=list)


def load_arbiter(impl_path: str | Path, deployment: Mapping[str, Any]) -> Any:
    """Build an arbiter from a package directory or a single Python file."""
    path = Path(impl_path)
    for name in [n for n in sys.modules if n == _ARBITER_MODULE or n.startswith(_ARBITER_MODULE + ".")]:
        del sys.modules[name]
    if path.is_dir():
        spec = importlib.util.spec_from_file_location(
            _ARBITER_MODULE, path / "__init__.py", submodule_search_locations=[str(path)]
        )
    else:
        spec = importlib.util.spec_from_file_location(_ARBITER_MODULE, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"arbiter implementation is not a loadable Python package or file: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ARBITER_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_ARBITER_MODULE, None)
        raise
    if not callable(getattr(module, "Arbiter", None)):
        raise ImportError(f"arbiter implementation does not define Arbiter: {impl_path}")
    return module.Arbiter(dict(deployment))


def _offer(slot: EngineSlot, now_ns: int, obs_frame: Any) -> dict[str, Any] | None:
    """Offer this engine's newest values with the time its policy action arrived.

    With nothing new this tick, the last values are re-offered with their original
    time, so the arbiter can judge them fresh or stale and count deadline misses. An
    engine that has never produced an action offers nothing (``starved``).
    """
    denied = False

    def note_denied() -> None:
        nonlocal denied
        denied = True

    had_need = slot.interpolator.needs_new_action()
    target = next_interpolated_action(slot.engine, slot.interpolator, obs_frame, hold=note_denied)
    if denied:
        slot.produced_ns, slot.last_values = None, None
        return None
    if target is not None:
        if had_need:
            # After an exhausted buffer the interpolator only returns a target when a
            # new policy action was added on this tick.
            slot.produced_ns = now_ns
        if len(target) != len(slot.action_keys):
            raise ValueError(
                f"engine {slot.engine_id!r} produced {len(target)} values for {len(slot.action_keys)} action keys"
            )
        slot.last_values = {key: float(target[i].item()) for i, key in enumerate(slot.action_keys)}
    if slot.last_values is None or slot.produced_ns is None:
        return None
    return {"values": dict(slot.last_values), "produced_ns": slot.produced_ns}


def multi_engine_tick(
    slots: list[EngineSlot],
    arbiter: Any,
    now_ns: int,
    obs_frame: Any,
    send: Callable[[dict[str, float]], None],
    last_command: dict[str, float],
    all_features: list[str],
) -> TickResult:
    """Run one control tick. Never waits on inference.

    ``last_command`` holds the most recent value sent for each feature; a held feature
    repeats it. If a held feature has never been commanded, nothing is sent this tick.
    The arbiter must report each selected engine in ``dispatch_sources``. After a
    send, only engines whose attributed features survived the final gate record a
    dispatch. ``decision`` retains the arbiter proposal; ``sent`` and ``contributing``
    describe the result after the permission checks.
    """
    by_id = {slot.engine_id: slot for slot in slots}
    if len(by_id) != len(slots):
        raise ValueError("engine slots must have unique engine_id values")
    offers = {slot.engine_id: _offer(slot, now_ns, obs_frame) for slot in slots}
    decision = arbiter.tick(now_ns, offers)
    dispatch: dict[str, float] = dict(decision["dispatch"])
    sources = decision.get("dispatch_sources")
    if not isinstance(sources, Mapping) or set(sources) != set(dispatch):
        raise ValueError("arbiter decision requires dispatch_sources for exactly the dispatched features")
    if not set(dispatch).issubset(all_features):
        raise ValueError("arbiter dispatched an unknown action feature")
    for feature, engine_id in sources.items():
        if not isinstance(engine_id, str) or engine_id not in by_id:
            raise ValueError(f"arbiter dispatch source for {feature!r} names an unknown engine")
        offer = offers[engine_id]
        if offer is None or feature not in offer["values"] or offer["values"][feature] != dispatch[feature]:
            raise ValueError(f"arbiter dispatch source for {feature!r} does not match the engine offer")
    # Source identity is supplied by the policy. Equal values convey no ownership.
    selected = set(sources.values())
    contributing = [slot for slot in slots if slot.engine_id in selected]
    # Final permission gate, as in ``commit_action``: a denying engine's values are
    # withdrawn and its features fall back to holding.
    permitted = []
    for slot in contributing:
        if slot.engine.dispatch_allowed():
            permitted.append(slot)
            continue
        slot.interpolator.reset()
        slot.produced_ns, slot.last_values = None, None
        slot.engine.acknowledge_hold()
        for key, engine_id in sources.items():
            if engine_id == slot.engine_id:
                del dispatch[key]

    command = {}
    for key in all_features:
        if key in dispatch:
            command[key] = dispatch[key]
        elif key in last_command:
            command[key] = last_command[key]
        else:
            return TickResult(decision=decision, sent=None, contributing=[])
    send(command)
    last_command.update(command)
    sent_sources = {sources[key] for key in dispatch}
    permitted = [slot for slot in permitted if slot.engine_id in sent_sources]
    for slot in permitted:
        slot.engine.record_dispatch()
    return TickResult(decision=decision, sent=command, contributing=[s.engine_id for s in permitted])
