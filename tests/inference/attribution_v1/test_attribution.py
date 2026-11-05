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
"""CPU regression fixtures for exact dispatch attribution and permission holds."""

from collections.abc import Callable
from typing import Any

import pytest
import torch

from lerobot.inference.multi_engine import EngineSlot, multi_engine_tick
from lerobot.utils.action_interpolator import ActionInterpolator


class Engine:
    def __init__(self, values: list[float]):
        self.actions = [torch.tensor(values, device="cpu")]
        self.allowed: Callable[[], bool] = lambda: True
        self.permission_checks = 0
        self.dispatches = 0
        self.holds = 0
        self.pulls = 0

    def dispatch_allowed(self) -> bool:
        self.permission_checks += 1
        return self.allowed()

    def acknowledge_hold(self) -> None:
        self.holds += 1

    def get_action(self, _frame: Any) -> torch.Tensor | None:
        self.pulls += 1
        return self.actions.pop(0) if self.actions else None

    def record_dispatch(self) -> None:
        self.dispatches += 1


class Decision:
    def __init__(self, dispatch: dict, sources: Any):
        self.dispatch, self.sources = dispatch, sources

    def tick(self, _now: int, _offers: dict) -> dict:
        return {"dispatch": self.dispatch, "dispatch_sources": self.sources, "hold": [], "events": []}


class EchoSelected:
    def tick(self, _now: int, offers: dict) -> dict:
        offer = offers["pi05"]
        dispatch = {} if offer is None else {"gripper": offer["values"]["gripper"]}
        sources = {} if offer is None else {"gripper": "pi05"}
        return {
            "dispatch": dispatch,
            "dispatch_sources": sources,
            "hold": [] if offer is not None else ["gripper"],
            "events": [],
        }


def slots():
    pi, smol = Engine([0.0, 0.0]), Engine([0.0, 0.0])
    return (
        [
            EngineSlot("pi05", pi, ActionInterpolator(1), ["arm", "gripper"]),
            EngineSlot("smolvla", smol, ActionInterpolator(1), ["base", "gripper"]),
        ],
        pi,
        smol,
    )


def test_equal_values_credit_only_the_selected_engine():
    engines, pi, smol = slots()
    sent = []
    result = multi_engine_tick(
        engines, Decision({"gripper": 0.0}, {"gripper": "pi05"}), 0, None, sent.append, {}, ["gripper"]
    )
    assert result.contributing == ["pi05"]
    assert (pi.dispatches, smol.dispatches) == (1, 0)


def test_a_denied_equal_valued_non_source_cannot_withdraw_the_selection():
    engines, pi, smol = slots()
    smol.allowed = lambda: smol.permission_checks <= 2
    sent = []
    result = multi_engine_tick(
        engines,
        Decision({"gripper": 0.0}, {"gripper": "pi05"}),
        0,
        None,
        sent.append,
        {"gripper": 0.75},
        ["gripper"],
    )
    assert sent == [{"gripper": 0.0}]
    assert result.contributing == ["pi05"]
    assert smol.permission_checks == 2 and smol.holds == 0 and smol.dispatches == 0
    assert pi.dispatches == 1


def test_final_denial_withdraws_only_that_engines_attributed_features():
    engines, pi, smol = slots()
    smol.allowed = lambda: smol.permission_checks <= 2
    sent = []
    result = multi_engine_tick(
        engines,
        Decision({"arm": 0.0, "base": 0.0}, {"arm": "pi05", "base": "smolvla"}),
        0,
        None,
        sent.append,
        {"arm": 9.0, "base": 8.0},
        ["arm", "base"],
    )
    assert sent == [{"arm": 0.0, "base": 8.0}]
    assert result.contributing == ["pi05"]
    assert smol.holds == 1 and smol.dispatches == 0 and pi.dispatches == 1


def test_initial_denial_invalidates_the_cached_offer():
    engines, pi, _smol = slots()
    sent, last = [], {}
    multi_engine_tick(engines, EchoSelected(), 0, None, sent.append, last, ["gripper"])
    pi.permission_checks = 0
    pi.allowed = lambda: pi.permission_checks > 1
    result = multi_engine_tick(engines, EchoSelected(), 1, None, sent.append, last, ["gripper"])
    assert result.contributing == [] and pi.dispatches == 1 and pi.holds == 1
    assert engines[0].last_values is None and engines[0].produced_ns is None
    pi.allowed = lambda: True
    result = multi_engine_tick(engines, EchoSelected(), 2, None, sent.append, last, ["gripper"])
    assert result.contributing == [] and pi.dispatches == 1


def test_final_denial_does_not_resurrect_an_action_on_the_next_tick():
    engines, pi, _smol = slots()
    pi.allowed = lambda: pi.permission_checks <= 2
    sent, last = [], {"gripper": 0.75}
    multi_engine_tick(engines, EchoSelected(), 0, None, sent.append, last, ["gripper"])
    pi.allowed = lambda: True
    result = multi_engine_tick(engines, EchoSelected(), 1, None, sent.append, last, ["gripper"])
    assert sent == [{"gripper": 0.75}, {"gripper": 0.75}]
    assert result.contributing == [] and pi.dispatches == 0


@pytest.mark.parametrize("sources", [None, {}, {"other": "pi05"}, {"gripper": "unknown"}, {"gripper": []}])
def test_invalid_source_metadata_fails_before_send(sources):
    engines, _pi, _smol = slots()
    sent, last = [], {"gripper": 0.75}
    with pytest.raises(ValueError, match="dispatch_sources|dispatch source"):
        multi_engine_tick(
            engines, Decision({"gripper": 0.0}, sources), 0, None, sent.append, last, ["gripper"]
        )
    assert sent == [] and last == {"gripper": 0.75}


def test_source_must_have_offered_the_selected_value():
    engines, _pi, smol = slots()
    smol.actions = [torch.tensor([0.0, 0.5], device="cpu")]
    sent = []
    with pytest.raises(ValueError, match="does not match"):
        multi_engine_tick(
            engines, Decision({"gripper": 0.0}, {"gripper": "smolvla"}), 0, None, sent.append, {}, ["gripper"]
        )
    assert sent == []


def test_duplicate_slot_ids_fail_before_consuming_actions():
    engines, pi, smol = slots()
    engines[1].engine_id = "pi05"
    with pytest.raises(ValueError, match="unique engine_id"):
        multi_engine_tick(engines, Decision({}, {}), 0, None, lambda _: None, {}, ["gripper"])
    assert pi.pulls == smol.pulls == 0
