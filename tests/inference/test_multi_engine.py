#!/usr/bin/env python

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
"""Tests for the multi-engine control tick (lerobot.inference.multi_engine)."""

import os
import textwrap

import pytest
import torch

from lerobot.inference.multi_engine import EngineSlot, load_arbiter, multi_engine_tick
from lerobot.utils.action_interpolator import ActionInterpolator

# A small arbiter honoring the interface contract: owned features go to their owner,
# shared features to the highest-priority fresh engine.
FAKE_ARBITER = textwrap.dedent(
    """
    class Arbiter:
        def __init__(self, deployment):
            self.engines = {e["engine_id"]: e for e in deployment["engines"]}
            self.features = deployment["features"]
            self.ticks = 0

        def tick(self, now_ns, offers):
            self.ticks += 1
            fresh = {
                eid: o for eid, o in offers.items()
                if o is not None and now_ns - o["produced_ns"] <= self.engines[eid]["deadline_ns"]
            }
            dispatch, sources, hold, events = {}, {}, [], []
            for eid, o in offers.items():
                if o is None:
                    events.append({"kind": "starved", "engine_id": eid})
            for feature, owner in self.features.items():
                if owner == "shared":
                    ranked = sorted(fresh, key=lambda e: -self.engines[e]["priority"])
                    src = ranked[0] if ranked else None
                else:
                    src = owner if owner in fresh else None
                if src is None:
                    hold.append(feature)
                else:
                    dispatch[feature] = fresh[src]["values"][feature]
                    sources[feature] = src
            return {"dispatch": dispatch, "dispatch_sources": sources, "hold": sorted(hold), "events": events}

        def summary(self):
            return {"ticks": self.ticks}
    """
)

DEPLOYMENT = {
    "engines": [
        {"engine_id": "pi05", "priority": 10, "deadline_ns": 50_000_000},
        {"engine_id": "smolvla", "priority": 1, "deadline_ns": 50_000_000},
    ],
    "features": {"arm.pos": "pi05", "base.vel": "smolvla", "gripper.pos": "shared"},
}


class FakeEngine:
    """The subset of InferenceEngine the tick uses."""

    def __init__(self, actions, allowed=True):
        self.actions = list(actions)
        self.allowed = allowed
        self.dispatches = 0
        self.holds = 0

    def dispatch_allowed(self):
        return self.allowed

    def acknowledge_hold(self):
        self.holds += 1

    def get_action(self, frame):
        return self.actions.pop(0) if self.actions else None

    def record_dispatch(self):
        self.dispatches += 1


@pytest.fixture
def arbiter(tmp_path):
    impl = tmp_path / "arbiter.py"
    impl.write_text(FAKE_ARBITER)
    return load_arbiter(impl, DEPLOYMENT)


def _slots(pi_actions, smol_actions, pi_allowed=True):
    pi = FakeEngine(pi_actions, allowed=pi_allowed)
    smol = FakeEngine(smol_actions)
    return (
        [
            EngineSlot("pi05", pi, ActionInterpolator(1), ["arm.pos", "gripper.pos"]),
            EngineSlot("smolvla", smol, ActionInterpolator(1), ["base.vel", "gripper.pos"]),
        ],
        pi,
        smol,
    )


FEATURES = ["arm.pos", "base.vel", "gripper.pos"]


def test_both_engines_fresh_merges_by_ownership_then_priority(arbiter):
    slots, pi, smol = _slots([torch.tensor([1.0, 0.5])], [torch.tensor([2.0, 0.9])])
    sent, last = [], {}
    result = multi_engine_tick(slots, arbiter, 1_000, None, sent.append, last, FEATURES)
    assert sent == [{"arm.pos": 1.0, "base.vel": 2.0, "gripper.pos": 0.5}]
    assert sorted(result.contributing) == ["pi05", "smolvla"]
    assert pi.dispatches == 1 and smol.dispatches == 1


def test_starved_engine_holds_its_features_with_last_command(arbiter):
    slots, pi, smol = _slots([torch.tensor([1.0, 0.5])], [])
    last = {"arm.pos": 0.0, "base.vel": 7.0, "gripper.pos": 0.0}
    sent = []
    result = multi_engine_tick(slots, arbiter, 1_000, None, sent.append, last, FEATURES)
    assert sent == [{"arm.pos": 1.0, "base.vel": 7.0, "gripper.pos": 0.5}]
    assert {"kind": "starved", "engine_id": "smolvla"} in result.decision["events"]
    assert smol.dispatches == 0


def test_nothing_sent_when_a_held_feature_was_never_commanded(arbiter):
    slots, _, _ = _slots([torch.tensor([1.0, 0.5])], [])
    sent = []
    result = multi_engine_tick(slots, arbiter, 1_000, None, sent.append, {}, FEATURES)
    assert sent == [] and result.sent is None


def test_denied_engine_is_withdrawn_at_the_final_gate(arbiter):
    slots, pi, smol = _slots([torch.tensor([1.0, 0.5])], [torch.tensor([2.0, 0.9])], pi_allowed=True)
    last = {"arm.pos": 0.0, "base.vel": 0.0, "gripper.pos": 0.0}
    # pi05 permits the pull, then denies at the final gate.
    calls = {"n": 0}

    def allowed():
        calls["n"] += 1
        return calls["n"] <= 2

    pi.dispatch_allowed = allowed
    sent = []
    result = multi_engine_tick(slots, arbiter, 1_000, None, sent.append, last, FEATURES)
    assert sent == [{"arm.pos": 0.0, "base.vel": 2.0, "gripper.pos": 0.0}]
    assert result.contributing == ["smolvla"]
    assert pi.holds == 1 and pi.dispatches == 0


def test_stale_high_priority_action_yields_shared_feature(arbiter):
    slots, pi, smol = _slots([torch.tensor([1.0, 0.5])], [torch.tensor([2.0, 0.9]), torch.tensor([3.0, 0.8])])
    last = {}
    sent = []
    multi_engine_tick(slots, arbiter, 0, None, sent.append, last, FEATURES)
    # 60 ms later pi05 has nothing new, so its 0 ns action is re-offered and is stale
    # (deadline 50 ms); smolvla offers a fresh one, which takes the shared feature.
    result = multi_engine_tick(slots, arbiter, 60_000_000, None, sent.append, last, FEATURES)
    assert sent[-1]["gripper.pos"] == pytest.approx(0.8)
    assert sent[-1]["base.vel"] == pytest.approx(3.0)
    assert sent[-1]["arm.pos"] == pytest.approx(1.0)  # owned by stale pi05: held at last command
    assert result.contributing == ["smolvla"]


def test_reoffered_action_within_deadline_still_counts_as_fresh(arbiter):
    slots, pi, smol = _slots([torch.tensor([1.0, 0.5])], [torch.tensor([2.0, 0.9])])
    last, sent = {}, []
    multi_engine_tick(slots, arbiter, 0, None, sent.append, last, FEATURES)
    multi_engine_tick(slots, arbiter, 10_000_000, None, sent.append, last, FEATURES)
    assert sent[-1]["gripper.pos"] == pytest.approx(0.5)  # pi05 still fresh at 10 ms, wins shared


@pytest.mark.skipif(
    not os.environ.get("TRINITY_ARBITER_IMPL"), reason="set TRINITY_ARBITER_IMPL to the generated arbiter"
)
def test_generated_arbiter_drives_a_tick():
    arbiter = load_arbiter(os.environ["TRINITY_ARBITER_IMPL"], DEPLOYMENT)
    slots, _, _ = _slots([torch.tensor([1.0, 0.5])], [torch.tensor([2.0, 0.9])])
    sent = []
    multi_engine_tick(slots, arbiter, 1_000, None, sent.append, {}, FEATURES)
    assert sent == [{"arm.pos": 1.0, "base.vel": 2.0, "gripper.pos": 0.5}]
    assert arbiter.summary()["continuity"] == 1.0
