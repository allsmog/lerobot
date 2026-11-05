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
"""Tests for the optional external-scheduler hold points (lerobot.utils.trinity_gate)."""

import textwrap

import pytest
import torch

from lerobot.policies.common import flow_matching
from lerobot.policies.common.flow_matching import euler_integrate
from lerobot.utils import trinity_gate

FAKE_IMPL = textwrap.dedent(
    """
    import os

    CALLS = []

    def gate_boundary(synchronize=None):
        if synchronize is not None:
            synchronize()
        CALLS.append(os.environ["TSG_CB"])
        return {"outcome": "pass", "path": os.environ["TSG_CB"]}
    """
)


@pytest.fixture(autouse=True)
def _clean_gate(monkeypatch):
    monkeypatch.delenv("TSG_CB", raising=False)
    monkeypatch.delenv("TSG_GATE_IMPL", raising=False)
    trinity_gate.reset_for_tests()
    yield
    trinity_gate.reset_for_tests()


def test_gate_off_by_default_is_noop(monkeypatch):
    # A bogus implementation path proves nothing is loaded while TSG_CB is unset.
    monkeypatch.setenv("TSG_GATE_IMPL", "/nonexistent/main.py")
    assert not trinity_gate.gate_enabled()
    assert trinity_gate.boundary("cpu") is None


def test_gate_on_loads_implementation_and_passes_control_block(monkeypatch, tmp_path):
    impl = tmp_path / "main.py"
    impl.write_text(FAKE_IMPL)
    cb = tmp_path / "tsg.cb"
    monkeypatch.setenv("TSG_CB", str(cb))
    monkeypatch.setenv("TSG_GATE_IMPL", str(impl))
    result = trinity_gate.boundary("cpu")
    assert result == {"outcome": "pass", "path": str(cb)}
    # The implementation is loaded once for the same control block and file.
    assert trinity_gate._current_impl() is trinity_gate._current_impl()


def test_implementation_without_gate_boundary_is_rejected(monkeypatch, tmp_path):
    impl = tmp_path / "main.py"
    impl.write_text("class SegmentGate:\n    pass\n")
    monkeypatch.setenv("TSG_CB", str(tmp_path / "tsg.cb"))
    monkeypatch.setenv("TSG_GATE_IMPL", str(impl))
    with pytest.raises(ImportError, match="gate_boundary"):
        trinity_gate.boundary("cpu")


def test_gate_on_without_implementation_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setenv("TSG_CB", str(tmp_path / "tsg.cb"))
    with pytest.raises(RuntimeError, match="TSG_GATE_IMPL"):
        trinity_gate.boundary("cpu")


def test_euler_integrate_has_one_hold_point_per_step(monkeypatch):
    calls = []
    monkeypatch.setattr(flow_matching, "gate_boundary", lambda device: calls.append(device))
    noise = torch.zeros(2, 4, 3)
    out = euler_integrate(lambda x_t, t: torch.ones_like(x_t), noise, 10)
    assert len(calls) == 10
    assert all(device == noise.device for device in calls)
    assert out.shape == noise.shape


def test_euler_integrate_output_identical_with_gate_on_and_never_held(monkeypatch, tmp_path):
    noise = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(0))

    def velocity(x_t, t):
        return x_t * 0.5 + t[:, None, None]

    gate_off = euler_integrate(velocity, noise.clone(), 10)

    impl = tmp_path / "main.py"
    impl.write_text(FAKE_IMPL)
    monkeypatch.setenv("TSG_CB", str(tmp_path / "tsg.cb"))
    monkeypatch.setenv("TSG_GATE_IMPL", str(impl))
    gate_on = euler_integrate(velocity, noise.clone(), 10)

    torch.testing.assert_close(gate_on, gate_off, rtol=0, atol=0)
    fake_calls = trinity_gate._current_impl().CALLS
    assert len(fake_calls) == 10


def test_gate_loads_package_with_relative_imports(monkeypatch, tmp_path):
    pkg = tmp_path / "fake_gate_pkg"
    pkg.mkdir()
    (pkg / "engine.py").write_text(FAKE_IMPL)
    (pkg / "__init__.py").write_text("from .engine import CALLS, gate_boundary\n")
    cb = tmp_path / "tsg.cb"
    monkeypatch.setenv("TSG_CB", str(cb))
    monkeypatch.setenv("TSG_GATE_IMPL", str(pkg))
    assert trinity_gate.boundary("cpu") == {"outcome": "pass", "path": str(cb)}
    assert [str(cb)] == trinity_gate._current_impl().CALLS


class _Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = torch.nn.Identity()
        self.lin = torch.nn.Linear(3, 3)

    def forward(self, x):
        return self.lin(self.input_layernorm(x))


def _fake_policy(vision_layers, text_layers):
    """Shaped like SmolVLA: model.vlm_with_expert.vlm.model.{vision_model.encoder,text_model}.layers."""
    root = torch.nn.Module()
    root.model = torch.nn.Module()
    root.model.vlm_with_expert = torch.nn.Module()
    root.model.vlm_with_expert.vlm = torch.nn.Module()
    vlm = root.model.vlm_with_expert.vlm.model = torch.nn.Module()
    vlm.vision_model = torch.nn.Module()
    vlm.vision_model.encoder = torch.nn.Module()
    vlm.vision_model.encoder.layers = torch.nn.ModuleList(_Layer() for _ in range(vision_layers))
    vlm.text_model = torch.nn.Module()
    vlm.text_model.layers = torch.nn.ModuleList(_Layer() for _ in range(text_layers))
    return root


def test_prefix_boundaries_hook_each_prefix_layer(monkeypatch):
    policy = _fake_policy(vision_layers=2, text_layers=3)
    assert trinity_gate.install_prefix_boundaries(policy, "cpu") == 5
    calls = []
    monkeypatch.setattr(trinity_gate, "boundary", lambda device=None: calls.append(device))
    x = torch.zeros(1, 3)
    vlm = policy.model.vlm_with_expert.vlm.model
    for layer in vlm.vision_model.encoder.layers:
        x = layer(x)
    for layer in vlm.text_model.layers:
        x = layer(x)
    assert calls == ["cpu"] * 5


def test_prefix_boundaries_are_noop_and_output_identical_with_gate_off():
    torch.manual_seed(0)
    policy = _fake_policy(vision_layers=1, text_layers=2)
    x = torch.randn(2, 3)
    layers = list(policy.model.vlm_with_expert.vlm.model.text_model.layers)
    before = layers[1](layers[0](x))
    trinity_gate.install_prefix_boundaries(policy, "cpu")
    after = layers[1](layers[0](x))
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_prefix_boundaries_skip_unknown_policies():
    assert trinity_gate.install_prefix_boundaries(torch.nn.Linear(2, 2), "cpu") == 0


def test_prefix_boundaries_keep_module_inputs_with_gate_on(monkeypatch, tmp_path):
    # The real gate returns a result dict; a pre-hook must not pass it on as the module input.
    impl = tmp_path / "main.py"
    impl.write_text(FAKE_IMPL)
    monkeypatch.setenv("TSG_CB", str(tmp_path / "tsg.cb"))
    monkeypatch.setenv("TSG_GATE_IMPL", str(impl))
    torch.manual_seed(0)
    policy = _fake_policy(vision_layers=1, text_layers=1)
    layer = policy.model.vlm_with_expert.vlm.model.vision_model.encoder.layers[0]
    x = torch.randn(2, 3)
    before = layer(x)
    trinity_gate.install_prefix_boundaries(policy, "cpu")
    after = layer(x)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    assert len(trinity_gate._current_impl().CALLS) == 1
