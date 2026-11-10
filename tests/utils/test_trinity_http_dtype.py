"""Check public policy precision options without creating model parameters."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.utils.trinity_http_server import LiberoChunkPolicy


@pytest.mark.parametrize(
    ("override", "expected"),
    [("bfloat16", torch.bfloat16), ("float32", torch.float32), (None, torch.bfloat16)],
)
def test_precision_reaches_policy_factory_as_torch_dtype(
    monkeypatch: pytest.MonkeyPatch, override: str | None, expected: torch.dtype
) -> None:
    cfg = SimpleNamespace(device="cpu", dtype=torch.bfloat16)
    seen = []

    def make_policy(*, cfg: Any, env_cfg: LiberoEnv, **kwargs: Any) -> SimpleNamespace:
        assert isinstance(env_cfg, LiberoEnv)
        seen.append(cfg.dtype)
        return SimpleNamespace(config=cfg, eval=lambda: None)

    policies = ModuleType("lerobot.policies")
    policies.make_policy = make_policy
    policies.make_pre_post_processors = lambda **kwargs: (None, None)
    factory = ModuleType("lerobot.envs.factory")
    factory.make_env_pre_post_processors = lambda **kwargs: (None, None)
    gate = ModuleType("lerobot.utils.trinity_gate")
    gate.install_prefix_boundaries = lambda *args: 0
    monkeypatch.setitem(sys.modules, policies.__name__, policies)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    monkeypatch.setitem(sys.modules, gate.__name__, gate)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *args: cfg)
    LiberoChunkPolicy("unused", device="cpu", dtype=override)
    assert seen == [expected]
    assert cfg.dtype is expected


def test_invalid_precision_is_rejected_before_policy_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = SimpleNamespace(device="cpu", dtype=torch.bfloat16)
    policies = ModuleType("lerobot.policies")

    def forbidden(**kwargs: Any) -> None:
        pytest.fail("invalid precision reached the policy factory")

    policies.make_policy = forbidden
    policies.make_pre_post_processors = lambda **kwargs: (None, None)
    factory = ModuleType("lerobot.envs.factory")
    factory.make_env_pre_post_processors = lambda **kwargs: (None, None)
    monkeypatch.setitem(sys.modules, policies.__name__, policies)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *args: cfg)
    with pytest.raises(ValueError, match="dtype"):
        LiberoChunkPolicy("unused", device="cpu", dtype="not_a_dtype")
