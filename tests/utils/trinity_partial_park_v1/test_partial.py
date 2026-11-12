"""Real CPU storage/value checks for the opt-in retained-weight park path."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

from lerobot.utils import trinity_http_server as server, trinity_residency_shadow as shadows


class Weights(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.large = torch.nn.Parameter(torch.arange(16, dtype=torch.float32))
        self.small = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.alias = self.large
        self.register_buffer("mutable", torch.tensor([3.0]))
        self.eval()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.large.sum() * value + self.small.sum() + self.alias.sum() + self.mutable


class Views(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = torch.arange(16, dtype=torch.float32)
        self.first = torch.nn.Parameter(base[:4])
        self.second = torch.nn.Parameter(base[4:8])
        self.eval()


def cache(model: torch.nn.Module, limit: int = 0) -> shadows.ParameterShadow:
    with patch.object(Path, "read_text", return_value="MemAvailable: 100663296 kB\nMemTotal: 134217728 kB\n"):
        return shadows.ParameterShadow(model, torch, "cpu", 14 * 1024**3, park_retain_bytes=limit)


def test_kept_storage_aliases_values_mutable_buffers_and_rng_survive_cycles() -> None:
    model = Weights()
    value = torch.tensor([2.0])
    expected = model(value).detach().clone()
    rng = torch.get_rng_state().clone()
    original_large, original_small = model.large.data_ptr(), model.small.data_ptr()
    store = cache(model, 64)
    for _ in range(3):
        store.park()
        assert model.large.data_ptr() == original_large
        assert model.small.data_ptr() == store.shadows["small"].data_ptr()
        assert model.small.data_ptr() != original_small
        assert model.alias is model.large
        assert torch.equal(model(value), expected)
        assert store.metadata()["last_park_retained_storage_bytes"] == 64
        assert store.metadata()["last_park_retained_parameters"] == 1
        assert store.metadata()["park_retain_limit_bytes"] == 64
        store.resumed()
    assert torch.equal(torch.get_rng_state(), rng)
    assert not torch.cuda.is_initialized()


def test_changed_and_replaced_parameters_refresh_before_partial_park() -> None:
    model = Weights()
    store = cache(model, 64)
    with torch.no_grad():
        model.large.add_(5)
        model.small.add_(7)
        model.mutable.add_(9)
    expected = model(torch.tensor([1.0])).detach().clone()
    store.park()
    assert store.last_refreshed == 2
    assert torch.equal(model(torch.tensor([1.0])), expected)
    store.resumed()
    model.large = torch.nn.Parameter(torch.full_like(model.large, 17))
    model.alias = model.large
    expected = model(torch.tensor([1.0])).detach().clone()
    replacement = model.large.data_ptr()
    store.park()
    assert store.last_refreshed == 1
    assert model.large.data_ptr() == replacement
    assert torch.equal(model(torch.tensor([1.0])), expected)


def test_views_charge_the_entire_shared_storage_and_keep_or_evict_together() -> None:
    model = Views()
    original = model.first.untyped_storage().data_ptr()
    assert original == model.second.untyped_storage().data_ptr()
    assert model.first.numel() * model.first.element_size() == 16
    assert model.first.untyped_storage().nbytes() == 64
    store = cache(model, 32)
    first, second = model.first.detach().clone(), model.second.detach().clone()
    store.park()
    assert store.metadata()["last_park_retained_storage_bytes"] == 0
    assert model.first.untyped_storage().data_ptr() != original
    assert model.second.untyped_storage().data_ptr() != original
    assert torch.equal(model.first, first) and torch.equal(model.second, second)
    model = Views()
    original = model.first.untyped_storage().data_ptr()
    store = cache(model, 64)
    store.park()
    assert store.metadata()["last_park_retained_storage_bytes"] == 64
    assert store.metadata()["last_park_retained_parameters"] == 2
    assert model.first.untyped_storage().data_ptr() == original
    assert model.second.untyped_storage().data_ptr() == original


def test_default_metadata_and_full_parking_remain_unchanged() -> None:
    model = Weights()
    store = cache(model)
    store.park()
    assert model.large.data_ptr() == store.shadows["large"].data_ptr()
    assert model.small.data_ptr() == store.shadows["small"].data_ptr()
    assert set(store.metadata()) == {
        "parameter_count",
        "parameter_bytes",
        "pinned",
        "host_reserve_bytes",
        "last_refreshed_parameters",
    }


@pytest.mark.parametrize("limit", [-1, True, 1.5, "64"])
def test_invalid_storage_cap_is_refused_before_shadow_allocation(limit: Any) -> None:
    model = Weights()
    original = model.large.data_ptr()
    with (
        patch.object(torch, "empty_like", side_effect=AssertionError("unexpected shadow allocation")),
        pytest.raises(ValueError, match="retention"),
    ):
        cache(model, limit)
    assert model.large.data_ptr() == original


@pytest.mark.parametrize("limit", [-1, True, 1.5, "64"])
def test_invalid_policy_option_is_refused_before_policy_loading(limit: Any) -> None:
    with pytest.raises(ValueError, match="retention"):
        server.LiberoChunkPolicy("unused", allow_residency=True, residency_retain_mib=limit)


def test_partial_option_requires_residency_before_policy_loading() -> None:
    with pytest.raises(ValueError, match="requires owned residency"):
        server.LiberoChunkPolicy("unused", residency_retain_mib=4096)


@pytest.mark.parametrize("limit", [0, 4096])
def test_prepare_forwards_the_exact_byte_cap_only_when_enabled(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    seen = []

    def shadow(*args: Any, **kwargs: Any) -> Any:
        seen.append((args, kwargs))
        return SimpleNamespace(metadata=lambda: {"fixture": True})

    monkeypatch.setattr(shadows, "ParameterShadow", shadow)
    model = object.__new__(server.LiberoChunkPolicy)
    model.policy = SimpleNamespace(config=SimpleNamespace(type="pi05"))
    model.torch, model.device = torch, "cpu"
    model.residency_enabled, model.residency_state = True, "resident"
    model.residency_retain_bytes = limit * 1024**2
    model._residency_shadow = None
    assert model.prepare_residency_shadow(14 * 1024**3) == {"fixture": True}
    assert seen[0][0] == (model.policy, torch, "cpu", 14 * 1024**3)
    assert seen[0][1] == ({"park_retain_bytes": limit * 1024**2} if limit else {})


@pytest.mark.parametrize(
    "arguments", [["--residency-retain-mib", "4"], ["--residency", "--residency-retain-mib", "-1"]]
)
def test_invalid_cli_never_constructs_a_policy(monkeypatch: pytest.MonkeyPatch, arguments: list[str]) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("invalid CLI retention reached policy construction")

    monkeypatch.setattr(server, "LiberoChunkPolicy", forbidden)
    with pytest.raises(SystemExit) as error:
        server.main(["--policy-path", "unused", "--port", "1999", *arguments])
    assert error.value.code == 2


@pytest.mark.parametrize("limit", [0, 4096])
def test_cli_preserves_default_constructor_calls_and_forwards_explicit_cap(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    seen = []

    def constructor(*args: Any, **kwargs: Any) -> None:
        seen.append(kwargs)
        raise LookupError("constructor probe")

    monkeypatch.setenv("TRINITY_LEROBOT_OWNER_NONCE", "a" * 32)
    monkeypatch.setattr(server, "LiberoChunkPolicy", constructor)
    with pytest.raises(LookupError, match="constructor probe"):
        server.main(
            ["--policy-path", "unused", "--port", "1999", "--residency", "--residency-retain-mib", str(limit)]
        )
    assert seen == [dict(allow_residency=True, **({"residency_retain_mib": limit} if limit else {}))]
