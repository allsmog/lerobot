"""Actual Torch module/alias/version tests with CUDA explicitly disabled."""

import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lerobot.utils.trinity_residency_shadow import ParameterShadow


class Pair(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(3, 2, bias=False)
        self.second = torch.nn.Linear(3, 2, bias=False)
        self.second.weight = self.first.weight
        self.register_buffer("mutable", torch.tensor([3.0]))
        with torch.no_grad():
            self.first.weight.copy_(torch.arange(6).reshape(2, 3))
        self.eval()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.first(value) + self.second(value) + self.mutable


class TestShadow(unittest.TestCase):
    def setUp(self) -> None:
        # Resource forecasts remain deterministic on Linux and macOS. This
        # fixture grants CPU memory only and never initializes a device.
        self.memory = patch.object(
            Path, "read_text", return_value="MemAvailable: 100663296 kB\nMemTotal: 134217728 kB\n"
        )
        self.memory.start()
        self.addCleanup(self.memory.stop)

    def store(self, model: Pair) -> ParameterShadow:
        return ParameterShadow(model, torch, "cpu", 14 * 1024**3)

    def test_alias_values_buffers_and_rng_survive(self) -> None:
        model = Pair()
        x = torch.tensor([1.0, 2.0, 3.0])
        expected = model(x).detach().clone()
        rng = torch.get_rng_state().clone()
        store = self.store(model)
        self.assertEqual(store.metadata()["parameter_count"], 1)
        with torch.no_grad():
            model.mutable.add_(4)
        store.park()
        self.assertIs(model.first.weight, model.second.weight)
        self.assertTrue(torch.equal(model(x), expected + 4))
        self.assertEqual(model.mutable.item(), 7)
        self.assertEqual(store.last_refreshed, 0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        store.resumed()
        self.assertFalse(torch.cuda.is_initialized())

    def test_changed_parameter_version_refreshes_cached_values(self) -> None:
        model = Pair()
        store = self.store(model)
        with torch.no_grad():
            model.first.weight.add_(5)
        expected = model.first.weight.detach().clone()
        store.park()
        self.assertEqual(store.last_refreshed, 1)
        self.assertTrue(torch.equal(model.first.weight, expected))
        self.assertIs(model.first.weight, model.second.weight)

    def test_replacement_parameter_refreshes_even_with_same_version(self) -> None:
        model = Pair()
        store = self.store(model)
        replacement = torch.nn.Parameter(torch.full((2, 3), 17.0))
        model.first.weight = replacement
        model.second.weight = replacement
        store.park()
        self.assertEqual(store.last_refreshed, 1)
        self.assertTrue(torch.equal(model.first.weight, torch.full((2, 3), 17.0)))

    def test_topology_change_is_refused(self) -> None:
        model = Pair()
        store = self.store(model)
        model.register_parameter("extra", torch.nn.Parameter(torch.ones(1)))
        with self.assertRaisesRegex(RuntimeError, "topology"):
            store.park()

    def test_training_or_gradient_state_is_refused(self) -> None:
        model = Pair()
        model.train()
        with self.assertRaisesRegex(ValueError, "inference"):
            self.store(model)
        model.eval()
        model.first.weight.grad = torch.ones_like(model.first.weight)
        with self.assertRaisesRegex(ValueError, "gradients"):
            self.store(model)

    def test_unaffordable_cache_is_refused_before_mutating_model(self) -> None:
        model = Pair()
        original = model.first.weight
        with self.assertRaises(MemoryError):
            ParameterShadow(model, torch, "cpu", 2**60)
        self.assertIs(model.first.weight, original)
        self.assertIs(model.first.weight, model.second.weight)


if __name__ == "__main__":
    program = unittest.main(verbosity=2, exit=False)
    assert program.result.wasSuccessful()
    assert not torch.cuda.is_initialized()
