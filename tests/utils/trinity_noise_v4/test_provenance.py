"""Additional provenance ownership cases over the already passing CPU fixture."""

from typing import Any

import numpy as np
import pytest
import torch
from trinity_noise_v2 import test_noise as noise_cases

from lerobot.utils.trinity_policy_noise import NOISE_ARRAY, array_sha256

# Export the existing pytest fixture as a module binding.
policy = noise_cases.policy
explicit_noise = noise_cases.explicit_noise
observation = noise_cases.observation


def test_noise_identity_binds_the_initial_tensor_even_if_policy_mutates_it(policy: Any) -> None:
    p, _, _ = policy
    noise = explicit_noise()
    initial = noise.copy()

    def mutate(batch: dict[str, Any], *, noise: torch.Tensor) -> torch.Tensor:
        noise.add_(1)
        return noise[:, :, :7]

    p.policy.predict_action_chunk = mutate
    result = p.chunk("task", {**observation(), NOISE_ARRAY: noise})
    assert p.last_inference_metadata["noise_sha256"] == array_sha256(initial)
    assert np.array_equal(noise, initial)
    assert np.array_equal(result, initial[0, :, :7] + 1)


def test_failed_prediction_clears_prior_provenance_without_replaying(policy: Any) -> None:
    p, _, _ = policy
    p.chunk("task", {**observation(), NOISE_ARRAY: explicit_noise()})
    assert p.last_inference_metadata is not None
    calls = []

    def fail(batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        calls.append(1)
        raise LookupError("controlled original prediction failure")

    p.policy.predict_action_chunk = fail
    with pytest.raises(LookupError, match="controlled original prediction failure"):
        p.chunk("task", {**observation(), NOISE_ARRAY: explicit_noise()})
    assert calls == [1]
    assert p.last_inference_metadata is None and p.last_policy_chunk is None


def test_default_request_clears_previous_controlled_metadata(policy: Any) -> None:
    p, _, calls = policy
    p.chunk("task", {**observation(), NOISE_ARRAY: explicit_noise()})
    assert p.last_inference_metadata is not None and p.last_policy_chunk is not None
    p.chunk("task", observation())
    assert calls[-1] == {}
    assert p.last_inference_metadata is None and p.last_policy_chunk is None
