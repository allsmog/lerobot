"""Keep model predictions distinct from processed simulator commands."""

from typing import Any

import numpy as np
from trinity_noise_v2 import test_noise as noise_cases

from lerobot.utils.constants import ACTION
from lerobot.utils.trinity_policy_noise import NOISE_ARRAY, array_sha256

policy = noise_cases.policy


def test_prediction_provenance_keeps_the_full_unprocessed_policy_chunk(policy: Any) -> None:
    p, _, _ = policy
    noise = noise_cases.explicit_noise()
    p.postprocessor = lambda value: value * 2
    p.env_postprocessor = lambda value: {ACTION: value[ACTION] + 3}
    actions = p.chunk("task", {**noise_cases.observation(), NOISE_ARRAY: noise})
    predictions = noise[0, :, :7]
    assert np.array_equal(p.last_policy_chunk, predictions)
    assert np.array_equal(actions, predictions * 2 + 3)
    metadata = p.last_inference_metadata
    assert metadata["prediction_space"] == "policy_before_postprocessing"
    assert metadata["actions_space"] == "libero_environment_after_postprocessing"
    assert metadata["predicted_sha256"] == array_sha256(predictions)
    assert metadata["actions_sha256"] == array_sha256(actions)
