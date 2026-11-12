"""Exercise the real policy unpadding boundary with CPU denoiser ports."""

from typing import Any

import numpy as np
import pytest
import torch
from trinity_noise_v2 import test_noise as noise_cases

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.trinity_libero_noise_adapter import CONTEXT_SCHEMA, ControlledLeRobotLiberoAdapter, unpack
from lerobot.utils.trinity_policy_noise import NOISE_ARRAY, array_sha256

policy = noise_cases.policy


def real_policy_port(policy: Any) -> tuple[Any, list[dict[str, Any]]]:
    server, config, _ = policy
    model: Any = PI05Policy.__new__(PI05Policy)
    torch.nn.Module.__init__(model)
    config.image_features = {}
    config.output_features = {ACTION: type("Feature", (), {"shape": (7,)})()}
    model.config = config
    model._preprocess_images = lambda batch: ([], [])
    model._prepare_memory_states = lambda batch: (None, None)
    calls: list[dict[str, Any]] = []

    def sample(*args: Any, **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        noise = kwargs["noise"]
        return torch.zeros(1, 50, 32) if noise is None else noise + 0.25

    model.model = type("Sampler", (), {"sample_actions": staticmethod(sample)})()
    server.policy = model
    server.preprocessor = lambda batch: {
        OBS_LANGUAGE_TOKENS: torch.zeros((1, 2), dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones((1, 2), dtype=torch.bool),
    }
    return server, calls


def test_real_policy_keeps_default_public_unpadding_without_observer(policy: Any) -> None:
    server, calls = real_policy_port(policy)
    before = torch.random.get_rng_state().clone()
    output = server.chunk("task", noise_cases.observation())
    assert np.array_equal(output, np.zeros((50, 7), np.float32))
    assert calls[0]["noise"] is None and "action_chunk_observer" not in calls[0]
    assert torch.equal(before, torch.random.get_rng_state())
    assert server.last_policy_chunk is None and server.last_inference_metadata is None


def test_real_unpadding_boundary_captures_all_denoised_channels_before_postprocessing(policy: Any) -> None:
    server, calls = real_policy_port(policy)
    noise = noise_cases.explicit_noise()
    before = noise.copy()

    def mutate_action(action: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            return action.add_(3)

    server.postprocessor = mutate_action
    output = server.chunk("task", {**noise_cases.observation(), NOISE_ARRAY: noise})
    assert len(calls) == 1 and np.array_equal(noise, before)
    assert np.array_equal(server.last_policy_chunk, before[0] + 0.25)
    assert server.last_policy_chunk.shape == (50, 32)
    assert np.array_equal(output, before[0, :, :7] + 0.25 + 3)
    assert server.last_inference_metadata["predicted_sha256"] == array_sha256(server.last_policy_chunk)


def test_actual_policy_http_reply_passes_the_strict_32_channel_adapter(policy: Any) -> None:
    server, calls = real_policy_port(policy)
    adapter = ControlledLeRobotLiberoAdapter({})
    adapter.episode("task")
    context = {
        "schema": CONTEXT_SCHEMA,
        "seed": 20261112,
        "task_id": 0,
        "state_id": 1,
        "episode": 0,
        "phase": "control",
        "tick": 0,
        "model": "pi05",
    }
    request = adapter.encode(
        {"raw": noise_cases.observation(), "captured_ns": 19, "policy_noise_context": context}, 7
    )
    running = noise_cases.RunningServer()
    running.http.RequestHandlerClass = noise_cases.server.make_handler(
        server, running.lock, noise_cases.OWNER
    )
    try:
        code, _, reply = running.post(request)
        assert code == 200
        actions = adapter.decode(reply, 7)
        _, arrays = unpack(reply)
        assert arrays["predicted_actions"].shape == (50, 32)
        assert actions.shape == (50, 7)
        assert np.array_equal(arrays["predicted_actions"], calls[0]["noise"].numpy()[0] + 0.25)
    finally:
        running.close()


def test_observer_exception_preserves_the_original_error_and_no_prior_metadata(policy: Any) -> None:
    server, _ = real_policy_port(policy)
    server.chunk("task", {**noise_cases.observation(), NOISE_ARRAY: noise_cases.explicit_noise()})

    def fail(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise LookupError("original denoiser error")

    server.policy.model.sample_actions = fail
    with pytest.raises(LookupError, match="original denoiser error"):
        server.chunk("task", {**noise_cases.observation(), NOISE_ARRAY: noise_cases.explicit_noise()})
    assert server.last_policy_chunk is None and server.last_inference_metadata is None
