"""Exact stateless processor parity and fail-closed opt-in chunk handling."""

from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from trinity_noise_padded_v1.test_padded import real_policy_port
from trinity_noise_v2 import test_noise as noise_cases

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor import (
    AbsoluteActionsProcessorStep,
    DeviceProcessorStep,
    PolicyProcessorPipeline,
    UnnormalizerProcessorStep,
    make_policy_processor_pipelines,
)
from lerobot.utils.constants import ACTION
from lerobot.utils.trinity_action_postprocess import check_batch_postprocess
from lerobot.utils.trinity_http_server import LiberoChunkPolicy
from lerobot.utils.trinity_libero_noise_adapter import (
    CONTEXT_SCHEMA,
    ControlledLeRobotLiberoAdapter,
    unpack,
)
from lerobot.utils.trinity_policy_noise import NOISE_ARRAY

policy = noise_cases.policy


def processors() -> tuple[Any, Any]:
    stats = {ACTION: {"mean": torch.linspace(-0.7, 0.9, 7), "std": torch.linspace(0.1, 1.7, 7)}}
    unnormalize = UnnormalizerProcessorStep(
        features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats=stats,
    )
    _, post = make_policy_processor_pipelines([], [unnormalize, DeviceProcessorStep(device="cpu")])
    return post, PolicyProcessorPipeline(steps=[])


@pytest.mark.parametrize("count", [1, 8, 50])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_processors_preserve_every_action_bit(count: int, dtype: torch.dtype) -> None:
    post, env = processors()
    check_batch_postprocess(SimpleNamespace(type="pi05"), post, env)
    chunk = torch.linspace(-100, 100, count * 7, dtype=dtype).reshape(1, count, 7)
    original = chunk.clone()
    before = torch.random.get_rng_state().clone()
    expected = torch.stack([env({ACTION: post(chunk[:, tick])})[ACTION] for tick in range(count)], dim=1)
    actual = env({ACTION: post(chunk)})[ACTION]
    assert torch.equal(expected, actual)
    assert torch.equal(chunk, original) and torch.equal(before, torch.random.get_rng_state())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("steps", [1, 8, 50])
def test_actual_chunk_and_full_prediction_provenance_are_identical(
    policy: Any, monkeypatch: pytest.MonkeyPatch, steps: int
) -> None:
    server, _ = real_policy_port(policy)
    server.policy.config.n_action_steps = steps
    server.postprocessor, server.env_postprocessor = processors()
    arrays = {**noise_cases.observation(), NOISE_ARRAY: noise_cases.explicit_noise()}
    calls = []
    original = UnnormalizerProcessorStep.__call__

    def observed(step: Any, transition: Any) -> Any:
        calls.append(1)
        return original(step, transition)

    monkeypatch.setattr(UnnormalizerProcessorStep, "__call__", observed)
    expected = server.chunk("task", arrays)
    metadata = deepcopy(server.last_inference_metadata)
    prediction = server.last_policy_chunk.copy()
    assert len(calls) == steps
    calls.clear()
    server.batch_action_postprocess = True
    actual = server.chunk("task", arrays)
    assert len(calls) == 1
    assert np.array_equal(expected, actual)
    assert np.array_equal(prediction, server.last_policy_chunk)
    assert server.last_inference_metadata == metadata


def test_opt_in_http_reply_preserves_strict_full_prediction_and_action_provenance(policy: Any) -> None:
    server, _ = real_policy_port(policy)
    server.postprocessor, server.env_postprocessor = processors()
    adapter = ControlledLeRobotLiberoAdapter({})
    adapter.episode("task")
    context = {
        "schema": CONTEXT_SCHEMA,
        "seed": 20261112,
        "task_id": 0,
        "state_id": 1,
        "episode": 0,
        "phase": "control",
        "tick": 40,
        "model": "pi05",
    }
    request = adapter.encode(
        {"raw": noise_cases.observation(), "captured_ns": 19, "policy_noise_context": context}, 7
    )
    header, arrays = unpack(request)
    expected = server.chunk(header["task"], arrays)
    metadata = deepcopy(server.last_inference_metadata)
    server.batch_action_postprocess = True
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
        assert np.array_equal(actions, expected)
        assert server.last_inference_metadata == metadata
    finally:
        running.close()


@pytest.mark.parametrize("mutation", ["relative", "hook", "environment", "converter", "statistics", "dtype"])
def test_unsupported_or_mutated_pipeline_is_refused_before_inference(policy: Any, mutation: str) -> None:
    server, calls = real_policy_port(policy)
    server.postprocessor, server.env_postprocessor = processors()
    server.batch_action_postprocess = True
    if mutation == "relative":
        server.postprocessor.steps.insert(1, AbsoluteActionsProcessorStep(enabled=True))
    elif mutation == "hook":
        server.postprocessor.before_step_hooks.append(lambda *_: None)
    elif mutation == "environment":
        server.env_postprocessor.steps.append(DeviceProcessorStep())
    elif mutation == "converter":
        server.postprocessor.to_output = lambda value: value
    elif mutation == "statistics":
        server.postprocessor.steps[0]._tensor_stats[ACTION]["mean"] = torch.zeros(50, 7)
    elif mutation == "dtype":
        server.postprocessor.steps[-1].float_dtype = "bfloat16"
    with pytest.raises(ValueError, match="batch action"):
        server.chunk("task", noise_cases.observation())
    assert calls == []
    assert server.last_inference_metadata is None and server.last_policy_chunk is None


def test_batch_option_is_strict_and_default_custom_processors_still_work(policy: Any) -> None:
    with pytest.raises(ValueError, match="boolean"):
        LiberoChunkPolicy("unused", batch_action_postprocess=1)
    server, _ = real_policy_port(policy)
    server.postprocessor = lambda action: action + 3
    expected = server.chunk("task", noise_cases.observation())
    assert np.array_equal(expected, np.full((50, 7), 3, dtype=np.float32))


def test_disabled_absolute_stage_is_supported() -> None:
    post, env = processors()
    post.steps.insert(1, AbsoluteActionsProcessorStep(enabled=False))
    check_batch_postprocess(SimpleNamespace(type="pi05"), post, env)


def test_unsupported_policy_or_relative_configuration_is_refused() -> None:
    post, env = processors()
    for config in [SimpleNamespace(type="pi0"), SimpleNamespace(type="pi05", use_relative_actions=True)]:
        with pytest.raises(ValueError, match="stateless pi05"):
            check_batch_postprocess(config, post, env)


@pytest.mark.parametrize(
    "config", [SimpleNamespace(type="pi0"), SimpleNamespace(type="pi05", use_relative_actions=True)]
)
def test_unsupported_configuration_is_refused_before_loading_weights(
    config: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *_: config)
    with pytest.raises(ValueError, match="stateless pi05"):
        LiberoChunkPolicy("unused", device="cpu", batch_action_postprocess=True)
