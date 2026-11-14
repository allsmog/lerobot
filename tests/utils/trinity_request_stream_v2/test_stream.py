"""CPU plumbing checks with observable CUDA ports, never a GPU qualification."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from trinity_noise_padded_v1.test_padded import real_policy_port
from trinity_noise_v2 import test_noise as noise_cases

from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils import trinity_http_server as server
from lerobot.utils.trinity_policy_noise import NOISE_ARRAY

policy = noise_cases.policy


class ObservableStream:
    """A labeled CPU port that records context, dependency and drain ordering."""

    cuda_stream = 1234

    def __init__(self) -> None:
        self.local = threading.local()
        self.entries = self.exits = self.drains = 0
        self.dependencies: list[Any] = []
        self.drain_entered = threading.Event()
        self.drain_release = threading.Event()
        self.drain_release.set()
        self.error: BaseException | None = None

    @property
    def active(self) -> bool:
        return getattr(self.local, "active", False)

    def wait_stream(self, other: Any) -> None:
        self.dependencies.append(other)

    @contextmanager
    def context(self, stream: Any) -> Iterator[None]:
        assert stream is self
        previous = self.active
        self.local.active = True
        self.entries += 1
        try:
            yield
        finally:
            self.local.active = previous
            self.exits += 1

    def synchronize(self) -> None:
        assert self.active
        self.drains += 1
        self.drain_entered.set()
        assert self.drain_release.wait(2), "CPU fixture drain bound"
        if self.error is not None:
            raise self.error


def enable(monkeypatch: pytest.MonkeyPatch, model: Any, stream: ObservableStream) -> None:
    model.dedicated_cuda_stream = True
    model._request_stream = stream
    monkeypatch.setattr(torch.cuda, "stream", stream.context)


def test_complete_real_chunk_includes_noise_observer_and_postprocess(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, calls = real_policy_port(policy)
    stream = ObservableStream()
    enable(monkeypatch, model, stream)
    events: list[str] = []
    original_to = torch.Tensor.to

    def copy(value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        assert stream.active, "a chunk tensor transfer escaped the request context"
        events.append("transfer")
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", copy)

    def observe(value: Any, label: str) -> Any:
        assert stream.active, label
        events.append(label)
        return value

    for name in ["env_preprocessor", "preprocessor", "postprocessor", "env_postprocessor"]:
        original = getattr(model, name)

        def process(value: Any, fn: Any = original, label: str = name) -> Any:
            return fn(observe(value, label))

        setattr(model, name, process)
    monkeypatch.setattr(
        "lerobot.envs.utils.preprocess_observation", lambda value: observe(value, "raw_observation")
    )
    original_sample = model.policy.model.sample_actions

    def sample(*args: Any, **kwargs: Any) -> torch.Tensor:
        observe(None, "prediction")
        return original_sample(*args, **kwargs)

    model.policy.model.sample_actions = sample
    noise = noise_cases.explicit_noise()
    output = model.chunk("task", {**noise_cases.observation(), NOISE_ARRAY: noise})
    assert events[0] == "transfer"
    assert events.index("raw_observation") < events.index("prediction") < events.index("postprocessor")
    assert events.count("transfer") >= 52  # Noise, padded observer and all 50 output copies.
    assert np.array_equal(model.last_policy_chunk, noise[0] + 0.25)
    assert np.array_equal(output, noise[0, :, :7] + 0.25)
    assert len(calls) == 1
    assert stream.entries == stream.exits == stream.drains == 1 and not stream.active


def test_request_error_still_drains_and_restores_context(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = real_policy_port(policy)
    stream = ObservableStream()
    enable(monkeypatch, model, stream)

    def fail(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise LookupError("original prediction error")

    model.policy.model.sample_actions = fail
    with pytest.raises(LookupError, match="original prediction error"):
        model.chunk("task", {**noise_cases.observation(), NOISE_ARRAY: noise_cases.explicit_noise()})
    assert stream.entries == stream.exits == stream.drains == 1 and not stream.active
    assert model.last_policy_chunk is None and model.last_inference_metadata is None
    assert model.residency_state == "resident"


def test_drain_failure_faults_the_policy_and_refuses_later_requests(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = real_policy_port(policy)
    stream = ObservableStream()
    stream.error = RuntimeError("controlled stream drain failure")
    enable(monkeypatch, model, stream)
    arrays = {**noise_cases.observation(), NOISE_ARRAY: noise_cases.explicit_noise()}
    with pytest.raises(RuntimeError, match="stream drain failure"):
        model.chunk("task", arrays)
    assert model.residency_state == "fault"
    assert model.last_policy_chunk is None and model.last_inference_metadata is None
    assert stream.entries == stream.exits == stream.drains == 1 and not stream.active
    with pytest.raises(RuntimeError, match="not resident: fault"):
        model.chunk("task", arrays)
    assert stream.entries == stream.drains == 1


def test_http_inference_lock_and_reply_wait_for_stream_drain(
    policy: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, _ = real_policy_port(policy)
    stream = ObservableStream()
    stream.drain_release.clear()
    enable(monkeypatch, model, stream)
    running = noise_cases.RunningServer()
    running.http.RequestHandlerClass = server.make_handler(model, running.lock, noise_cases.OWNER)
    replies, errors = [], []
    wire = server.pack(
        {"task": "task", "request_id": 7, "captured_ns": 19},
        {**noise_cases.observation(), NOISE_ARRAY: noise_cases.explicit_noise()},
    )

    def request() -> None:
        try:
            replies.append(running.post(wire))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=request)
    thread.start()
    try:
        assert stream.drain_entered.wait(1)
        assert running.lock.locked() and not replies
        stream.drain_release.set()
        thread.join(2)
        assert not thread.is_alive() and not errors
        assert replies[0][0] == 200 and replies[0][1]["X-Trinity-Generation"] == "1"
        assert stream.entries == stream.exits == stream.drains == 1
    finally:
        stream.drain_release.set()
        thread.join(2)
        running.close()


def test_default_chunk_does_not_select_a_cuda_context(policy: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    model, _, _ = policy

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("default chunk selected a CUDA context")

    monkeypatch.setattr(torch.cuda, "stream", forbidden)
    output = model.chunk("task", noise_cases.observation())
    assert output.shape == (50, 7)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_non_cuda_selection_fails_before_policy_loading(device: str, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("invalid stream selection loaded a policy")

    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", forbidden)
    monkeypatch.setattr(torch.cuda, "Stream", forbidden)
    with pytest.raises(ValueError, match="requires a CUDA device"):
        server.LiberoChunkPolicy("unused", device=device, dedicated_cuda_stream=True)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_stream_selection_requires_a_boolean(value: Any) -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        server.LiberoChunkPolicy("unused", dedicated_cuda_stream=value)


@pytest.mark.parametrize("enabled", [False, True])
def test_constructor_stream_waits_for_model_initialization(
    enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    events: list[str] = []
    stream, initial = ObservableStream(), object()
    config = SimpleNamespace(device="cuda:0", dtype=torch.float32)
    policies = ModuleType("lerobot.policies")

    def make_policy(**kwargs: Any) -> Any:
        events.append("model")
        return SimpleNamespace(config=config, eval=lambda: None)

    policies.make_policy = make_policy
    policies.make_pre_post_processors = lambda **kwargs: (None, None)
    factory = ModuleType("lerobot.envs.factory")
    factory.make_env_pre_post_processors = lambda **kwargs: (None, None)
    gate = ModuleType("lerobot.utils.trinity_gate")
    gate.install_prefix_boundaries = lambda *args: events.append("prefix") or 0
    for module in [policies, factory, gate]:
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *args: config)

    def create(*, device: Any) -> ObservableStream:
        assert device == "cuda:0"
        events.append("stream")
        return stream

    monkeypatch.setattr(torch.cuda, "Stream", create)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: initial)
    model = server.LiberoChunkPolicy("unused", device="cuda:0", dedicated_cuda_stream=enabled)
    assert events == (["model", "prefix", "stream"] if enabled else ["model", "prefix"])
    assert stream.dependencies == ([initial] if enabled else [])
    assert model._request_stream is (stream if enabled else None)


def test_cli_forwards_the_explicit_stream_option(monkeypatch: pytest.MonkeyPatch) -> None:
    class StopBeforeLoadingError(Exception):
        pass

    seen = []

    def construct(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        raise StopBeforeLoadingError

    monkeypatch.setattr(server, "LiberoChunkPolicy", construct)
    with pytest.raises(StopBeforeLoadingError):
        server.main(["--policy-path", "unused", "--port", "12345", "--dedicated-cuda-stream"])
    assert seen[0]["dedicated_cuda_stream"] is True
