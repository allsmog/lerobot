"""Exercise real HTTP/chunk plumbing with CPU-only policy ports, without model weights."""

from __future__ import annotations

import http.client
import http.server
import threading
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from lerobot.utils import trinity_http_server as server
from lerobot.utils.trinity_policy_noise import NOISE_ARRAY, SCHEMA, array_sha256, observation_sha256

OWNER = "a" * 32


class DefaultPolicy:
    residency_enabled = True
    residency_state = "resident"
    residency_epoch = 0

    def __init__(self) -> None:
        self.calls = 0

    def chunk(self, task: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
        self.calls += 1
        return np.arange(14, dtype=np.float32).reshape(2, 7)


class RunningServer:
    def __init__(self) -> None:
        self.model = DefaultPolicy()
        self.lock = threading.Lock()
        self.http = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), server.make_handler(self.model, self.lock, OWNER)
        )
        self.thread = threading.Thread(target=lambda: self.http.serve_forever(poll_interval=0.01))
        try:
            self.thread.start()
        except BaseException:
            self.http.server_close()
            raise

    def post(self, body: bytes, owner: str = OWNER) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_address[1], timeout=2)
        try:
            conn.request("POST", "/", body, {"X-Trinity-Owner": owner, "X-Trinity-Generation": "1"})
            response = conn.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            conn.close()

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(2)
        assert not self.thread.is_alive()


def observation() -> dict[str, np.ndarray]:
    return {
        "agentview_image": np.zeros((2, 2, 3), np.uint8),
        "robot0_eye_in_hand_image": np.ones((2, 2, 3), np.uint8),
        "robot0_eef_pos": np.zeros(3, np.float32),
        "robot0_eef_quat": np.array([1, 0, 0, 0], np.float32),
        "robot0_gripper_qpos": np.zeros(2, np.float32),
    }


def explicit_noise() -> np.ndarray:
    return np.arange(1600, dtype=np.float32).reshape(1, 50, 32) / 1000


@pytest.fixture
def policy(monkeypatch: pytest.MonkeyPatch) -> tuple[server.LiberoChunkPolicy, Any, list[dict[str, Any]]]:
    cfg = SimpleNamespace(
        type="pi05",
        chunk_size=50,
        max_action_dim=32,
        n_action_steps=50,
        dtype=torch.float32,
        use_visual_memory=False,
        use_proprioceptive_memory=False,
    )
    calls: list[dict[str, Any]] = []

    def predict(batch: dict[str, Any], **kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        noise = kwargs.get("noise")
        return torch.zeros(1, 50, 7) if noise is None else noise[:, :, :7]

    p = server.LiberoChunkPolicy.__new__(server.LiberoChunkPolicy)
    p.torch, p.device = torch, "cpu"
    p.residency_state, p.residency_epoch, p.residency_enabled = "resident", 0, True
    p.policy = SimpleNamespace(config=cfg, predict_action_chunk=predict)
    p.env_preprocessor = p.preprocessor = lambda value: value
    p.postprocessor = lambda value: value
    p.env_postprocessor = lambda value: value
    monkeypatch.setattr("lerobot.envs.utils.preprocess_observation", lambda value: value)
    monkeypatch.setattr("lerobot.utils.trinity_gate.boundary", lambda device: None)
    return p, cfg, calls


def test_default_prediction_has_no_noise_keyword_or_extra_metadata(policy: Any) -> None:
    p, _, calls = policy
    out = p.chunk("task", observation())
    assert out.shape == (50, 7) and np.array_equal(out, np.zeros((50, 7), np.float32))
    assert calls == [{}]
    assert p.last_inference_metadata is None and p.last_policy_chunk is None


def test_explicit_noise_reaches_policy_without_global_rng_changes(policy: Any) -> None:
    p, _, calls = policy
    noise = explicit_noise()
    arrays = {**observation(), NOISE_ARRAY: noise}
    before = torch.random.get_rng_state().clone()
    out = p.chunk("task", arrays)
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(calls) == 1 and set(calls[0]) == {"noise"}
    forwarded = calls[0]["noise"]
    assert forwarded.device.type == "cpu" and forwarded.dtype == torch.float32
    assert np.array_equal(forwarded.numpy(), noise)
    assert np.array_equal(out, noise[0, :, :7])
    metadata = p.last_inference_metadata
    assert metadata["schema"] == SCHEMA
    assert metadata["noise_sha256"] == array_sha256(noise)
    assert metadata["actions_sha256"] == array_sha256(out)
    assert metadata["predicted_sha256"] == array_sha256(p.last_policy_chunk)


@pytest.mark.parametrize(
    "invalid", ["shape", "dtype", "nan", "inf", "other_policy", "visual_memory", "state_memory"]
)
def test_invalid_controlled_noise_is_rejected_before_prediction(policy: Any, invalid: str) -> None:
    p, cfg, calls = policy
    noise = explicit_noise()
    if invalid == "shape":
        noise = noise[0]
    elif invalid == "dtype":
        noise = noise.astype(np.float64)
    elif invalid == "nan":
        noise[0, 0, 0] = np.nan
    elif invalid == "inf":
        noise[0, 0, 0] = np.inf
    elif invalid == "other_policy":
        cfg.type = "smolvla"
    elif invalid == "visual_memory":
        cfg.use_visual_memory = True
    elif invalid == "state_memory":
        cfg.use_proprioceptive_memory = True
    with pytest.raises(ValueError):
        p.chunk("task", {**observation(), NOISE_ARRAY: noise})
    assert calls == [] and p.last_inference_metadata is None and p.last_policy_chunk is None


def test_observation_identity_separates_content_from_noise_and_header_time() -> None:
    arrays = observation()
    original = observation_sha256("task", arrays, server.REQUEST_ARRAYS)
    for timestamp, noise in [(17, explicit_noise()), (9999, -explicit_noise())]:
        _, unpacked = server.unpack(server.pack({"captured_ns": timestamp}, {**arrays, NOISE_ARRAY: noise}))
        assert observation_sha256("task", unpacked, server.REQUEST_ARRAYS) == original
    assert observation_sha256("different task", arrays, server.REQUEST_ARRAYS) != original
    arrays["robot0_eef_pos"][0] = 1
    assert observation_sha256("task", arrays, server.REQUEST_ARRAYS) != original


def test_owned_http_keeps_full_predictions_and_provenance_when_return_is_sliced(policy: Any) -> None:
    p, cfg, calls = policy
    cfg.n_action_steps = 4
    running = RunningServer()
    running.http.RequestHandlerClass = server.make_handler(p, running.lock, OWNER)
    noise = explicit_noise()
    try:
        wire = server.pack(
            {"request_id": 7, "captured_ns": 19, "task": "task"}, {**observation(), NOISE_ARRAY: noise}
        )
        code, headers, body = running.post(wire)
        header, arrays = server.unpack(body)
        assert code == 200 and headers["X-Trinity-Owner"] == OWNER
        assert headers["X-Trinity-Generation"] == "1"
        assert header["request_id"] == 7 and header["captured_ns"] == 19 and header["error"] is None
        assert set(arrays) == {"actions", "predicted_actions"}
        assert arrays["actions"].shape == (4, 7) and arrays["predicted_actions"].shape == (50, 7)
        metadata = header["policy_provenance"]
        assert metadata["noise_sha256"] == array_sha256(noise)
        assert metadata["predicted_sha256"] == array_sha256(arrays["predicted_actions"])
        assert metadata["actions_sha256"] == array_sha256(arrays["actions"])
        assert len(calls) == 1
    finally:
        running.close()


def test_foreign_owned_request_cannot_reach_the_noise_policy(policy: Any) -> None:
    p, _, calls = policy
    running = RunningServer()
    running.http.RequestHandlerClass = server.make_handler(p, running.lock, OWNER)
    try:
        wire = server.pack({"request_id": 7}, {**observation(), NOISE_ARRAY: explicit_noise()})
        code, headers, body = running.post(wire, owner="b" * 32)
        assert code == 409 and "X-Trinity-Owner" not in headers
        assert "policy_provenance" not in server.unpack(body)[0] and calls == []
    finally:
        running.close()


def test_unsupported_http_provider_cannot_silently_ignore_noise() -> None:
    running = RunningServer()
    try:
        wire = server.pack({"request_id": 7}, {NOISE_ARRAY: explicit_noise()})
        code, _, body = running.post(wire)
        assert code == 409 and "does not support explicit noise" in server.unpack(body)[0]["error"]
        assert running.model.calls == 0
    finally:
        running.close()
