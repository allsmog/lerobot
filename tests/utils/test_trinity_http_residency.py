"""Exercise the owned HTTP server without loading any model or using a device."""

from __future__ import annotations

import http.client
import http.server
import json
import threading
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.utils import trinity_http_server as server

OWNER = "a" * 32


class InertPolicy:
    residency_enabled = True

    def __init__(self) -> None:
        self.residency_state, self.residency_epoch = "resident", 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.calls, self.reserves = 0, []
        self.fail_park = False

    def chunk(self, task: str, arrays: dict[str, np.ndarray]) -> np.ndarray:
        if self.residency_state != "resident":
            raise RuntimeError("not resident")
        self.calls += 1
        self.entered.set()
        if not self.release.wait(2):
            raise TimeoutError("owned CPU fixture bound")
        return np.arange(14, dtype=np.float32).reshape(2, 7)

    def residency(self, action: str) -> dict[str, Any]:
        if action not in ("status", "park", "resume"):
            raise ValueError("unknown operation")
        if self.fail_park and action == "park":
            self.residency_state = "fault"
            raise RuntimeError("controlled migration failure")
        target = {"park": "parked", "resume": "resident"}.get(action)
        if target is not None and target != self.residency_state:
            self.residency_epoch += 1
            self.residency_state = target
        return {"state": self.residency_state, "epoch": self.residency_epoch}

    def prepare_residency_shadow(self, reserve: int) -> dict[str, int]:
        self.reserves.append(reserve)
        return {"host_reserve_bytes": reserve}


class RunningServer:
    def __init__(self) -> None:
        self.model = InertPolicy()
        self.lock = threading.Lock()
        self.http = http.server.ThreadingHTTPServer(
            ("127.0.0.1", 0), server.make_handler(self.model, self.lock, OWNER)
        )
        self.thread = threading.Thread(target=lambda: self.http.serve_forever(poll_interval=0.01))
        self.thread.start()

    def post(
        self, body: bytes, path: str = "/", owner: str = OWNER, generation: int = 1, length: int | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection("127.0.0.1", self.http.server_address[1], timeout=2)
        headers = {"X-Trinity-Owner": owner, "X-Trinity-Generation": str(generation)}
        if length is not None:
            headers["Content-Length"] = str(length)
        try:
            conn.request("POST", path, body, headers)
            reply = conn.getresponse()
            return reply.status, dict(reply.getheaders()), reply.read()
        finally:
            conn.close()

    def control(self, action: str, generation: int = 1) -> dict[str, Any]:
        code, _, body = self.post(
            json.dumps({"action": action}).encode(), "/trinity/residency", generation=generation
        )
        assert code == 200
        return json.loads(body)

    def close(self) -> None:
        self.model.release.set()
        self.http.shutdown()
        self.http.server_close()
        self.thread.join(2)
        assert not self.thread.is_alive()


@pytest.fixture
def running() -> Iterator[RunningServer]:
    instance = RunningServer()
    try:
        yield instance
    finally:
        instance.close()


def payload() -> bytes:
    return server.pack({"request_id": 7, "captured_ns": 17, "task": "task"}, {"image": np.zeros(2, np.uint8)})


def test_original_binary_envelope_is_preserved(running: RunningServer) -> None:
    code, headers, body = running.post(payload())
    expected = server.pack(
        {"request_id": 7, "captured_ns": 17, "error": None},
        {"actions": np.arange(14, dtype=np.float32).reshape(2, 7)},
    )
    assert code == 200 and body == expected
    assert headers["X-Trinity-Owner"] == OWNER
    assert headers["X-Trinity-Generation"] == "1"
    assert running.model.calls == 1


def test_park_resume_epochs_are_idempotent(running: RunningServer) -> None:
    assert running.control("status")["generation"] == 1
    assert running.control("park")["generation"] == 2
    assert running.control("park", 2)["generation"] == 2
    code, _, body = running.post(payload(), generation=2)
    assert code == 409 and server.unpack(body)[0]["error"].endswith("not resident")
    assert running.control("resume", 2)["generation"] == 3
    assert running.control("resume", 3)["generation"] == 3
    assert running.post(payload(), generation=3)[0] == 200


def test_foreign_owner_and_stale_generation_cannot_mutate(running: RunningServer) -> None:
    command = b'{"action":"park"}'
    code, headers, body = running.post(command, "/trinity/residency", owner="b" * 32)
    assert code == 409 and OWNER.encode() not in body
    assert "X-Trinity-Owner" not in headers
    assert running.model.residency_state == "resident"
    assert running.control("park")["generation"] == 2
    code, _, _ = running.post(b'{"action":"resume"}', "/trinity/residency", generation=1)
    assert code == 409 and running.model.residency_state == "parked"


def test_invalid_control_bodies_do_not_change_residency(running: RunningServer) -> None:
    for body, length in [(b"[]", None), (b"{", None), (b"", 65537), (b"", 0)]:
        assert running.post(body, "/trinity/residency", length=length)[0] == 409
        assert running.model.residency_state == "resident"
        assert running.model.residency_epoch == 0


def test_cache_reserve_is_explicit_and_does_not_increment_epoch(running: RunningServer) -> None:
    reserve = 14 * 1024**3
    code, _, body = running.post(
        json.dumps({"action": "prepare_shadow", "host_reserve_bytes": reserve}).encode(), "/trinity/residency"
    )
    result = json.loads(body)
    assert code == 200 and result["generation"] == 1
    assert result["residency"]["host_reserve_bytes"] == reserve
    assert running.model.reserves == [reserve]


def test_transfer_failure_refuses_later_inference(running: RunningServer) -> None:
    running.model.fail_park = True
    assert running.post(b'{"action":"park"}', "/trinity/residency")[0] == 409
    assert running.model.residency_state == "fault"
    assert running.post(payload())[0] == 409
    assert running.model.calls == 0


def test_inference_lock_serializes_transfer_and_keeps_reply_generation(running: RunningServer) -> None:
    running.model.release.clear()
    inference, control, errors = [], [], []
    control_done = threading.Event()

    def infer() -> None:
        try:
            inference.append(running.post(payload()))
        except Exception as error:
            errors.append(error)

    def park() -> None:
        try:
            control.append(running.control("park"))
        except Exception as error:
            errors.append(error)
        finally:
            control_done.set()

    threads = [threading.Thread(target=infer), threading.Thread(target=park)]
    threads[0].start()
    try:
        assert running.model.entered.wait(1)
        assert running.lock.locked()
        threads[1].start()
        assert not control_done.wait(0.02)
    finally:
        running.model.release.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(2)
                assert not thread.is_alive()
    assert not errors
    assert inference[0][0] == 200 and inference[0][1]["X-Trinity-Generation"] == "1"
    assert control[0]["generation"] == 2 and running.model.residency_state == "parked"


def test_owner_is_checked_before_policy_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRINITY_LEROBOT_OWNER_NONCE", raising=False)
    with (
        patch.object(server, "LiberoChunkPolicy", side_effect=AssertionError("must not load policy")),
        pytest.raises(ValueError, match="owner nonce"),
    ):
        server.main(["--policy-path", "unused", "--port", "7460", "--residency"])
