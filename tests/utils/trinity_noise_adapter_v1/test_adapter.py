"""Counterfactual capture identity and lossless request/reply ownership."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from trinity_noise_v2 import test_noise as noise_cases

from lerobot.utils import trinity_http_server as server, trinity_libero_noise_adapter as adapter

policy = noise_cases.policy


def context(tick: int = 40, **changes: Any) -> dict[str, Any]:
    return dict(
        schema=adapter.CONTEXT_SCHEMA,
        seed=20261112,
        task_id=0,
        state_id=1,
        episode=2,
        phase="control",
        tick=tick,
        model="pi05",
        **changes,
    )


def observation(tick: int = 40, captured: int = 19) -> dict[str, Any]:
    return {"raw": noise_cases.observation(), "captured_ns": captured, "policy_noise_context": context(tick)}


def test_different_request_cadence_evaluates_the_same_exogenous_field() -> None:
    sparse = adapter.ControlledLeRobotLiberoAdapter({})
    dense = adapter.ControlledLeRobotLiberoAdapter({})
    sparse.episode("task")
    dense.episode("task")
    a, b = {}, {}
    for ordinal, tick in enumerate([0, 40, 80]):
        a[tick] = adapter.unpack(sparse.encode(observation(tick, 20 + tick), ordinal + 1))[1]["policy_noise"]
    for ordinal, tick in enumerate([80, 60, 40, 20, 0]):
        b[tick] = adapter.unpack(dense.encode(observation(tick, 999 + tick), ordinal + 100))[1][
            "policy_noise"
        ]
    assert all(np.array_equal(a[tick], b[tick]) for tick in a)
    assert not np.array_equal(b[0], b[20])


@pytest.mark.parametrize("field", ["seed", "task_id", "state_id", "episode", "tick"])
def test_independent_semantic_identities_get_distinct_noise(field: str) -> None:
    first = context()
    second = {**first, field: first[field] + 1}
    assert not np.array_equal(adapter.policy_noise(first, 50, 32), adapter.policy_noise(second, 50, 32))


def test_field_does_not_consume_the_shared_numpy_generator() -> None:
    before = np.random.get_state()
    adapter.policy_noise(context(), 50, 32)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]


@pytest.mark.parametrize(
    "change",
    [{"arm": "protected"}, {"request_id": 2}, {"captured_ns": 10}, {"tick": True}, {"phase": "prefill"}],
)
def test_outcome_or_transport_keys_cannot_replace_semantic_capture_context(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        adapter.policy_noise({**context(), **change}, 50, 32)


def encoded_reply(p: Any, client: Any, request_id: int = 7) -> tuple[bytes, bytes]:
    p.policy.predict_action_chunk = lambda batch, **options: options["noise"]
    wire = client.encode(observation(), request_id)
    header, arrays = server.unpack(wire)
    actions = p.chunk("task", arrays)
    reply = server.pack(
        {
            "request_id": request_id,
            "captured_ns": header["captured_ns"],
            "error": None,
            "policy_provenance": p.last_inference_metadata,
        },
        {"actions": actions, "predicted_actions": p.last_policy_chunk},
    )
    return wire, reply


def test_real_chunk_interface_keeps_full_predictions_and_lossless_wire_evidence(
    policy: Any, tmp_path: Path
) -> None:
    client = adapter.ControlledLeRobotLiberoAdapter({})
    client.episode("task")
    wire, reply = encoded_reply(policy[0], client)
    actions = client.decode(reply, 7)
    assert actions.shape == (50, 7)
    assert adapter.unpack(reply)[1]["predicted_actions"].shape == (50, 32)
    client.close_episode(tmp_path)
    root = tmp_path / "policy-provenance"
    assert (root / "7.request.bin").read_bytes() == wire
    assert (root / "7.reply.bin").read_bytes() == reply
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["records"][0]["context"] == context() and manifest["records"][0]["verified"]
    client.episode("next episode")


@pytest.mark.parametrize("bad", ["noise", "observation", "capture", "prediction", "configuration"])
def test_stale_or_mismatched_provenance_is_rejected_and_preserved(
    policy: Any, tmp_path: Path, bad: str
) -> None:
    client = adapter.ControlledLeRobotLiberoAdapter({})
    client.episode("task")
    _, reply = encoded_reply(policy[0], client)
    header, arrays = server.unpack(reply)
    metadata = header["policy_provenance"]
    if bad == "noise":
        metadata["noise_sha256"] = "0" * 64
    elif bad == "observation":
        metadata["raw_observation_sha256"] = "0" * 64
    elif bad == "capture":
        header["captured_ns"] += 1
    elif bad == "prediction":
        arrays["predicted_actions"] = arrays["predicted_actions"].copy()
        arrays["predicted_actions"][0, 0] += 1
    else:
        metadata["policy_config"]["chunk_size"] += 1
    rejected = server.pack(header, arrays)
    with pytest.raises(ValueError):
        client.decode(rejected, 7)
    client.close_episode(tmp_path)
    assert (tmp_path / "policy-provenance/7.reply.bin").read_bytes() == rejected


def test_provenance_has_explicit_memory_and_episode_ownership_bounds(tmp_path: Path) -> None:
    client = adapter.ControlledLeRobotLiberoAdapter({"max_records": 1})
    client.episode("task")
    client.encode(observation(), 1)
    with pytest.raises(RuntimeError, match="previous episode"):
        client.episode("foreign task")
    with pytest.raises(RuntimeError, match="allocation exhausted"):
        client.encode(observation(80), 2)
    client.close_episode(tmp_path)
    small = adapter.ControlledLeRobotLiberoAdapter({"max_bytes": 1})
    with pytest.raises(RuntimeError, match="allocation exhausted"):
        small.encode(observation(), 1)


def test_adapter_can_load_by_path_without_importing_lerobot() -> None:
    # The independent scheduler loads this file, without adopting child imports.
    spec = importlib.util.spec_from_file_location("standalone_noise_adapter", adapter.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.ControlledLeRobotLiberoAdapter({}).chunk == 50
