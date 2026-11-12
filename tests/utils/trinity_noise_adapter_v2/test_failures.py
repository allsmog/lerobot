"""Rejected transport evidence remains available without replaying inference."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from trinity_noise_adapter_v1.test_adapter import observation

from lerobot.utils.trinity_libero_noise_adapter import ControlledLeRobotLiberoAdapter


def test_malformed_reply_is_preserved_before_its_header_is_parsed(tmp_path: Path) -> None:
    client = ControlledLeRobotLiberoAdapter({})
    client.encode(observation(), 1)
    malformed = b"\x01\x02"
    with pytest.raises(ValueError, match="shorter"):
        client.decode(malformed, 1)
    client.close_episode(tmp_path)
    root = tmp_path / "policy-provenance"
    assert (root / "1.reply.bin").read_bytes() == malformed
    assert not json.loads((root / "manifest.json").read_text())["records"][0].get("verified")


@pytest.mark.parametrize("request_id", [True, 0, -1])
def test_invalid_request_identity_cannot_enter_the_record_set(request_id: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ControlledLeRobotLiberoAdapter({}).encode(observation(), request_id)
