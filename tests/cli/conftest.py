"""Shared fixtures for tests/cli/.

The CLI NATS paths fail-close when the operator creds file is absent (B7).
For unit tests that monkeypatch the NATS client, we never touch the wire —
opt out of the creds check at the env level so tests focus on subject/payload
shape rather than infra-setup state.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _allow_anonymous_nats(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default CLI tests to LLMCLI_NATS_SKIP_CREDS=1.

    A test that wants to exercise the fail-closed path explicitly removes the
    env var with monkeypatch.delenv before invoking the CLI.
    """
    monkeypatch.setenv("LLMCLI_NATS_SKIP_CREDS", "1")
