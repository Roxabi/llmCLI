"""RED tests for LifecycleMixin engine=remote rejection — issue #34, Slice 2, T13.

These tests MUST FAIL until _lifecycle.py and roxabi_contracts.llm are implemented
(Wave 2, T17; Slice 1, T7/T9). Imports will fail at collection time — expected RED.

Spec trace: C3, E4
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from llmcli.nats._lifecycle import LifecycleMixin
from roxabi_contracts.llm import LifecycleRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_swap_request(model_name: str = "qwen3-remote") -> LifecycleRequest:
    return LifecycleRequest(
        contract_version="1",
        trace_id="trace-remote-reject",
        issued_at="2026-05-21T00:00:00Z",
        request_id="req-remote-001",
        op="swap",
        model_name=model_name,
        host=None,
    )


def _make_fake_msg(reply: str = "_inbox.llm-operator.test") -> SimpleNamespace:
    return SimpleNamespace(reply=reply, subject="lyra.llm.lifecycle.swap")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.nats
async def test_swap_remote_engine_returns_lifecycle_rejected():
    """Swapping a model with engine='remote' replies with llm.lifecycle_rejected.

    Arrange: catalog contains a model with engine='remote'. Publish lifecycle.swap
    for that model.
    Assert: reply has worker_error.code == 'llm.lifecycle_rejected' and
    retryable == False. No engine start is attempted.

    Negative: removing the engine=='remote' guard in _do_swap causes this test
    to fail — the code would attempt to start a remote engine locally, either
    crashing or returning an unexpected success.
    """

    class TestAdapter(LifecycleMixin):
        def __init__(self):
            self.__init_lifecycle__()
            self._sem = asyncio.Semaphore(2)
            self.drain_timeout = 5.0
            self._instances: dict = {}
            # Catalog with one remote-engine model
            remote_spec = MagicMock()
            remote_spec.engine = "remote"
            self._catalog = MagicMock()
            self._catalog.models = {"qwen3-remote": remote_spec}
            self._catalog.host = MagicMock()
            self._engine_start_calls: list = []

        async def _wait_sem_idle(self) -> None:
            pass

        def _engine_for_spec(self, spec):
            engine = MagicMock()
            engine.start = MagicMock(side_effect=lambda s: self._engine_start_calls.append(s))
            return engine

        async def _reply_ok(self, msg, req, *, data=None):
            self._reply_data = {"ok": True, "data": data}

        async def _reply_err(self, msg, req, code, message, *, retryable=True):
            self._reply_data = {
                "ok": False,
                "worker_error": {"code": code, "message": message, "retryable": retryable},
            }

    adapter = TestAdapter()
    req = _make_swap_request(model_name="qwen3-remote")
    msg = _make_fake_msg()

    await adapter._do_swap(msg, req)

    # Must have replied with an error
    assert hasattr(adapter, "_reply_data"), "_do_swap did not call _reply_ok or _reply_err"
    assert adapter._reply_data["ok"] is False, (
        "Expected rejection for engine='remote' model, but got ok=True"
    )

    worker_error = adapter._reply_data.get("worker_error", {})

    assert worker_error.get("code") == "llm.lifecycle_rejected", (
        f"Expected code 'llm.lifecycle_rejected', got: {worker_error.get('code')!r}"
    )
    assert worker_error.get("retryable") is False, (
        "Remote engine rejection must be retryable=False (client-side misconfiguration)"
    )

    # No engine start must have been attempted
    assert adapter._engine_start_calls == [], (
        f"Engine.start() was called despite engine='remote' guard: {adapter._engine_start_calls}"
    )
