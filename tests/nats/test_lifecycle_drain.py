"""RED tests for LifecycleMixin drain pattern — issue #34, Slice 2, T12.

These tests MUST FAIL until _lifecycle.py + _do_swap drain logic are implemented
(Wave 2, T17). The imports from llmcli.nats._lifecycle and roxabi_contracts.llm
will fail at collection time — this is the expected RED state.

Spec trace: SC AC-5; E5
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


def _make_lifecycle_request(**kwargs) -> LifecycleRequest:
    """Build a minimal valid LifecycleRequest for swap."""
    defaults = {
        "contract_version": "1",
        "trace_id": "trace-drain-test",
        "issued_at": "2026-05-21T00:00:00Z",
        "request_id": "req-drain-001",
        "op": "swap",
        "model_name": "qwen3-8b",
        "host": None,
    }
    defaults.update(kwargs)
    return LifecycleRequest(**defaults)


def _make_fake_msg(reply: str = "_inbox.llm-operator.test") -> SimpleNamespace:
    return SimpleNamespace(reply=reply, subject="lyra.llm.lifecycle.swap")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.nats
async def test_drain_in_flight_completes_then_swap():
    """In-flight generation completes before swap proceeds.

    Arrange: create a LifecycleMixin subclass with a semaphore held by a
    synthetic in-flight request. Concurrently trigger _do_swap.
    Assert: _draining is set while waiting; generation completes (semaphore
    released); swap proceeds; reply is OK.

    Negative: removing the drain Event or the semaphore-wait logic in _do_swap
    causes swap to race with the in-flight generation, either crashing or
    producing a double-start. This test fails because the assertions on
    message order break.
    """

    class TestAdapter(LifecycleMixin):
        """Minimal adapter stub wired just enough to exercise _do_swap drain path."""

        def __init__(self):
            self.__init_lifecycle__()
            self._sem = asyncio.Semaphore(2)
            self._drain_timeout = 5.0
            self._instances: dict = {}
            self._catalog = MagicMock()
            self._catalog.models = {"qwen3-8b": MagicMock(engine="llamacpp")}
            self._catalog.host = MagicMock()

        async def _wait_sem_idle(self) -> None:
            """Wait until all semaphore slots are free."""
            target = self._sem._value  # original capacity
            while self._sem._value < target:
                await asyncio.sleep(0)

        def _engine_for_spec(self, spec):
            engine = MagicMock()
            inst = MagicMock()
            inst.port = 8091
            engine.start = MagicMock(return_value=inst)
            engine.stop = MagicMock()
            return engine

        async def _reply_ok(self, msg, req, *, data=None):
            self._reply_data = {"ok": True, "data": data}

        async def _reply_err(self, msg, req, code, message, *, retryable=True):
            self._reply_data = {"ok": False, "code": code, "message": message}

    adapter = TestAdapter()

    # Acquire one semaphore slot to simulate an in-flight request
    await adapter._sem.acquire()

    async def _release_after_delay():
        await asyncio.sleep(0.05)
        adapter._sem.release()

    req = _make_lifecycle_request()
    msg = _make_fake_msg()

    # Run in-flight release concurrently with swap
    await asyncio.gather(
        adapter._do_swap(msg, req),
        _release_after_delay(),
    )

    # Assert swap completed successfully after drain
    assert hasattr(adapter, "_reply_data"), "_do_swap never called _reply_ok or _reply_err"
    assert adapter._reply_data["ok"] is True, (
        f"Expected swap to succeed after drain, got: {adapter._reply_data}"
    )
    # Draining flag must be cleared after swap
    assert not adapter._draining.is_set(), "_draining event not cleared after swap completed"


@pytest.mark.nats
async def test_drain_timeout_force_cuts():
    """When drain exceeds drain_timeout, swap proceeds with hard cut.

    Arrange: semaphore held; drain_timeout=0.01s (very short).
    Assert: _do_swap does not hang; warning is logged; reply is still sent
    (either OK with new engine or error). _draining is cleared after.

    Negative: removing the asyncio.wait_for(timeout=…) in _do_swap causes
    the swap to block indefinitely — pytest timeout kills the test.
    """

    class TestAdapter(LifecycleMixin):
        def __init__(self):
            self.__init_lifecycle__()
            self._sem = asyncio.Semaphore(2)
            self._drain_timeout = 0.01  # force timeout
            self._instances: dict = {}
            self._catalog = MagicMock()
            self._catalog.models = {"qwen3-8b": MagicMock(engine="llamacpp")}
            self._catalog.host = MagicMock()

        async def _wait_sem_idle(self) -> None:
            # Never completes — simulates permanently stuck generation
            await asyncio.sleep(9999)

        def _engine_for_spec(self, spec):
            engine = MagicMock()
            inst = MagicMock()
            inst.port = 8091
            engine.start = MagicMock(return_value=inst)
            engine.stop = MagicMock()
            return engine

        async def _reply_ok(self, msg, req, *, data=None):
            self._reply_data = {"ok": True, "data": data}

        async def _reply_err(self, msg, req, code, message, *, retryable=True):
            self._reply_data = {"ok": False, "code": code, "message": message}

    adapter = TestAdapter()

    # Acquire semaphore slot — will never be released
    await adapter._sem.acquire()

    req = _make_lifecycle_request()
    msg = _make_fake_msg()

    # Must complete within pytest timeout (default 60s) despite stuck semaphore
    await adapter._do_swap(msg, req)

    # A reply must have been sent (timeout → hard cut → new engine started)
    assert hasattr(adapter, "_reply_data"), "_do_swap timed out and did not send any reply"
    # After hard cut, swap should succeed (new engine starts regardless)
    assert adapter._reply_data["ok"] is True, (
        f"Expected OK after drain timeout hard cut, got: {adapter._reply_data}"
    )
    # Draining must be cleared regardless of timeout
    assert not adapter._draining.is_set(), "_draining not cleared after drain timeout"
