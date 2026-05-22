"""Integration tests for LifecycleMixin crash recovery — issue #34, Slice 5, T37.

E10 scenario: engine fails mid-swap → adapter survives → status returns empty →
subsequent swap succeeds.

Tests marked @pytest.mark.nats skip without a NATS broker (CI provides one via T38).
The in-process crash injection does NOT require a broker — it patches _engine_for_spec
to raise mid-call.  The @pytest.mark.nats annotation is required because these tests
are part of the Slice 5 integration suite that runs together with broker tests.

Spec trace: E10 (crash recovery), SC AC-1
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcli.nats._lifecycle import LifecycleMixin
from roxabi_contracts.llm import LifecycleRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(op: str, model_name: str | None = None) -> LifecycleRequest:
    return LifecycleRequest(
        contract_version="1",
        trace_id="trace-crash-recovery",
        issued_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        request_id=f"req-crash-{op}-001",
        op=op,
        model_name=model_name,
        host=None,
    )


def _make_msg(subject: str = "lyra.llm.lifecycle.swap") -> SimpleNamespace:
    return SimpleNamespace(reply="_inbox.llm-operator.crash", subject=subject)


def _make_instance(port: int = 8091) -> MagicMock:
    inst = MagicMock()
    inst.port = port
    return inst


def _make_spec(engine: str = "llamacpp", name: str = "model", vram_gib: float = 8.0) -> MagicMock:
    spec = MagicMock()
    spec.engine = engine
    spec.name = name
    spec.vram_gib = vram_gib
    return spec


class _CrashRecoveryAdapter(LifecycleMixin):
    """Adapter designed to test mid-swap crash recovery.

    Exposes `engine_mode` to configure whether start() succeeds or raises,
    enabling the test to flip from crash → success for the recovery scenario.
    """

    def __init__(
        self,
        catalog_models: dict,
        initial_instances: dict | None = None,
        start_port: int = 8091,
    ) -> None:
        self.__init_lifecycle__()
        self._sem = asyncio.Semaphore(2)
        self._max_concurrent = 2
        self.drain_timeout = 5.0
        self._instances: dict = initial_instances or {}
        self._nc = MagicMock()
        self._nc.publish = AsyncMock()
        self._vram_monitor = None

        self._catalog = MagicMock()
        self._catalog.host = MagicMock()
        self._catalog.host.vram_budget_gib = None
        self._catalog.models = catalog_models

        self._start_port = start_port

        # Control flag: set to True to make engine.start() raise
        self.engine_start_should_crash: bool = False
        self.engine_start_calls: list[str] = []
        self.engine_stop_calls: list[str] = []

        self._reply_ok_calls: list[dict] = []
        self._reply_err_calls: list[dict] = []

    async def _wait_sem_idle(self) -> None:
        pass

    def _engine_for_spec(self, spec):
        engine = MagicMock()
        inst = _make_instance(port=self._start_port)

        def _start(s):
            name = getattr(s, "name", str(s))
            self.engine_start_calls.append(name)
            if self.engine_start_should_crash:
                raise RuntimeError(f"GPU OOM starting {name}")
            return inst

        def _stop(i):
            self.engine_stop_calls.append(str(i))

        engine.start = MagicMock(side_effect=_start)
        engine.stop = MagicMock(side_effect=_stop)
        return engine

    async def _reply_ok(self, msg, req, *, data=None) -> None:
        self._reply_ok_calls.append({"msg": msg, "req": req, "data": data})

    async def _reply_err(self, msg, req, code, message, *, retryable=True) -> None:
        self._reply_err_calls.append(
            {
                "msg": msg,
                "req": req,
                "code": code,
                "message": message,
                "retryable": retryable,
            }
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.nats
class TestCrashRecovery:
    """E10: adapter survives engine crash mid-swap; subsequent swap succeeds."""

    @pytest.mark.asyncio
    async def test_worker_survives_engine_crash_during_swap(self) -> None:
        """When engine.start() raises mid-swap, the worker does not crash.

        Arrange: adapter with a model in catalog; engine.start() raises RuntimeError.
        Act: call _do_swap.
        Assert: _reply_err sent (worker.crash); no exception escapes _do_swap.

        Negative: removing the try/except around engine.start() in _do_swap lets the
        RuntimeError propagate, terminating the asyncio task and crashing the worker.
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _CrashRecoveryAdapter(catalog_models={"qwen3-8b": spec})
        adapter.engine_start_should_crash = True

        req = _make_request("swap", model_name="qwen3-8b")
        msg = _make_msg()

        # Act — must NOT raise
        await adapter._do_swap(msg, req)

        # Assert — error reply sent; worker alive
        assert len(adapter._reply_err_calls) == 1, (
            f"Expected worker.crash reply after engine crash, got: {adapter._reply_err_calls}"
        )
        assert adapter._reply_err_calls[0]["code"] == "worker.crash"
        assert adapter._reply_err_calls[0]["retryable"] is True

    @pytest.mark.asyncio
    async def test_status_returns_empty_after_crash(self) -> None:
        """After a crash during swap, _do_status reports model=None (empty state).

        Arrange: crash the first swap (engine fails); check status after.
        Assert: status data has model=None, confirming the adapter is in a clean
        empty state (old engine stopped, new engine not started).

        Negative: if instances were not cleared after the crash (missing
        `del instances[old_name]`), status would incorrectly report the old model
        as running even though its process was already killed.
        """
        # Arrange — old model was running; swap to new model crashes
        old_spec = _make_spec(engine="llamacpp", name="qwen3-4b")
        new_spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        old_inst = _make_instance(port=8091)

        adapter = _CrashRecoveryAdapter(
            catalog_models={"qwen3-4b": old_spec, "qwen3-8b": new_spec},
            initial_instances={"qwen3-4b": old_inst},
        )
        adapter.engine_start_should_crash = True

        swap_req = _make_request("swap", model_name="qwen3-8b")
        swap_msg = _make_msg()

        # Act — failed swap
        await adapter._do_swap(swap_msg, swap_req)

        # Clear state and check status
        adapter._reply_ok_calls.clear()
        adapter._reply_err_calls.clear()

        status_req = _make_request("status")
        status_msg = _make_msg("lyra.llm.lifecycle.status")
        await adapter._do_status(status_msg, status_req)

        # Assert — model=None (empty state)
        assert len(adapter._reply_ok_calls) == 1, (
            f"Expected _do_status to reply ok after crash, got: {adapter._reply_ok_calls}"
        )
        data = adapter._reply_ok_calls[0]["data"]
        assert data == {"model": None}, f"After crash, status must be {{model: None}}, got: {data}"

    @pytest.mark.asyncio
    async def test_subsequent_swap_succeeds_after_crash(self) -> None:
        """After a failed swap, a follow-up swap with the same model succeeds.

        This is the full E10 recovery loop:
        1. Initial state: no models loaded
        2. Swap attempt: engine crashes → worker.crash reply, instances empty
        3. Recovery swap: engine now succeeds → ok reply, model registered

        Negative: if _draining is not cleared in the crash path (missing finally block),
        the adapter remains in draining mode and the recovery swap hangs waiting for
        semaphore idle.
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _CrashRecoveryAdapter(
            catalog_models={"qwen3-8b": spec},
            start_port=8091,
        )
        # First swap crashes
        adapter.engine_start_should_crash = True

        req1 = _make_request("swap", model_name="qwen3-8b")
        msg1 = _make_msg()
        await adapter._do_swap(msg1, req1)

        assert len(adapter._reply_err_calls) == 1, "First swap must have failed"
        assert adapter._reply_err_calls[0]["code"] == "worker.crash"

        # Reset for recovery
        adapter.engine_start_should_crash = False
        adapter._reply_ok_calls.clear()
        adapter._reply_err_calls.clear()

        # Act — recovery swap
        req2 = _make_request("swap", model_name="qwen3-8b")
        msg2 = _make_msg()
        await adapter._do_swap(msg2, req2)

        # Assert — recovery succeeded
        assert len(adapter._reply_ok_calls) == 1, (
            f"Expected ok reply on recovery swap, got errs: {adapter._reply_err_calls}"
        )
        data = adapter._reply_ok_calls[0]["data"]
        assert data["model"] == "qwen3-8b", (
            f"Recovery swap must show model='qwen3-8b', got: {data.get('model')!r}"
        )
        assert data["port"] == 8091

    @pytest.mark.asyncio
    async def test_draining_flag_cleared_after_crash(self) -> None:
        """_draining is cleared even when engine.start() raises.

        If _draining remains set after a crash, generation requests that check
        `if adapter._draining.is_set(): wait` will block forever.
        Negative: removing finally: self._draining.clear() causes _draining to stay set.
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _CrashRecoveryAdapter(catalog_models={"qwen3-8b": spec})
        adapter.engine_start_should_crash = True

        req = _make_request("swap", model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert
        assert not adapter._draining.is_set(), (
            "_draining must be cleared after engine.start() crash (finally block)"
        )

    @pytest.mark.asyncio
    async def test_lifecycle_lock_released_after_crash(self) -> None:
        """_lifecycle_lock is released after a crash — subsequent ops can proceed.

        If the lock is not released (e.g. exception bypasses the async with block),
        all subsequent lifecycle operations deadlock.
        Negative: wrapping _do_swap in a bare try/finally outside the lock would
        break this guarantee; the async with ctx manager ensures release on exception.
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _CrashRecoveryAdapter(catalog_models={"qwen3-8b": spec})
        adapter.engine_start_should_crash = True

        req_swap = _make_request("swap", model_name="qwen3-8b")
        msg_swap = _make_msg()

        # Act — crash the swap
        await adapter._dispatch_lifecycle_op("swap", msg_swap, req_swap)

        # Assert — lock is not held (can be acquired immediately)
        # asyncio.Lock.locked() returns True if the lock is currently acquired
        assert not adapter._lifecycle_lock.locked(), (
            "_lifecycle_lock must be released after crash; "
            "if locked, subsequent lifecycle ops will deadlock"
        )

    @pytest.mark.asyncio
    async def test_concurrent_swap_and_crash_recovery(self) -> None:
        """Multiple sequential swap attempts work correctly after a crash.

        Scenario: crash → recover → crash again → recover again.
        Verifies the adapter can cycle through multiple crash/recovery sequences.
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _CrashRecoveryAdapter(catalog_models={"qwen3-8b": spec})

        for cycle in range(2):
            # Crash phase
            adapter.engine_start_should_crash = True
            adapter._reply_ok_calls.clear()
            adapter._reply_err_calls.clear()

            req_fail = _make_request("swap", model_name="qwen3-8b")
            msg_fail = _make_msg()
            await adapter._do_swap(msg_fail, req_fail)

            assert len(adapter._reply_err_calls) == 1, (
                f"Cycle {cycle}: expected crash reply, got: {adapter._reply_err_calls}"
            )
            assert not adapter._draining.is_set(), f"Cycle {cycle}: draining not cleared"

            # Recovery phase
            adapter.engine_start_should_crash = False
            adapter._reply_ok_calls.clear()
            adapter._reply_err_calls.clear()

            req_ok = _make_request("swap", model_name="qwen3-8b")
            msg_ok = _make_msg()
            await adapter._do_swap(msg_ok, req_ok)

            assert len(adapter._reply_ok_calls) == 1, (
                f"Cycle {cycle}: expected ok reply on recovery, "
                f"got errs: {adapter._reply_err_calls}"
            )
            # Clear instances for next cycle (simulate stop between cycles)
            adapter._instances.clear()
