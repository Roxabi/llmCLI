"""Integration tests for LifecycleMixin._do_swap — issue #34, Slice 5, T35.

These tests are marked @pytest.mark.nats; they will SKIP without a NATS broker.
CI provides a broker via T38.  For local development without a broker, run:
  uv run pytest tests/nats/test_lifecycle_swap.py -m "not nats"
to confirm imports + collection only (no tests to collect — that is correct).

Spec trace: SC AC-1 (swap happy path), E10 (idempotent same-model fast-path)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcli.config import Catalog, HostSettings, ModelSpec
from llmcli.nats._lifecycle import LifecycleMixin
from roxabi_contracts.llm import LifecycleRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_swap_request(model_name: str = "qwen3-8b", host: str | None = None) -> LifecycleRequest:
    return LifecycleRequest(
        contract_version="1",
        trace_id="trace-swap-integration",
        issued_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        request_id="req-swap-001",
        op="swap",
        model_name=model_name,
        host=host,
    )


def _make_status_request() -> LifecycleRequest:
    return LifecycleRequest(
        contract_version="1",
        trace_id="trace-status-integration",
        issued_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        request_id="req-status-001",
        op="status",
        host=None,
    )


def _make_msg(subject: str = "lyra.llm.lifecycle.swap") -> SimpleNamespace:
    return SimpleNamespace(reply="_inbox.llm-operator.test", subject=subject)


def _make_instance(port: int = 8091) -> MagicMock:
    inst = MagicMock()
    inst.port = port
    return inst


class _IntegrationAdapter(LifecycleMixin):
    """Test adapter wired to exercise _do_swap with a real engine mock.

    Extends the minimal stub pattern from test_lifecycle_drain.py to add:
    - Configurable catalog with multiple models
    - Engine start/stop call tracking for assertions
    - _vram_monitor stub for VRAM field assertions
    """

    def __init__(
        self,
        catalog_models: dict | None = None,
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

        # VRAM monitor stub — returns deterministic values
        self._vram_monitor = MagicMock()
        self._vram_monitor.sample = MagicMock(return_value=(8000.0, 4096.0))

        # Catalog
        self._catalog = MagicMock()
        self._catalog.host = MagicMock()
        # host has vram_budget_gib to avoid TypeError in check_vram_budget
        self._catalog.host.vram_budget_gib = None
        self._catalog.models = catalog_models or {}

        # Engine tracking
        self._engine_start_calls: list[str] = []
        self._engine_stop_calls: list[str] = []
        self._start_port = start_port

        # Reply capture
        self._reply_ok_calls: list[dict] = []
        self._reply_err_calls: list[dict] = []

    async def _wait_sem_idle(self) -> None:
        """No in-flight generations in tests — resolves immediately."""
        pass

    def _engine_for_spec(self, spec):
        engine = MagicMock()
        inst = _make_instance(port=self._start_port)

        def _start(s):
            self._engine_start_calls.append(getattr(s, "name", str(s)))
            return inst

        def _stop(i):
            self._engine_stop_calls.append(str(i))

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


def _make_spec(engine: str = "llamacpp", vram_gib: float = 8.0, name: str = "test") -> MagicMock:
    spec = MagicMock()
    spec.engine = engine
    spec.vram_gib = vram_gib
    spec.name = name
    return spec


# ---------------------------------------------------------------------------
# Tests — happy path swap
# ---------------------------------------------------------------------------


@pytest.mark.nats
class TestSwapHappyPath:
    """_do_swap with a valid non-remote model → ok reply with model/port/vram_used_mb.

    These tests are marked @pytest.mark.nats.  Collection succeeds without a broker;
    execution requires one (CI provides it via T38).
    """

    @pytest.mark.asyncio
    async def test_swap_unknown_model_replies_lifecycle_rejected(self) -> None:
        """Swapping an unknown model replies with llm.lifecycle_rejected.

        Negative: removing the `spec is None` guard in _do_swap causes the code to
        proceed with a None spec, raising AttributeError instead of a clean error reply.
        """
        # Arrange
        adapter = _IntegrationAdapter(catalog_models={})
        req = _make_swap_request(model_name="nonexistent-model")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert
        assert len(adapter._reply_err_calls) == 1, (
            f"Expected one _reply_err for unknown model, got: {adapter._reply_err_calls}"
        )
        err = adapter._reply_err_calls[0]
        assert err["code"] == "llm.lifecycle_rejected", (
            f"Expected 'llm.lifecycle_rejected', got: {err['code']!r}"
        )
        assert err["retryable"] is False

    @pytest.mark.asyncio
    async def test_swap_valid_model_replies_ok_with_model_port_vram(self) -> None:
        """Swapping a valid non-remote model replies ok with required data shape.

        Spec: data must contain model, port, vram_used_mb.
        Negative: removing the _reply_ok call at the end of _do_swap causes no reply
        to be sent — callers (CLI/lyra) wait for timeout.
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", vram_gib=8.0, name="qwen3-8b")
        adapter = _IntegrationAdapter(
            catalog_models={"qwen3-8b": spec},
            start_port=8091,
        )
        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1, (
            f"Expected one _reply_ok, got err calls: {adapter._reply_err_calls}"
        )
        data = adapter._reply_ok_calls[0]["data"]
        assert data["model"] == "qwen3-8b", f"Expected model='qwen3-8b', got: {data.get('model')!r}"
        assert data["port"] == 8091, f"Expected port=8091, got: {data.get('port')!r}"
        assert "vram_used_mb" in data, f"Expected 'vram_used_mb' in data, got: {data}"
        assert isinstance(data["vram_used_mb"], int), (
            f"vram_used_mb must be int, got: {type(data['vram_used_mb'])}"
        )

    @pytest.mark.asyncio
    async def test_swap_registers_new_instance_in_instances_dict(self) -> None:
        """After a successful swap, the new model is present in _instances."""
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _IntegrationAdapter(catalog_models={"qwen3-8b": spec})
        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert
        assert "qwen3-8b" in adapter._instances, (
            f"New model must be registered in _instances after swap, got: {adapter._instances}"
        )

    @pytest.mark.asyncio
    async def test_swap_stops_old_model_before_starting_new(self) -> None:
        """When a model is already running, _do_swap stops the old one then starts the new one.

        This exercises the stop-before-start ordering that prevents port conflicts.
        Negative: removing the old-engine stop loop causes the old engine to keep running
        while the new one also tries to bind the same port.
        """
        # Arrange — start with qwen3-4b already running
        old_spec = _make_spec(engine="llamacpp", name="qwen3-4b")
        new_spec = _make_spec(engine="llamacpp", name="qwen3-8b")

        old_inst = _make_instance(port=8091)
        adapter = _IntegrationAdapter(
            catalog_models={"qwen3-4b": old_spec, "qwen3-8b": new_spec},
            initial_instances={"qwen3-4b": old_inst},
        )
        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1
        assert "qwen3-4b" not in adapter._instances, (
            "Old model qwen3-4b must be removed from instances after swap"
        )
        assert "qwen3-8b" in adapter._instances, (
            "New model qwen3-8b must be in instances after swap"
        )

    @pytest.mark.asyncio
    async def test_swap_draining_flag_cleared_after_completion(self) -> None:
        """_draining event is cleared after a successful swap.

        Negative: removing the `self._draining.clear()` in the finally block of
        _do_swap causes subsequent generation requests to be blocked permanently
        (they check _draining before accepting new work).
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _IntegrationAdapter(catalog_models={"qwen3-8b": spec})
        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert
        assert not adapter._draining.is_set(), "_draining must be cleared after swap completes"


# ---------------------------------------------------------------------------
# Tests — idempotent same-model fast-path (E10)
# ---------------------------------------------------------------------------


@pytest.mark.nats
class TestSwapIdempotentFastPath:
    """Same-model swap returns ok immediately without restarting the engine (E10).

    Negative: removing the `if model_name in instances` guard causes the same model
    to be stopped-then-restarted on every swap, adding unnecessary downtime.
    """

    @pytest.mark.asyncio
    async def test_same_model_swap_returns_ok_without_engine_stop(self) -> None:
        """Swapping the already-running model replies ok and does NOT call engine.stop().

        Spec E10: idempotent swap fast-path.
        """
        # Arrange — model already in instances
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        inst = _make_instance(port=8091)
        adapter = _IntegrationAdapter(
            catalog_models={"qwen3-8b": spec},
            initial_instances={"qwen3-8b": inst},
        )
        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert — replied ok
        assert len(adapter._reply_ok_calls) == 1, (
            f"Expected ok reply for idempotent swap, got errs: {adapter._reply_err_calls}"
        )
        data = adapter._reply_ok_calls[0]["data"]
        assert data["model"] == "qwen3-8b"
        # Port is preserved from the already-running instance
        assert data["port"] == 8091

    @pytest.mark.asyncio
    async def test_same_model_swap_does_not_restart_engine(self) -> None:
        """Idempotent swap does not call engine.start() — no disruption to running model."""
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        inst = _make_instance(port=8091)
        adapter = _IntegrationAdapter(
            catalog_models={"qwen3-8b": spec},
            initial_instances={"qwen3-8b": inst},
        )
        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert — no engine.start() was called for the same-model fast-path
        assert adapter._engine_start_calls == [], (
            f"engine.start() must NOT be called for same-model idempotent swap, "
            f"got: {adapter._engine_start_calls}"
        )

    @pytest.mark.asyncio
    async def test_status_after_swap_shows_new_model(self) -> None:
        """After swapping from model-A to model-B, _do_status reflects model-B.

        This is the post-swap state check from the integration scenario: swap succeeds
        → status correctly reports the new model.
        """
        # Arrange — start with model-a, swap to model-b
        spec_a = _make_spec(engine="llamacpp", name="qwen3-4b")
        spec_b = _make_spec(engine="llamacpp", name="qwen3-8b")
        inst_a = _make_instance(port=8091)

        adapter = _IntegrationAdapter(
            catalog_models={"qwen3-4b": spec_a, "qwen3-8b": spec_b},
            initial_instances={"qwen3-4b": inst_a},
            start_port=8092,
        )
        swap_req = _make_swap_request(model_name="qwen3-8b")
        swap_msg = _make_msg()

        # Act — swap to new model
        await adapter._do_swap(swap_msg, swap_req)

        # Assert — clear reply_ok calls, then check status
        adapter._reply_ok_calls.clear()
        status_req = _make_status_request()
        status_msg = _make_msg("lyra.llm.lifecycle.status")
        await adapter._do_status(status_msg, status_req)

        assert len(adapter._reply_ok_calls) == 1
        status_data = adapter._reply_ok_calls[0]["data"]
        assert status_data["model"] == "qwen3-8b", (
            f"Status after swap must show new model 'qwen3-8b', got: {status_data['model']!r}"
        )
        assert status_data["port"] == 8092, (
            f"Status must show new port 8092, got: {status_data['port']!r}"
        )


# ---------------------------------------------------------------------------
# Tests — remote engine rejection
# ---------------------------------------------------------------------------


@pytest.mark.nats
class TestSwapRemoteEngineRejection:
    """engine=remote models are rejected with lifecycle_rejected (not started locally)."""

    @pytest.mark.asyncio
    async def test_swap_remote_engine_replies_lifecycle_rejected(self) -> None:
        """Swapping a model with engine='remote' replies with llm.lifecycle_rejected.

        Mirrors test_lifecycle_remote_reject.py::test_swap_remote_engine_returns_lifecycle_rejected
        but runs within the integration adapter scaffold for consistency.
        Negative: removing the engine=='remote' guard in _do_swap causes the adapter
        to attempt starting a remote model locally — either crashes or silently misroutes.
        """
        # Arrange
        remote_spec = _make_spec(engine="remote", name="qwen3-remote")
        adapter = _IntegrationAdapter(catalog_models={"qwen3-remote": remote_spec})
        req = _make_swap_request(model_name="qwen3-remote")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert
        assert len(adapter._reply_err_calls) == 1, (
            f"Expected rejection for engine='remote', got: {adapter._reply_err_calls}"
        )
        err = adapter._reply_err_calls[0]
        assert err["code"] == "llm.lifecycle_rejected"
        assert err["retryable"] is False

    @pytest.mark.asyncio
    async def test_swap_remote_engine_does_not_call_engine_start(self) -> None:
        """Remote model rejection must not attempt engine.start().

        Negative: if the guard is removed, engine.start() is called for a remote model,
        causing the local worker to try to start a model it does not manage.
        """
        # Arrange
        remote_spec = _make_spec(engine="remote", name="qwen3-remote")
        adapter = _IntegrationAdapter(catalog_models={"qwen3-remote": remote_spec})
        req = _make_swap_request(model_name="qwen3-remote")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert
        assert adapter._engine_start_calls == [], (
            f"engine.start() must NOT be called for engine='remote' swap, "
            f"got: {adapter._engine_start_calls}"
        )


# ---------------------------------------------------------------------------
# Tests — engine crash recovery (simplified, no broker)
# ---------------------------------------------------------------------------


class TestSwapEngineStartFailure:
    """Engine.start() raises mid-swap → reply error; draining cleared; adapter survives.

    These tests do NOT require a broker (no @pytest.mark.nats) — they exercise the
    error path of _do_swap using the in-process stub adapter.  Full crash/recovery
    with NATS round-trip is in test_lifecycle_crash_recovery.py.
    """

    @pytest.mark.asyncio
    async def test_engine_start_failure_replies_worker_crash(self) -> None:
        """If engine.start() raises, _do_swap replies with worker.crash.

        Negative: removing the try/except around engine.start() in _do_swap causes
        the exception to propagate unhandled, leaving no reply and _draining set.
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _IntegrationAdapter(catalog_models={"qwen3-8b": spec})

        # Override _engine_for_spec to raise on start
        def _crashing_engine(s):
            engine = MagicMock()
            engine.stop = MagicMock()
            engine.start = MagicMock(side_effect=RuntimeError("GPU OOM"))
            return engine

        adapter._engine_for_spec = _crashing_engine  # type: ignore[method-assign]

        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert — error reply sent
        assert len(adapter._reply_err_calls) == 1, (
            f"Expected one _reply_err on engine crash, got: {adapter._reply_err_calls}"
        )
        err = adapter._reply_err_calls[0]
        assert err["code"] == "worker.crash", f"Expected 'worker.crash' code, got: {err['code']!r}"
        assert err["retryable"] is True, "worker.crash must be retryable=True"

    @pytest.mark.asyncio
    async def test_engine_start_failure_clears_draining_flag(self) -> None:
        """_draining is cleared even when engine.start() raises (finally block).

        Negative: removing the finally: self._draining.clear() in _do_swap causes
        the draining event to remain set after a crash — all subsequent generation
        requests are blocked forever.
        """
        # Arrange
        spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        adapter = _IntegrationAdapter(catalog_models={"qwen3-8b": spec})

        def _crashing_engine(s):
            engine = MagicMock()
            engine.stop = MagicMock()
            engine.start = MagicMock(side_effect=RuntimeError("OOM"))
            return engine

        adapter._engine_for_spec = _crashing_engine  # type: ignore[method-assign]

        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert — draining cleared in finally block
        assert not adapter._draining.is_set(), (
            "_draining event must be cleared even after engine.start() raises"
        )

    @pytest.mark.asyncio
    async def test_engine_start_failure_leaves_empty_instances(self) -> None:
        """After engine.start() failure, _instances is empty (old model stopped, new not started)."""
        # Arrange — old model was running
        old_spec = _make_spec(engine="llamacpp", name="qwen3-4b")
        new_spec = _make_spec(engine="llamacpp", name="qwen3-8b")
        old_inst = _make_instance(port=8091)

        adapter = _IntegrationAdapter(
            catalog_models={"qwen3-4b": old_spec, "qwen3-8b": new_spec},
            initial_instances={"qwen3-4b": old_inst},
        )

        def _crashing_engine(s):
            engine = MagicMock()
            engine.stop = MagicMock()
            engine.start = MagicMock(side_effect=RuntimeError("OOM"))
            return engine

        adapter._engine_for_spec = _crashing_engine  # type: ignore[method-assign]

        req = _make_swap_request(model_name="qwen3-8b")
        msg = _make_msg()

        # Act
        await adapter._do_swap(msg, req)

        # Assert — old model removed; new model not registered
        assert adapter._instances == {}, (
            f"After crash, instances must be empty (old stopped, new failed), "
            f"got: {adapter._instances}"
        )


# ---------------------------------------------------------------------------
# VRAM rejection with real HostSettings (B5)
# ---------------------------------------------------------------------------


class _VramTestAdapter(LifecycleMixin):
    """Test adapter with a real Catalog/HostSettings so check_vram_budget runs properly.

    Separate from _IntegrationAdapter to avoid mutating the shared fixture for other tests.
    """

    def __init__(self, catalog: Catalog) -> None:
        self.__init_lifecycle__()
        self._sem = asyncio.Semaphore(2)
        self._max_concurrent = 2
        self.drain_timeout = 5.0
        self._instances: dict = {}
        self._nc = MagicMock()
        self._nc.publish = AsyncMock()
        self._vram_monitor = None
        self._catalog = catalog
        self._reply_ok_calls: list[dict] = []
        self._reply_err_calls: list[dict] = []

    async def _wait_sem_idle(self) -> None:
        pass

    def _engine_for_spec(self, spec):
        engine = MagicMock()
        inst = _make_instance(port=8091)
        engine.start = MagicMock(return_value=inst)
        engine.stop = MagicMock()
        return engine

    async def _reply_ok(self, msg, req, *, data=None) -> None:
        self._reply_ok_calls.append({"msg": msg, "req": req, "data": data})

    async def _reply_err(self, msg, req, code, message, *, retryable=True) -> None:
        self._reply_err_calls.append(
            {"msg": msg, "req": req, "code": code, "message": message, "retryable": retryable}
        )


@pytest.mark.nats
class TestSwapVramRejection:
    """VRAM budget exceeded → lifecycle_rejected, not TypeError swallow (B5).

    Spec: check_vram_budget raises ValueError when spec.vram_gib > host.vram_budget_gib.
    The TypeError swallow in _lifecycle.py was removed in B5; these tests verify that
    the check runs and produces the correct rejection for real HostSettings objects.
    """

    @pytest.mark.asyncio
    async def test_vram_budget_exceeded_replies_lifecycle_rejected(self) -> None:
        """Swap a model that exceeds the host VRAM budget → llm.lifecycle_rejected.

        Arrange: host.vram_budget_gib=10.0, model.vram_gib=13.0.
        Assert: _reply_err called with code='llm.lifecycle_rejected', retryable=False.
        """
        host = HostSettings(
            bind="0.0.0.0",
            public_base_url="http://localhost",
            api_key_env="LLMCLI_API_KEY",
            vram_budget_gib=10.0,
        )
        spec = ModelSpec(
            name="big-model",
            engine="llamacpp",
            repo="Org/Big-GGUF",
            file="big.gguf",
            port=8091,
            vram_gib=13.0,
        )
        catalog = Catalog(host=host, models={"big-model": spec})
        adapter = _VramTestAdapter(catalog=catalog)

        req = _make_swap_request(model_name="big-model")
        msg = _make_msg()

        await adapter._do_swap(msg, req)

        assert len(adapter._reply_err_calls) == 1, (
            f"Expected one _reply_err for VRAM exceeded, got: {adapter._reply_err_calls}"
        )
        err = adapter._reply_err_calls[0]
        assert err["code"] == "llm.lifecycle_rejected", (
            f"Expected 'llm.lifecycle_rejected', got: {err['code']!r}"
        )
        assert err["retryable"] is False, "VRAM rejection must be retryable=False"

    @pytest.mark.asyncio
    async def test_vram_within_budget_does_not_reject(self) -> None:
        """Swap a model that fits in the host VRAM budget → ok reply (no rejection).

        Arrange: host.vram_budget_gib=16.0, model.vram_gib=8.0.
        Assert: _reply_ok called, _reply_err not called.
        """
        host = HostSettings(
            bind="0.0.0.0",
            public_base_url="http://localhost",
            api_key_env="LLMCLI_API_KEY",
            vram_budget_gib=16.0,
        )
        spec = ModelSpec(
            name="small-model",
            engine="llamacpp",
            repo="Org/Small-GGUF",
            file="small.gguf",
            port=8091,
            vram_gib=8.0,
        )
        catalog = Catalog(host=host, models={"small-model": spec})
        adapter = _VramTestAdapter(catalog=catalog)

        req = _make_swap_request(model_name="small-model")
        msg = _make_msg()

        await adapter._do_swap(msg, req)

        assert adapter._reply_err_calls == [], (
            f"Expected no rejection for model within budget, got: {adapter._reply_err_calls}"
        )
        assert len(adapter._reply_ok_calls) == 1, (
            f"Expected ok reply for model within budget, got: {adapter._reply_ok_calls}"
        )
