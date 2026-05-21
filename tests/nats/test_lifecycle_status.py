"""Unit tests for LifecycleMixin status/list/stop/reload-catalog handlers — issue #34, Slice 5, T36.

No broker required — calls _do_<op> directly on minimal adapter stubs.
Tests verify the LifecycleResponse.data shape for each op against the spec.

Spec trace: SC AC-1, AC-2, AC-3, AC-11
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcli.nats._lifecycle import LifecycleMixin
from roxabi_contracts.llm import LifecycleRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(op: str, **kwargs) -> LifecycleRequest:
    defaults = {
        "contract_version": "1",
        "trace_id": "trace-unit-test",
        "issued_at": datetime(2026, 5, 21, tzinfo=timezone.utc),
        "request_id": "req-unit-001",
        "op": op,
        "host": None,
    }
    defaults.update(kwargs)
    return LifecycleRequest(**defaults)


def _make_msg(subject: str = "lyra.llm.lifecycle.status") -> SimpleNamespace:
    return SimpleNamespace(reply="_inbox.llm-operator.test", subject=subject)


def _make_instance(port: int = 8091) -> MagicMock:
    inst = MagicMock()
    inst.port = port
    return inst


class _TestAdapter(LifecycleMixin):
    """Minimal adapter capturing _reply_ok and _reply_err calls.

    Pattern matches test_lifecycle_reload_toml.py and test_lifecycle_host_filter.py.
    """

    def __init__(self, instances: dict | None = None, catalog_models: dict | None = None) -> None:
        self.__init_lifecycle__()
        self._sem = asyncio.Semaphore(2)
        self._max_concurrent = 2
        self._drain_timeout = 5.0
        self._instances: dict = instances or {}
        self._nc = MagicMock()
        self._nc.publish = AsyncMock()
        self._engine_stop_calls: list[str] = []

        # Build catalog mock
        self._catalog = MagicMock()
        self._catalog.host = MagicMock()
        if catalog_models is not None:
            self._catalog.models = catalog_models
        else:
            self._catalog.models = {}

        # Capture reply calls
        self._reply_ok_calls: list[dict] = []
        self._reply_err_calls: list[dict] = []

    async def _reply_ok(self, msg, req, *, data=None) -> None:
        self._reply_ok_calls.append({"msg": msg, "req": req, "data": data})

    async def _reply_err(self, msg, req, code, message, *, retryable=True) -> None:
        self._reply_err_calls.append({
            "msg": msg, "req": req, "code": code,
            "message": message, "retryable": retryable,
        })

    def _engine_for_spec(self, spec):
        engine = MagicMock()
        engine.stop = MagicMock(side_effect=lambda inst: self._engine_stop_calls.append(spec))
        engine.start = MagicMock(return_value=_make_instance())
        return engine


# ---------------------------------------------------------------------------
# Tests: _do_status
# ---------------------------------------------------------------------------


class TestDoStatus:
    """_do_status data shape — SC AC-1."""

    @pytest.mark.asyncio
    async def test_status_empty_instances_returns_model_none(self) -> None:
        """When no model is running, _do_status replies with data={"model": None}.

        Negative: removing the `if not instances` branch causes the code to try
        to iterate an empty dict, producing KeyError or an incorrect data shape.
        """
        # Arrange
        adapter = _TestAdapter(instances={})
        req = _make_request("status")
        msg = _make_msg("lyra.llm.lifecycle.status")

        # Act
        await adapter._do_status(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1, (
            f"Expected one _reply_ok call, got {len(adapter._reply_ok_calls)}"
        )
        data = adapter._reply_ok_calls[0]["data"]
        assert data == {"model": None}, (
            f"Expected {{model: None}} when no instances, got: {data}"
        )

    @pytest.mark.asyncio
    async def test_status_with_running_model_returns_model_port_vram(self) -> None:
        """When a model is running, _do_status replies with model/port/vram_used_mb.

        Negative: removing the VRAM probe or the port lookup causes the data shape to
        be missing required keys — callers (CLI/lyra) cannot display status correctly.
        """
        # Arrange
        inst = _make_instance(port=8091)
        adapter = _TestAdapter(instances={"qwen3-8b": inst})
        req = _make_request("status")
        msg = _make_msg()

        # Act
        await adapter._do_status(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1
        data = adapter._reply_ok_calls[0]["data"]
        assert data["model"] == "qwen3-8b", f"Expected model='qwen3-8b', got: {data.get('model')!r}"
        assert data["port"] == 8091, f"Expected port=8091, got: {data.get('port')!r}"
        assert "vram_used_mb" in data, f"Expected 'vram_used_mb' key in data, got: {data}"
        assert isinstance(data["vram_used_mb"], int), (
            f"vram_used_mb must be int, got: {type(data['vram_used_mb'])}"
        )

    @pytest.mark.asyncio
    async def test_status_vram_monitor_sample_used(self) -> None:
        """When _vram_monitor is set, its sample() value is used in the response."""
        # Arrange
        inst = _make_instance(port=8091)
        adapter = _TestAdapter(instances={"qwen3-8b": inst})

        vram_monitor = MagicMock()
        vram_monitor.sample = MagicMock(return_value=(9000.0, 5120.0))
        adapter._vram_monitor = vram_monitor

        req = _make_request("status")
        msg = _make_msg()

        # Act
        await adapter._do_status(msg, req)

        # Assert
        data = adapter._reply_ok_calls[0]["data"]
        assert data["vram_used_mb"] == 5120, (
            f"Expected vram_used_mb=5120 from monitor.sample(), got: {data['vram_used_mb']}"
        )

    @pytest.mark.asyncio
    async def test_status_no_vram_monitor_returns_zero(self) -> None:
        """Without _vram_monitor, vram_used_mb is 0 (not an error or missing key)."""
        # Arrange
        inst = _make_instance(port=8091)
        adapter = _TestAdapter(instances={"qwen3-8b": inst})
        adapter._vram_monitor = None

        req = _make_request("status")
        msg = _make_msg()

        # Act
        await adapter._do_status(msg, req)

        # Assert
        data = adapter._reply_ok_calls[0]["data"]
        assert data["vram_used_mb"] == 0, (
            f"Expected vram_used_mb=0 when no monitor, got: {data['vram_used_mb']}"
        )


# ---------------------------------------------------------------------------
# Tests: _do_list
# ---------------------------------------------------------------------------


class TestDoList:
    """_do_list data shape — SC AC-2."""

    @pytest.mark.asyncio
    async def test_list_empty_catalog_returns_empty_models(self) -> None:
        """An empty catalog yields data={"models": []}."""
        # Arrange
        adapter = _TestAdapter(catalog_models={})
        req = _make_request("list")
        msg = _make_msg("lyra.llm.lifecycle.list")

        # Act
        await adapter._do_list(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1
        data = adapter._reply_ok_calls[0]["data"]
        assert "models" in data, f"Expected 'models' key, got: {data}"
        assert data["models"] == [], f"Expected empty list for empty catalog, got: {data['models']}"

    @pytest.mark.asyncio
    async def test_list_returns_model_shape_with_running_flag(self) -> None:
        """_do_list returns list of {name, running, engine, vram_gib} with correct running flag.

        Negative: removing the `name in instances` check causes all models to show
        running=False even when one is loaded — caller cannot distinguish loaded from idle.
        """
        # Arrange
        spec_a = MagicMock()
        spec_a.engine = "llamacpp"
        spec_a.vram_gib = 8.0

        spec_b = MagicMock()
        spec_b.engine = "vllm"
        spec_b.vram_gib = 12.0

        inst_a = _make_instance(port=8091)
        catalog_models = {"qwen3-8b": spec_a, "qwen3-32b": spec_b}
        adapter = _TestAdapter(instances={"qwen3-8b": inst_a}, catalog_models=catalog_models)

        req = _make_request("list")
        msg = _make_msg("lyra.llm.lifecycle.list")

        # Act
        await adapter._do_list(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1
        models = adapter._reply_ok_calls[0]["data"]["models"]
        assert len(models) == 2, f"Expected 2 models in list, got: {len(models)}"

        by_name = {m["name"]: m for m in models}
        assert "qwen3-8b" in by_name, "qwen3-8b missing from list"
        assert "qwen3-32b" in by_name, "qwen3-32b missing from list"

        assert by_name["qwen3-8b"]["running"] is True, (
            "qwen3-8b is in instances — running must be True"
        )
        assert by_name["qwen3-32b"]["running"] is False, (
            "qwen3-32b is NOT in instances — running must be False"
        )

    @pytest.mark.asyncio
    async def test_list_model_entry_has_all_required_fields(self) -> None:
        """Every model entry has name/running/engine/vram_gib fields (spec data model)."""
        # Arrange
        spec = MagicMock()
        spec.engine = "llamacpp"
        spec.vram_gib = 4.5
        adapter = _TestAdapter(catalog_models={"test-model": spec})

        req = _make_request("list")
        msg = _make_msg("lyra.llm.lifecycle.list")

        # Act
        await adapter._do_list(msg, req)

        # Assert
        models = adapter._reply_ok_calls[0]["data"]["models"]
        assert len(models) == 1
        entry = models[0]
        for field in ("name", "running", "engine", "vram_gib"):
            assert field in entry, f"Missing field '{field}' in model entry: {entry}"
        assert entry["name"] == "test-model"
        assert entry["engine"] == "llamacpp"
        assert entry["vram_gib"] == 4.5


# ---------------------------------------------------------------------------
# Tests: _do_stop
# ---------------------------------------------------------------------------


class TestDoStop:
    """_do_stop behaviour — SC AC-3."""

    @pytest.mark.asyncio
    async def test_stop_with_no_running_instances_replies_ok(self) -> None:
        """_do_stop with no running models replies ok with empty data dict."""
        # Arrange
        adapter = _TestAdapter(instances={})
        req = _make_request("stop")
        msg = _make_msg("lyra.llm.lifecycle.stop")

        # Act
        await adapter._do_stop(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1, (
            f"Expected one _reply_ok call even with no instances, got: {len(adapter._reply_ok_calls)}"
        )
        data = adapter._reply_ok_calls[0]["data"]
        assert data == {}, f"Expected empty dict data on stop, got: {data}"

    @pytest.mark.asyncio
    async def test_stop_calls_engine_stop_for_each_running_instance(self) -> None:
        """_do_stop calls engine.stop() for each running instance and clears instances dict.

        Negative: removing the engine.stop() call in _do_stop means instances remain
        running after the lifecycle stop command — memory/port leak.
        """
        # Arrange
        spec_a = MagicMock()
        spec_a.engine = "llamacpp"
        spec_b = MagicMock()
        spec_b.engine = "llamacpp"

        inst_a = _make_instance(port=8091)
        inst_b = _make_instance(port=8092)

        catalog_models = {"qwen3-8b": spec_a, "qwen3-4b": spec_b}
        instances = {"qwen3-8b": inst_a, "qwen3-4b": inst_b}
        adapter = _TestAdapter(instances=instances, catalog_models=catalog_models)

        req = _make_request("stop")
        msg = _make_msg("lyra.llm.lifecycle.stop")

        # Act
        await adapter._do_stop(msg, req)

        # Assert — both engine.stop() calls were made
        assert len(adapter._engine_stop_calls) == 2, (
            f"Expected 2 engine.stop() calls for 2 running instances, "
            f"got: {adapter._engine_stop_calls}"
        )

    @pytest.mark.asyncio
    async def test_stop_clears_instances_dict(self) -> None:
        """After _do_stop, _instances is empty (all models unregistered).

        Negative: removing the `del instances[old_name]` line causes instances to
        persist in memory — a subsequent status command incorrectly shows a running model.
        """
        # Arrange
        spec = MagicMock()
        spec.engine = "llamacpp"
        inst = _make_instance(port=8091)
        adapter = _TestAdapter(instances={"qwen3-8b": inst}, catalog_models={"qwen3-8b": spec})

        req = _make_request("stop")
        msg = _make_msg("lyra.llm.lifecycle.stop")

        # Act
        await adapter._do_stop(msg, req)

        # Assert
        assert adapter._instances == {}, (
            f"_instances must be empty after stop, got: {adapter._instances}"
        )

    @pytest.mark.asyncio
    async def test_stop_replies_ok_after_clearing_instances(self) -> None:
        """_do_stop sends ok reply (not error) after stopping all models."""
        # Arrange
        spec = MagicMock()
        spec.engine = "llamacpp"
        inst = _make_instance(port=8091)
        adapter = _TestAdapter(instances={"qwen3-8b": inst}, catalog_models={"qwen3-8b": spec})

        req = _make_request("stop")
        msg = _make_msg("lyra.llm.lifecycle.stop")

        # Act
        await adapter._do_stop(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1
        assert len(adapter._reply_err_calls) == 0


# ---------------------------------------------------------------------------
# Tests: _do_reload_catalog
# ---------------------------------------------------------------------------


class TestDoReloadCatalog:
    """_do_reload_catalog happy path — SC AC-11.

    Error paths (TOML parse error, catalog unchanged) are covered in
    tests/nats/test_lifecycle_reload_toml.py.
    """

    @pytest.mark.asyncio
    async def test_reload_catalog_success_replies_models_loaded_count(self) -> None:
        """Successful reload returns data={"models_loaded": N}."""
        # Arrange
        adapter = _TestAdapter()
        req = _make_request("reload-catalog")
        msg = _make_msg("lyra.llm.lifecycle.reload-catalog")

        new_catalog = MagicMock()
        new_catalog.models = {
            "model-a": MagicMock(),
            "model-b": MagicMock(),
            "model-c": MagicMock(),
        }

        with patch("llmcli.nats._lifecycle.load_catalog", return_value=new_catalog):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1
        data = adapter._reply_ok_calls[0]["data"]
        assert data is not None, "data must not be None on successful reload"
        assert "models_loaded" in data, f"Expected 'models_loaded' key, got: {data}"
        assert data["models_loaded"] == 3, (
            f"Expected models_loaded=3 for 3-model catalog, got: {data['models_loaded']}"
        )

    @pytest.mark.asyncio
    async def test_reload_catalog_success_replaces_in_memory_catalog(self) -> None:
        """After successful reload, self._catalog points to the new catalog object."""
        # Arrange
        original_catalog = MagicMock(name="original")
        original_catalog.models = {}
        adapter = _TestAdapter()
        adapter._catalog = original_catalog

        req = _make_request("reload-catalog")
        msg = _make_msg("lyra.llm.lifecycle.reload-catalog")

        new_catalog = MagicMock(name="new_catalog")
        new_catalog.models = {"model-x": MagicMock()}

        with patch("llmcli.nats._lifecycle.load_catalog", return_value=new_catalog):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert
        assert adapter._catalog is new_catalog, (
            "self._catalog must be replaced with the new catalog on success"
        )

    @pytest.mark.asyncio
    async def test_reload_catalog_does_not_reply_err_on_success(self) -> None:
        """Successful reload must not call _reply_err."""
        # Arrange
        adapter = _TestAdapter()
        req = _make_request("reload-catalog")
        msg = _make_msg("lyra.llm.lifecycle.reload-catalog")

        new_catalog = MagicMock()
        new_catalog.models = {"model-a": MagicMock()}

        with patch("llmcli.nats._lifecycle.load_catalog", return_value=new_catalog):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert
        assert adapter._reply_err_calls == [], (
            f"_reply_err must not be called on success, got: {adapter._reply_err_calls}"
        )


# ---------------------------------------------------------------------------
# Tests: dispatch routing (_dispatch_lifecycle_op)
# ---------------------------------------------------------------------------


class TestDispatchRouting:
    """_dispatch_lifecycle_op routes each op to the correct handler."""

    @pytest.mark.asyncio
    async def test_unknown_op_is_ignored_without_reply(self) -> None:
        """An unknown op string is logged and ignored — no reply sent.

        Negative: removing the `if handler is None: return` guard causes an
        AttributeError when the handler lookup returns None and the code tries
        to call it.
        """
        # Arrange
        adapter = _TestAdapter()
        # Build a request manually since LifecycleRequest validates op
        req = MagicMock()
        req.op = "definitely-unknown-op"
        msg = _make_msg()

        # Act — must not raise
        await adapter._dispatch_lifecycle_op("definitely-unknown-op", msg, req)

        # Assert — no reply
        assert adapter._reply_ok_calls == [], (
            "Unknown op must not trigger _reply_ok"
        )
        assert adapter._reply_err_calls == [], (
            "Unknown op must not trigger _reply_err (silent drop, not an error reply)"
        )

    @pytest.mark.asyncio
    async def test_status_op_routes_to_do_status(self) -> None:
        """op='status' routes to _do_status which replies with model data shape."""
        # Arrange
        inst = _make_instance(port=8091)
        adapter = _TestAdapter(instances={"qwen3-8b": inst})
        req = _make_request("status")
        msg = _make_msg()

        # Act
        await adapter._dispatch_lifecycle_op("status", msg, req)

        # Assert — _do_status was called: reply_ok with model key
        assert len(adapter._reply_ok_calls) == 1
        assert "model" in adapter._reply_ok_calls[0]["data"]

    @pytest.mark.asyncio
    async def test_list_op_routes_to_do_list(self) -> None:
        """op='list' routes to _do_list which replies with models list."""
        # Arrange
        adapter = _TestAdapter(catalog_models={})
        req = _make_request("list")
        msg = _make_msg("lyra.llm.lifecycle.list")

        # Act
        await adapter._dispatch_lifecycle_op("list", msg, req)

        # Assert — _do_list was called: reply_ok with models key
        assert len(adapter._reply_ok_calls) == 1
        assert "models" in adapter._reply_ok_calls[0]["data"]

    @pytest.mark.asyncio
    async def test_stop_op_routes_to_do_stop(self) -> None:
        """op='stop' routes to _do_stop which replies with empty data."""
        # Arrange
        adapter = _TestAdapter(instances={})
        req = _make_request("stop")
        msg = _make_msg("lyra.llm.lifecycle.stop")

        # Act
        await adapter._dispatch_lifecycle_op("stop", msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1
        assert adapter._reply_ok_calls[0]["data"] == {}
