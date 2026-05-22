"""Tests for LifecycleMixin._do_reload_catalog TOML parse error — issue #34, Slice 5, T34.

Spec trace: E11 — reload-catalog fails with TOML parse error → reply rejected,
in-memory catalog unchanged (no partial state, no service interruption).

No broker required — calls _do_reload_catalog directly on a minimal adapter.
Implemented in T19 (Wave 2).

Negative pattern: removing the `except tomllib.TOMLDecodeError` block causes this
test to fail — the exception propagates unhandled and _do_reload_catalog never calls
_reply_err, so the assertions on worker_error.code fail.
"""

from __future__ import annotations

import asyncio
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llmcli.nats._lifecycle import LifecycleMixin
from roxabi_contracts.llm import LifecycleRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_reload_request() -> LifecycleRequest:
    return LifecycleRequest(
        contract_version="1",
        trace_id="trace-reload-toml",
        issued_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        request_id="req-reload-001",
        op="reload-catalog",
        host=None,
    )


def _make_msg(reply: str = "_inbox.llm-operator.test") -> SimpleNamespace:
    return SimpleNamespace(reply=reply, subject="lyra.llm.lifecycle.reload-catalog")


class _TestAdapter(LifecycleMixin):
    """Minimal adapter capturing _reply_ok and _reply_err calls."""

    def __init__(self, initial_catalog=None) -> None:
        self.__init_lifecycle__()
        self._sem = asyncio.Semaphore(2)
        self.drain_timeout = 5.0
        self._instances: dict = {}
        # Use a sentinel catalog to verify it remains unchanged after a failed reload
        self._catalog = initial_catalog or MagicMock()
        self._nc = MagicMock()
        self._nc.publish = AsyncMock()
        # Capture reply calls
        self._reply_ok_calls: list[dict] = []
        self._reply_err_calls: list[dict] = []

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

    def _engine_for_spec(self, spec):
        return MagicMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReloadCatalogTomlError:
    """_do_reload_catalog with a malformed TOML file."""

    @pytest.mark.asyncio
    async def test_malformed_toml_replies_lifecycle_rejected(self, tmp_path: Path) -> None:
        """_do_reload_catalog returns llm.lifecycle_rejected on TOML parse error.

        Arrange: patch config.load to raise tomllib.TOMLDecodeError.
        Act: call _do_reload_catalog.
        Assert: reply has code == 'llm.lifecycle_rejected' and retryable == False.

        Negative: removing the except tomllib.TOMLDecodeError block means the error
        propagates unhandled and _reply_err is never called — this assertion fails.
        """
        # Arrange
        original_catalog = MagicMock(name="original_catalog")
        original_catalog.models = {"qwen3-8b": MagicMock()}
        adapter = _TestAdapter(initial_catalog=original_catalog)

        req = _make_reload_request()
        msg = _make_msg()

        # Simulate load_catalog() raising a TOMLDecodeError
        toml_error = tomllib.TOMLDecodeError("invalid TOML syntax at line 1 col 1")
        with patch("llmcli.nats._lifecycle.load_catalog", side_effect=toml_error):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert — must have replied with an error
        assert len(adapter._reply_err_calls) == 1, (
            f"Expected exactly one _reply_err call, got {len(adapter._reply_err_calls)}. "
            f"reply_ok_calls: {adapter._reply_ok_calls}"
        )
        err = adapter._reply_err_calls[0]
        assert err["code"] == "llm.lifecycle_rejected", (
            f"Expected code 'llm.lifecycle_rejected', got: {err['code']!r}"
        )
        assert err["retryable"] is False, (
            "TOML parse error must be retryable=False (client-side misconfiguration)"
        )

    @pytest.mark.asyncio
    async def test_malformed_toml_error_message_contains_context(self, tmp_path: Path) -> None:
        """Error message contains 'catalog' and 'parse' hints per spec E11."""
        # Arrange
        adapter = _TestAdapter()
        req = _make_reload_request()
        msg = _make_msg()

        toml_error = tomllib.TOMLDecodeError("unexpected token '}' at line 2 col 5")
        with patch("llmcli.nats._lifecycle.load_catalog", side_effect=toml_error):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert — message must provide diagnostic context
        assert len(adapter._reply_err_calls) == 1
        message = adapter._reply_err_calls[0]["message"].lower()
        # Spec E11: "catalog parse error: <detail>"
        assert "catalog" in message or "parse" in message, (
            f"Error message must mention 'catalog' or 'parse', got: {message!r}"
        )

    @pytest.mark.asyncio
    async def test_malformed_toml_catalog_remains_unchanged(self, tmp_path: Path) -> None:
        """In-memory catalog is unchanged after a failed reload (E11 invariant).

        Negative: if the implementation assigns the partial/failed result to
        self._catalog before catching the error, this test fails because the
        sentinel catalog is replaced.
        """
        # Arrange — use a sentinel so we can verify identity
        sentinel_catalog = MagicMock(name="sentinel_catalog")
        sentinel_catalog.models = {"original-model": MagicMock()}
        adapter = _TestAdapter(initial_catalog=sentinel_catalog)

        req = _make_reload_request()
        msg = _make_msg()

        toml_error = tomllib.TOMLDecodeError("malformed TOML")
        with patch("llmcli.nats._lifecycle.load_catalog", side_effect=toml_error):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert — catalog must be the exact same object (not replaced)
        assert adapter._catalog is sentinel_catalog, (
            "In-memory catalog must remain unchanged after a failed reload. "
            f"Expected sentinel_catalog (id={id(sentinel_catalog)}), "
            f"got: {adapter._catalog!r} (id={id(adapter._catalog)})"
        )

    @pytest.mark.asyncio
    async def test_malformed_toml_does_not_reply_ok(self, tmp_path: Path) -> None:
        """_do_reload_catalog must not call _reply_ok when TOML parse fails."""
        # Arrange
        adapter = _TestAdapter()
        req = _make_reload_request()
        msg = _make_msg()

        toml_error = tomllib.TOMLDecodeError("parse failure")
        with patch("llmcli.nats._lifecycle.load_catalog", side_effect=toml_error):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert
        assert adapter._reply_ok_calls == [], (
            "Must not call _reply_ok on TOML parse error. "
            f"Got _reply_ok calls: {adapter._reply_ok_calls}"
        )


class TestReloadCatalogSuccess:
    """_do_reload_catalog happy path — valid catalog → reply ok."""

    @pytest.mark.asyncio
    async def test_valid_catalog_replies_ok(self) -> None:
        """_do_reload_catalog replies ok with models_loaded count on success."""
        # Arrange
        adapter = _TestAdapter()
        req = _make_reload_request()
        msg = _make_msg()

        new_catalog = MagicMock()
        new_catalog.models = {"model-a": MagicMock(), "model-b": MagicMock()}

        with patch("llmcli.nats._lifecycle.load_catalog", return_value=new_catalog):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert
        assert len(adapter._reply_ok_calls) == 1, (
            f"Expected one _reply_ok call, got {len(adapter._reply_ok_calls)}"
        )
        data = adapter._reply_ok_calls[0]["data"]
        assert data is not None, "_reply_ok data must not be None"
        assert data.get("models_loaded") == 2, f"Expected models_loaded=2, got: {data}"

    @pytest.mark.asyncio
    async def test_valid_catalog_updates_in_memory_catalog(self) -> None:
        """After a successful reload, self._catalog is replaced with the new one."""
        # Arrange
        original_catalog = MagicMock(name="original")
        original_catalog.models = {}
        adapter = _TestAdapter(initial_catalog=original_catalog)

        req = _make_reload_request()
        msg = _make_msg()

        new_catalog = MagicMock(name="new_catalog")
        new_catalog.models = {"model-x": MagicMock()}

        with patch("llmcli.nats._lifecycle.load_catalog", return_value=new_catalog):
            # Act
            await adapter._do_reload_catalog(msg, req)

        # Assert — catalog must be replaced
        assert adapter._catalog is new_catalog, (
            "In-memory catalog must be updated to new_catalog after successful reload."
        )
