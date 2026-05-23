"""Tests for NatsClient.request_fleet() — issue #61, T6.

Monkeypatches nats.aio.client.Client so no broker is required.
request_fleet() is exercised directly against a fake NATS client that
simulates multiple replies, timeouts, and error payloads.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llmcli.cli._nats_client import FleetResult, NatsClient
from roxabi_contracts.errors import WorkerError
from roxabi_contracts.llm import LifecycleRequest, LifecycleResponse


def _make_ok(host: str, data: dict | None = None) -> bytes:
    resp = LifecycleResponse(
        contract_version="1",
        trace_id="trace-ok",
        issued_at=datetime.now(timezone.utc).isoformat(),
        request_id="req-ok",
        ok=True,
        host=host,
        data=data or {},
    )
    return resp.model_dump_json().encode()


def _make_err(host: str, code: str, message: str) -> bytes:
    resp = LifecycleResponse(
        contract_version="1",
        trace_id="trace-err",
        issued_at=datetime.now(timezone.utc).isoformat(),
        request_id="req-err",
        ok=False,
        host=host,
        worker_error=WorkerError(code=code, message=message, retryable=False),
    )
    return resp.model_dump_json().encode()


class _FakeNATSClientFleet:
    """NATS client stub that yields pre-staged replies for request_fleet()."""

    def __init__(self, replies: list[bytes]) -> None:
        self._replies = replies
        self._reply_idx = 0
        self.published_subject: str | None = None
        self.published_payload: bytes | None = None
        self.connect = AsyncMock(return_value=None)
        self.drain = AsyncMock(return_value=None)

    async def new_inbox(self):
        return "_inbox.test.1"

    async def subscribe(self, inbox):
        client = self

        class _Sub:
            async def next_msg(self, timeout=10):
                if client._reply_idx < len(client._replies):
                    r = client._replies[client._reply_idx]
                    client._reply_idx += 1
                    return SimpleNamespace(data=r)
                raise asyncio.TimeoutError

            async def unsubscribe(self):
                pass

        return _Sub()

    async def publish(self, subject, payload, reply=None):
        self.published_subject = subject
        self.published_payload = payload


def _patch_nats_fleet(replies: list[bytes]):
    client = _FakeNATSClientFleet(replies)
    return patch("nats.aio.client.Client", return_value=client), client


class TestNatsClientRequestFleet:
    async def test_collects_multiple_replies(self):
        # Arrange
        replies = [
            _make_ok("host-a", {"model": "qwen3-8b", "port": 8091, "vram_used_mb": 5120}),
            _make_ok("host-b", {"model": "qwen3-4b", "port": 8092, "vram_used_mb": 2048}),
        ]
        patcher, fake = _patch_nats_fleet(replies)

        # Act
        with patcher:
            client = NatsClient(allow_anonymous=True)
            await client.connect()
            result = await client.request_fleet("test.subject", "status", timeout=5.0)

        # Assert
        assert isinstance(result, FleetResult)
        assert len(result.responses) == 2
        assert result.responses[0].host == "host-a"
        assert result.responses[1].host == "host-b"
        assert not result.timeout_reached
        assert result.errors == []
        assert result.elapsed_ms > 0
        assert fake.published_subject == "test.subject"

    async def test_timeout_no_replies(self):
        # Arrange
        patcher, fake = _patch_nats_fleet([])

        # Act
        with patcher:
            client = NatsClient(allow_anonymous=True)
            await client.connect()
            result = await client.request_fleet("test.subject", "status", timeout=0.1)

        # Assert
        assert result.responses == []
        assert result.timeout_reached is True
        assert result.errors == []

    async def test_error_replies_added_to_errors(self):
        # Arrange
        replies = [
            _make_ok("host-a", {"model": "qwen3-8b"}),
            _make_err("host-b", "llm.no_engine", "no engine running"),
        ]
        patcher, fake = _patch_nats_fleet(replies)

        # Act
        with patcher:
            client = NatsClient(allow_anonymous=True)
            await client.connect()
            result = await client.request_fleet("test.subject", "status", timeout=5.0)

        # Assert
        assert len(result.responses) == 1
        assert result.responses[0].host == "host-a"
        assert len(result.errors) == 1
        host, we = result.errors[0]
        assert host == "host-b"
        assert we.code == "llm.no_engine"
        assert we.message == "no engine running"

    async def test_mixed_replies(self):
        # Arrange
        replies = [
            _make_ok("host-a", {"model": "qwen3-8b"}),
            _make_err("host-b", "llm.no_engine", "no engine running"),
            _make_ok("host-c", {"model": "qwen3-4b"}),
        ]
        patcher, fake = _patch_nats_fleet(replies)

        # Act
        with patcher:
            client = NatsClient(allow_anonymous=True)
            await client.connect()
            result = await client.request_fleet("test.subject", "status", timeout=5.0)

        # Assert
        assert len(result.responses) == 2
        assert len(result.errors) == 1
        assert not result.timeout_reached
        assert fake.published_subject == "test.subject"
        assert fake.published_payload is not None
        req = LifecycleRequest.model_validate_json(fake.published_payload)
        assert req.host == "*"
        assert req.op == "status"

    async def test_invalid_reply_skipped(self):
        # Arrange — malformed JSON in the middle should be silently dropped
        replies = [
            _make_ok("host-a", {"model": "qwen3-8b"}),
            b"not-valid-json",
            _make_ok("host-c", {"model": "qwen3-4b"}),
        ]
        patcher, fake = _patch_nats_fleet(replies)

        # Act
        with patcher:
            client = NatsClient(allow_anonymous=True)
            await client.connect()
            result = await client.request_fleet("test.subject", "status", timeout=5.0)

        # Assert — only the two valid replies are collected
        assert len(result.responses) == 2
        assert result.responses[0].host == "host-a"
        assert result.responses[1].host == "host-c"
        assert not result.timeout_reached
        assert result.errors == []
