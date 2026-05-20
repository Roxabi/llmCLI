"""Fixtures for NATS adapter unit tests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from llmcli.nats.llm_adapter import LlmNatsAdapter


@pytest.fixture
def fake_msg_factory():
    """Build a NATS-message-shaped object with .data (bytes) and .reply (str).

    Tests pass a payload dict; we encode it to JSON bytes mirroring
    NatsAdapterBase._dispatch's input. .reply is a fixed inbox so callers
    can assert publishes.
    """

    def _build(payload: dict, reply: str = "_inbox.test.1") -> SimpleNamespace:
        return SimpleNamespace(data=json.dumps(payload).encode(), reply=reply)

    return _build


@pytest.fixture
def make_request_payload():
    """Factory for valid LlmRequest-shaped payload dicts.

    Defaults satisfy NatsAdapterBase envelope checks (contract_version +
    schema_version) and request_id pattern (^[A-Za-z0-9_-]{1,128}$).
    Tests override per-case.
    """
    from roxabi_contracts.envelope import CONTRACT_VERSION

    def _build(*, stream: bool = True, **overrides: Any) -> dict:
        base = {
            "contract_version": CONTRACT_VERSION,
            "schema_version": 1,
            "trace_id": "trace-xyz",
            "issued_at": datetime.now(timezone.utc).isoformat(),
            "request_id": "req-001",
            "messages": [{"role": "user", "content": "hi"}],
            "model": "qwen3-8b",
            "system_prompt": None,
            "stream": stream,
            "max_tokens": 16,
            "temperature": 0.7,
        }
        base.update(overrides)
        return base

    return _build


@pytest.fixture
def adapter(monkeypatch) -> Iterator[LlmNatsAdapter]:
    """LlmNatsAdapter instance with daemon SWAP/STATUS bypassed and shared
    httpx client replaced by AsyncMock so individual tests can program responses.

    Tests interact with `adapter._client.post` and `adapter._client.stream`
    via AsyncMock.return_value to drive _blocking_response / _stream_response.
    """

    # Bypass _ensure_model so tests don't need a live daemon socket.
    monkeypatch.setattr(LlmNatsAdapter, "_ensure_model", lambda self: None)

    a = LlmNatsAdapter(
        model_name="qwen3-8b",
        litellm_url="http://litellm.test/v1",
        litellm_key="test-key",
        max_concurrent=2,
    )
    # State that _ensure_model would have set:
    a._loaded_model = "qwen3-8b"

    # Replace the real httpx client with an AsyncMock — tests configure responses.
    a._client = MagicMock()
    a._client.post = AsyncMock()
    a._client.stream = MagicMock()  # context manager returned by stream() is configured per-test
    a._client.aclose = AsyncMock()

    # Provide a stand-in NATS connection so reply()/publish() calls don't blow up.
    fake_nc = MagicMock()
    fake_nc.publish = AsyncMock()
    fake_nc.is_connected = True
    a._nc = fake_nc

    yield a


@pytest.fixture
def stream_lines():
    """Helper to build SSE line iterators for adapter._client.stream mocking.

    Returns a function: stream_lines(chunks: list[str]) -> async iterator emitting
    `data: {...}` lines per chunk plus `data: [DONE]`. Each chunk is wrapped in the
    OpenAI delta shape so the existing adapter parser extracts it.
    """

    def _build(chunks: list[str]):
        async def _agen():
            for c in chunks:
                payload = {"choices": [{"delta": {"content": c}}]}
                yield f"data: {json.dumps(payload)}"
            yield "data: [DONE]"

        return _agen()

    return _build
