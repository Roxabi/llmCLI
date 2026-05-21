"""Unit tests for LlmNatsAdapter — issue #12."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from llmcli.nats.llm_adapter import LlmNatsAdapter


def _decode_publish(call) -> dict:
    """Extract JSON body from a nc.publish(subject, data) call."""
    return json.loads(call.args[1].decode())


# ---------- Slice 2 — request parsing & happy paths ----------


@pytest.mark.asyncio
async def test_request_parse_invalid_request_id_emits_transport_parse(
    adapter, fake_msg_factory, make_request_payload
):
    # Use a 129-char all-valid-char request_id: fails _REQUEST_ID_RE (max 128 chars)
    # but _err slices it to [:128] before building LlmResponse, so serialization succeeds.
    # Using "../../bad path"-style IDs would crash LlmResponse Pydantic validation.
    payload = make_request_payload(stream=False, request_id="a" * 129)
    msg = fake_msg_factory(payload)

    await adapter.handle(msg, payload)

    assert adapter._nc.publish.await_count == 1
    body = _decode_publish(adapter._nc.publish.await_args_list[0])
    assert body["ok"] is False
    assert body["worker_error"]["code"] == "transport.parse"
    assert body["worker_error"]["retryable"] is False


@pytest.mark.asyncio
async def test_blocking_reply_shape(adapter, fake_msg_factory, make_request_payload):
    payload = make_request_payload(stream=False)
    msg = fake_msg_factory(payload)

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.json = MagicMock(return_value={"choices": [{"message": {"content": "hello world"}}]})
    adapter._client.post.return_value = fake_resp

    await adapter.handle(msg, payload)

    # 1 publish: the LlmResponse via NatsAdapterBase.reply
    assert adapter._nc.publish.await_count == 1
    body = _decode_publish(adapter._nc.publish.await_args_list[0])
    assert body["ok"] is True
    assert body["text"] == "hello world"
    assert "duration_ms" in body and body["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_stream_chunks_and_terminator(
    adapter, fake_msg_factory, make_request_payload, stream_lines
):
    payload = make_request_payload(stream=True)
    msg = fake_msg_factory(payload)

    # The async with self._client.stream(...) returns a response-like context
    # whose .aiter_lines() yields our SSE lines and whose .raise_for_status() noops.
    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.aiter_lines = lambda: stream_lines(["hello ", "world", "!"])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=fake_resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    adapter._client.stream.return_value = cm

    await adapter.handle(msg, payload)

    # 3 chunk publishes + 1 terminator publish = 4 total
    assert adapter._nc.publish.await_count == 4
    bodies = [_decode_publish(c) for c in adapter._nc.publish.await_args_list]
    deltas = [b.get("delta") for b in bodies if not b.get("done")]
    assert deltas == ["hello ", "world", "!"]
    terminator = bodies[-1]
    assert terminator["done"] is True
    assert terminator.get("delta") in (None, "")
    assert terminator.get("duration_ms", -1) >= 0


# ---------- Slice 2 — error classifier (5 codes) ----------


@pytest.mark.asyncio
async def test_error_timeout_emits_worker_timeout(adapter, fake_msg_factory, make_request_payload):
    payload = make_request_payload(stream=False)
    msg = fake_msg_factory(payload)
    adapter._client.post.side_effect = httpx.TimeoutException("upstream timeout")

    await adapter.handle(msg, payload)

    body = _decode_publish(adapter._nc.publish.await_args_list[-1])
    assert body["ok"] is False
    assert body["worker_error"]["code"] == "worker.timeout"
    assert body["worker_error"]["retryable"] is True


@pytest.mark.asyncio
async def test_error_5xx_emits_upstream_5xx(adapter, fake_msg_factory, make_request_payload):
    payload = make_request_payload(stream=False)
    msg = fake_msg_factory(payload)

    request = httpx.Request("POST", "http://litellm.test/v1/chat/completions")
    response = httpx.Response(503, request=request)
    adapter._client.post.side_effect = httpx.HTTPStatusError(
        "5xx", request=request, response=response
    )

    await adapter.handle(msg, payload)

    body = _decode_publish(adapter._nc.publish.await_args_list[-1])
    assert body["worker_error"]["code"] == "upstream.5xx"
    assert body["worker_error"]["retryable"] is True


@pytest.mark.asyncio
async def test_error_connect_emits_upstream_unavailable(
    adapter, fake_msg_factory, make_request_payload
):
    payload = make_request_payload(stream=False)
    msg = fake_msg_factory(payload)
    adapter._client.post.side_effect = httpx.ConnectError("refused")

    await adapter.handle(msg, payload)

    body = _decode_publish(adapter._nc.publish.await_args_list[-1])
    assert body["worker_error"]["code"] == "upstream.unavailable"
    assert body["worker_error"]["retryable"] is True


@pytest.mark.asyncio
async def test_error_parse_emits_transport_parse(adapter, fake_msg_factory, make_request_payload):
    payload = make_request_payload(stream=True)
    msg = fake_msg_factory(payload)

    # The per-line try/except in _stream_response swallows json.JSONDecodeError on
    # individual lines. To reach the transport.parse branch in _run_generation's
    # classifier, we raise JSONDecodeError directly from the async iterator's
    # __anext__ — this escapes the per-line guard and propagates to _run_generation.
    async def _raising_lines():
        raise json.JSONDecodeError("bad", "{not valid", 0)
        yield  # make it a generator

    fake_resp = MagicMock()
    fake_resp.raise_for_status = MagicMock()
    fake_resp.aiter_lines = lambda: _raising_lines()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=fake_resp)
    cm.__aexit__ = AsyncMock(return_value=None)
    adapter._client.stream.return_value = cm

    await adapter.handle(msg, payload)

    # Streaming errors are published as LlmChunkEvent with is_error=True (not LlmResponse).
    last = _decode_publish(adapter._nc.publish.await_args_list[-1])
    assert last.get("is_error") is True
    assert last["worker_error"]["code"] == "transport.parse"
    assert last["worker_error"]["retryable"] is False


@pytest.mark.asyncio
async def test_error_generic_emits_worker_internal(adapter, fake_msg_factory, make_request_payload):
    payload = make_request_payload(stream=False)
    msg = fake_msg_factory(payload)
    adapter._client.post.side_effect = RuntimeError("kaboom")

    await adapter.handle(msg, payload)

    body = _decode_publish(adapter._nc.publish.await_args_list[-1])
    assert body["worker_error"]["code"] == "worker.internal"
    assert body["worker_error"]["retryable"] is False


# ---------- Slice 2 — _ensure_model daemon-optional skip paths (#28) ----------


def test_ensure_model_socket_missing_sets_loaded_model(tmp_path):
    """Socket absent → _loaded_model populated from _model_name (not None)."""
    a = LlmNatsAdapter(
        model_name="qwen3-8b",
        litellm_url="http://litellm.test/v1",
        litellm_key="test-key",
        socket_path=tmp_path / "nonexistent.sock",
    )
    a._ensure_model()
    assert a._loaded_model == "qwen3-8b"


def test_ensure_model_oserror_sets_loaded_model(tmp_path, monkeypatch):
    """OSError mid-connect → _loaded_model populated from _model_name (not None)."""
    import llmcli.daemon as daemon_mod

    sock = tmp_path / "fake.sock"
    sock.touch()  # exists() returns True → enters try block

    monkeypatch.setattr(daemon_mod, "daemon_request", MagicMock(side_effect=OSError("refused")))

    a = LlmNatsAdapter(
        model_name="qwen3-8b",
        litellm_url="http://litellm.test/v1",
        litellm_key="test-key",
        socket_path=sock,
    )
    a._ensure_model()
    assert a._loaded_model == "qwen3-8b"


# ---------- Slice 2 — heartbeat enrichment ----------


def test_heartbeat_payload_has_vram_keys(adapter, monkeypatch):
    # Patch VRAMMonitor.sample to return deterministic (free_mb, used_mb).
    from llmcli.gpu import VRAMMonitor

    monitor = VRAMMonitor()
    monkeypatch.setattr(monitor, "sample", lambda: (10240.0, 6144.0))
    adapter._vram_monitor = monitor

    p = adapter.heartbeat_payload()

    assert p["model_loaded"] == "qwen3-8b"
    assert p["active_requests"] == 0
    assert p["vram_free_mb"] == 10240  # 10.0 GiB in MB
    assert p["vram_used_mb"] == 6144  # 6.0 GiB in MB
    assert "worker_id" in p


# ---------- Issue #20 — capacity, shutdown, nvml fallback ----------


@pytest.mark.asyncio
async def test_reject_when_full_emits_worker_internal_retryable(
    adapter, fake_msg_factory, make_request_payload
):
    """Saturated semaphore + reject_when_full=True → worker.internal retryable."""
    adapter._reject_when_full = True
    # Drain the semaphore through its public API (max_concurrent=2 in the fixture)
    # so .locked() returns True without touching CPython internals.
    await adapter._sem.acquire()
    await adapter._sem.acquire()
    assert adapter._sem.locked()

    payload = make_request_payload(stream=False)
    msg = fake_msg_factory(payload)

    await adapter.handle(msg, payload)

    # publish.await_count == 1 proves the reject path didn't fall through to
    # _run_generation (which would publish a second reply on top of _err).
    assert adapter._nc.publish.await_count == 1
    body = _decode_publish(adapter._nc.publish.await_args_list[0])
    assert body["ok"] is False
    assert body["worker_error"]["code"] == "worker.internal"
    assert body["worker_error"]["retryable"] is True


@pytest.mark.asyncio
async def test_handle_during_drain_emits_worker_capacity_retryable(
    adapter, fake_msg_factory, make_request_payload
):
    """AC-5: generation during _draining → worker.capacity retryable, no fallthrough.

    Regression for the missing drain-guard in handle(): without the guard a
    generation arriving mid-swap would acquire the semaphore and extend the
    drain window indefinitely.
    """
    adapter._draining.set()

    payload = make_request_payload(stream=False)
    msg = fake_msg_factory(payload)

    await adapter.handle(msg, payload)

    # One publish (the reject), no fallthrough to _run_generation
    assert adapter._nc.publish.await_count == 1
    body = _decode_publish(adapter._nc.publish.await_args_list[0])
    assert body["ok"] is False
    assert body["worker_error"]["code"] == "worker.capacity"
    assert body["worker_error"]["retryable"] is True
    # Semaphore must NOT have been touched — its full capacity stays available
    assert not adapter._sem.locked()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "initial_monitor",
    [None, "with_monitor"],
    ids=["no_monitor", "with_monitor"],
)
async def test_shutdown_closes_http_client_and_resets_vram_monitor(
    adapter, monkeypatch, initial_monitor
):
    """_shutdown closes httpx client; VRAMMonitor.__exit__ fires only when monitor is set."""
    from llmcli.gpu import VRAMMonitor
    from roxabi_nats.adapter_base import NatsAdapterBase

    base_shutdown = AsyncMock(return_value=None)
    monkeypatch.setattr(NatsAdapterBase, "_shutdown", base_shutdown)

    exit_calls: list[str] = []

    if initial_monitor == "with_monitor":
        monitor = VRAMMonitor()
        monkeypatch.setattr(monitor, "__exit__", lambda *a: exit_calls.append("exit"))
        adapter._vram_monitor = monitor
    else:
        adapter._vram_monitor = None

    await adapter._shutdown()

    assert adapter._client.aclose.await_count == 1
    assert adapter._vram_monitor is None
    assert base_shutdown.await_count == 1
    if initial_monitor == "with_monitor":
        assert exit_calls == ["exit"]
    else:
        assert exit_calls == []


def test_heartbeat_vram_zero_when_monitor_not_set(adapter):
    """No VRAMMonitor → vram_free_mb and vram_used_mb both 0."""
    adapter._vram_monitor = None

    p = adapter.heartbeat_payload()

    assert p["vram_free_mb"] == 0
    assert p["vram_used_mb"] == 0


def test_heartbeat_vram_used_zero_when_sample_returns_zeros(adapter, monkeypatch):
    """VRAMMonitor.sample() returning (0.0, 0.0) → both fields 0."""
    from llmcli.gpu import VRAMMonitor

    monitor = VRAMMonitor()
    monkeypatch.setattr(monitor, "sample", lambda: (0.0, 0.0))
    adapter._vram_monitor = monitor

    p = adapter.heartbeat_payload()

    assert p["vram_free_mb"] == 0
    assert p["vram_used_mb"] == 0
