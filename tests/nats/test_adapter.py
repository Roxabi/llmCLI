"""Unit tests for LlmNatsAdapter — issue #12."""

from __future__ import annotations

import json
import sys
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
    # Patch VRAM probe to a deterministic value (avoids real nvidia-smi / pynvml).
    import llmcli.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "probe_free_vram_gib", lambda: 10.0)

    # Patch pynvml in sys.modules so heartbeat_payload's `import pynvml` branch
    # returns a known total VRAM of 16 GiB.
    fake_pynvml = MagicMock()
    fake_pynvml.nvmlInit.return_value = None
    fake_pynvml.nvmlShutdown.return_value = None
    fake_pynvml.nvmlDeviceGetHandleByIndex.return_value = "handle"
    mem_info = MagicMock()
    mem_info.total = 16 * 1024 * 1024 * 1024  # 16 GiB in bytes
    fake_pynvml.nvmlDeviceGetMemoryInfo.return_value = mem_info
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    p = adapter.heartbeat_payload()

    assert p["model_loaded"] == "qwen3-8b"
    assert p["active_requests"] == 0
    assert p["vram_free_mb"] == int(10.0 * 1024)  # 10240
    assert p["vram_used_mb"] == 16 * 1024 - int(10.0 * 1024)  # 6144
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
@pytest.mark.parametrize(
    ("initial_handle", "expected_shutdown_calls"),
    [(None, 0), ("fake-handle", 1)],
    ids=["no_handle", "with_handle"],
)
async def test_shutdown_closes_http_client_and_resets_nvml(
    adapter, monkeypatch, initial_handle, expected_shutdown_calls
):
    """_shutdown closes httpx client; nvmlShutdown fires only when handle is cached."""
    from roxabi_nats.adapter_base import NatsAdapterBase

    base_shutdown = AsyncMock(return_value=None)
    monkeypatch.setattr(NatsAdapterBase, "_shutdown", base_shutdown)

    fake_pynvml = MagicMock()
    fake_pynvml.nvmlShutdown.return_value = None
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    adapter._nvml_handle = initial_handle

    await adapter._shutdown()

    assert adapter._client.aclose.await_count == 1
    assert fake_pynvml.nvmlShutdown.call_count == expected_shutdown_calls
    assert adapter._nvml_handle is None
    assert base_shutdown.await_count == 1


def test_heartbeat_vram_used_zero_when_nvml_unavailable(adapter, monkeypatch):
    """nvml handle unavailable → vram_used_mb falls back to 0 (else branch)."""
    import llmcli.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "probe_free_vram_gib", lambda: 8.0)
    monkeypatch.setattr(LlmNatsAdapter, "_get_nvml_handle", lambda self: None)

    p = adapter.heartbeat_payload()

    assert p["vram_free_mb"] == int(8.0 * 1024)
    assert p["vram_used_mb"] == 0


def test_heartbeat_vram_used_zero_when_nvml_query_fails(adapter, monkeypatch):
    """nvml handle present but memory query raises → vram_used_mb falls back to 0 (except branch)."""
    import llmcli.gpu as gpu_mod

    monkeypatch.setattr(gpu_mod, "probe_free_vram_gib", lambda: 8.0)
    monkeypatch.setattr(LlmNatsAdapter, "_get_nvml_handle", lambda self: "fake-handle")

    fake_pynvml = MagicMock()
    fake_pynvml.nvmlDeviceGetMemoryInfo.side_effect = RuntimeError("nvml query failed")
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

    p = adapter.heartbeat_payload()

    assert p["vram_free_mb"] == int(8.0 * 1024)
    assert p["vram_used_mb"] == 0
