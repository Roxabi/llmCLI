"""GenerationMixin — HTTP generation logic for LlmNatsAdapter.

Extracted from llm_adapter.py to keep both files under the 300-line
quality gate.  The mixin accesses state defined on LlmNatsAdapter
(``_client``, ``_loaded_model``, ``_nc``) via ``self``; it carries no
instance state of its own.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from roxabi_contracts.envelope import CONTRACT_VERSION
from roxabi_contracts.errors import WorkerError
from roxabi_contracts.llm.builders import build_llm_chunk, build_llm_response
from roxabi_contracts.llm.models import LlmChunkEvent, LlmResponse

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from nats.aio.client import Client as NatsClient
    from nats.aio.msg import Msg as NatsMsg

log = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class GenerationMixin:
    """HTTP generation methods for the LlmNatsAdapter.

    Requires the host class to expose:
    - ``self._client``   — ``httpx.AsyncClient``
    - ``self._loaded_model`` — ``str | None``
    - ``self._nc``       — NATS connection (with ``.publish``)
    - ``self.reply``     — coroutine from NatsAdapterBase
    """

    if TYPE_CHECKING:
        # Host-provided attributes (NatsAdapterBase + LlmNatsAdapter.__init__).
        # Declared here so Pyright can resolve attribute access on self.
        _nc: NatsClient | None
        _client: httpx.AsyncClient
        _loaded_model: str | None
        reply: Callable[[NatsMsg, bytes], Awaitable[None]]

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_worker_error(code: str, msg: str, retryable: bool) -> WorkerError:
        return WorkerError(code=code, message=msg, retryable=retryable)

    @staticmethod
    def _build_error_response(payload: dict, we: WorkerError, *, stream: bool) -> bytes:
        """Build and serialize an error envelope (LlmChunkEvent or LlmResponse).

        Sanitizes request_id to prevent Pydantic validation errors from illegal chars.
        """
        rid = str(payload.get("request_id", "unknown"))[:128]
        safe_id = rid if _REQUEST_ID_RE.match(rid) else "unknown"
        trace_id = payload.get("trace_id") or safe_id
        # builders.py does not accept worker_error= — build envelope manually.
        if stream:
            return (
                LlmChunkEvent(
                    contract_version=CONTRACT_VERSION,
                    trace_id=trace_id,
                    issued_at=datetime.now(timezone.utc),
                    request_id=safe_id,
                    done=True,
                    is_error=True,
                    error=we.message,
                    worker_error=we,
                )
                .model_dump_json(exclude_none=True)
                .encode()
            )
        return (
            LlmResponse(
                contract_version=CONTRACT_VERSION,
                trace_id=trace_id,
                issued_at=datetime.now(timezone.utc),
                request_id=safe_id,
                ok=False,
                error=we.message,
                worker_error=we,
            )
            .model_dump_json(exclude_none=True)
            .encode()
        )

    async def _err(self, msg, payload: dict, worker_error: WorkerError) -> None:
        data = self._build_error_response(payload, worker_error, stream=False)
        await self.reply(msg, data)

    # ------------------------------------------------------------------
    # Generation dispatch
    # ------------------------------------------------------------------

    async def _run_generation(self, msg, payload: dict, request_id: str) -> None:
        messages: list[dict] = payload.get("messages") or []
        system_prompt: str | None = payload.get("system_prompt")
        stream: bool = bool(payload.get("stream", True))
        max_tokens: int | None = payload.get("max_tokens")
        temperature: float | None = payload.get("temperature")

        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, *messages]

        body: dict = {"model": self._loaded_model or "default", "messages": messages}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        t0 = time.monotonic()

        try:
            if stream:
                await self._stream_response(msg, payload, body, t0)
            else:
                await self._blocking_response(msg, payload, body, t0)
            return
        except httpx.TimeoutException:
            we = self._make_worker_error("worker.timeout", "upstream timeout", retryable=True)
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status >= 500:
                code, retry = "upstream.5xx", True
            elif status == 429:
                code, retry = "upstream.unavailable", True  # rate limited — retryable
            else:
                code, retry = "upstream.unavailable", False  # 4xx client error — non-retryable
            we = self._make_worker_error(code, f"upstream {status}", retryable=retry)
        except httpx.ConnectError:
            we = self._make_worker_error(
                "upstream.unavailable", "upstream unavailable", retryable=True
            )
        except json.JSONDecodeError:
            we = self._make_worker_error("transport.parse", "invalid SSE chunk", retryable=False)
        except Exception:  # noqa: BLE001
            we = self._make_worker_error("worker.internal", "internal error", retryable=False)

        log.error(
            "llm_adapter: generation error request_id=%s code=%s: %s",
            request_id,
            we.code,
            we.message,
        )
        if stream and msg.reply and self._nc:
            try:
                await self._nc.publish(
                    msg.reply,
                    self._build_error_response(payload, we, stream=True),
                )
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                await self._err(msg, payload, we)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # HTTP calls
    # ------------------------------------------------------------------

    async def _stream_response(self, msg, payload: dict, body: dict, t0: float) -> None:
        body = {**body, "stream": True}
        nc = self._nc
        assert nc is not None  # _nc is connected before any generation is dispatched

        chunk_count = 0
        saw_done = False
        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    chunk_data = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta and msg.reply:
                    await nc.publish(
                        msg.reply,
                        build_llm_chunk(payload, delta=delta).encode(),
                    )
                    chunk_count += 1

        if chunk_count == 0 and not saw_done:
            raise json.JSONDecodeError("no parseable SSE chunks", "", 0)
        if chunk_count == 0 and saw_done:
            log.warning(
                "llm_adapter: empty response (only [DONE]) request_id=%s",
                payload.get("request_id", "unknown"),
            )

        duration_ms = int((time.monotonic() - t0) * 1000)
        if msg.reply:
            await nc.publish(
                msg.reply,
                build_llm_chunk(payload, done=True, duration_ms=duration_ms).encode(),
            )

    async def _blocking_response(self, msg, payload: dict, body: dict, t0: float) -> None:
        body = {**body, "stream": False}
        resp = await self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        text: str = data["choices"][0]["message"]["content"]

        duration_ms = int((time.monotonic() - t0) * 1000)
        await self.reply(
            msg,
            build_llm_response(payload, ok=True, text=text, duration_ms=duration_ms).encode(),
        )
