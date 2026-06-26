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
from roxabi_contracts.llm.models import LlmChunkEvent, LlmResponse
from roxabi_satellite.llm.replies import build_llm_error_reply

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
        return build_llm_error_reply(payload, we, stream=stream)

    async def _err(self, msg, payload: dict, worker_error: WorkerError) -> None:
        await self.reply(msg, self._build_error_response(payload, worker_error, stream=False))

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
                    # Conditional kwarg exclusion: omit job_id when absent so
                    # default_factory fires instead of passing None to _validate_job_id.
                    _jid: dict = {}
                    if _v := payload.get("job_id"):
                        _jid["job_id"] = _v
                    await nc.publish(
                        msg.reply,
                        LlmChunkEvent(
                            contract_version=CONTRACT_VERSION,
                            trace_id=payload.get("trace_id") or str(payload.get("request_id", "")),
                            issued_at=datetime.now(timezone.utc),
                            request_id=str(payload.get("request_id", "")),
                            delta=delta,
                            **_jid,
                        )
                        .model_dump_json(exclude_none=True)
                        .encode(),
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
            _jid2: dict = {}
            if _v2 := payload.get("job_id"):
                _jid2["job_id"] = _v2
            await nc.publish(
                msg.reply,
                LlmChunkEvent(
                    contract_version=CONTRACT_VERSION,
                    trace_id=payload.get("trace_id") or str(payload.get("request_id", "")),
                    issued_at=datetime.now(timezone.utc),
                    request_id=str(payload.get("request_id", "")),
                    done=True,
                    duration_ms=duration_ms,
                    **_jid2,
                )
                .model_dump_json(exclude_none=True)
                .encode(),
            )

    async def _blocking_response(self, msg, payload: dict, body: dict, t0: float) -> None:
        body = {**body, "stream": False}
        resp = await self._client.post("/chat/completions", json=body)
        resp.raise_for_status()
        data = resp.json()
        text: str = data["choices"][0]["message"]["content"]

        duration_ms = int((time.monotonic() - t0) * 1000)
        # Conditional kwarg exclusion: omit job_id when absent so
        # default_factory fires instead of passing None to _validate_job_id.
        _jid3: dict = {}
        if _v3 := payload.get("job_id"):
            _jid3["job_id"] = _v3
        await self.reply(
            msg,
            LlmResponse(
                contract_version=CONTRACT_VERSION,
                trace_id=payload.get("trace_id") or str(payload.get("request_id", "")),
                issued_at=datetime.now(timezone.utc),
                request_id=str(payload.get("request_id", "")),
                ok=True,
                text=text,
                duration_ms=duration_ms,
                **_jid3,
            )
            .model_dump_json(exclude_none=True)
            .encode(),
        )
