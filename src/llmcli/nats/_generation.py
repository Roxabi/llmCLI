"""GenerationMixin — HTTP generation logic for LlmNatsAdapter.

Extracted from llm_adapter.py to keep both files under the 300-line
quality gate.  The mixin accesses state defined on LlmNatsAdapter
(``_client``, ``_loaded_model``, ``_nc``) via ``self``; it carries no
instance state of its own.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import httpx

from roxabi_contracts.envelope import CONTRACT_VERSION
from roxabi_contracts.errors import WorkerError
from roxabi_contracts.llm.builders import build_llm_chunk, build_llm_response
from roxabi_contracts.llm.models import LlmChunkEvent, LlmResponse

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


class GenerationMixin:
    """HTTP generation methods for the LlmNatsAdapter.

    Requires the host class to expose:
    - ``self._client``   — ``httpx.AsyncClient``
    - ``self._loaded_model`` — ``str | None``
    - ``self._nc``       — NATS connection (with ``.publish``)
    - ``self.reply``     — coroutine from NatsAdapterBase
    """

    # ------------------------------------------------------------------
    # Error helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_worker_error(code: str, msg: str, retryable: bool) -> WorkerError:
        return WorkerError(code=code, message=msg, retryable=retryable)

    async def _err(self, msg, payload: dict, worker_error: WorkerError) -> None:
        safe_payload = {
            "request_id": str(payload.get("request_id", "unknown"))[:128],
            "trace_id": payload.get("trace_id"),
        }
        # builders.py does not accept worker_error= — build envelope manually.
        data = (
            LlmResponse(
                contract_version=CONTRACT_VERSION,
                trace_id=safe_payload.get("trace_id") or safe_payload["request_id"],
                issued_at=datetime.now(timezone.utc),
                request_id=safe_payload["request_id"],
                ok=False,
                error=worker_error.message,
                worker_error=worker_error,
            )
            .model_dump_json(exclude_none=True)
            .encode()
        )
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
        except httpx.TimeoutException as exc:
            we = self._make_worker_error(
                "worker.timeout", str(exc) or "upstream timeout", retryable=True
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status >= 500:
                code, retry = "upstream.5xx", True
            elif status == 429:
                code, retry = "upstream.unavailable", True  # rate limited — retryable
            else:
                code, retry = "upstream.unavailable", False  # 4xx client error — non-retryable
            we = self._make_worker_error(code, f"upstream {status}", retryable=retry)
        except httpx.ConnectError as exc:
            we = self._make_worker_error(
                "upstream.unavailable", str(exc) or "upstream unavailable", retryable=True
            )
        except json.JSONDecodeError as exc:
            we = self._make_worker_error(
                "transport.parse", str(exc) or "invalid SSE chunk", retryable=False
            )
        except Exception as exc:  # noqa: BLE001
            we = self._make_worker_error(
                "worker.internal", str(exc) or "internal error", retryable=False
            )

        log.error(
            "llm_adapter: generation error request_id=%s code=%s: %s",
            request_id,
            we.code,
            we.message,
        )
        if stream and msg.reply and self._nc:
            try:
                # builders.py does not accept worker_error= — build envelope manually.
                data = (
                    LlmChunkEvent(
                        contract_version=CONTRACT_VERSION,
                        trace_id=payload.get("trace_id") or request_id,
                        issued_at=datetime.now(timezone.utc),
                        request_id=request_id,
                        done=True,
                        is_error=True,
                        error=we.message,
                        worker_error=we,
                    )
                    .model_dump_json(exclude_none=True)
                    .encode()
                )
                await self._nc.publish(msg.reply, data)
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
