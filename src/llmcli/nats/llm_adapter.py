"""LlmNatsAdapter — llmCLI satellite for lyra.llm.generate.request.

Subscribes to the NATS queue group ``llm-workers``, receives LlmRequest
messages, routes them through the local llmCLI daemon (SWAP + STATUS),
then forwards to the running engine's OpenAI-compatible HTTP API.

Streaming requests publish LlmChunkEvent messages to the reply inbox.
Non-streaming requests publish a single LlmResponse.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from roxabi_nats.adapter_base import NatsAdapterBase

from llmcli.daemon import SOCKET_PATH, daemon_request
from roxabi_contracts.llm import SUBJECTS
from roxabi_contracts.llm.builders import build_llm_chunk, build_llm_response

log = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_STATUS_PORT_RE = re.compile(r"port=(\d+)")
_STATUS_MODEL_RE = re.compile(r"model=(\S+)")


class LlmNatsAdapter(NatsAdapterBase):
    """NATS satellite adapter for llmCLI.

    Receives ``LlmRequest`` messages, SWAPs to the configured model on
    startup, then calls the local OpenAI-compatible HTTP endpoint.
    """

    def __init__(
        self,
        *,
        model_name: str,
        litellm_url: str,
        litellm_key: str,
        socket_path: Path | None = None,
        max_concurrent: int = 4,
        reject_when_full: bool = False,
        heartbeat_interval: float = 5.0,
        drain_timeout: float = 30.0,
    ) -> None:
        super().__init__(
            SUBJECTS.generate_request,
            SUBJECTS.llm_workers,
            envelope_name="llm",
            schema_version=1,
            heartbeat_subject=SUBJECTS.heartbeat,
            heartbeat_interval=heartbeat_interval,
            drain_timeout=drain_timeout,
            inbox_prefix="_inbox.llmcli-llm",
        )
        self._model_name = model_name
        self._socket_path = socket_path or SOCKET_PATH
        self._max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._reject_when_full = reject_when_full
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._port: int | None = None
        self._loaded_model: str | None = None
        self._client = httpx.AsyncClient(
            base_url=litellm_url,
            headers={"Authorization": f"Bearer {litellm_key}"},
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    # ------------------------------------------------------------------
    # Lifecycle — SWAP before subscribing
    # ------------------------------------------------------------------

    async def run(self, nats_url: str, stop: asyncio.Event | None = None) -> None:
        await asyncio.get_event_loop().run_in_executor(
            self._executor, self._ensure_model
        )
        await super().run(nats_url, stop)

    def _ensure_model(self) -> None:
        """SWAP to configured model via daemon socket. Runs in executor."""
        reply = daemon_request(f"SWAP {self._model_name}", socket_path=self._socket_path)
        if not reply.startswith("OK"):
            raise RuntimeError(f"llmCLI daemon SWAP failed: {reply}")
        status = daemon_request("STATUS", socket_path=self._socket_path)
        self._port = self._parse_port(status)
        self._loaded_model = self._parse_model(status)
        log.info("llm_adapter: model=%s port=%d ready", self._loaded_model, self._port)

    # ------------------------------------------------------------------
    # NatsAdapterBase overrides
    # ------------------------------------------------------------------

    def _extra_subjects(self) -> list[str]:
        return [f"{self.subject}.{self._worker_id}"]

    def heartbeat_payload(self) -> dict:
        payload = super().heartbeat_payload()
        payload["model_loaded"] = self._loaded_model
        payload["active_requests"] = self._active_requests()
        return payload

    def _active_requests(self) -> int:
        return self._max_concurrent - self._sem._value  # type: ignore[attr-defined]

    async def _shutdown(self) -> None:
        try:
            await self._client.aclose()
        finally:
            await super()._shutdown()

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def handle(self, msg, payload: dict) -> None:
        request_id = str(payload.get("request_id", ""))
        if not _REQUEST_ID_RE.match(request_id):
            await self._err(msg, payload, "malformed_request: invalid request_id")
            return

        if self._reject_when_full and self._sem.locked():
            await self._err(msg, payload, "capacity_exceeded")
            return

        async with self._sem:
            await self._run_generation(msg, payload, request_id)

    async def _err(self, msg, payload: dict, error: str) -> None:
        safe_payload = {
            "request_id": str(payload.get("request_id", "unknown"))[:128],
            "trace_id": payload.get("trace_id"),
        }
        await self.reply(msg, build_llm_response(safe_payload, ok=False, error=error).encode())

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
        except Exception as exc:  # noqa: BLE001
            log.error("llm_adapter: generation error request_id=%s: %s", request_id, exc)
            if stream and msg.reply:
                try:
                    await self._nc.publish(
                        msg.reply,
                        build_llm_chunk(payload, done=True, is_error=True, error="generation_failed").encode(),
                    )
                except Exception:  # noqa: BLE001
                    pass
            else:
                try:
                    await self._err(msg, payload, "generation_failed")
                except Exception:  # noqa: BLE001
                    pass

    async def _stream_response(
        self, msg, payload: dict, body: dict, t0: float
    ) -> None:
        body = {**body, "stream": True}
        nc = self._nc

        async with self._client.stream("POST", "/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = (
                    chunk_data.get("choices", [{}])[0]
                    .get("delta", {})
                    .get("content")
                )
                if delta and msg.reply:
                    await nc.publish(
                        msg.reply,
                        build_llm_chunk(payload, delta=delta).encode(),
                    )

        duration_ms = int((time.monotonic() - t0) * 1000)
        if msg.reply:
            await nc.publish(
                msg.reply,
                build_llm_chunk(payload, done=True, duration_ms=duration_ms).encode(),
            )

    async def _blocking_response(
        self, msg, payload: dict, body: dict, t0: float
    ) -> None:
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_port(status: str) -> int:
        m = _STATUS_PORT_RE.search(status)
        if not m or m.group(1) == "none":
            raise RuntimeError(f"llmCLI STATUS did not return a port: {status!r}")
        return int(m.group(1))

    @staticmethod
    def _parse_model(status: str) -> str:
        m = _STATUS_MODEL_RE.search(status)
        return m.group(1) if m else "unknown"
