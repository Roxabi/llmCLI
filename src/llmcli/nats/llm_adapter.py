"""LlmNatsAdapter — llmCLI satellite for lyra.llm.generate.request.

Subscribes to the NATS queue group ``llm-workers``, receives LlmRequest
messages, routes them through the local llmCLI daemon (SWAP + STATUS),
then forwards to the LiteLLM proxy (``LLMCLI_LITELLM_URL``) with Bearer
auth, which owns catalog/aliasing/fallback.

Streaming requests publish LlmChunkEvent messages to the reply inbox.
Non-streaming requests publish a single LlmResponse.

HTTP generation logic lives in ``_generation.GenerationMixin``; only
NATS lifecycle, model-swap, and heartbeat remain here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from roxabi_nats.adapter_base import NatsAdapterBase

from llmcli.daemon import SOCKET_PATH, daemon_request
from llmcli.nats._generation import GenerationMixin
from roxabi_contracts.llm import SUBJECTS

log = logging.getLogger(__name__)

_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_STATUS_MODEL_RE = re.compile(r"model=(\S+)")


class LlmNatsAdapter(GenerationMixin, NatsAdapterBase):
    """NATS satellite adapter for llmCLI.

    Receives ``LlmRequest`` messages, SWAPs to the configured model on
    startup, then calls the local OpenAI-compatible HTTP endpoint.

    Generation methods (``_run_generation``, ``_stream_response``,
    ``_blocking_response``, ``_err``, ``_make_worker_error``) are
    provided by ``GenerationMixin``.
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
        self._loaded_model: str | None = None
        self._nvml_handle: object | None = None
        self._nvml_init_failed = False
        self._client = httpx.AsyncClient(
            base_url=litellm_url,
            headers={"Authorization": f"Bearer {litellm_key}"},
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )

    # ------------------------------------------------------------------
    # Lifecycle — SWAP before subscribing
    # ------------------------------------------------------------------

    async def run(self, nats_url: str, stop: asyncio.Event | None = None) -> None:
        await asyncio.get_running_loop().run_in_executor(self._executor, self._ensure_model)
        await super().run(nats_url, stop)

    def _ensure_model(self) -> None:
        """SWAP to configured model via daemon socket. Runs in executor.

        When the socket is absent the worker is assumed to be running in
        remote-worker mode where the model is loaded externally (e.g. by
        llama-server started by the operator or supervisor). The SWAP/STATUS
        pre-check is skipped so the container starts without a host daemon.
        """
        if not Path(self._socket_path).exists():
            log.info(
                "llm_adapter: daemon socket not found at %s — "
                "skipping SWAP/STATUS (remote-worker mode, model assumed loaded externally)",
                self._socket_path,
            )
            self._loaded_model = self._model_name
            return
        try:
            reply = daemon_request(f"SWAP {self._model_name}", socket_path=self._socket_path)
            if not reply.startswith("OK"):
                raise RuntimeError(f"llmCLI daemon SWAP failed: {reply}")
            status = daemon_request("STATUS", socket_path=self._socket_path)
            self._loaded_model = self._parse_model(status)
            log.info("llm_adapter: model=%s ready", self._loaded_model)
        except OSError as exc:
            # Socket disappeared mid-flight (daemon restarted, socket removed).
            # Log and continue — model is presumed loaded; don't crash the worker.
            log.info(
                "llm_adapter: daemon unreachable (%s) — "
                "skipping SWAP/STATUS, assuming model already loaded",
                exc,
            )
            self._loaded_model = self._model_name

    # ------------------------------------------------------------------
    # NatsAdapterBase overrides
    # ------------------------------------------------------------------

    def _extra_subjects(self) -> list[str]:
        return []

    def _get_nvml_handle(self) -> object | None:
        """Lazy-init nvml device handle; cached for heartbeat reuse."""
        if self._nvml_init_failed:
            return None
        if self._nvml_handle is not None:
            return self._nvml_handle
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return self._nvml_handle
        except Exception:  # noqa: BLE001
            self._nvml_init_failed = True
            return None

    def heartbeat_payload(self) -> dict:
        payload = super().heartbeat_payload()
        payload["model_loaded"] = self._loaded_model
        payload["active_requests"] = self._active_requests()

        from llmcli.gpu import probe_free_vram_gib

        free_gib = probe_free_vram_gib()
        payload["vram_free_mb"] = int(free_gib * 1024)
        handle = self._get_nvml_handle()
        if handle is not None:
            try:
                import pynvml  # type: ignore[import-untyped]

                total_mb = pynvml.nvmlDeviceGetMemoryInfo(handle).total // (1024 * 1024)
                payload["vram_used_mb"] = max(0, int(total_mb) - payload["vram_free_mb"])
            except Exception:  # noqa: BLE001
                payload["vram_used_mb"] = 0
        else:
            payload["vram_used_mb"] = 0
        return payload

    def _active_requests(self) -> int:
        return self._max_concurrent - self._sem._value  # type: ignore[attr-defined]

    async def _shutdown(self) -> None:
        try:
            if self._nvml_handle is not None:
                try:
                    import pynvml  # type: ignore[import-untyped]

                    pynvml.nvmlShutdown()
                except Exception:  # noqa: BLE001
                    pass
                self._nvml_handle = None
            await self._client.aclose()
        finally:
            await super()._shutdown()

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def handle(self, msg, payload: dict) -> None:
        request_id = str(payload.get("request_id", ""))
        if not _REQUEST_ID_RE.match(request_id):
            await self._err(
                msg,
                payload,
                self._make_worker_error("transport.parse", "invalid request_id", retryable=False),
            )
            return

        if self._reject_when_full and self._sem.locked():
            await self._err(
                msg,
                payload,
                self._make_worker_error("worker.internal", "capacity_exceeded", retryable=True),
            )
            return

        async with self._sem:
            await self._run_generation(msg, payload, request_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_model(status: str) -> str | None:
        m = _STATUS_MODEL_RE.search(status)
        if not m:
            return None
        val = m.group(1)
        return None if val == "none" else val
