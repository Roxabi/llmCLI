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
from concurrent.futures import ThreadPoolExecutor

import httpx
from roxabi_contracts.telemetry import ATTR_MODEL, MessageLifecycleHooks
from roxabi_nats.adapter_base import NatsAdapterBase

from llmcli.config import load as load_catalog
from llmcli.gpu import VRAMMonitor
from llmcli.nats._generation import GenerationMixin, _REQUEST_ID_RE
from llmcli.nats._lifecycle import LIFECYCLE_SUBJECTS, LifecycleMixin
from roxabi_contracts.llm import SUBJECTS

log = logging.getLogger(__name__)


class LlmNatsAdapter(LifecycleMixin, GenerationMixin, NatsAdapterBase):
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
        max_concurrent: int = 4,
        reject_when_full: bool = False,
        heartbeat_interval: float = 5.0,
        drain_timeout: float = 30.0,
        lifecycle_hooks: MessageLifecycleHooks | None = None,
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
            wait_ready=False,  # C2: workers skip JetStream KV probe
            lifecycle_hooks=lifecycle_hooks,
        )
        self.__init_lifecycle__()  # sets _draining + _lifecycle_lock
        self._model_name = model_name
        self._max_concurrent = max_concurrent
        self._sem = asyncio.Semaphore(max_concurrent)
        self._reject_when_full = reject_when_full
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._loaded_model: str | None = None
        self._vram_monitor: VRAMMonitor | None = None
        self._client = httpx.AsyncClient(
            base_url=litellm_url,
            headers={"Authorization": f"Bearer {litellm_key}"},
            timeout=httpx.Timeout(connect=5.0, read=120.0, write=10.0, pool=5.0),
        )
        # Lifecycle state — _instances and _catalog populated on run()
        self._instances: dict = {}
        self._catalog = None

    # ------------------------------------------------------------------
    # Lifecycle — SWAP before subscribing
    # ------------------------------------------------------------------

    async def run(self, nats_url: str, stop: asyncio.Event | None = None) -> None:
        # B1: load catalog into _catalog so lifecycle handlers can dereference it.
        # Without this, _do_swap/_do_list/_do_stop crash with AttributeError on first
        # use until a reload-catalog op arrives.
        self._catalog = load_catalog()
        await asyncio.get_running_loop().run_in_executor(self._executor, self._ensure_model)
        self._vram_monitor = VRAMMonitor()
        self._vram_monitor.open()
        await super().run(nats_url, stop)

    def _ensure_model(self) -> None:
        """Record configured model name. Runs in executor.

        Slice 6 (AF_UNIX daemon removed, #34): the daemon socket no longer exists.
        Model loading is managed externally (llama-server started by the operator).
        """
        self._loaded_model = self._model_name
        log.info("llm_adapter: model=%s assumed loaded externally", self._loaded_model)

    # ------------------------------------------------------------------
    # NatsAdapterBase overrides
    # ------------------------------------------------------------------

    def telemetry_attributes(
        self, payload: dict, result: object | None
    ) -> dict[str, str]:
        del result
        return {ATTR_MODEL: str(self._loaded_model or self._model_name)}

    def _engine_for_spec(self, spec):
        """Dispatch on spec.engine — same remote-guard as daemon._engine_for_spec."""
        if spec.engine == "remote":
            raise ValueError(
                f"Model '{spec.name}' uses engine='remote' — managed by LiteLLM proxy."
            )
        from llmcli.engines import get_engine

        return get_engine(spec)

    def heartbeat_payload(self) -> dict:
        payload = super().heartbeat_payload()
        payload["model_loaded"] = self._loaded_model
        payload["active_requests"] = self._active_requests()

        free_mb, used_mb = (
            self._vram_monitor.sample() if self._vram_monitor is not None else (0.0, 0.0)
        )
        payload["vram_free_mb"] = int(free_mb)
        payload["vram_used_mb"] = int(used_mb)
        return payload

    def _active_requests(self) -> int:
        return self._max_concurrent - self._sem._value  # type: ignore[attr-defined]

    async def _shutdown(self) -> None:
        try:
            if self._vram_monitor is not None:
                self._vram_monitor.close()
                self._vram_monitor = None
            await self._client.aclose()
        finally:
            await super()._shutdown()

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    async def handle(self, msg, payload: dict) -> None:
        if getattr(msg, "subject", None) in LIFECYCLE_SUBJECTS:
            await self.handle_lifecycle(msg, payload)
            return
        # B5 / AC-5: reject new generations while a swap drain is in flight so the
        # drain window can't be extended indefinitely by incoming requests.
        if self._draining.is_set():
            await self._err(
                msg,
                payload,
                self._make_worker_error("worker.capacity", "drain in progress", retryable=True),
            )
            return
        # Generation path (preserves MRO chain — S5)
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
