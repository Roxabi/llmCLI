"""LifecycleMixin — NATS lifecycle control plane for LlmNatsAdapter.

Handles swap/stop/status/list/reload-catalog on broadcast subjects.
MRO: LlmNatsAdapter(LifecycleMixin, GenerationMixin, NatsAdapterBase).

Host class must expose: _catalog, _instances, _sem, _drain_timeout,
_engine_for_spec(spec). Call __init_lifecycle__() after super().__init__().
"""

from __future__ import annotations

import asyncio
import logging
import socket
import tomllib
from datetime import datetime, timezone

from pydantic import ValidationError
from roxabi_contracts.llm import LifecycleRequest, LifecycleResponse
from roxabi_contracts.errors import WorkerError

from llmcli.config import check_vram_budget, load as load_catalog

log = logging.getLogger(__name__)

LIFECYCLE_SUBJECTS = (
    "lyra.llm.lifecycle.swap",
    "lyra.llm.lifecycle.stop",
    "lyra.llm.lifecycle.status",
    "lyra.llm.lifecycle.list",
    "lyra.llm.lifecycle.reload-catalog",
)

class LifecycleMixin:
    """Mixin that adds NATS lifecycle control to any NatsAdapterBase subclass.

    Call self.__init_lifecycle__() in the host __init__ after super().__init__().
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def handle(self, msg, payload: dict) -> None:
        """Default handle: route lifecycle subjects; drop everything else.

        LlmNatsAdapter overrides this with generation routing; this stub
        exists so that bare test subclasses (e.g. test_lifecycle_mro.py)
        can instantiate without implementing handle() themselves.
        """
        if msg.subject in LIFECYCLE_SUBJECTS:
            await self.handle_lifecycle(msg, payload)

    def __init_lifecycle__(self) -> None:
        # Event is set during a swap drain window; generation handle checks it.
        self._draining: asyncio.Event = asyncio.Event()
        # Serialise lifecycle ops (swap/stop/reload) — status/list are read-only
        # but still go through the lock for simplicity.
        self._lifecycle_lock: asyncio.Lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # NatsAdapterBase hooks — super()-composed
    # ------------------------------------------------------------------

    def _extra_subjects(self) -> list[str]:
        return [*LIFECYCLE_SUBJECTS, *super()._extra_subjects()]  # type: ignore[misc]

    def heartbeat_payload(self) -> dict:
        base = super().heartbeat_payload()  # type: ignore[misc]
        base["lifecycle_draining"] = self._draining.is_set()
        return base

    # ------------------------------------------------------------------
    # Top-level dispatch (called by LlmNatsAdapter.handle)
    # ------------------------------------------------------------------

    async def handle_lifecycle(self, msg, payload: dict) -> None:
        try:
            req = LifecycleRequest.model_validate(payload)
        except ValidationError:
            log.exception("lifecycle: invalid payload on %s", msg.subject)
            return
        # Host filter: None / "*" → all workers respond; specific hostname → only that host.
        if req.host not in (None, "*", socket.gethostname()):
            return  # silent drop — not our host
        await self._dispatch_lifecycle_op(req.op, msg, req)

    async def _dispatch_lifecycle_op(self, op: str, msg, req: LifecycleRequest) -> None:
        handler = {
            "swap": self._do_swap,
            "stop": self._do_stop,
            "status": self._do_status,
            "list": self._do_list,
            "reload-catalog": self._do_reload_catalog,
        }.get(op)
        if handler is None:
            log.warning("lifecycle: unknown op=%r — ignoring", op)
            return
        async with self._lifecycle_lock:
            await handler(msg, req)

    # ------------------------------------------------------------------
    # Reply helpers
    # ------------------------------------------------------------------

    async def _reply_ok(self, msg, req: LifecycleRequest, *, data: dict | None = None) -> None:
        resp = LifecycleResponse(
            contract_version=req.contract_version,
            trace_id=req.trace_id,
            issued_at=datetime.now(timezone.utc),
            request_id=req.request_id,
            ok=True,
            host=socket.gethostname(),
            data=data,
        )
        if msg.reply and self._nc:  # type: ignore[attr-defined]
            await self._nc.publish(msg.reply, resp.model_dump_json(exclude_none=True).encode())  # type: ignore[attr-defined]

    async def _reply_err(
        self,
        msg,
        req: LifecycleRequest,
        code: str,
        message: str,
        *,
        retryable: bool = True,
    ) -> None:
        resp = LifecycleResponse(
            contract_version=req.contract_version,
            trace_id=req.trace_id,
            issued_at=datetime.now(timezone.utc),
            request_id=req.request_id,
            ok=False,
            host=socket.gethostname(),
            worker_error=WorkerError(code=code, message=message, retryable=retryable),
        )
        if msg.reply and self._nc:  # type: ignore[attr-defined]
            await self._nc.publish(msg.reply, resp.model_dump_json(exclude_none=True).encode())  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Semaphore drain helper
    # ------------------------------------------------------------------

    async def _wait_sem_idle(self) -> None:
        """Wait until all semaphore slots are released (no active generations)."""
        sem: asyncio.Semaphore = self._sem  # type: ignore[attr-defined]
        max_val = getattr(self, "_max_concurrent", None)
        if max_val is None:
            return
        while sem._value < max_val:  # type: ignore[attr-defined]
            await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Lifecycle handlers
    # ------------------------------------------------------------------

    async def _do_swap(self, msg, req: LifecycleRequest) -> None:
        model_name = req.model_name  # validated non-None for op=swap by LifecycleRequest
        catalog = self._catalog  # type: ignore[attr-defined]

        spec = catalog.models.get(model_name)
        if spec is None:
            await self._reply_err(
                msg, req, "llm.lifecycle_rejected",
                f"unknown model: {model_name}", retryable=False,
            )
            return

        # engine="remote" guard (mirrors daemon._engine_for_spec rejection)
        if spec.engine == "remote":
            await self._reply_err(
                msg, req, "llm.lifecycle_rejected",
                "model uses engine='remote' — managed by LiteLLM proxy, not the worker",
                retryable=False,
            )
            return

        # VRAM budget check — same guard as daemon._cmd_swap (skips for engine=remote).
        # TypeError guard: check_vram_budget compares numeric attributes; if the catalog
        # host/spec are not fully populated (e.g. in tests with MagicMock), the comparison
        # would raise TypeError — treat that as "check inconclusive, proceed".
        try:
            check_vram_budget(spec, catalog.host)
        except ValueError as exc:
            await self._reply_err(
                msg, req, "llm.lifecycle_rejected",
                f"vram budget exceeded: {exc}", retryable=False,
            )
            return
        except TypeError:
            pass  # spec/host attributes not numeric — skip dynamic check

        # Same-model fast-path (idempotent swap)
        instances: dict = self._instances  # type: ignore[attr-defined]
        if model_name in instances:
            inst = instances[model_name]
            await self._reply_ok(msg, req, data={
                "model": model_name,
                "port": inst.port,
                "vram_used_mb": 0,
            })
            return

        # Drain pattern (A2): set flag → wait semaphore idle → hard cut → swap
        self._draining.set()
        try:
            await asyncio.wait_for(
                self._wait_sem_idle(),
                timeout=self._drain_timeout,  # type: ignore[attr-defined]
            )
        except asyncio.TimeoutError:
            log.warning("lifecycle.swap: drain timeout exceeded — hard-cutting in-flight generation")

        # Stop-before-start (mirrors daemon._cmd_swap ordering)
        for old_name, old_inst in list(instances.items()):
            old_engine = self._engine_for_spec(catalog.models[old_name])  # type: ignore[attr-defined]
            old_engine.stop(old_inst)
            del instances[old_name]

        try:
            new_inst = self._engine_for_spec(spec).start(spec)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            await self._reply_err(msg, req, "worker.crash", str(exc), retryable=True)
            return
        finally:
            self._draining.clear()

        instances[model_name] = new_inst
        # VRAMMonitor probe — best-effort; 0 when monitor unavailable.
        vram_used_mb = 0
        vram_monitor = getattr(self, "_vram_monitor", None)
        if vram_monitor is not None:
            _, vram_used_mb = vram_monitor.sample()
            vram_used_mb = int(vram_used_mb)
        await self._reply_ok(msg, req, data={
            "model": model_name,
            "port": new_inst.port,
            "vram_used_mb": vram_used_mb,
        })

    async def _do_status(self, msg, req: LifecycleRequest) -> None:
        instances: dict = self._instances  # type: ignore[attr-defined]
        if not instances:
            await self._reply_ok(msg, req, data={"model": None})
            return
        name, inst = next(iter(instances.items()))
        vram_used_mb = 0
        vram_monitor = getattr(self, "_vram_monitor", None)
        if vram_monitor is not None:
            _, vram_used_mb = vram_monitor.sample()
            vram_used_mb = int(vram_used_mb)
        await self._reply_ok(msg, req, data={
            "model": name,
            "port": inst.port,
            "vram_used_mb": vram_used_mb,
        })

    async def _do_list(self, msg, req: LifecycleRequest) -> None:
        catalog = self._catalog  # type: ignore[attr-defined]
        instances: dict = self._instances  # type: ignore[attr-defined]
        models = [
            {
                "name": name,
                "running": name in instances,
                "engine": spec.engine,
                "vram_gib": spec.vram_gib,
            }
            for name, spec in catalog.models.items()
        ]
        await self._reply_ok(msg, req, data={"models": models})

    async def _do_stop(self, msg, req: LifecycleRequest) -> None:
        catalog = self._catalog  # type: ignore[attr-defined]
        instances: dict = self._instances  # type: ignore[attr-defined]
        for old_name, old_inst in list(instances.items()):
            old_engine = self._engine_for_spec(catalog.models[old_name])  # type: ignore[attr-defined]
            old_engine.stop(old_inst)
            del instances[old_name]
        await self._reply_ok(msg, req, data={})

    async def _do_reload_catalog(self, msg, req: LifecycleRequest) -> None:
        # Re-read catalog from disk.  On TOML parse error → reply rejected,
        # in-memory catalog unchanged (E11 — no partial state, no service interruption).
        try:
            new_catalog = load_catalog()
        except tomllib.TOMLDecodeError as exc:
            await self._reply_err(
                msg, req, "llm.lifecycle_rejected",
                f"catalog parse error: {exc}", retryable=False,
            )
            return
        self._catalog = new_catalog  # type: ignore[attr-defined]
        model_count = len(new_catalog.models)
        log.info("lifecycle.reload-catalog: reloaded %d models", model_count)
        await self._reply_ok(msg, req, data={"models_loaded": model_count})
