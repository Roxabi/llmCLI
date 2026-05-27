"""LifecycleMixin — NATS lifecycle control plane for LlmNatsAdapter.

Handles swap/stop/status/list/reload-catalog on broadcast subjects.
MRO: LlmNatsAdapter(LifecycleMixin, GenerationMixin, NatsAdapterBase).

Host class must expose: _catalog, _instances, _sem, drain_timeout,
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

from llmcli.auth.store import XAI_CREDENTIALS_PATH
from llmcli.config import check_vram_budget, load as load_catalog
from llmcli.support.litellm_config import _XAI_OAUTH_MODELS

log = logging.getLogger(__name__)

# B3: wire-safe error messages keyed by exception class name.
# str(exc) is never sent on the wire — it may contain file paths, addresses,
# or internal state that operators should not expose to callers.
_CRASH_MESSAGES: dict[str, str] = {
    "FileNotFoundError": "model file not found",
    "PermissionError": "permission denied",
    "TimeoutError": "engine startup timed out",
    "ConnectionError": "engine connection failed",
    "TOMLDecodeError": "catalog parse error",
    "ValueError": "invalid configuration",
}
_CRASH_FALLBACK = "engine start failed"

LIFECYCLE_SUBJECTS = (
    "lyra.llm.lifecycle.swap",
    "lyra.llm.lifecycle.stop",
    "lyra.llm.lifecycle.status",
    "lyra.llm.lifecycle.list",
    "lyra.llm.lifecycle.reload-catalog",
)


class LifecycleMixin:
    """Mixin that adds NATS lifecycle control to any NatsAdapterBase subclass."""

    async def handle(self, msg, payload: dict) -> None:
        # Default stub for bare test subclasses; LlmNatsAdapter overrides this
        # with generation routing.
        if msg.subject in LIFECYCLE_SUBJECTS:
            await self.handle_lifecycle(msg, payload)

    def __init_lifecycle__(self) -> None:
        self._draining: asyncio.Event = asyncio.Event()
        self._lifecycle_lock: asyncio.Lock = asyncio.Lock()

    def _extra_subjects(self) -> list[str]:
        return [*LIFECYCLE_SUBJECTS, *super()._extra_subjects()]  # type: ignore[misc]

    def heartbeat_payload(self) -> dict:
        base = super().heartbeat_payload()  # type: ignore[misc]
        base["lifecycle_draining"] = self._draining.is_set()
        return base

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

    async def _wait_sem_idle(self) -> None:
        """Wait until all semaphore slots are released (no active generations)."""
        sem: asyncio.Semaphore = self._sem  # type: ignore[attr-defined]
        max_val = getattr(self, "_max_concurrent", None)
        if max_val is None:
            return
        while sem._value < max_val:  # type: ignore[attr-defined]
            await asyncio.sleep(0)

    async def _do_swap(self, msg, req: LifecycleRequest) -> None:
        model_name = req.model_name
        log.info(
            "lifecycle.swap: start trace_id=%s request_id=%s model=%s host=%s",
            req.trace_id,
            req.request_id,
            model_name,
            req.host,
        )
        catalog = self._catalog  # type: ignore[attr-defined]

        spec = catalog.models.get(model_name)
        if spec is None:
            await self._reply_err(
                msg,
                req,
                "llm.lifecycle_rejected",
                f"unknown model: {model_name}",
                retryable=False,
            )
            return

        if spec.engine == "remote":
            await self._reply_err(
                msg,
                req,
                "llm.lifecycle_rejected",
                "model uses engine='remote' — managed by LiteLLM proxy, not the worker",
                retryable=False,
            )
            return

        # ADR-006 Override Protocol: refuse swap when the engine does not support it.
        _cap_engine = self._engine_for_spec(spec)  # type: ignore[attr-defined]
        if not _cap_engine.supports_swap():
            await self._reply_err(
                msg,
                req,
                "llm.unsupported_operation",
                f"engine '{spec.engine}' does not support swap",
                retryable=False,
            )
            return

        try:
            check_vram_budget(spec, catalog.host)
        except ValueError as exc:
            await self._reply_err(
                msg,
                req,
                "llm.lifecycle_rejected",
                f"vram budget exceeded: {exc}",
                retryable=False,
            )
            return

        await self._swap_drain_and_replace(msg, req, model_name, spec, _cap_engine)

    async def _swap_drain_and_replace(
        self,
        msg,
        req: LifecycleRequest,
        model_name: str,
        spec,
        _cap_engine,
    ) -> None:
        """Drain active requests, stop old instances, start new one, and reply."""
        from llmcli.nats._swap import _swap_drain_and_replace as _swap_impl

        await _swap_impl(
            self,
            msg,
            req,
            model_name,
            spec,
            _cap_engine,
            crash_messages=_CRASH_MESSAGES,
            crash_fallback=_CRASH_FALLBACK,
        )

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
        await self._reply_ok(
            msg,
            req,
            data={
                "model": name,
                "port": inst.port,
                "vram_used_mb": vram_used_mb,
            },
        )

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
        if XAI_CREDENTIALS_PATH.exists():
            for model_name in _XAI_OAUTH_MODELS:
                models.append({
                    "name": model_name,
                    "running": False,  # forwarder liveness checked via /health, not surfaced here
                    "engine": "oauth-forwarder",
                    "vram_gib": 0,
                })
        await self._reply_ok(msg, req, data={"models": models})

    async def _do_stop(self, msg, req: LifecycleRequest) -> None:
        log.info(
            "lifecycle.stop: start trace_id=%s request_id=%s host=%s",
            req.trace_id,
            req.request_id,
            req.host,
        )
        catalog = self._catalog  # type: ignore[attr-defined]
        instances: dict = self._instances  # type: ignore[attr-defined]
        loop = asyncio.get_running_loop()
        executor = getattr(self, "_executor", None)
        stopped: list[str] = []
        for old_name, old_inst in list(instances.items()):
            old_engine = self._engine_for_spec(catalog.models[old_name])  # type: ignore[attr-defined]
            await loop.run_in_executor(executor, old_engine.stop, old_inst)
            del instances[old_name]
            stopped.append(old_name)
        await self._reply_ok(msg, req, data={})
        log.info("lifecycle.stop: done trace_id=%s stopped=%s", req.trace_id, stopped)

    async def _do_reload_catalog(self, msg, req: LifecycleRequest) -> None:
        # load_catalog() can raise TOMLDecodeError (malformed TOML),
        # FileNotFoundError (missing file), or ValueError (ModelSpec/HostSettings
        # validation) — all three must reply rather than escape the lock.
        try:
            new_catalog = load_catalog()
        except (tomllib.TOMLDecodeError, FileNotFoundError, ValueError) as exc:
            await self._reply_err(
                msg,
                req,
                "llm.lifecycle_rejected",
                f"catalog load error: {exc}",
                retryable=False,
            )
            return
        self._catalog = new_catalog  # type: ignore[attr-defined]
        model_count = len(new_catalog.models)
        log.info("lifecycle.reload-catalog: reloaded %d models", model_count)
        await self._reply_ok(msg, req, data={"models_loaded": model_count})
