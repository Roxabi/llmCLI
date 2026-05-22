"""Swap execution helper for LifecycleMixin."""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


async def _swap_drain_and_replace(
    mixin,
    msg,
    req,
    model_name: str,
    spec,
    _cap_engine,
    *,
    crash_messages: dict[str, str],
    crash_fallback: str,
) -> None:
    """Drain active requests, stop old instances, start new one, and reply."""
    instances: dict = mixin._instances  # type: ignore[attr-defined]
    catalog = mixin._catalog  # type: ignore[attr-defined]

    if model_name in instances:
        inst = instances[model_name]
        await mixin._reply_ok(
            msg,
            req,
            data={
                "model": model_name,
                "port": inst.port,
                "vram_used_mb": 0,
            },
        )
        log.info(
            "lifecycle.swap: noop trace_id=%s model=%s (same model)",
            req.trace_id,
            model_name,
        )
        return

    loop = asyncio.get_running_loop()
    executor = getattr(mixin, "_executor", None)

    mixin._draining.set()
    new_inst = None
    try:
        try:
            await asyncio.wait_for(
                mixin._wait_sem_idle(),
                timeout=mixin.drain_timeout,
            )
        except asyncio.TimeoutError:
            log.warning("lifecycle.swap: drain timeout — hard-cutting in-flight")

        for old_name, old_inst in list(instances.items()):
            old_engine = mixin._engine_for_spec(catalog.models[old_name])  # type: ignore[attr-defined]
            await loop.run_in_executor(executor, old_engine.stop, old_inst)
            del instances[old_name]

        new_inst = await loop.run_in_executor(executor, _cap_engine.start, spec)
    except Exception as exc:  # noqa: BLE001
        wire_msg = crash_messages.get(type(exc).__name__, crash_fallback)
        await mixin._reply_err(msg, req, "worker.crash", wire_msg, retryable=True)
        log.exception("worker.crash on swap to %s", model_name)
        return
    finally:
        mixin._draining.clear()

    instances[model_name] = new_inst
    vram_used_mb = 0
    vram_monitor = getattr(mixin, "_vram_monitor", None)
    if vram_monitor is not None:
        _, vram_used_mb = vram_monitor.sample()
        vram_used_mb = int(vram_used_mb)
    await mixin._reply_ok(
        msg,
        req,
        data={
            "model": model_name,
            "port": new_inst.port,
            "vram_used_mb": vram_used_mb,
        },
    )
    log.info(
        "lifecycle.swap: done trace_id=%s model=%s port=%s vram_used_mb=%d",
        req.trace_id,
        model_name,
        new_inst.port,
        vram_used_mb,
    )
