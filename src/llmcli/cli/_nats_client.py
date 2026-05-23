"""Reusable NATS client helper for llmCLI lifecycle commands."""

from __future__ import annotations

import asyncio
import os
import socket
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import typer

from roxabi_contracts.errors import WorkerError
from roxabi_contracts.llm import LifecycleRequest, LifecycleResponse

from llmcli.cli._app import err_console
from llmcli.config import apply_nats_env_from_config


@dataclass
class FleetResult:
    responses: list[LifecycleResponse] = field(default_factory=list)
    errors: list[tuple[str, WorkerError]] = field(default_factory=list)
    timeout_reached: bool = False
    elapsed_ms: float = 0.0


class NatsClient:
    def __init__(self, *, allow_anonymous: bool = False) -> None:
        self.allow_anonymous = allow_anonymous
        self._nc: "NATS" | None = None

    async def connect(self) -> None:
        from nats.aio.client import Client as NATS

        self._nc = NATS()
        creds_path = Path("~/.roxabi/llmcli/nkeys/operator.creds").expanduser()
        apply_nats_env_from_config()
        nats_url = os.environ.get("LLMCLI_NATS_URL", "nats://localhost:4222")

        if not creds_path.exists() and not self.allow_anonymous:
            err_console.print(
                f"[red]NATS operator credentials not found at {creds_path}. "
                f"Run lyra-acl genkeys (Slice 0) or pass "
                f"--allow-anonymous to connect without credentials (CI/dev only).[/red]"
            )
            raise typer.Exit(code=1)

        await self._nc.connect(
            nats_url,
            nkeys_seed_str=creds_path.read_text().strip() if creds_path.exists() else None,
            inbox_prefix="_inbox.llm-operator",
        )

    async def request(
        self,
        subject: str,
        op: str,
        host: str | None,
        timeout: float,
        *,
        model_name: str | None = None,
    ) -> LifecycleResponse:
        if self._nc is None:
            raise RuntimeError("NatsClient not connected. Call connect() first.")

        req = LifecycleRequest(
            contract_version="1",
            trace_id=uuid4().hex,
            issued_at=datetime.now(timezone.utc).isoformat(),
            request_id=uuid4().hex,
            host=host or socket.gethostname(),
            op=op,
            model_name=model_name,
        )
        msg = await self._nc.request(subject, req.model_dump_json().encode(), timeout=timeout)
        return LifecycleResponse.model_validate_json(msg.data)

    async def request_fleet(
        self,
        subject: str,
        op: str,
        timeout: float,
        *,
        model_name: str | None = None,
    ) -> FleetResult:
        if self._nc is None:
            raise RuntimeError("NatsClient not connected. Call connect() first.")

        req = LifecycleRequest(
            contract_version="1",
            trace_id=uuid4().hex,
            issued_at=datetime.now(timezone.utc).isoformat(),
            request_id=uuid4().hex,
            host="*",
            op=op,
            model_name=model_name,
        )
        payload = req.model_dump_json().encode()

        inbox = self._nc.new_inbox()
        sub = await self._nc.subscribe(inbox)
        try:
            await self._nc.publish(subject, payload, reply=inbox)

            result = FleetResult()
            start = time.perf_counter()

            while True:
                elapsed = time.perf_counter() - start
                remaining = timeout - elapsed
                if remaining <= 0:
                    result.timeout_reached = True
                    break

                try:
                    msg = await asyncio.wait_for(sub.next_msg(), timeout=remaining)
                except asyncio.TimeoutError:
                    result.timeout_reached = True
                    break

                try:
                    resp = LifecycleResponse.model_validate_json(msg.data)
                except Exception:
                    continue

                if resp.ok:
                    result.responses.append(resp)
                else:
                    we = resp.worker_error
                    if we is None:
                        we = WorkerError(
                            code="unknown",
                            message=resp.error or "unknown error",
                        )
                    result.errors.append((resp.host or "unknown", we))

            result.elapsed_ms = (time.perf_counter() - start) * 1000
            return result
        finally:
            await sub.unsubscribe()

    async def close(self) -> None:
        if self._nc is not None:
            await self._nc.drain()
            self._nc = None
