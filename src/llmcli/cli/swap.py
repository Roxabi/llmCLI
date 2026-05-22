from __future__ import annotations

import asyncio
import os
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

import typer

from llmcli.cli._app import app, console, err_console


# ---------------------------------------------------------------------------
# NATS swap implementation (inline per DP2 consensus — no _nats_client.py)
# ---------------------------------------------------------------------------


async def _swap_via_nats(name: str, host: str, timeout: float) -> None:
    from nats.aio.client import Client as NATS  # type: ignore[import]
    from roxabi_contracts.llm import LifecycleRequest, LifecycleResponse
    from roxabi_contracts.llm.subjects import SUBJECTS

    nc = NATS()
    creds_path = Path("~/.config/llmcli/nkeys/operator.creds").expanduser()
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    # Fail-closed: missing creds connects anonymously against permissive brokers.
    # CI / pre-Slice-0 dev set LLMCLI_NATS_SKIP_CREDS=1 to opt in explicitly.
    if not creds_path.exists() and os.environ.get("LLMCLI_NATS_SKIP_CREDS", "").lower() not in (
        "1",
        "true",
    ):
        err_console.print(
            f"[red]NATS operator credentials not found at {creds_path}. "
            f"Run lyra-acl genkeys (Slice 0) or export "
            f"LLMCLI_NATS_SKIP_CREDS=1 to allow anonymous connect (dev/CI only).[/red]"
        )
        raise typer.Exit(code=1)
    await nc.connect(
        nats_url,
        user_credentials=str(creds_path) if creds_path.exists() else None,
    )
    try:
        req = LifecycleRequest(
            contract_version="1",
            trace_id=uuid4().hex,
            issued_at=datetime.now(timezone.utc).isoformat(),
            request_id=uuid4().hex,
            host=host,
            op="swap",
            model_name=name,
        )
        try:
            msg = await nc.request(
                SUBJECTS.lifecycle_swap,
                req.model_dump_json().encode(),
                timeout=timeout,
            )
        except Exception as exc:
            err_console.print(f"[red]NATS unreachable or no worker responded: {exc}[/red]")
            raise typer.Exit(code=1) from exc

        resp = LifecycleResponse.model_validate_json(msg.data)
        if not resp.ok:
            we = resp.worker_error
            if we:
                err_console.print(f"[red]{we.code}: {we.message}[/red]")
            else:
                err_console.print(f"[red]swap failed: {resp.error}[/red]")
            raise typer.Exit(code=1)

        data = resp.data or {}
        console.print(
            f"OK swapped to {data.get('model', name)} "
            f"(port={data.get('port', '?')}, "
            f"vram={data.get('vram_used_mb', 0)}MB)"
        )
    finally:
        await nc.drain()


# ---------------------------------------------------------------------------
# swap
# ---------------------------------------------------------------------------


@app.command()
def swap(
    name: str,
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Target hostname; default = local hostname",
    ),
    timeout: float = typer.Option(
        300.0,
        "--timeout",
        help="Seconds to wait for the daemon to load the model (default: 300s for large models).",
    ),
) -> None:
    """Hot-swap the running model via NATS."""
    import llmcli.cli as _cli

    catalog = _cli.config.load()

    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(f"[red]Unknown model '{name}'. Available: {available}[/red]")
        raise typer.Exit(code=1)

    asyncio.run(_swap_via_nats(name, host=host or socket.gethostname(), timeout=timeout))
