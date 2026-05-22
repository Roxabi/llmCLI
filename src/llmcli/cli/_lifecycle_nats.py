"""NATS lifecycle request helper — shared by lifecycle.py and lifecycle_extra.py.

Inline per DP2 consensus: no cli/_nats_client.py in PR-1.
Tech debt: extract + consolidate in v2 (Roxabi/llmCLI#61).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import typer

from llmcli.cli._app import err_console


async def lifecycle_nats_request(
    subject: str,
    op: str,
    host: str | None,
    timeout: float,
    *,
    model_name: str | None = None,
    allow_anonymous: bool = False,
):
    """Send a LifecycleRequest via NATS and return the LifecycleResponse.

    On timeout or no-reply, prints a warning and raises typer.Exit(1).
    Pass allow_anonymous=True to skip the operator creds check (CI/dev only).
    """
    from nats.aio.client import Client as NATS  # type: ignore[import]
    from roxabi_contracts.llm import LifecycleRequest, LifecycleResponse

    nc = NATS()
    creds_path = Path("~/.roxabi/llmcli/nkeys/operator.creds").expanduser()
    nats_url = os.environ.get("NATS_URL", "nats://localhost:4222")
    # Fail-closed: missing creds requires explicit --allow-anonymous flag to connect.
    # Use --allow-anonymous for CI/dev only — do not use in production.
    if not creds_path.exists() and not allow_anonymous:
        err_console.print(
            f"[red]NATS operator credentials not found at {creds_path}. "
            f"Run lyra-acl genkeys (Slice 0) or pass "
            f"--allow-anonymous to connect without credentials (CI/dev only).[/red]"
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
            op=op,
            model_name=model_name,
        )
        try:
            msg = await nc.request(subject, req.model_dump_json().encode(), timeout=timeout)
        except Exception as exc:
            err_console.print(
                f"[yellow]No worker responded for host={host!r}. "
                f"Check hostname or NATS connectivity: {exc}[/yellow]"
            )
            raise typer.Exit(code=1) from exc
        return LifecycleResponse.model_validate_json(msg.data)
    finally:
        await nc.drain()
