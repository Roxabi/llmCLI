from __future__ import annotations

import asyncio
import socket
from typing import Optional

import typer

from llmcli.cli._app import app, console, err_console
from llmcli.cli._nats_client import NatsClient
from roxabi_contracts.llm.subjects import SUBJECTS


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
    allow_anonymous: bool = typer.Option(
        False,
        "--allow-anonymous",
        help="Connect to NATS without operator credentials. CI/dev only — do not use in production.",
        hidden=True,
    ),
    fleet: bool = typer.Option(
        False,
        "--fleet",
        help="Swap on all workers in the fleet.",
    ),
) -> None:
    """Hot-swap the running model via NATS."""
    import llmcli.cli as _cli

    catalog = _cli.config.load()

    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(f"[red]Unknown model '{name}'. Available: {available}[/red]")
        raise typer.Exit(code=1)

    async def _run() -> None:
        client = NatsClient(allow_anonymous=allow_anonymous)
        await client.connect()
        try:
            if fleet:
                result = await client.request_fleet(
                    SUBJECTS.lifecycle_swap, "swap", timeout, model_name=name
                )
                any_failed = False
                for resp in result.responses:
                    data = resp.data or {}
                    console.print(
                        f"OK {resp.host or '?'} swapped to {data.get('model', name)} "
                        f"(port={data.get('port', '?')}, "
                        f"vram={data.get('vram_used_mb', 0)}MB)"
                    )
                if result.errors:
                    any_failed = True
                    for h, we in result.errors:
                        err_console.print(f"[red]{h}: {we.code} - {we.message}[/red]")
                if result.timeout_reached and not result.responses:
                    err_console.print("[yellow]Fleet swap timed out with no responses.[/yellow]")
                    any_failed = True
                if any_failed:
                    raise typer.Exit(code=1)
            else:
                try:
                    resp = await client.request(
                        SUBJECTS.lifecycle_swap,
                        "swap",
                        host or socket.gethostname(),
                        timeout,
                        model_name=name,
                    )
                except Exception as exc:
                    err_console.print(f"[red]NATS unreachable or no worker responded: {exc}[/red]")
                    raise typer.Exit(code=1) from exc

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
            await client.close()

    asyncio.run(_run())
