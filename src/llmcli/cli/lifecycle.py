"""CLI lifecycle commands: serve, stop, status.

list and reload-catalog live in lifecycle_extra.py (split for 300-line cap).
NATS request helper: _lifecycle_nats.py.
AF_UNIX daemon path removed in Slice 6 cutover (#34).
"""

from __future__ import annotations

import asyncio
import socket
from typing import Optional

import typer
from rich.table import Table

from llmcli.cli._app import app, console, err_console
from llmcli.cli._lifecycle_nats import lifecycle_nats_request


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    name: Optional[str] = typer.Option(None, "--name", help="Model name to serve"),
) -> None:
    """[Removed] llmcli serve is no longer available.

    # T3 (Slice 6 cutover, #34): option (b) — stub for discoverability.
    # The AF_UNIX daemon is gone; operators start the NATS worker via systemd.
    """
    _ = name  # unused; keep parameter so CLI signature is backward-compatible
    err_console.print(
        "[yellow]llmcli serve has been removed.[/yellow]\n"
        "Start the NATS worker with:\n"
        "  [bold]systemctl --user start llmcli-nats-worker[/bold]"
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# stop (T24)
# ---------------------------------------------------------------------------


@app.command()
def stop(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Target hostname; default = local hostname",
    ),
    timeout: float = typer.Option(30.0, "--timeout", help="Request timeout in seconds."),
    allow_anonymous: bool = typer.Option(
        False,
        "--allow-anonymous",
        help="Connect to NATS without operator credentials. CI/dev only — do not use in production.",
        hidden=True,
    ),
) -> None:
    """Stop the running engine on the target host (via NATS)."""
    from roxabi_contracts.llm.subjects import SUBJECTS

    resp = asyncio.run(
        lifecycle_nats_request(
            SUBJECTS.lifecycle_stop,
            "stop",
            host or socket.gethostname(),
            timeout,
            allow_anonymous=allow_anonymous,
        )
    )
    if not resp.ok:
        we = resp.worker_error
        if we:
            err_console.print(f"[red]{we.code}: {we.message}[/red]")
        else:
            err_console.print(f"[red]stop failed: {resp.error}[/red]")
        raise typer.Exit(code=1)
    console.print("OK stopped")


# ---------------------------------------------------------------------------
# status (T23)
# ---------------------------------------------------------------------------


@app.command()
def status(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Target hostname; default = local hostname",
    ),
    timeout: float = typer.Option(30.0, "--timeout", help="Request timeout in seconds."),
    allow_anonymous: bool = typer.Option(
        False,
        "--allow-anonymous",
        help="Connect to NATS without operator credentials. CI/dev only — do not use in production.",
        hidden=True,
    ),
    fleet: bool = typer.Option(
        False,
        "--fleet",
        help="Query all workers in the fleet.",
    ),
) -> None:
    """Show engine status, ports, VRAM, uptime."""
    from roxabi_contracts.llm.subjects import SUBJECTS
    from llmcli.cli._nats_client import NatsClient

    async def _run() -> None:
        client = NatsClient(allow_anonymous=allow_anonymous)
        await client.connect()
        try:
            if fleet:
                result = await client.request_fleet(
                    SUBJECTS.lifecycle_status, "status", timeout
                )
                if result.responses:
                    table = Table(title="Fleet status")
                    table.add_column("host", style="cyan")
                    table.add_column("model")
                    table.add_column("port")
                    table.add_column("vram_used_mb")
                    for resp in result.responses:
                        data = resp.data or {}
                        table.add_row(
                            resp.host or "?",
                            str(data.get("model", "?")),
                            str(data.get("port", "?")),
                            str(data.get("vram_used_mb", 0)),
                        )
                    console.print(table)
                if result.errors:
                    for h, we in result.errors:
                        err_console.print(f"[red]{h}: {we.code} - {we.message}[/red]")
                if result.timeout_reached and not result.responses:
                    err_console.print("[yellow]Fleet query timed out with no responses.[/yellow]")
            else:
                try:
                    resp = await client.request(
                        SUBJECTS.lifecycle_status,
                        "status",
                        host or socket.gethostname(),
                        timeout,
                    )
                except Exception as exc:
                    err_console.print(
                        f"[yellow]No worker responded for host={host or socket.gethostname()!r}. "
                        f"Check hostname or NATS connectivity: {exc}[/yellow]"
                    )
                    raise typer.Exit(code=1) from exc
                if not resp.ok:
                    we = resp.worker_error
                    if we:
                        err_console.print(f"[red]{we.code}: {we.message}[/red]")
                    else:
                        err_console.print(f"[red]status failed: {resp.error}[/red]")
                    raise typer.Exit(code=1)
                data = resp.data or {}
                if not data.get("model"):
                    console.print("No engines running.")
                    return
                table = Table(title="Running engines")
                table.add_column("model")
                table.add_column("port")
                table.add_column("vram_used_mb")
                table.add_row(
                    str(data.get("model", "?")),
                    str(data.get("port", "?")),
                    str(data.get("vram_used_mb", 0)),
                )
                console.print(table)
        finally:
            await client.close()

    asyncio.run(_run())
