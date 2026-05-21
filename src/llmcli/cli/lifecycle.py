"""CLI lifecycle commands: serve, stop, status.

list and reload-catalog live in lifecycle_extra.py (split for 300-line cap).
Feature-flag helper and NATS request helper: _lifecycle_nats.py.
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from llmcli.cli._app import app, console, err_console
from llmcli.cli._lifecycle_nats import _use_nats_lifecycle, lifecycle_nats_request


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    name: Optional[str] = typer.Option(None, "--name", help="Model name to serve"),
) -> None:
    """Start the daemon + serve the default (or named) model."""
    import llmcli.cli as _cli

    catalog = _cli.config.load()

    model_name = name or catalog.host.default_model
    if model_name is None:
        err_console.print("[red]No model specified and no default_model set in catalog.[/red]")
        raise typer.Exit(code=1)

    if model_name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(f"[red]Unknown model '{model_name}'. Available: {available}[/red]")
        raise typer.Exit(code=1)

    spec = catalog.models[model_name]

    # VRAM guard (C2 / SC-13) — Remote specs need no local GPU; skip VRAM check.
    try:
        if spec.engine != "remote":
            _cli.config.check_vram_budget(spec, catalog.host)
    except ValueError as exc:
        err_console.print(
            Panel(
                f"[bold]{exc}[/bold]\n\n"
                f"[dim]See [link=docs/guides/deployment.md]docs/guides/deployment.md[/link] "
                "for VRAM budgeting.[/dim]",
                title="[red]VRAM budget exceeded[/red]",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)

    console.print(f"Starting daemon for model [cyan]{model_name}[/cyan] …")
    daemon = _cli.Daemon(catalog=catalog)
    daemon.serve(model_name)


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
) -> None:
    """Stop the daemon and any running engine."""
    import llmcli.cli as _cli

    if _use_nats_lifecycle():
        from roxabi_contracts.llm.subjects import SUBJECTS

        resp = asyncio.run(
            lifecycle_nats_request(
                SUBJECTS.lifecycle_stop,
                "stop",
                host or socket.gethostname(),
                timeout,
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
        return

    # AF_UNIX daemon path (E1 rollback path).
    try:
        resp_str = _cli.daemon_request("SHUTDOWN")
        console.print(f"Daemon: {resp_str}")
    except Exception as exc:
        console.print(f"[yellow]Daemon not running or unreachable: {exc}[/yellow]")


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
) -> None:
    """Show engine status, ports, VRAM, uptime."""
    import llmcli.cli as _cli

    if _use_nats_lifecycle():
        from roxabi_contracts.llm.subjects import SUBJECTS

        resp = asyncio.run(
            lifecycle_nats_request(
                SUBJECTS.lifecycle_status,
                "status",
                host or socket.gethostname(),
                timeout,
            )
        )
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
        return

    # AF_UNIX daemon path (E1 rollback path).
    try:
        raw = _cli.daemon_request("STATUS")
    except Exception as exc:
        console.print(f"[yellow]Daemon not running: {exc}[/yellow]")
        return

    # Try JSON dict format (rich status payload)
    if raw.startswith("{"):
        try:
            instances = json.loads(raw)
            if not instances:
                console.print("No engines running.")
                return
            table = Table(title="Running engines")
            table.add_column("model")
            table.add_column("pid")
            table.add_column("port")
            for model_name, info in instances.items():
                table.add_row(
                    model_name,
                    str(info.get("pid", "?")),
                    str(info.get("port", "?")),
                )
            console.print(table)
            return
        except json.JSONDecodeError:
            pass

    # Plain text "OK model=... port=... uptime=..." format
    console.print(raw)
