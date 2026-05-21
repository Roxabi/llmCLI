"""CLI lifecycle extra commands: list, reload-catalog (T25, T26).

New NATS-aware commands split from lifecycle.py for 300-line cap.
AF_UNIX daemon path preserved for list (E1 rollback).
reload-catalog is NATS-only (no AF_UNIX equivalent).
"""

from __future__ import annotations

import asyncio
import json
import socket
from typing import Optional

import typer
from rich.table import Table

from llmcli.cli._app import app, console, err_console
from llmcli.cli._lifecycle_nats import _use_nats_lifecycle, lifecycle_nats_request


# ---------------------------------------------------------------------------
# list (T25)
# PR-1: single-host view. Fleet broadcast + aggregate response → v2 (#61).
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_models(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Target hostname; default = local hostname",
    ),
    timeout: float = typer.Option(30.0, "--timeout", help="Request timeout in seconds."),
) -> None:
    """Show catalog models + running state + VRAM."""
    import llmcli.cli as _cli

    if _use_nats_lifecycle():
        from roxabi_contracts.llm.subjects import SUBJECTS

        # PR-1: single-host list. Fleet broadcast + aggregate → v2 (Roxabi/llmCLI#61).
        resp = asyncio.run(
            lifecycle_nats_request(
                SUBJECTS.lifecycle_list,
                "list",
                host or socket.gethostname(),
                timeout,
            )
        )
        if not resp.ok:
            we = resp.worker_error
            if we:
                err_console.print(f"[red]{we.code}: {we.message}[/red]")
            else:
                err_console.print(f"[red]list failed: {resp.error}[/red]")
            raise typer.Exit(code=1)
        data = resp.data or {}
        models = data.get("models", [])
        table = Table(title="llmCLI models")
        table.add_column("name", style="cyan")
        table.add_column("engine")
        table.add_column("vram_gib")
        table.add_column("running?")
        for m in models:
            table.add_row(
                m.get("name", "?"),
                m.get("engine", "?"),
                str(m.get("vram_gib", "?")),
                "yes" if m.get("running") else "no",
            )
        console.print(table)
        return

    # AF_UNIX daemon path (E1 rollback) — mirrors original catalog.py list_cmd.
    catalog = _cli.config.load()
    running: dict = {}
    try:
        raw = _cli.daemon_request("STATUS")
        if raw.startswith("{"):
            running = json.loads(raw)
    except Exception:
        pass

    table = Table(title="llmCLI models")
    table.add_column("name", style="cyan")
    table.add_column("engine")
    table.add_column("vram_gib")
    table.add_column("port")
    table.add_column("repo")
    table.add_column("running?")

    for name, spec in catalog.models.items():
        is_running = "yes" if name in running else "no"
        table.add_row(
            name,
            spec.engine,
            str(spec.vram_gib),
            str(spec.port),
            spec.repo,
            is_running,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# reload-catalog (T26)
# NATS-only: no AF_UNIX equivalent.
# PR-1: single-host reload. Fleet broadcast + aggregate → v2 (Roxabi/llmCLI#61).
# ---------------------------------------------------------------------------


@app.command(name="reload-catalog")
def reload_catalog(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Target hostname; default = local hostname",
    ),
    timeout: float = typer.Option(30.0, "--timeout", help="Request timeout in seconds."),
) -> None:
    """Trigger worker-side catalog reload (re-reads llmcli.toml).

    Requires LLMCLI_LIFECYCLE_VIA_NATS=1. No AF_UNIX equivalent.
    PR-1: single-host reload. Fleet broadcast + aggregate → v2 (Roxabi/llmCLI#61).
    """
    if not _use_nats_lifecycle():
        err_console.print(
            "[yellow]reload-catalog requires LLMCLI_LIFECYCLE_VIA_NATS=1 "
            "(NATS path only — no AF_UNIX equivalent).[/yellow]"
        )
        raise typer.Exit(code=1)

    from roxabi_contracts.llm.subjects import SUBJECTS

    # PR-1: single-host reload. Fleet broadcast + aggregate response → v2 (Roxabi/llmCLI#61).
    resp = asyncio.run(
        lifecycle_nats_request(
            SUBJECTS.lifecycle_reload_catalog,
            "reload-catalog",
            host or socket.gethostname(),
            timeout,
        )
    )
    if not resp.ok:
        we = resp.worker_error
        if we:
            err_console.print(f"[red]{we.code}: {we.message}[/red]")
        else:
            err_console.print(f"[red]reload-catalog failed: {resp.error}[/red]")
        raise typer.Exit(code=1)

    data = resp.data or {}
    n = data.get("models_loaded", "?")
    console.print(f"OK reloaded {n} models")
