"""CLI lifecycle extra commands: list, reload-catalog (T25, T26).

New NATS-aware commands split from lifecycle.py for 300-line cap.
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
    allow_anonymous: bool = typer.Option(
        False,
        "--allow-anonymous",
        help="Connect to NATS without operator credentials. CI/dev only — do not use in production.",
        hidden=True,
    ),
) -> None:
    """Show catalog models + running state + VRAM."""
    from roxabi_contracts.llm.subjects import SUBJECTS

    # PR-1: single-host list. Fleet broadcast + aggregate → v2 (Roxabi/llmCLI#61).
    resp = asyncio.run(
        lifecycle_nats_request(
            SUBJECTS.lifecycle_list,
            "list",
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
    allow_anonymous: bool = typer.Option(
        False,
        "--allow-anonymous",
        help="Connect to NATS without operator credentials. CI/dev only — do not use in production.",
        hidden=True,
    ),
) -> None:
    """Trigger worker-side catalog reload (re-reads llmcli.toml).

    PR-1: single-host reload. Fleet broadcast + aggregate → v2 (Roxabi/llmCLI#61).
    """
    from roxabi_contracts.llm.subjects import SUBJECTS

    # Spec N5/U5: reload-catalog is a broadcast — host=None so every worker reloads.
    # --host flag is accepted for CLI symmetry but intentionally ignored here.
    resp = asyncio.run(
        lifecycle_nats_request(
            SUBJECTS.lifecycle_reload_catalog,
            "reload-catalog",
            None,
            timeout,
            allow_anonymous=allow_anonymous,
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
