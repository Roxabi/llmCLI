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
    fleet: bool = typer.Option(
        False,
        "--fleet",
        help="Query all workers in the fleet.",
    ),
) -> None:
    """Show catalog models + running state + VRAM."""
    from roxabi_contracts.llm.subjects import SUBJECTS
    from llmcli.cli._nats_client import NatsClient

    async def _run() -> None:
        client = NatsClient(allow_anonymous=allow_anonymous)
        await client.connect()
        try:
            if fleet:
                result = await client.request_fleet(
                    SUBJECTS.lifecycle_list, "list", timeout
                )
                # Merge model lists from all hosts
                models: dict[str, dict] = {}
                for resp in result.responses:
                    for m in (resp.data or {}).get("models", []):
                        name = m.get("name", "?")
                        if name not in models:
                            models[name] = {
                                "engine": m.get("engine", "?"),
                                "vram_gib": m.get("vram_gib", "?"),
                                "running_on": [],
                            }
                        if m.get("running"):
                            models[name]["running_on"].append(resp.host or "?")
                if models:
                    table = Table(title="Fleet models")
                    table.add_column("name", style="cyan")
                    table.add_column("engine")
                    table.add_column("vram_gib")
                    table.add_column("running")
                    for name, info in sorted(models.items()):
                        running = ", ".join(info["running_on"]) if info["running_on"] else "no"
                        table.add_row(name, info["engine"], str(info["vram_gib"]), running)
                    console.print(table)
                if result.errors:
                    for h, we in result.errors:
                        err_console.print(f"[red]{h}: {we.code} - {we.message}[/red]")
                if result.timeout_reached and not result.responses:
                    err_console.print("[yellow]Fleet query timed out with no responses.[/yellow]")
            else:
                try:
                    resp = await client.request(
                        SUBJECTS.lifecycle_list,
                        "list",
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
        finally:
            await client.close()

    asyncio.run(_run())


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
