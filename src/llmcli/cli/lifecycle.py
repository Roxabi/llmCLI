from __future__ import annotations

import json
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from llmcli.cli._app import app, console, err_console


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
        err_console.print(
            f"[red]Unknown model '{model_name}'. Available: {available}[/red]"
        )
        raise typer.Exit(code=1)

    spec = catalog.models[model_name]

    # VRAM guard (C2 / SC-13)
    try:
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
# stop
# ---------------------------------------------------------------------------


@app.command()
def stop() -> None:
    """Stop the daemon and any running engine."""
    import llmcli.cli as _cli

    try:
        resp = _cli.daemon_request("SHUTDOWN")
        console.print(f"Daemon: {resp}")
    except Exception as exc:
        console.print(f"[yellow]Daemon not running or unreachable: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Show engine status, ports, VRAM, uptime."""
    import llmcli.cli as _cli

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
