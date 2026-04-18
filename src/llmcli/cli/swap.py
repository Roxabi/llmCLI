from __future__ import annotations

import typer

from llmcli.cli._app import app, console, err_console


# ---------------------------------------------------------------------------
# swap
# ---------------------------------------------------------------------------


@app.command()
def swap(
    name: str,
    timeout: float = typer.Option(
        300.0,
        "--timeout",
        help="Seconds to wait for the daemon to load the model (default: 300s for large models).",
    ),
) -> None:
    """Hot-swap the running model via the daemon socket."""
    import llmcli.cli as _cli

    catalog = _cli.config.load()

    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(f"[red]Unknown model '{name}'. Available: {available}[/red]")
        raise typer.Exit(code=1)

    try:
        resp = _cli.daemon_request(f"SWAP {name}", timeout=timeout)
    except Exception as exc:
        console.print(f"[yellow]Daemon not running or unreachable: {exc}[/yellow]")
        raise typer.Exit(code=1)

    if resp.startswith("ERR"):
        err_console.print(f"[red]{resp}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Daemon: {resp}")
