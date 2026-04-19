from __future__ import annotations

import os

import typer

from llmcli.cli._app import app, console, err_console


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_cmd() -> None:
    """Show catalog + running state + VRAM."""
    import json

    import llmcli.cli as _cli
    from rich.table import Table

    catalog = _cli.config.load()

    # Try to get running state from daemon; silently ignore if down.
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
# pull
# ---------------------------------------------------------------------------


@app.command()
def pull(name: str) -> None:
    """Download a model from HF into the shared hub cache."""
    import llmcli.cli as _cli

    catalog = _cli.config.load()

    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(f"[red]Unknown model '{name}'. Available: {available}[/red]")
        raise typer.Exit(code=1)

    spec = catalog.models[name]
    hf_home = os.environ.get("HF_HOME", str(os.path.expanduser("~/.cache/huggingface")))
    cache_dir = os.path.join(hf_home, "hub")

    console.print(f"Pulling [cyan]{spec.repo}[/cyan] / [yellow]{spec.file}[/yellow] …")
    path = _cli.hf_hub_download(
        repo_id=spec.repo,
        filename=spec.file,
        cache_dir=cache_dir,
    )
    console.print(f"Saved to [green]{path}[/green]")
