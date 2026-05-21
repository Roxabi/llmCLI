from __future__ import annotations

import os

import typer

from llmcli.cli._app import app, console, err_console


# NOTE: The `list` command has been consolidated into lifecycle.py (Slice 3, T25).
# It supports both AF_UNIX daemon path and the new NATS path behind
# LLMCLI_LIFECYCLE_VIA_NATS feature flag.


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
