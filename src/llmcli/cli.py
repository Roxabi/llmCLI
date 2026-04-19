from __future__ import annotations

from dataclasses import replace

import httpx
import typer
from rich.console import Console
from rich.table import Table

from .config import Catalog, ConfigNotFoundError, load
from .engines import LlamaCppEngine, LlamaCppTQ3Engine
from .engines.base import BinaryNotFoundError

app = typer.Typer(add_completion=False, help="llmCLI — local LLM serving")
console = Console()

_ENGINES = {
    "llamacpp": LlamaCppEngine,
    "llamacpp_tq3": LlamaCppTQ3Engine,
}


def _probe(port: int, timeout: float = 0.25) -> bool:
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _load_catalog() -> Catalog:
    try:
        return load()
    except ConfigNotFoundError as e:
        console.print(f"[red]Config not found:[/red] {e.path}")
        console.print("Copy [cyan]llmcli.example.toml[/cyan] to that path and customize.")
        raise typer.Exit(code=1) from e


def _resolve_name(name: str | None, catalog: Catalog) -> str:
    if name is not None:
        return name
    if catalog.host.default_model:
        return catalog.host.default_model
    console.print("[red]No model name given and no host.default_model in catalog.[/red]")
    raise typer.Exit(code=2)


@app.command(name="list")
def list_cmd() -> None:
    """Show catalog + running state + VRAM."""
    catalog = _load_catalog()
    table = Table(title="llmCLI catalog")
    table.add_column("name")
    table.add_column("engine")
    table.add_column("port", justify="right")
    table.add_column("vram_gib", justify="right")
    table.add_column("status")
    for name, spec in sorted(catalog.models.items()):
        up = _probe(spec.port)
        status = "[green]up[/green]" if up else "[dim]down[/dim]"
        table.add_row(name, spec.engine, str(spec.port), f"{spec.vram_gib:.1f}", status)
    console.print(table)


@app.command()
def pull(name: str) -> None:
    """Download a model from HF into the shared hub cache."""
    raise typer.Exit(code=0)


@app.command()
def serve(
    name: str | None = typer.Argument(None, help="Model name (defaults to host.default_model)."),
    host: str | None = typer.Option(None, "--host", help="Override bind address."),
) -> None:
    """Serve a model. Foreground, blocking — supervisor adopts llama-server as direct child."""
    catalog = _load_catalog()
    target = _resolve_name(name, catalog)
    spec = catalog.models.get(target)
    if spec is None:
        console.print(f"[red]Unknown model: {target}[/red]")
        raise typer.Exit(code=2)
    engine_cls = _ENGINES.get(spec.engine)
    if engine_cls is None:
        console.print(f"[red]Unknown engine: {spec.engine}[/red]")
        raise typer.Exit(code=2)
    host_settings = replace(catalog.host, bind=host) if host else catalog.host
    try:
        engine_cls(host_settings).start(spec)
    except BinaryNotFoundError as e:
        console.print(f"[red]Binary not found on PATH:[/red] {e.binary}")
        raise typer.Exit(code=1) from e


@app.command()
def stop() -> None:
    """Stop the daemon and any running engine."""
    raise typer.Exit(code=0)


@app.command()
def status() -> None:
    """Show engine health per catalog port."""
    catalog = _load_catalog()
    table = Table(title="llmCLI status")
    table.add_column("name")
    table.add_column("engine")
    table.add_column("port", justify="right")
    table.add_column("health")
    for name, spec in sorted(catalog.models.items()):
        up = _probe(spec.port)
        health = "[green]up[/green]" if up else "[dim]down[/dim]"
        table.add_row(name, spec.engine, str(spec.port), health)
    console.print(table)


@app.command()
def swap(name: str) -> None:
    """Hot-swap the running model via the daemon socket."""
    raise typer.Exit(code=0)


@app.command()
def chat(name: str, prompt: str) -> None:
    """One-shot OpenAI chat call, bypassing the proxy."""
    raise typer.Exit(code=0)


@app.command(name="register-proxy")
def register_proxy() -> None:
    """Refresh the llmCLI block in ~/.litellm/config.yaml and reload."""
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
