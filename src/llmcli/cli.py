from __future__ import annotations

import json
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from llmcli import config
from llmcli.daemon import Daemon, daemon_request
from llmcli.litellm_config import build_block, write_block

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]

app = typer.Typer(add_completion=False, help="llmCLI — local LLM serving")
console = Console()
err_console = Console(stderr=True)


def _load_catalog():
    """Load catalog from default config path."""
    return config.load()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_cmd() -> None:
    """Show catalog + running state + VRAM."""
    catalog = _load_catalog()

    # Try to get running state from daemon; silently ignore if down.
    running: dict = {}
    try:
        raw = daemon_request("STATUS")
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
    catalog = _load_catalog()

    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(
            f"[red]Unknown model '{name}'. Available: {available}[/red]"
        )
        raise typer.Exit(code=1)

    spec = catalog.models[name]
    hf_home = os.environ.get("HF_HOME", str(os.path.expanduser("~/.cache/huggingface")))
    cache_dir = os.path.join(hf_home, "hub")

    console.print(f"Pulling [cyan]{spec.repo}[/cyan] / [yellow]{spec.file}[/yellow] …")
    path = hf_hub_download(
        repo_id=spec.repo,
        filename=spec.file,
        cache_dir=cache_dir,
    )
    console.print(f"Saved to [green]{path}[/green]")


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    name: Optional[str] = typer.Option(None, "--name", help="Model name to serve"),
) -> None:
    """Start the daemon + serve the default (or named) model."""
    catalog = _load_catalog()

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

    # VRAM guard (C2)
    try:
        config.check_vram_budget(spec, catalog.host)
    except ValueError as exc:
        err_console.print(
            f"[red]VRAM budget exceeded: {exc}[/red]\n"
            f"Model [yellow]{spec.name}[/yellow] requires [bold]{spec.vram_gib}[/bold] GiB, "
            f"budget is [bold]{catalog.host.vram_budget_gib}[/bold] GiB."
        )
        raise typer.Exit(code=1)

    console.print(f"Starting daemon for model [cyan]{model_name}[/cyan] …")
    daemon = Daemon(catalog=catalog)
    daemon.serve(model_name)


# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------


@app.command()
def stop() -> None:
    """Stop the daemon and any running engine."""
    try:
        resp = daemon_request("SHUTDOWN")
        console.print(f"Daemon: {resp}")
    except Exception as exc:
        console.print(f"[yellow]Daemon not running or unreachable: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status() -> None:
    """Show engine status, ports, VRAM, uptime."""
    try:
        raw = daemon_request("STATUS")
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


# ---------------------------------------------------------------------------
# swap
# ---------------------------------------------------------------------------


@app.command()
def swap(name: str) -> None:
    """Hot-swap the running model via the daemon socket."""
    try:
        resp = daemon_request(f"SWAP {name}")
        console.print(f"Daemon: {resp}")
    except Exception as exc:
        console.print(f"[yellow]Daemon not running or unreachable: {exc}[/yellow]")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@app.command()
def chat(name: str, prompt: str) -> None:
    """One-shot OpenAI chat call, bypassing the proxy."""
    catalog = _load_catalog()

    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(
            f"[red]Unknown model '{name}'. Available: {available}[/red]"
        )
        raise typer.Exit(code=1)

    spec = catalog.models[name]

    # Determine base_url: try daemon STATUS first, fall back to catalog port.
    base_url = f"http://localhost:{spec.port}/v1"
    try:
        raw = daemon_request("STATUS")
        if raw.startswith("{"):
            instances = json.loads(raw)
            if name in instances:
                port = instances[name].get("port", spec.port)
                base_url = f"http://localhost:{port}/v1"
    except Exception:
        pass

    api_key = os.environ.get(catalog.host.api_key_env, "no-key")

    client = openai.OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=name,
        messages=[{"role": "user", "content": prompt}],
    )
    console.print(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# register-proxy
# ---------------------------------------------------------------------------


@app.command(name="register-proxy")
def register_proxy() -> None:
    """Refresh the llmCLI block in ~/.litellm/config.yaml and reload."""
    catalog = _load_catalog()

    block = build_block(catalog, catalog.host.public_base_url)
    write_block(block)
    console.print("[green]LiteLLM proxy config updated.[/green]")


if __name__ == "__main__":
    app()
