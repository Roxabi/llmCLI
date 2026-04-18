from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmcli import config
from llmcli.daemon import Daemon, daemon_request
from llmcli.litellm_config import build_block, reload_proxy, write_block

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

    # VRAM guard (C2 / SC-13)
    try:
        config.check_vram_budget(spec, catalog.host)
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
    catalog = _load_catalog()

    if name not in catalog.models:
        available = ", ".join(catalog.models.keys())
        err_console.print(
            f"[red]Unknown model '{name}'. Available: {available}[/red]"
        )
        raise typer.Exit(code=1)

    try:
        resp = daemon_request(f"SWAP {name}")
    except Exception as exc:
        console.print(f"[yellow]Daemon not running or unreachable: {exc}[/yellow]")
        raise typer.Exit(code=1)

    if resp.startswith("ERR"):
        err_console.print(f"[red]{resp}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Daemon: {resp}")


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
def register_proxy(
    config_path: Optional[str] = typer.Option(
        None,
        "--config",
        envvar="LITELLM_CONFIG_PATH",
        help="Path to LiteLLM config.yaml (default: ~/.litellm/config.yaml)",
    ),
) -> None:
    """Refresh the llmCLI block in ~/.litellm/config.yaml and reload."""
    # 1. Load catalog
    catalog = _load_catalog()

    # 2. Determine host
    hostname = socket.gethostname()

    # 3. Resolve config path: --config flag > env var > default
    resolved_path = Path(config_path) if config_path else Path.home() / ".litellm" / "config.yaml"

    # 4. Friendly error when parent directory doesn't exist or is not writable
    if not resolved_path.parent.exists():
        err_console.print(
            f"[red]Config directory does not exist: {resolved_path.parent}[/red]\n"
            f"Create it first:  mkdir -p {resolved_path.parent}"
        )
        raise typer.Exit(code=1)

    # 5. Build and write block
    block = build_block(catalog, catalog.host.public_base_url)
    try:
        write_block(block, resolved_path)
    except PermissionError as exc:
        err_console.print(
            f"[red]Permission denied writing to {resolved_path}: {exc}[/red]\n"
            f"Check file permissions or run with appropriate privileges."
        )
        raise typer.Exit(code=1)
    except OSError as exc:
        err_console.print(f"[red]Failed to write config: {exc}[/red]")
        raise typer.Exit(code=1)

    model_count = len(catalog.models)

    # 6. Reload proxy — warn on failure, don't fail the command
    try:
        reload_proxy()
        reload_status = "[green]reloaded[/green]"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError) as exc:
        reload_status = f"[yellow]reload failed (write succeeded): {exc}[/yellow]"
        err_console.print(
            f"[yellow]Warning: proxy reload failed — {exc}[/yellow]\n"
            "The config file was updated successfully. Reload the proxy manually."
        )

    # 7. Confirmation output
    console.print(
        f"[green]LiteLLM proxy config updated.[/green] "
        f"host=[cyan]{hostname}[/cyan] "
        f"path=[cyan]{resolved_path}[/cyan] "
        f"models=[bold]{model_count}[/bold] "
        f"reload={reload_status}"
    )


if __name__ == "__main__":
    app()
