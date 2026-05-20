from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Optional

import typer

from llmcli.cli._app import app, console, err_console
from llmcli.config import Catalog
from llmcli.providers import PROVIDERS


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
    import llmcli.cli as _cli

    # 1. Load catalog
    catalog = _cli.config.load()

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
    block = _cli.build_block(catalog, catalog.host.public_base_url)
    try:
        _cli.write_block(block, resolved_path)
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
        _cli.reload_proxy()
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


# ---------------------------------------------------------------------------
# proxy
# ---------------------------------------------------------------------------


@app.command()
def proxy(
    port: int = typer.Option(18091, "--port", envvar="LLMCLI_PROXY_PORT"),
    host: str = typer.Option("0.0.0.0", "--host", envvar="LLMCLI_PROXY_HOST"),
    config_out: Optional[Path] = typer.Option(None, "--config-out", help="Write generated YAML to PATH and exit (dry-run)."),
) -> None:
    """Run a managed LiteLLM proxy bound to :{port} from the llmCLI catalog."""
    raise NotImplementedError  # body filled in T8 and T14


def _validate_provider_keys(catalog: Catalog, hostname: str | None = None) -> list[str]:
    """Return list of missing-provider-key error messages; empty list = OK.

    Skips local engines. For each engine="remote" spec on the effective host
    (machines filter applied), looks up PROVIDERS[spec.provider].key_env and
    checks os.environ; missing → append an actionable error string.
    """
    effective = hostname or socket.gethostname()
    errors: list[str] = []
    for name, spec in catalog.models.items():
        if spec.engine != "remote":
            continue
        if spec.machines and effective not in spec.machines:
            continue
        provider = PROVIDERS.get(spec.provider)
        if provider is None:
            continue  # surfaced as ValueError by build_full_config later
        if not os.environ.get(provider.key_env):
            errors.append(
                f"Missing provider key for '{name}': set {provider.key_env} "
                "(in environment or ~/.litellm/.env)"
            )
    return errors


def _spawn_litellm(config_path: Path, port: int, host: str) -> subprocess.Popen:
    """Spawn the litellm proxy subprocess with inherited stdout/stderr.

    Locates the binary via shutil.which. Missing → typer.echo to stderr
    and typer.Exit(127). Inherits parent stdout/stderr (LiteLLM logs are
    structured JSON; no Rich wrapping post-spawn).

    Args:
        config_path: Path to the generated proxy config YAML.
        port: TCP port (e.g. 18091).
        host: Bind host (e.g. "0.0.0.0").

    Returns:
        subprocess.Popen handle for caller to wait()/signal.
    """
    binary = shutil.which("litellm")
    if binary is None:
        err_console.print(
            "[red]litellm binary not found on PATH.[/red] "
            "Install with: uv tool install 'litellm[proxy]' "
            "or `uv add 'litellm[proxy]'`"
        )
        raise typer.Exit(127)
    return subprocess.Popen(  # noqa: S603
        [binary, "--config", str(config_path), "--port", str(port), "--host", host]
    )
