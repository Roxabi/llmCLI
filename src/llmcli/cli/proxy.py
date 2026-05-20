from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional

import typer
import yaml

from llmcli.cli._app import app, console, err_console
from llmcli.config import Catalog
from llmcli.litellm_config import build_full_config
from llmcli.providers import PROVIDERS

# Import style note: pure functions (build_full_config, PROVIDERS) are imported
# directly because tests never patch them through the `llmcli.cli` namespace.
# Patchable surface (config, build_block, write_block, reload_proxy) goes via
# the lazy `_cli = llmcli.cli` indirection inside the command bodies — see the
# patchable-surface docstring in `llmcli/cli/__init__.py`.


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
    config_out: Optional[Path] = typer.Option(
        None, "--config-out", help="Write generated YAML to PATH and exit (dry-run)."
    ),
) -> None:
    """Run a managed LiteLLM proxy bound to :{port} from the llmCLI catalog."""
    import llmcli.cli as _cli

    # 1. Load catalog
    try:
        catalog = _cli.config.load()
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    # 2. Validate provider keys
    errors = _validate_provider_keys(catalog)
    if errors:
        for err in errors:
            err_console.print(f"[red]{err}[/red]")
        raise typer.Exit(1)

    # 3. Build config dict + serialize to YAML
    cfg = build_full_config(catalog, catalog.host.public_base_url)
    yaml_text = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)

    # 4. Choose target path
    if config_out is not None:
        target = config_out
    else:
        target = Path.home() / ".local" / "state" / "llmcli" / "proxy.config.yaml"

    # 5. Write with 0o700 dir mode, 0o600 file mode
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target.write_text(yaml_text)
    target.chmod(0o600)

    # 6. If --config-out, dry-run exit
    if config_out is not None:
        if port != 18091:
            err_console.print(
                f"[yellow]--port {port} ignored in --config-out (dry-run) mode[/yellow]"
            )
        console.print(f"[green]Wrote proxy config to {target}[/green]")
        raise typer.Exit(0)

    # 7. Spawn litellm, install signal handlers, wait, propagate exit code
    child = _spawn_litellm(target, port, host)
    _install_signal_handlers(child)
    returncode = child.wait()
    # POSIX convention: negative return = killed by signal N → exit 128+N
    # Specifically, -9 (SIGKILL, e.g. OOM) → 137
    if returncode < 0:
        raise typer.Exit(128 + abs(returncode))
    raise typer.Exit(returncode)


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
            errors.append(
                f"Unknown provider '{spec.provider}' in model '{name}': "
                f"valid providers are {sorted(PROVIDERS.keys())}."
            )
            continue
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


def _install_signal_handlers(child: subprocess.Popen, drain_timeout: float = 10.0) -> None:
    """Forward SIGTERM/SIGINT to the litellm child with a poll-loop drain.

    First signal: child.terminate() → poll child.poll() in 0.1s ticks until
    child exits or drain_timeout elapses; if still alive → child.kill().

    Reentrant: a second SIGINT during drain triggers immediate child.kill()
    and raises SystemExit(130) (POSIX convention for SIGINT).
    """
    drain_state = {"active": False}

    def handler(signum, frame):  # noqa: ARG001 (signature required by signal.signal)
        if drain_state["active"] and signum == signal.SIGINT:
            child.kill()
            raise SystemExit(130)
        drain_state["active"] = True
        child.terminate()
        deadline = time.monotonic() + drain_timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                return
            time.sleep(0.1)
        child.kill()

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
