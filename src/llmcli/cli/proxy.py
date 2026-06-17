from __future__ import annotations

import logging
import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import typer
import yaml

from llmcli.cli._app import app, console, err_console
from llmcli.config import Catalog
from llmcli.support.litellm_config import (
    build_model_list,
    clear_model_cache,
    emit_xai_oauth_warning_if_absent,
    load_proxy_base,
    merge_proxy_config,
    register_model_refresh_callback,
)
from llmcli.support.providers import PROVIDERS

log = logging.getLogger(__name__)

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

    # 5a. Warn if xAI credentials are absent (discoverability hint)
    emit_xai_oauth_warning_if_absent(err_console)

    # 5b. Write custom auth module so LiteLLM can resolve it
    _write_custom_auth_module(resolved_path.parent)
    # 5c. Warn if custom_auth is not yet in the persistent config
    if resolved_path.exists() and "custom_auth" not in resolved_path.read_text():
        err_console.print(
            "[yellow]Note: add `custom_auth: proxy_custom_auth.custom_auth` to "
            "general_settings in your config to fix the 'No connected db' error.[/yellow]"
        )

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
    port: Optional[int] = typer.Option(
        None, "--port", help="Proxy TCP port (env LLMCLI_PROXY_PORT > --port > catalog > 18091)."
    ),
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

    # 1a. Resolve port via precedence: env > flag > catalog > default(18091)
    env_port_raw = os.environ.get("LLMCLI_PROXY_PORT")
    env_port: int | None
    if env_port_raw:
        try:
            env_port = int(env_port_raw)
        except ValueError:
            err_console.print(
                f"[red]LLMCLI_PROXY_PORT={env_port_raw!r} is not a valid integer.[/red]"
            )
            raise typer.Exit(code=1)
    else:
        env_port = None
    resolved_port = _resolve_port(
        env_val=env_port,
        flag_val=port,
        catalog_port=catalog.host.port,
    )

    # 2. Validate provider keys
    errors = _validate_provider_keys(catalog)
    if errors:
        for err in errors:
            err_console.print(f"[red]{err}[/red]")
        raise typer.Exit(1)

    # 3. Load optional proxy-base.yaml
    proxy_base_path = Path.home() / ".roxabi" / "llmcli" / "proxy-base.yaml"
    try:
        base = load_proxy_base(proxy_base_path)
    except yaml.YAMLError as exc:
        err_console.print(f"[red]proxy-base.yaml: {exc}[/red]")
        raise typer.Exit(code=1)

    # 4. Choose target path
    if config_out is not None:
        target = config_out
    else:
        target = Path.home() / ".local" / "state" / "llmcli" / "proxy.config.yaml"

    # 4a. Spawn path: validate litellm binary exists BEFORE writing the config file
    # so we never leave a stale ~/.local/state/llmcli/proxy.config.yaml on PATH failure.
    if config_out is None:
        if shutil.which("litellm") is None:
            err_console.print(
                "[red]litellm binary not found on PATH.[/red] "
                "Install with: uv tool install 'litellm[proxy]' or `uv add 'litellm[proxy]'`"
            )
            raise typer.Exit(127)

    # 5. Build model_list + merge into layered config and write to disk
    _write_proxy_config(catalog, target, base)

    # 6. If --config-out, dry-run exit
    if config_out is not None:
        if resolved_port != 18091:
            err_console.print(
                f"[yellow]--port {resolved_port} ignored in --config-out (dry-run) mode[/yellow]"
            )
        console.print(f"[green]Wrote proxy config to {target}[/green]")
        raise typer.Exit(0)

    # 7. Spawn litellm, background refresh, install signal handlers, wait
    child_state: dict[str, Any] = {"child": _spawn_litellm(target, resolved_port, host), "stop": False}

    def _refresh_now() -> None:
        try:
            _write_proxy_config(catalog, target, base)
            child_state["child"] = _reload_litellm_child(
                child_state["child"], target, resolved_port, host
            )
        except Exception as exc:
            log.warning("model catalogue refresh failed: %s", exc)

    register_model_refresh_callback(_refresh_now)
    refresh_interval = _parse_model_refresh_interval()
    refresh_thread = _start_model_refresh_loop(
        child_state,
        catalog=catalog,
        target=target,
        base=base,
        port=resolved_port,
        host=host,
        interval_secs=refresh_interval,
    )

    def _shutdown_handler(signum, frame):  # noqa: ARG001
        child_state["stop"] = True
        register_model_refresh_callback(None)
        refresh_thread.join(timeout=2.0)

    _install_signal_handlers(child_state["child"], pre_handler=_shutdown_handler)
    returncode = child_state["child"].wait()
    # POSIX convention: negative return = killed by signal N → exit 128+N
    # Specifically, -9 (SIGKILL, e.g. OOM) → 137
    if returncode < 0:
        raise typer.Exit(128 + abs(returncode))
    raise typer.Exit(returncode)


def _resolve_port(env_val: int | None, flag_val: int | None, catalog_port: int | None) -> int:
    """Resolve final proxy port via precedence: env > flag > catalog > default(18091)."""
    if env_val is not None:
        return env_val
    if flag_val is not None:
        return flag_val
    if catalog_port is not None:
        return catalog_port
    return 18091


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
    # start_new_session=True: child runs in its own process group so terminal
    # SIGINT/SIGTERM hits only the parent. The handler in _install_signal_handlers
    # then forwards deterministically with a drain timeout — without isolation,
    # the kernel would deliver the signal to both parent and child via the
    # foreground group, racing the drain logic.
    return subprocess.Popen(  # noqa: S603
        [binary, "--config", str(config_path), "--port", str(port), "--host", host],
        start_new_session=True,
    )


def _parse_model_refresh_interval() -> float:
    """Parse LLMCLI_MODEL_REFRESH_SECS (default 60); reject negative values."""
    raw = os.environ.get("LLMCLI_MODEL_REFRESH_SECS", "60")
    try:
        value = float(raw)
    except ValueError:
        log.warning("LLMCLI_MODEL_REFRESH_SECS=%r invalid; using 60", raw)
        return 60.0
    if value < 0:
        err_console.print(
            f"[red]LLMCLI_MODEL_REFRESH_SECS={raw} must be non-negative.[/red]"
        )
        raise typer.Exit(code=1)
    return value if value > 0 else 60.0


def _write_proxy_config(catalog: Catalog, target: Path, base: dict[str, Any]) -> None:
    """Build merged proxy YAML from catalog + base transport config and write to target."""
    model_list = build_model_list(catalog, catalog.host.public_base_url)
    cfg = merge_proxy_config(base, model_list, api_key_env=catalog.host.api_key_env)
    yaml_text = yaml.safe_dump(cfg, default_flow_style=False, sort_keys=False)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(target, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(yaml_text)
    target.chmod(0o600)
    _write_custom_auth_module(target.parent)


def _reload_litellm_child(
    child: subprocess.Popen,
    config_path: Path,
    port: int,
    host: str,
    *,
    drain_timeout: float = 5.0,
) -> subprocess.Popen:
    """Reload litellm by SIGHUP when possible; otherwise terminate and respawn."""
    if child.poll() is None:
        try:
            child.send_signal(signal.SIGHUP)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if child.poll() is not None:
                    break
                time.sleep(0.1)
            if child.poll() is None:
                return child
        except OSError as exc:
            log.debug("SIGHUP reload failed, respawning litellm child: %s", exc)

    if child.poll() is None:
        child.terminate()
        deadline = time.monotonic() + drain_timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                break
            time.sleep(0.1)
        if child.poll() is None:
            child.kill()
    return _spawn_litellm(config_path, port, host)


def _start_model_refresh_loop(
    child_state: dict[str, Any],
    *,
    catalog: Catalog,
    target: Path,
    base: dict[str, Any],
    port: int,
    host: str,
    interval_secs: float,
) -> threading.Thread:
    """Daemon thread: periodic model_list regen + litellm child reload."""

    def _loop() -> None:
        while not child_state.get("stop"):
            time.sleep(interval_secs)
            if child_state.get("stop"):
                break
            try:
                clear_model_cache()
                _write_proxy_config(catalog, target, base)
                child = child_state.get("child")
                if child is not None and child.poll() is None:
                    child_state["child"] = _reload_litellm_child(child, target, port, host)
            except Exception as exc:
                log.warning("background model refresh failed: %s", exc)

    thread = threading.Thread(target=_loop, name="llmcli-model-refresh", daemon=True)
    thread.start()
    return thread


def _install_signal_handlers(
    child: subprocess.Popen,
    drain_timeout: float = 10.0,
    *,
    pre_handler: Any | None = None,
) -> None:
    """Forward SIGTERM/SIGINT to the litellm child with a poll-loop drain.

    First signal: child.terminate() → poll child.poll() in 0.1s ticks until
    child exits or drain_timeout elapses; if still alive → child.kill().

    Reentrant: a second SIGINT during drain triggers immediate child.kill()
    and raises SystemExit(130) (POSIX convention for SIGINT).
    """
    drain_state = {"active": False}

    def handler(signum, frame):  # noqa: ARG001 (signature required by signal.signal)
        if pre_handler is not None:
            pre_handler(signum, frame)
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


def _write_custom_auth_module(target_dir: Path) -> None:
    """Copy the LiteLLM custom auth module to the config directory.

    LiteLLM's get_instance_fn resolves custom_auth modules relative to the
    config file path. Writing proxy_custom_auth.py next to the config ensures
    it is importable regardless of which venv runs litellm.
    """
    auth_module_src = Path(__file__).parent.parent / "proxy_custom_auth.py"
    if auth_module_src.exists():
        shutil.copy2(auth_module_src, target_dir / "proxy_custom_auth.py")
