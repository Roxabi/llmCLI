"""xAI / SuperGrok OAuth credentials CLI subcommands."""

from __future__ import annotations

import json

import typer

from llmcli.auth import login_flow, store
from llmcli.auth.store import CredentialsCorruptError
from llmcli.cli._app import console, err_console
from llmcli.support.litellm_config import invalidate_model_cache

xai_app = typer.Typer(help="xAI / SuperGrok OAuth credentials")


@xai_app.command("login")
def login_cmd(
    manual: bool = typer.Option(
        False,
        "--manual",
        help="Headless mode: print the auth URL and paste the redirected code "
        "(no loopback listener / SSH tunnel needed).",
    ),
) -> None:
    """Run PKCE OAuth flow against auth.x.ai; stores credentials at xai.json."""
    creds = login_flow(manual=manual)
    invalidate_model_cache()
    console.print(f"[green]✓[/green] Logged in. expires_at={creds.expires_at}")


@xai_app.command("logout")
def logout_cmd() -> None:
    """Delete cached xai.json credentials (silent no-op if absent)."""
    store.XAI_CREDENTIALS_PATH.unlink(missing_ok=True)
    invalidate_model_cache()
    console.print("[green]✓[/green] Credentials removed.")


@xai_app.command("status")
def status_cmd() -> None:
    """Report logged_in + expires_at + scope. Never prints token material."""
    try:
        creds = store.load()
    except CredentialsCorruptError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(2)
    if creds is None:
        console.print(json.dumps({"logged_in": False}))
        raise typer.Exit(1)
    # AC3: stdout must NOT contain eyJ (JWT prefix) or xai- substring
    console.print(
        json.dumps({"logged_in": True, "expires_at": creds.expires_at, "scope": creds.scope})
    )
