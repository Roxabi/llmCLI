from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(add_completion=False, help="llmCLI — local LLM serving")
console = Console()
err_console = Console(stderr=True)

from llmcli.cli.xai import xai_app  # noqa: E402
app.add_typer(xai_app, name="xai")
