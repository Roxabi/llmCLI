"""llmcli.cli package — Typer app + all command groups.

Public surface (stable):
  from llmcli.cli import app   ← entry point used by [project.scripts] and tests

Patched names (tests use `patch("llmcli.cli.X", create=True)`):
  config, daemon_request, hf_hub_download, Daemon, openai,
  build_block, write_block, reload_proxy
"""

from __future__ import annotations

# Core app instance — must come first (submodules import _app.app)
from llmcli.cli._app import app, console, err_console  # noqa: F401

# Re-export patched names so `patch("llmcli.cli.X")` resolves correctly.
# Submodules do `import llmcli.cli as _cli` and call `_cli.X(...)` at
# runtime, so the mock injected here is what they see.
from llmcli import config  # noqa: F401
from llmcli.daemon import Daemon, daemon_request  # noqa: F401
from llmcli.litellm_config import build_block, reload_proxy, write_block  # noqa: F401

try:
    from huggingface_hub import hf_hub_download  # noqa: F401
except ImportError:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]

try:
    import openai  # noqa: F401
except ImportError:  # pragma: no cover
    openai = None  # type: ignore[assignment]

# Import submodules AFTER the re-exports above are in place.
# Each submodule calls `@app.command()` at import time, registering commands.
from llmcli.cli import catalog, chat, lifecycle, proxy, swap  # noqa: F401

__all__ = ["app"]
