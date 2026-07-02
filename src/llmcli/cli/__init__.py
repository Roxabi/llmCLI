"""llmcli.cli package — Typer app + all command groups.

Public surface (stable):
  from llmcli.cli import app   ← entry point used by [project.scripts] and tests

Patched names (tests use `patch("llmcli.cli.X", create=True)`):
  config, hf_hub_download, openai,
  build_block, write_block, reload_proxy
"""

from __future__ import annotations

# Core app instance — must come first (submodules import _app.app)
from llmcli.cli._app import app, console, err_console  # noqa: F401

# Re-export patched names so `patch("llmcli.cli.X")` resolves correctly.
# Submodules do `import llmcli.cli as _cli` and call `_cli.X(...)` at
# runtime, so the mock injected here is what they see.
from llmcli import config  # noqa: F401
from llmcli.support.litellm_config import build_block, reload_proxy, write_block  # noqa: F401

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
# lifecycle_extra must come after lifecycle (both import _lifecycle_nats).
# catalog must come before lifecycle_extra ('list' command moved from catalog to lifecycle_extra).
from llmcli.cli import bench, catalog, chat, lifecycle, lifecycle_extra, proxy, swap  # noqa: F401

# NATS sub-app — registered lazily so nats-py is only required when used.
try:
    from llmcli.cli_nats import nats_app  # noqa: F401

    app.add_typer(nats_app, name="nats-serve")
except ImportError:  # pragma: no cover
    pass

__all__ = ["app"]
