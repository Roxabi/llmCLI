"""Regression guard for #49 — litellm[proxy] runtime deps must be importable.

A breakage here means `llmcli proxy` would crash-loop at container start
(as happened in T8 smoke of #44 before PR #50 fixed the missing [proxy] sub-extra).
"""

from __future__ import annotations


def test_litellm_proxy_runtime_imports() -> None:
    import apscheduler  # noqa: F401  # pyright: ignore[reportMissingImports]
    import uvloop  # noqa: F401  # pyright: ignore[reportMissingImports]
    import websockets  # noqa: F401  # pyright: ignore[reportMissingImports]
    from litellm.proxy.proxy_server import app  # pyright: ignore[reportMissingImports]

    assert app is not None
