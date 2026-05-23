"""Regression guard for #49 — litellm[proxy] runtime deps must be importable.

A breakage here means `llmcli proxy` would crash-loop at container start
(as happened in T8 smoke of #44 before PR #50 fixed the missing [proxy] sub-extra).
"""

from __future__ import annotations

import pytest


def test_litellm_proxy_runtime_imports() -> None:
    pytest.importorskip("apscheduler")
    pytest.importorskip("uvloop")
    pytest.importorskip("websockets")
    pytest.importorskip("litellm")
    from litellm.proxy.proxy_server import app  # pyright: ignore[reportMissingImports]

    assert app is not None
