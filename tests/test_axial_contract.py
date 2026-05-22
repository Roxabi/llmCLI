"""Axial decomposition contract tests (ADR-006).

Ensures engine leaves compose shared stage primitives from `_common` (drift
signal) and that engines with stage divergence declare it via the override
protocol (supports_swap / supports_hot_reload).
"""

from __future__ import annotations

import re
from pathlib import Path

ENGINES_DIR = Path(__file__).parent.parent / "src" / "llmcli" / "engines"
LEAVES = {"llamacpp.py", "llamacpp_tq3.py", "vllm.py"}


def test_engines_import_common():
    """Every engine leaf must import from _common (directly or via sibling that does) — enforces
    stage primitive reuse (ADR-006).

    llamacpp_tq3.py is a thin subclass of LlamaCppEngine and delegates all
    stage primitives via inheritance; its parent (llamacpp.py) imports _common.
    The check therefore accepts either a direct _common import or an import from
    .llamacpp (the only sibling that itself imports _common).
    """
    for leaf in LEAVES:
        src = (ENGINES_DIR / leaf).read_text()
        has_common = re.search(
            r"from \._common import|from \.\._common import|from llmcli\.engines\._common import",
            src,
        )
        has_llamacpp = re.search(r"from \.llamacpp import", src)
        assert has_common or has_llamacpp, (
            f"{leaf} does not import from _common or from .llamacpp "
            f"— stage logic must delegate (ADR-006 drift signal)"
        )


def test_protocol_overrides_present():
    """Engines with stage divergence must declare supports_* overrides (ADR-006 override
    protocol)."""
    from llmcli.engines.llamacpp_tq3 import LlamaCppTQ3Engine
    from llmcli.engines.vllm import VLLMEngine

    assert VLLMEngine().supports_swap() is False
    assert LlamaCppTQ3Engine().supports_hot_reload() is False
