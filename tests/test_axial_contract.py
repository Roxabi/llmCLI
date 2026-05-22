"""Axial decomposition contract tests (ADR-006).

Ensures engine leaves compose shared stage primitives from `_common` (drift
signal) and that engines with stage divergence declare it via the override
protocol (supports_swap / supports_hot_reload).
"""

from __future__ import annotations

import re
from pathlib import Path

ENGINES_DIR = Path(__file__).parent.parent / "src" / "llmcli" / "engines"

# Engine files that implement stage primitives directly (own stage logic).
# llamacpp_tq3.py is a thin subclass of LlamaCppEngine — it inherits all stage
# primitives via LlamaCppEngine which composes _common.  It has no stage logic
# of its own, so the import-from-_common contract does not apply to it.
LEAVES_WITH_STAGE_LOGIC = {"llamacpp.py", "vllm.py"}


def test_engines_import_common():
    """Engine files that own stage primitives must import from _common — enforces
    stage primitive reuse (ADR-006).

    Only engines in LEAVES_WITH_STAGE_LOGIC are checked.  llamacpp_tq3.py is
    excluded because it is a thin inheritor (no stage logic of its own); its
    parent llamacpp.py satisfies the contract on its behalf.
    """
    for leaf in LEAVES_WITH_STAGE_LOGIC:
        src = (ENGINES_DIR / leaf).read_text()
        has_common = re.search(
            r"from \._common import|from \.\._common import|from llmcli\.engines\._common import",
            src,
        )
        assert has_common, (
            f"{leaf} does not import from _common "
            f"— stage logic must delegate (ADR-006 drift signal)"
        )


def test_protocol_overrides_present():
    """Engines with stage divergence must declare supports_* overrides (False side).

    Pins the engines that explicitly disable a capability.  See also
    test_protocol_defaults_present for the True side.
    """
    from llmcli.engines.llamacpp_tq3 import LlamaCppTQ3Engine
    from llmcli.engines.vllm import VLLMEngine

    assert VLLMEngine().supports_swap() is False
    assert LlamaCppTQ3Engine().supports_hot_reload() is False


def test_protocol_defaults_present():
    """Pin the True side of all capability defaults (ADR-006 positive contract).

    Ensures that flipping a Protocol default from True to False cannot silently
    pass — both sides of every known capability state are asserted.

    Covered states (3 engines × 2 methods = 6):
      LlamaCppEngine:    supports_swap=True,  supports_hot_reload=True
      LlamaCppTQ3Engine: supports_swap=True,  supports_hot_reload=False  (False: test_protocol_overrides_present)
      VLLMEngine:        supports_swap=False,  supports_hot_reload=True   (False: test_protocol_overrides_present)
    """
    from llmcli.engines.llamacpp import LlamaCppEngine
    from llmcli.engines.llamacpp_tq3 import LlamaCppTQ3Engine
    from llmcli.engines.vllm import VLLMEngine

    assert LlamaCppEngine().supports_swap() is True
    assert LlamaCppEngine().supports_hot_reload() is True
    # LlamaCppTQ3Engine inherits supports_swap from LlamaCppEngine
    assert LlamaCppTQ3Engine().supports_swap() is True
    # VLLMEngine defines supports_hot_reload explicitly
    assert VLLMEngine().supports_hot_reload() is True
