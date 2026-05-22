from __future__ import annotations

import os

from .llamacpp import LlamaCppEngine


class LlamaCppTQ3Engine(LlamaCppEngine):
    """TurboQuant fork of llama.cpp — required for TQ3_4S mixed-quant models.

    Drop-in replacement for vanilla llama-server: same CLI args, different binary.
    The binary defaults to ``llama-server-tq3`` but can be overridden via the
    ``LLMCLI_TQ3_BINARY`` environment variable.
    """

    binary: str = os.environ.get("LLMCLI_TQ3_BINARY", "llama-server-tq3")

    def supports_hot_reload(self) -> bool:
        return False
