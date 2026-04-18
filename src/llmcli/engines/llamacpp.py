from __future__ import annotations

from ..config import ModelSpec
from ..engine import EngineInstance


class LlamaCppEngine:
    """Vanilla llama.cpp llama-server wrapper (standard GGUF)."""

    binary: str = "llama-server"

    def start(self, spec: ModelSpec) -> EngineInstance:
        raise NotImplementedError

    def stop(self, instance: EngineInstance) -> None:
        raise NotImplementedError

    def health(self, instance: EngineInstance) -> bool:
        raise NotImplementedError

    @property
    def base_url(self) -> str:
        raise NotImplementedError
