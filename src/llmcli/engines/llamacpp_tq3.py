from __future__ import annotations

from ..config import ModelSpec
from ..engine import EngineInstance


class LlamaCppTQ3Engine:
    """TurboQuant fork of llama.cpp — required for TQ3_4S mixed-quant models."""

    binary: str = "llama-server-tq3"

    def start(self, spec: ModelSpec) -> EngineInstance:
        raise NotImplementedError

    def stop(self, instance: EngineInstance) -> None:
        raise NotImplementedError

    def health(self, instance: EngineInstance) -> bool:
        raise NotImplementedError

    @property
    def base_url(self) -> str:
        raise NotImplementedError
