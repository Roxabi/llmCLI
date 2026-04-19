from __future__ import annotations

from typing import NoReturn

from ..config import HostSettings, ModelSpec
from ..engine import EngineInstance
from .base import spawn_llama_server


class LlamaCppEngine:
    """Vanilla llama.cpp llama-server wrapper (standard GGUF)."""

    binary: str = "llama-server"

    def __init__(self, host: HostSettings) -> None:
        self.host = host

    def start(self, spec: ModelSpec) -> NoReturn:
        spawn_llama_server(self.binary, spec, self.host)

    def stop(self, instance: EngineInstance) -> None:
        raise NotImplementedError

    def health(self, instance: EngineInstance) -> bool:
        raise NotImplementedError

    @property
    def base_url(self) -> str:
        raise NotImplementedError
