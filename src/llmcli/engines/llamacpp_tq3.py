from __future__ import annotations

from typing import NoReturn

from ..config import HostSettings, ModelSpec
from ..engine import EngineInstance
from .base import spawn_llama_server


class LlamaCppTQ3Engine:
    """TurboQuant fork of llama.cpp — required for TQ3_4S mixed-quant models."""

    binary: str = "llama-server-tq3"

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
