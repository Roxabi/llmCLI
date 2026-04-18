from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import ModelSpec


@dataclass
class EngineInstance:
    pid: int
    port: int
    model: str
    started_at: float


class Engine(Protocol):
    def start(self, spec: ModelSpec) -> EngineInstance: ...

    def stop(self, instance: EngineInstance) -> None: ...

    def health(self, instance: EngineInstance) -> bool: ...

    @property
    def base_url(self) -> str: ...
