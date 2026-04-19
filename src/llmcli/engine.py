from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn, Protocol

from .config import ModelSpec


@dataclass
class EngineInstance:
    pid: int
    port: int
    model: str
    started_at: float


class Engine(Protocol):
    def start(self, spec: ModelSpec) -> NoReturn: ...

    def stop(self, instance: EngineInstance) -> None: ...

    def health(self, instance: EngineInstance) -> bool: ...

    @property
    def base_url(self) -> str: ...
