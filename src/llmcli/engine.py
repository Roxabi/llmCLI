from dataclasses import dataclass, field
from typing import Protocol

from .config import ModelSpec


@dataclass
class EngineInstance:
    pid: int
    port: int
    model_name: str
    started_at: float = field(default=0.0)

    def __init__(
        self,
        pid: int,
        port: int,
        model_name: str = "",
        started_at: float = 0.0,
        *,
        model: str = "",
    ) -> None:
        self.pid = pid
        self.port = port
        # Accept 'model' as an alias for 'model_name' (test_llamacpp.py compat)
        self.model_name = model_name or model
        self.started_at = started_at

    @property
    def model(self) -> str:
        """Alias for model_name — compatibility with test_llamacpp.py fixtures."""
        return self.model_name

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"


class Engine(Protocol):
    def start(self, spec: ModelSpec) -> EngineInstance: ...

    def stop(self, instance: EngineInstance) -> None: ...

    def health(self, instance: EngineInstance) -> bool: ...

    # Override capability flags. Default = True (capable). Engines that diverge
    # from the canonical stage shape MUST override to False — see ADR-006.
    # NOTE: Protocol defaults below apply only to nominal subclasses
    # (class X(Engine)); structural implementors MUST define these concretely.
    def supports_swap(self) -> bool:
        return True

    def supports_hot_reload(self) -> bool:
        return True
