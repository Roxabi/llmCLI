from .llamacpp import LlamaCppEngine
from .llamacpp_tq3 import LlamaCppTQ3Engine
from .vllm import VLLMEngine
from ..config import ModelSpec
from ..engine import Engine

_ENGINE_REGISTRY: dict[str, type] = {
    "llamacpp": LlamaCppEngine,
    "llamacpp_tq3": LlamaCppTQ3Engine,
    "vllm": VLLMEngine,
}


def get_engine(spec: ModelSpec) -> Engine:
    """Return an engine instance for the given spec.engine value.

    Raises ValueError for unknown engine names.
    """
    engine_cls = _ENGINE_REGISTRY.get(spec.engine)
    if engine_cls is None:
        raise ValueError(
            f"Unknown engine '{spec.engine}' for model '{spec.name}'. "
            f"Valid engines: {sorted(_ENGINE_REGISTRY)}"
        )
    return engine_cls()


__all__ = ["LlamaCppEngine", "LlamaCppTQ3Engine", "VLLMEngine", "get_engine"]
