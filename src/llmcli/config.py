from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .gpu import kv_overhead_gib, probe_free_vram_gib

logger = logging.getLogger(__name__)


DEFAULT_CONFIG_PATH = Path(
    os.environ.get("LLMCLI_CONFIG", Path.home() / ".config" / "llmcli" / "llmcli.toml")
)


@dataclass(frozen=True)
class HostSettings:
    bind: str = "0.0.0.0"
    public_base_url: str = "http://localhost"
    api_key_env: str = "LLMCLI_API_KEY"
    default_model: str | None = None
    vram_budget_gib: float | None = None


@dataclass(frozen=True)
class ModelSpec:
    name: str
    engine: str
    repo: str
    file: str
    port: int
    vram_gib: float
    flags: list[str] = field(default_factory=list)
    mmproj: str | None = None


@dataclass(frozen=True)
class Catalog:
    host: HostSettings
    models: dict[str, ModelSpec]


def _parse_model_spec(name: str, spec: dict) -> ModelSpec:
    if "repo" not in spec:
        raise ValueError(
            f"Model '{name}' is missing required field 'repo'. "
            "Add a 'repo' key pointing to the HuggingFace repository (e.g. 'Org/Model-GGUF')."
        )
    return ModelSpec(name=name, **spec)


def load(path: Path = DEFAULT_CONFIG_PATH) -> Catalog:
    with path.open("rb") as f:
        data = tomllib.load(f)

    host_data = data.get("host", {})
    host = HostSettings(**{k: v for k, v in host_data.items() if k in HostSettings.__dataclass_fields__})

    models = {
        name: _parse_model_spec(name, spec) for name, spec in data.get("models", {}).items()
    }
    return Catalog(host=host, models=models)


def check_vram_budget(spec: ModelSpec, host: HostSettings) -> None:
    """Raise ValueError if spec.vram_gib exceeds the host budget or current free VRAM.

    Two-stage check:
    1. Static: compare spec.vram_gib against host.vram_budget_gib (catalog ceiling).
    2. Dynamic: probe actual free VRAM via pynvml/nvidia-smi and factor in KV-cache overhead.
       Skipped when the probe returns 0.0 (GPU tools unavailable — logged as a warning).

    No-op for stage 1 when host.vram_budget_gib is None (unconstrained host).
    """
    # Stage 1 — static catalog ceiling
    if host.vram_budget_gib is not None:
        if spec.vram_gib > host.vram_budget_gib:
            raise ValueError(
                f"Model '{spec.name}' requires {spec.vram_gib} GiB VRAM but this host's budget is "
                f"{host.vram_budget_gib} GiB. Choose a smaller model that fits within the budget."
            )

    # Stage 2 — dynamic free-VRAM probe
    free_gib = probe_free_vram_gib()
    if free_gib == 0.0:
        # Probe unavailable — skip dynamic check (warning already logged in probe_free_vram_gib)
        return

    overhead = kv_overhead_gib(spec.flags)
    required = spec.vram_gib + overhead
    if free_gib < required:
        held_gib = round(
            (host.vram_budget_gib - free_gib) if host.vram_budget_gib is not None else 0.0,
            2,
        )
        raise ValueError(
            f"Model '{spec.name}' requires {required:.2f} GiB "
            f"({spec.vram_gib} GiB model + {overhead:.2f} GiB KV cache); "
            f"only {free_gib:.2f} GiB free now "
            f"(desktop/other processes holding {held_gib:.2f} GiB). "
            "Free VRAM or pick a smaller model."
        )
