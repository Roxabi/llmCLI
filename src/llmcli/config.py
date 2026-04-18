from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_CONFIG_PATH = Path(
    os.environ.get("LLMCLI_CONFIG", Path.home() / ".config" / "llmcli" / "llmcli.toml")
)


@dataclass(frozen=True)
class HostSettings:
    bind: str = "0.0.0.0"
    public_base_url: str = "http://localhost"
    api_key_env: str = "LLMCLI_API_KEY"


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


def load(path: Path = DEFAULT_CONFIG_PATH) -> Catalog:
    with path.open("rb") as f:
        data = tomllib.load(f)

    host_data = data.get("host", {})
    host = HostSettings(**{k: v for k, v in host_data.items() if k in HostSettings.__annotations__})

    models = {
        name: ModelSpec(name=name, **spec) for name, spec in data.get("models", {}).items()
    }
    return Catalog(host=host, models=models)
