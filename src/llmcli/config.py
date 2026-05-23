from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .gpu import kv_overhead_gib, probe_free_vram_gib
from .support.providers import PROVIDERS

logger = logging.getLogger(__name__)


# Resolved at import time — set LLMCLI_CONFIG before import or pass path= to load().
DEFAULT_CONFIG_PATH = Path(
    os.environ.get("LLMCLI_CONFIG", Path.home() / ".roxabi" / "llmcli" / "llmcli.toml")
)

_VALID_PROTOCOLS = frozenset({"openai", "anthropic"})
_LOCAL_ENGINES = frozenset({"llamacpp", "llamacpp_tq3", "vllm"})
_REMOTE_LOCAL_FIELDS = frozenset({"repo", "file", "port", "vram_gib", "flags", "mmproj"})


@dataclass(frozen=True)
class HostSettings:
    bind: str = "0.0.0.0"
    public_base_url: str = "http://localhost"
    api_key_env: str = "LLMCLI_API_KEY"
    default_model: str | None = None
    vram_budget_gib: float | None = None
    port: int | None = None  # B2 — None = absent from TOML; resolver falls back to 18091


@dataclass(frozen=True)
class ModelSpec:
    name: str
    engine: str
    # Local-engine fields — optional (forbidden for engine="remote")
    repo: str = ""
    port: int = 0
    vram_gib: float = 0.0
    file: str = ""
    flags: list[str] = field(default_factory=list)
    mmproj: str | None = None
    # Remote-engine fields — forbidden for local engines
    provider: str = ""
    model_id: str = ""
    protocol: str = "openai"
    # Startup timeout override — None = use engine default (llamacpp=60s, vllm=180s)
    startup_timeout_s: int | None = None
    # Per-machine filter — empty = all hosts
    machines: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Catalog:
    host: HostSettings
    models: dict[str, ModelSpec]


def _parse_model_spec(name: str, spec: dict) -> ModelSpec:
    engine = spec.get("engine", "")
    valid_engines = _LOCAL_ENGINES | {"remote"}

    if "engine" not in spec:
        raise ValueError(
            f"Model '{name}' is missing required field 'engine'. "
            f"Valid engines: {sorted(valid_engines)}."
        )

    if engine not in valid_engines:
        raise ValueError(
            f"Model '{name}' has unknown engine '{engine}'. Valid engines: {sorted(valid_engines)}."
        )

    if engine == "remote":
        # Validate required remote fields
        provider = spec.get("provider", "")
        model_id = spec.get("model_id", "")
        protocol = spec.get("protocol", "openai")

        if not provider:
            raise ValueError(
                f"Model '{name}' with engine='remote' is missing required field 'provider'."
            )
        if provider not in PROVIDERS:
            raise ValueError(
                f"Model '{name}' references unknown provider '{provider}'. "
                f"Valid providers: {sorted(PROVIDERS.keys())}."
            )
        if not model_id:
            raise ValueError(
                f"Model '{name}' with engine='remote' is missing required field 'model_id'."
            )
        if protocol not in _VALID_PROTOCOLS:
            raise ValueError(
                f"Model '{name}' has invalid protocol '{protocol}'. "
                f"Valid protocols: {sorted(_VALID_PROTOCOLS)}."
            )
        # Cross-validate provider × protocol
        if provider == "anthropic" and protocol != "anthropic":
            raise ValueError(
                f"Model '{name}' uses provider='anthropic' which requires protocol='anthropic', "
                f"got protocol='{protocol}'."
            )
        if provider != "anthropic" and protocol == "anthropic":
            raise ValueError(
                f"Model '{name}' uses protocol='anthropic' which is only supported by "
                f"provider='anthropic', got provider='{provider}'."
            )
        # Reject mixing remote with local-only fields
        mixed = _REMOTE_LOCAL_FIELDS & spec.keys()
        if mixed:
            raise ValueError(
                f"Model '{name}' with engine='remote' must not set local-engine fields: "
                f"{sorted(mixed)}. Remove them or use a local engine."
            )
    else:
        # Local engine — require repo, reject remote fields
        if "repo" not in spec:
            raise ValueError(
                f"Model '{name}' is missing required field 'repo'. "
                "Add a 'repo' key pointing to the HuggingFace repository (e.g. 'Org/Model-GGUF')."
            )
        remote_fields = {"provider", "model_id", "protocol"} & spec.keys()
        if remote_fields:
            raise ValueError(
                f"Model '{name}' with engine='{engine}' must not set remote-engine fields: "
                f"{sorted(remote_fields)}. Remove them or use engine='remote'."
            )

    return ModelSpec(name=name, **spec)


def load(path: Path = DEFAULT_CONFIG_PATH) -> Catalog:
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"llmCLI catalog not found at {path}.\n"
            f"If you're upgrading from ~/.config/llmcli/, run:\n"
            f"  mv ~/.config/llmcli ~/.roxabi/llmcli\n"
            f"Or set LLMCLI_CONFIG=<path> to point at a custom location."
        ) from e

    host_data = data.get("host", {})
    host = HostSettings(
        **{k: v for k, v in host_data.items() if k in HostSettings.__dataclass_fields__}
    )

    # Inline models — kept for backward compat (single-file configs still work)
    models: dict[str, ModelSpec] = {
        name: _parse_model_spec(name, spec) for name, spec in data.get("models", {}).items()
    }

    # Per-file models: <config_dir>/models/<name>.toml — take precedence over inline
    models_dir = path.parent / "models"
    if models_dir.is_dir():
        for model_file in sorted(models_dir.glob("*.toml")):
            name = model_file.stem
            with model_file.open("rb") as f:
                spec_data = tomllib.load(f)
            if name in models:
                logger.warning(
                    "Model '%s' defined both inline and in models/; using models/ version", name
                )
            models[name] = _parse_model_spec(name, spec_data)

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


_KNOWN_NATS: dict[str, type] = {"url": str}


def load_nats_config(config: Path | None = None) -> dict:
    """Load the ``[nats]`` table from llmcli.toml.

    Returns ``{}`` when no config is found or the table is absent.
    """
    path = config if config is not None else DEFAULT_CONFIG_PATH
    if path is None or not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    raw = data.get("nats", {})
    result: dict = {}
    for key, expected_type in _KNOWN_NATS.items():
        if key in raw:
            try:
                result[key] = expected_type(raw[key])
            except (ValueError, TypeError):
                pass
    return result


def apply_nats_env_from_config(config: Path | None = None) -> None:
    """Populate ``LLMCLI_NATS_URL`` from ``[nats]`` toml when env is unset.

    Env var takes precedence — wrappers, CI, and ad-hoc overrides keep working.
    Call at the entrypoint of any NATS-using command before reading ``os.environ``.
    """
    cfg = load_nats_config(config)
    if "LLMCLI_NATS_URL" not in os.environ:
        url = cfg.get("url")
        if isinstance(url, str) and url:
            os.environ["LLMCLI_NATS_URL"] = url
