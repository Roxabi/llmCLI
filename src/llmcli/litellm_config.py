from __future__ import annotations

import socket
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .config import Catalog
from .providers import PROVIDERS

LITELLM_CONFIG = Path.home() / ".litellm" / "config.yaml"
BLOCK_START = "# --- llmCLI managed block start ---"
BLOCK_END = "# --- llmCLI managed block end ---"


def build_full_config(
    catalog: Catalog, public_base_url: str, *, hostname: str | None = None
) -> dict[str, Any]:
    """Build a complete LiteLLM proxy config dict from the catalog.

    Args:
        catalog: Loaded catalog with host settings and model specs.
        public_base_url: Base URL for the host (e.g. 'http://roxabitower.lan').
        hostname: Override hostname for machine filter (default: socket.gethostname()).

    Returns:
        Dict with keys: general_settings, litellm_settings, model_list.
        model_list is [] (not None) when the filtered catalog is empty.
    """
    effective_hostname = hostname if hostname is not None else socket.gethostname()
    api_key_ref = f"os.environ/{catalog.host.api_key_env}"

    model_list: list[dict[str, Any]] = []
    for name, spec in catalog.models.items():
        # Per-machine filter: skip if machines is set and hostname not in list
        if spec.machines and effective_hostname not in spec.machines:
            continue

        if spec.engine == "remote":
            provider_cfg = PROVIDERS.get(spec.provider)
            if provider_cfg is None:
                raise ValueError(
                    f"Unknown provider '{spec.provider}' in spec '{name}'. "
                    f"Valid providers: {sorted(PROVIDERS.keys())}."
                )
            provider = provider_cfg
            if spec.protocol == "anthropic":
                entry: dict[str, Any] = {
                    "model_name": name,
                    "litellm_params": {
                        "model": f"anthropic/{spec.model_id}",
                        "api_key": f"os.environ/{provider.key_env}",
                    },
                }
            else:
                # protocol == "openai"
                entry = {
                    "model_name": name,
                    "litellm_params": {
                        "model": f"openai/{spec.model_id}",
                        "api_base": provider.api_base,
                        "api_key": f"os.environ/{provider.key_env}",
                    },
                }
        else:
            # Local engines: llamacpp, llamacpp_tq3, vllm
            entry = {
                "model_name": name,
                "litellm_params": {
                    "model": f"openai/{name}",
                    "api_base": f"{public_base_url}:{spec.port}/v1",
                    "api_key": api_key_ref,
                },
            }
        model_list.append(entry)

    return {
        "general_settings": {"master_key": api_key_ref},
        "litellm_settings": {"drop_params": True},
        "model_list": model_list,
    }


def build_block(catalog: Catalog, public_base_url: str, *, hostname: str | None = None) -> str:
    """Build a namespaced model_list block for the proxy config.

    Args:
        catalog: Loaded catalog with host settings and model specs.
        public_base_url: Base URL for the host (e.g. 'http://roxabitower.lan').
        hostname: Override hostname for machine filter (default: socket.gethostname()).

    Returns:
        YAML string wrapped in llmCLI sentinel comments.
    """
    cfg = build_full_config(catalog, public_base_url, hostname=hostname)
    model_list = cfg["model_list"]
    if model_list:
        inner = yaml.safe_dump(
            {"model_list": model_list}, default_flow_style=False, sort_keys=False
        )
    else:
        # null (not []) preserves register-proxy backwards-compat sentinel form
        inner = yaml.safe_dump({"model_list": None}, default_flow_style=False)
    return f"{BLOCK_START}\n{inner}{BLOCK_END}\n"


def write_block(block: str, path: Path = LITELLM_CONFIG) -> None:
    """Idempotently replace the llmCLI block in the proxy config.

    Behaviour:
    - Always writes a .bak backup before modifying (even for new files).
    - File absent → create file containing only the block.
    - File present, no sentinels → append block (with preceding newline).
    - File present, sentinels present → splice new block in place.
    - Malformed (only one sentinel) → raise ValueError.

    Args:
        block: The full sentinel-wrapped YAML string from build_block().
        path: Destination config file path (default ~/.litellm/config.yaml).
    """
    backup = path.with_suffix(path.suffix + ".bak")

    # --- read existing content (may not exist) ---
    if path.exists():
        existing = path.read_text()
    else:
        existing = ""

    # --- always write backup first ---
    backup.write_text(existing)

    # --- detect sentinel positions ---
    has_start = BLOCK_START in existing
    has_end = BLOCK_END in existing

    if has_start and not has_end:
        raise ValueError(
            f"Malformed llmCLI config: found '{BLOCK_START}' without a matching '{BLOCK_END}' "
            f"in {path}. Fix the file manually before proceeding."
        )
    if has_end and not has_start:
        raise ValueError(
            f"Malformed llmCLI config: found '{BLOCK_END}' without a matching '{BLOCK_START}' "
            f"in {path}. Fix the file manually before proceeding."
        )

    if not has_start and not has_end:
        # No sentinels — either empty/absent file or append case
        if existing:
            # Ensure a single newline separator before the block
            separator = "" if existing.endswith("\n") else "\n"
            new_content = existing + separator + block
        else:
            new_content = block
    else:
        # Both sentinels present — splice block in place
        lines = existing.splitlines(keepends=True)

        start_idx: int | None = None
        end_idx: int | None = None
        for i, line in enumerate(lines):
            if BLOCK_START in line:
                start_idx = i
            if BLOCK_END in line:
                end_idx = i

        # start_idx and end_idx are guaranteed non-None here (both sentinels found)
        assert start_idx is not None and end_idx is not None  # noqa: S101 (guarded above)

        before = "".join(lines[:start_idx])
        after = "".join(lines[end_idx + 1 :])

        new_content = before + block + after

    path.write_text(new_content)


def reload_proxy() -> None:
    """Reload the LiteLLM proxy by running 'make litellm reload' in the lyra supervisor dir.

    This is a thin side-effect wrapper. Tests mock subprocess.run.
    """
    litellm_dir = Path.home() / ".litellm"
    subprocess.run(  # noqa: S603
        ["make", "litellm", "reload"],
        cwd=litellm_dir,
        check=True,
    )
