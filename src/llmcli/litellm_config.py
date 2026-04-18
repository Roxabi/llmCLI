from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from .config import Catalog

LITELLM_CONFIG = Path.home() / ".litellm" / "config.yaml"
BLOCK_START = "# --- llmCLI managed block start ---"
BLOCK_END = "# --- llmCLI managed block end ---"


def build_block(catalog: Catalog, public_base_url: str) -> str:
    """Build a namespaced model_list block for the proxy config.

    Args:
        catalog: Loaded catalog with host settings and model specs.
        public_base_url: Base URL for the host (e.g. 'http://roxabitower.lan').

    Returns:
        YAML string wrapped in llmCLI sentinel comments.
    """
    api_key_ref = f"os.environ/{catalog.host.api_key_env}"

    if catalog.models:
        model_list = [
            {
                "model_name": name,
                "litellm_params": {
                    "model": f"openai/{name}",
                    "api_base": f"{public_base_url}:{spec.port}/v1",
                    "api_key": api_key_ref,
                },
            }
            for name, spec in catalog.models.items()
        ]
        inner = yaml.safe_dump({"model_list": model_list}, default_flow_style=False, sort_keys=False)
    else:
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
