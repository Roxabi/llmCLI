from __future__ import annotations

from pathlib import Path

from .config import Catalog

LITELLM_CONFIG = Path.home() / ".litellm" / "config.yaml"
BLOCK_START = "# --- llmCLI managed block start ---"
BLOCK_END = "# --- llmCLI managed block end ---"


def build_block(catalog: Catalog, public_base_url: str) -> str:
    """Build a namespaced model_list block for the proxy config."""
    raise NotImplementedError


def write_block(block: str, path: Path = LITELLM_CONFIG) -> None:
    """Idempotently replace the llmCLI block in the proxy config. Never touch other entries."""
    raise NotImplementedError
