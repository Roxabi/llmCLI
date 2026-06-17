from __future__ import annotations

import copy
import logging
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import yaml

from llmcli.auth.store import XAI_CREDENTIALS_PATH as _XAI_CREDENTIALS_PATH
from llmcli.config import Catalog, ModelSpec
from llmcli.support.providers import PROVIDERS, Provider

log = logging.getLogger(__name__)

LITELLM_CONFIG = Path.home() / ".litellm" / "config.yaml"

_DEFAULT_MODEL_REFRESH_SECS = 60.0
_PROBE_TIMEOUT_SECS = 3.0
_XAI_FETCH_TIMEOUT_SECS = 3.0
BLOCK_START = "# --- llmCLI managed block start ---"
BLOCK_END = "# --- llmCLI managed block end ---"

# Default proxy base used as a fallback when no proxy-base.yaml exists and no catalog is loaded.
# build_full_config uses a dynamic api_key_ref derived from catalog.host.api_key_env instead.
_DEFAULT_PROXY_BASE: dict[str, Any] = {
    "general_settings": {"master_key": "os.environ/LLMCLI_API_KEY"},
    "litellm_settings": {"drop_params": True},
}


def load_proxy_base(path: Path) -> dict[str, Any]:
    """Load LiteLLM transport config from optional proxy-base.yaml.

    - File absent (FileNotFoundError) → return deep copy of _DEFAULT_PROXY_BASE.
    - File present but empty (yaml.safe_load → None) → warn + return deep copy of default.
    - File present + valid mapping → return parsed dict.
    - File present + non-mapping (int, list, scalar) → raise yaml.YAMLError.
    - YAML error (syntax or ConstructorError from unsafe tags) → re-raise yaml.YAMLError.
    """
    try:
        text = path.read_text()
    except FileNotFoundError:
        return copy.deepcopy(_DEFAULT_PROXY_BASE)
    parsed = yaml.safe_load(text)
    if parsed is None:
        log.warning("proxy-base.yaml at %s is empty; using built-in defaults", path)
        return copy.deepcopy(_DEFAULT_PROXY_BASE)
    if not isinstance(parsed, dict):
        raise yaml.YAMLError(
            f"proxy-base.yaml must be a YAML mapping (dict), got {type(parsed).__name__}"
        )
    return parsed


def merge_proxy_config(
    base: dict[str, Any],
    model_list: list[dict[str, Any]],
    *,
    api_key_env: str = "LLMCLI_API_KEY",
) -> dict[str, Any]:
    """Overlay catalog-derived model_list on base; backfill defaults for missing keys.

    The api_key_env param is used to build the dynamic master_key fallback
    (matches build_full_config behavior). Default 'LLMCLI_API_KEY' for no-catalog callers.
    """
    result = copy.deepcopy(base)  # B3 — deep copy to prevent mutation of caller's base
    gs = result.setdefault("general_settings", {})
    gs.setdefault("master_key", f"os.environ/{api_key_env}")
    gs.setdefault("custom_auth", "proxy_custom_auth.custom_auth")
    ls = result.setdefault("litellm_settings", {})
    ls.setdefault("drop_params", _DEFAULT_PROXY_BASE["litellm_settings"]["drop_params"])
    result["model_list"] = model_list
    return result


class ModelDiscoveryCache:
    """Thread-safe in-memory cache for merged model_list entries."""

    def __init__(self, *, ttl_secs: float = _DEFAULT_MODEL_REFRESH_SECS) -> None:
        self._ttl_secs = ttl_secs
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] | None = None
        self._cache_key: str | None = None
        self._fetched_at: float = 0.0

    def invalidate(self) -> None:
        with self._lock:
            self._entries = None
            self._cache_key = None
            self._fetched_at = 0.0

    def get_or_refresh(
        self,
        cache_key: str,
        builder: Callable[[], list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        with self._lock:
            now = time.monotonic()
            if (
                self._entries is not None
                and self._cache_key == cache_key
                and (now - self._fetched_at) < self._ttl_secs
            ):
                return copy.deepcopy(self._entries)
        built = builder()
        with self._lock:
            self._entries = copy.deepcopy(built)
            self._cache_key = cache_key
            self._fetched_at = time.monotonic()
            return copy.deepcopy(built)


_MODEL_DISCOVERY_CACHE = ModelDiscoveryCache()
_refresh_callback: Callable[[], None] | None = None


def register_model_refresh_callback(callback: Callable[[], None] | None) -> None:
    """Register a hook invoked after cache invalidation (e.g. proxy immediate refresh)."""
    global _refresh_callback
    _refresh_callback = callback


def clear_model_cache() -> None:
    """Clear cached model_list without invoking the refresh callback."""
    _MODEL_DISCOVERY_CACHE.invalidate()


def invalidate_model_cache() -> None:
    """Clear cached model_list; trigger optional immediate refresh callback."""
    clear_model_cache()
    if _refresh_callback is not None:
        _refresh_callback()


def fetch_xai_models(forwarder_base: str, *, timeout: float = _XAI_FETCH_TIMEOUT_SECS) -> list[str]:
    """Fetch live Grok model IDs from the xAI OAuth forwarder."""
    url = f"{forwarder_base.rstrip('/')}/models"
    try:
        resp = httpx.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        log.warning("xAI forwarder model fetch failed (%s): %s", url, exc)
        return []
    ids: list[str] = []
    for item in payload.get("data", []):
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            model_id = item["id"]
            if not model_id.startswith("grok-imagine-"):
                ids.append(model_id)
    return ids


def _xai_credentials_cache_token() -> str:
    """Cache-bust token for xai.json presence and mtime (manual edits outside CLI)."""
    try:
        stat = _XAI_CREDENTIALS_PATH.stat()
    except OSError:
        return "absent"
    return f"present:{stat.st_mtime_ns}"


def probe_remote_model(
    spec: ModelSpec,
    provider: Provider,
    *,
    timeout: float = _PROBE_TIMEOUT_SECS,
) -> bool:
    """Return True when the provider's ``GET /models`` endpoint responds 2xx.

    Provider-level liveness only — a 200 here does not guarantee a specific
    ``model_id`` (e.g. ``kimi-k2.6``) succeeds at completion time.
    """
    if provider.key_env == "_OAUTH_MANAGED":
        return True
    api_key = os.environ.get(provider.key_env)
    if not api_key:
        return False
    url = f"{provider.api_base.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        resp = httpx.get(url, headers=headers, timeout=timeout)
        return 200 <= resp.status_code < 300
    except Exception as exc:
        log.debug("upstream probe failed for %s: %s", spec.name, exc)
        return False


def _xai_model_entry(model_name: str, api_base: str) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "litellm_params": {
            "model": f"openai/responses/{model_name}",
            "api_base": api_base,
            "api_key": "dummy",
        },
    }


def _build_model_list_uncached(
    catalog: Catalog, public_base_url: str, *, hostname: str | None = None
) -> list[dict[str, Any]]:
    """Build the model_list entries from the catalog.

    Args:
        catalog: Loaded catalog with host settings and model specs.
        public_base_url: Base URL for the host (e.g. 'http://roxabitower.lan').
        hostname: Override hostname for machine filter (default: socket.gethostname()).

    Returns:
        List of model_list entry dicts. Empty list when the filtered catalog is empty.
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
            if not probe_remote_model(spec, provider):
                continue
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

    existing_names = {entry["model_name"] for entry in model_list}
    xai_provider = PROVIDERS.get("xai-oauth")
    if (
        xai_provider is not None
        and xai_provider.key_env == "_OAUTH_MANAGED"
        and _XAI_CREDENTIALS_PATH.exists()
    ):
        for model_name in fetch_xai_models(xai_provider.api_base):
            if model_name in existing_names:
                continue
            model_list.append(_xai_model_entry(model_name, xai_provider.api_base))
            existing_names.add(model_name)

    return model_list


def build_model_list(
    catalog: Catalog,
    public_base_url: str,
    *,
    hostname: str | None = None,
    cache: ModelDiscoveryCache | None = None,
) -> list[dict[str, Any]]:
    """Build merged model_list from catalog, health probes, and live xAI discovery."""
    effective_cache = cache if cache is not None else _MODEL_DISCOVERY_CACHE
    effective_hostname = hostname if hostname is not None else socket.gethostname()
    cache_key = (
        f"{effective_hostname}:{public_base_url}:"
        f"{','.join(sorted(catalog.models))}:{catalog.host.api_key_env}:"
        f"{_xai_credentials_cache_token()}"
    )

    def _builder() -> list[dict[str, Any]]:
        return _build_model_list_uncached(catalog, public_base_url, hostname=hostname)

    return effective_cache.get_or_refresh(cache_key, _builder)


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
    api_key_ref = f"os.environ/{catalog.host.api_key_env}"
    model_list = build_model_list(catalog, public_base_url, hostname=hostname)
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


def emit_xai_oauth_warning_if_absent(err_console: Any) -> None:
    """Print a stderr warning when xAI credentials are absent.

    Called from cli/proxy.py:register_proxy after the LiteLLM config write.
    No-op when credentials exist.
    """
    if not _XAI_CREDENTIALS_PATH.exists():
        err_console.print(
            "[yellow]WARNING:[/yellow] xAI credentials not found — "
            "run `llmcli xai login` to enable Grok models"
        )


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
