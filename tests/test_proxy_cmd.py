"""Tests for llmcli.cli.proxy — _validate_provider_keys."""

from __future__ import annotations

import pytest

from llmcli.config import Catalog, HostSettings, ModelSpec
from llmcli.cli.proxy import _validate_provider_keys
from llmcli.providers import PROVIDERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PUBLIC_BASE_URL = "http://roxabitower.lan"


def _make_catalog(models: dict[str, dict] | None = None) -> Catalog:
    """Build a Catalog with the given model specs."""
    host = HostSettings(
        bind="0.0.0.0",
        public_base_url=PUBLIC_BASE_URL,
        api_key_env="LLMCLI_API_KEY",
        default_model="qwen3-8b",
        vram_budget_gib=16.0,
    )
    if models is None:
        models = {}
    model_specs = {name: ModelSpec(name=name, **spec) for name, spec in models.items()}
    return Catalog(host=host, models=model_specs)


# ---------------------------------------------------------------------------
# TestValidateProviderKeys
# ---------------------------------------------------------------------------


class TestValidateProviderKeys:
    def test_all_keys_set_returns_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No errors when all remote provider keys are present in the environment."""
        # Arrange
        provider = PROVIDERS["fireworks"]
        monkeypatch.setenv(provider.key_env, "test-key")
        catalog = _make_catalog(
            models={
                "kimi-k2": dict(
                    engine="remote",
                    provider="fireworks",
                    model_id="accounts/fireworks/models/kimi",
                    protocol="openai",
                    machines=[],
                )
            }
        )
        # Act
        result = _validate_provider_keys(catalog, hostname="roxabitower")
        # Assert
        assert result == []

    def test_missing_remote_key_returns_one_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One error string is returned when the provider key is absent from the environment."""
        # Arrange
        provider = PROVIDERS["nvidia-nim"]
        monkeypatch.delenv(provider.key_env, raising=False)
        catalog = _make_catalog(
            models={
                "nvidia-llama": dict(
                    engine="remote",
                    provider="nvidia-nim",
                    model_id="meta/llama-3.1-8b-instruct",
                    protocol="openai",
                    machines=[],
                )
            }
        )
        # Act
        result = _validate_provider_keys(catalog, hostname="roxabitower")
        # Assert
        assert len(result) == 1
        assert provider.key_env in result[0]

    def test_local_models_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Local engine models (e.g. llamacpp) are skipped regardless of environment state."""
        # Arrange — ensure no stray provider keys that could mask a bug
        for p in PROVIDERS.values():
            monkeypatch.delenv(p.key_env, raising=False)
        catalog = _make_catalog(
            models={
                "qwen3-8b": dict(
                    engine="llamacpp",
                    repo="Org/Qwen3-8B-GGUF",
                    file="qwen3-8b-q4_k_m.gguf",
                    port=8091,
                    vram_gib=5.5,
                )
            }
        )
        # Act
        result = _validate_provider_keys(catalog, hostname="roxabitower")
        # Assert
        assert result == []
