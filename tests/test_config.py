"""Tests for llmcli.config — RED phase (T1.1).

These tests MUST fail against the current scaffold because:
- HostSettings is missing `default_model` field
- HostSettings is missing `vram_budget_gib` field
- `check_vram_budget()` helper does not exist

Expected: ImportError or AttributeError failures. GREEN phase (T1.8) adds the missing fields/helper.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from llmcli.config import (
    Catalog,
    HostSettings,
    ModelSpec,
    _parse_model_spec,
    load,
    check_vram_budget,
)
from llmcli.support.providers import PROVIDERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "llmcli.toml"
    p.write_text(textwrap.dedent(content))
    return p


MINIMAL_TOML = """\
    [host]
    bind              = "0.0.0.0"
    public_base_url   = "http://localhost"
    api_key_env       = "LLMCLI_API_KEY"
    default_model     = "small-q4"
    vram_budget_gib   = 10.0

    [models.small-q4]
    engine   = "llamacpp"
    repo     = "SomeOrg/small-model-GGUF"
    file     = "small-q4_k_m.gguf"
    port     = 8091
    vram_gib = 6.0
    flags    = ["-ngl", "99"]
"""

OVERSIZED_TOML = """\
    [host]
    bind              = "0.0.0.0"
    public_base_url   = "http://localhost"
    api_key_env       = "LLMCLI_API_KEY"
    default_model     = "big-model"
    vram_budget_gib   = 8.0

    [models.big-model]
    engine   = "llamacpp_tq3"
    repo     = "SomeOrg/big-model-GGUF"
    file     = "big-model.gguf"
    port     = 8092
    vram_gib = 13.0
    flags    = ["-ngl", "99"]
"""


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_default_config_path_matches_roxabi_convention(self) -> None:
        """DEFAULT_CONFIG_PATH resolves to ~/.roxabi/llmcli/llmcli.toml absent override.

        Verified in a subprocess to avoid module-reload side-effects on other tests.
        """
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os; os.environ.pop('LLMCLI_CONFIG', None); "
                    "import llmcli.config as cfg; from pathlib import Path; "
                    "expected = Path.home() / '.roxabi' / 'llmcli' / 'llmcli.toml'; "
                    "assert cfg.DEFAULT_CONFIG_PATH == expected, "
                    "f'Got {cfg.DEFAULT_CONFIG_PATH!r}, expected {expected!r}'"
                ),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# SC-1 / C10 — catalog load
# ---------------------------------------------------------------------------


class TestLoad:
    def test_load_returns_catalog_type(self, tmp_path: Path) -> None:
        """load() returns a Catalog instance."""
        # Arrange
        toml_path = _write_toml(tmp_path, MINIMAL_TOML)
        # Act
        catalog = load(toml_path)
        # Assert
        assert isinstance(catalog, Catalog)

    def test_load_host_settings(self, tmp_path: Path) -> None:
        """Catalog.host is a HostSettings with correct values."""
        # Arrange
        toml_path = _write_toml(tmp_path, MINIMAL_TOML)
        # Act
        catalog = load(toml_path)
        # Assert
        assert isinstance(catalog.host, HostSettings)
        assert catalog.host.bind == "0.0.0.0"
        assert catalog.host.public_base_url == "http://localhost"
        assert catalog.host.api_key_env == "LLMCLI_API_KEY"

    def test_load_models_dict(self, tmp_path: Path) -> None:
        """Catalog.models is a dict[str, ModelSpec]."""
        # Arrange
        toml_path = _write_toml(tmp_path, MINIMAL_TOML)
        # Act
        catalog = load(toml_path)
        # Assert
        assert "small-q4" in catalog.models
        spec = catalog.models["small-q4"]
        assert isinstance(spec, ModelSpec)
        assert spec.engine == "llamacpp"
        assert spec.repo == "SomeOrg/small-model-GGUF"
        assert spec.port == 8091
        assert spec.vram_gib == 6.0

    def test_load_model_flags_list(self, tmp_path: Path) -> None:
        """ModelSpec.flags is parsed as a list of strings."""
        # Arrange
        toml_path = _write_toml(tmp_path, MINIMAL_TOML)
        # Act
        catalog = load(toml_path)
        # Assert
        assert catalog.models["small-q4"].flags == ["-ngl", "99"]

    def test_load_example_toml_realistic(self) -> None:
        """Loading llmcli.example.toml from repo root succeeds and has expected structure.

        Models live in the sibling models/ directory — the example toml is host-only.
        """
        # Arrange — find repo root relative to this file
        repo_root = Path(__file__).parent.parent
        example_path = repo_root / "llmcli.example.toml"
        # Act
        catalog = load(example_path)
        # Assert
        assert isinstance(catalog, Catalog)
        assert isinstance(catalog.host, HostSettings)
        # models/ dir is sibling to llmcli.example.toml; must have at least one model
        assert len(catalog.models) >= 1
        valid_engines = {"llamacpp", "llamacpp_tq3", "vllm", "remote"}
        for name, spec in catalog.models.items():
            assert spec.name == name
            assert spec.engine in valid_engines
            if spec.engine != "remote":
                assert spec.port > 0
                assert spec.vram_gib > 0

    def test_load_models_dir_loaded(self, tmp_path: Path) -> None:
        """Models defined in models/*.toml are merged into the catalog."""
        # Arrange
        toml_path = _write_toml(tmp_path, '[host]\nbind = "0.0.0.0"\n')
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / "my-model.toml").write_text(
            'engine = "llamacpp"\nrepo = "Org/M-GGUF"\nfile = "m.gguf"\nport = 8099\nvram_gib = 6.0\n'
        )
        # Act
        catalog = load(toml_path)
        # Assert
        assert "my-model" in catalog.models
        assert catalog.models["my-model"].engine == "llamacpp"
        assert catalog.models["my-model"].port == 8099

    def test_load_models_dir_overrides_inline(self, tmp_path: Path) -> None:
        """A model file in models/ overrides an inline [models.*] entry of the same name."""
        # Arrange — inline model with port 8091
        toml_path = _write_toml(
            tmp_path,
            '[host]\nbind = "0.0.0.0"\n\n[models.m]\nengine = "llamacpp"\nrepo = "Org/M"\nfile = "m.gguf"\nport = 8091\nvram_gib = 6.0\n',
        )
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        # models/ version has port 9999
        (models_dir / "m.toml").write_text(
            'engine = "llamacpp"\nrepo = "Org/M"\nfile = "m.gguf"\nport = 9999\nvram_gib = 6.0\n'
        )
        # Act
        catalog = load(toml_path)
        # Assert — models/ version wins
        assert catalog.models["m"].port == 9999

    def test_load_missing_config_raises_friendly_error(self, tmp_path: Path) -> None:
        """load() raises FileNotFoundError (or similar) when config is absent."""
        # Arrange
        missing = tmp_path / "nonexistent.toml"
        # Act / Assert
        with pytest.raises((FileNotFoundError, OSError)):
            load(missing)


# ---------------------------------------------------------------------------
# T1.1 (new fields) — default_model + vram_budget_gib on HostSettings
# ---------------------------------------------------------------------------


class TestHostSettingsNewFields:
    def test_default_model_field_exists_on_hostsettings(self, tmp_path: Path) -> None:
        """HostSettings must have a `default_model` field (str | None)."""
        # Arrange
        toml_path = _write_toml(tmp_path, MINIMAL_TOML)
        # Act
        catalog = load(toml_path)
        # Assert — attribute must exist (GREEN: value is "small-q4")
        assert hasattr(catalog.host, "default_model")
        assert catalog.host.default_model == "small-q4"

    def test_vram_budget_gib_field_exists_on_hostsettings(self, tmp_path: Path) -> None:
        """HostSettings must have a `vram_budget_gib` field (float | None)."""
        # Arrange
        toml_path = _write_toml(tmp_path, MINIMAL_TOML)
        # Act
        catalog = load(toml_path)
        # Assert
        assert hasattr(catalog.host, "vram_budget_gib")
        assert catalog.host.vram_budget_gib == 10.0

    def test_default_model_defaults_to_none_when_absent(self, tmp_path: Path) -> None:
        """default_model should default to None when not in TOML."""
        # Arrange — no default_model, no vram_budget_gib
        toml = """\
            [host]
            bind = "0.0.0.0"

            [models.m]
            engine   = "llamacpp"
            repo     = "Org/m"
            file     = "m.gguf"
            port     = 8091
            vram_gib = 4.0
        """
        toml_path = _write_toml(tmp_path, toml)
        # Act
        catalog = load(toml_path)
        # Assert
        assert catalog.host.default_model is None

    def test_vram_budget_gib_defaults_to_none_when_absent(self, tmp_path: Path) -> None:
        """vram_budget_gib should default to None when not in TOML."""
        # Arrange
        toml = """\
            [host]
            bind = "0.0.0.0"

            [models.m]
            engine   = "llamacpp"
            repo     = "Org/m"
            file     = "m.gguf"
            port     = 8091
            vram_gib = 4.0
        """
        toml_path = _write_toml(tmp_path, toml)
        # Act
        catalog = load(toml_path)
        # Assert
        assert catalog.host.vram_budget_gib is None


# ---------------------------------------------------------------------------
# C2 / SC-14 — check_vram_budget helper
# ---------------------------------------------------------------------------


class TestCheckVramBudget:
    def test_check_vram_budget_passes_when_model_fits(self, tmp_path: Path) -> None:
        """check_vram_budget does not raise when model vram_gib <= budget."""
        # Arrange
        toml_path = _write_toml(tmp_path, MINIMAL_TOML)
        catalog = load(toml_path)
        spec = catalog.models["small-q4"]  # vram_gib=6 <= budget=10
        # Act / Assert — must not raise
        check_vram_budget(spec, catalog.host)

    def test_check_vram_budget_raises_when_model_exceeds_budget(self, tmp_path: Path) -> None:
        """check_vram_budget raises ValueError when model vram_gib > budget."""
        # Arrange
        toml_path = _write_toml(tmp_path, OVERSIZED_TOML)
        catalog = load(toml_path)
        spec = catalog.models["big-model"]  # vram_gib=13 > budget=8
        # Act / Assert
        with pytest.raises(ValueError):
            check_vram_budget(spec, catalog.host)

    def test_vram_guard_error_names_model_and_budget(self, tmp_path: Path) -> None:
        """ValueError message must mention the model name and the budget (SC-14)."""
        # Arrange
        toml_path = _write_toml(tmp_path, OVERSIZED_TOML)
        catalog = load(toml_path)
        spec = catalog.models["big-model"]
        # Act / Assert
        with pytest.raises(ValueError, match=r"big-model"):
            check_vram_budget(spec, catalog.host)

    def test_vram_guard_passes_when_budget_is_none(self, tmp_path: Path) -> None:
        """check_vram_budget static stage is a no-op when host.vram_budget_gib is None.

        The dynamic probe is mocked to return sufficient free VRAM so this test
        only exercises the static (catalog) check — not GPU state.
        """
        from unittest.mock import patch

        # Arrange
        toml = """\
            [host]
            bind = "0.0.0.0"

            [models.huge]
            engine   = "llamacpp"
            repo     = "Org/huge"
            file     = "huge.gguf"
            port     = 8091
            vram_gib = 999.0
        """
        toml_path = _write_toml(tmp_path, toml)
        catalog = load(toml_path)
        spec = catalog.models["huge"]
        # Act / Assert — no static budget set → static check must not raise.
        # Dynamic probe mocked to 0.0 (GPU tools unavailable) so it is skipped too.
        with patch("llmcli.config.probe_free_vram_gib", return_value=0.0):
            check_vram_budget(spec, catalog.host)


# ---------------------------------------------------------------------------
# ModelSpec field defaults — vLLM models have no GGUF file
# ---------------------------------------------------------------------------


class TestModelSpecDefaults:
    def test_model_spec_file_defaults_to_empty_string(self) -> None:
        """vLLM models have no GGUF file — file must default to '' when absent."""
        # Arrange
        raw: dict = {
            "engine": "vllm",
            "repo": "kaitchup/Qwen3.6-27B-autoround-nvfp4-linearattn-BF16",
            "port": 8093,
            "vram_gib": 15.0,
        }
        # Act
        spec = _parse_model_spec("qwen3-27b-nvfp4", raw)
        # Assert
        assert spec.file == ""
        assert spec.name == "qwen3-27b-nvfp4"
        assert spec.engine == "vllm"


# ---------------------------------------------------------------------------
# Friendly error messages
# ---------------------------------------------------------------------------


class TestFriendlyErrors:
    def test_unknown_model_name_raises_with_available_names(self, tmp_path: Path) -> None:
        """Accessing a nonexistent model key raises KeyError (or similar) with available names."""
        # Arrange
        toml_path = _write_toml(tmp_path, MINIMAL_TOML)
        catalog = load(toml_path)
        # Act / Assert — catalog.models["does-not-exist"] must raise; the error should
        # ideally mention the available names ("small-q4"). We test the raise at minimum.
        with pytest.raises((KeyError, ValueError, LookupError)):
            _ = catalog.models["does-not-exist"]

    def test_missing_repo_id_raises_descriptive_error(self, tmp_path: Path) -> None:
        """A model entry missing `repo` must raise a descriptive error on load."""
        # Arrange — intentionally omit required `repo` field
        bad_toml = """\
            [host]
            bind = "0.0.0.0"

            [models.broken]
            engine   = "llamacpp"
            file     = "broken.gguf"
            port     = 8091
            vram_gib = 4.0
        """
        toml_path = _write_toml(tmp_path, bad_toml)
        # Act / Assert
        with pytest.raises((TypeError, KeyError, ValueError)):
            load(toml_path)


# ---------------------------------------------------------------------------
# Remote engine specs (issue #36)
# ---------------------------------------------------------------------------


class TestRemoteEngineSpec:
    def test_remote_openai_protocol_loads(self, tmp_path: Path) -> None:
        """A remote spec with provider=fireworks and protocol=openai loads successfully."""
        # Arrange
        toml_path = _write_toml(
            tmp_path,
            '[host]\nbind = "0.0.0.0"\n\n[models.kimi]\nengine = "remote"\n'
            'provider = "fireworks"\nmodel_id = "accounts/fireworks/models/kimi"\nprotocol = "openai"\n',
        )
        # Act
        catalog = load(toml_path)
        # Assert
        spec = catalog.models["kimi"]
        assert spec.engine == "remote"
        assert spec.provider == "fireworks"
        assert spec.model_id == "accounts/fireworks/models/kimi"
        assert spec.protocol == "openai"

    def test_remote_anthropic_protocol_loads(self, tmp_path: Path) -> None:
        """A remote spec with provider=anthropic and protocol=anthropic loads successfully."""
        # Arrange
        toml_path = _write_toml(
            tmp_path,
            '[host]\nbind = "0.0.0.0"\n\n[models.claude]\nengine = "remote"\n'
            'provider = "anthropic"\nmodel_id = "claude-sonnet-4-6"\nprotocol = "anthropic"\n',
        )
        # Act
        catalog = load(toml_path)
        # Assert
        spec = catalog.models["claude"]
        assert spec.engine == "remote"
        assert spec.provider == "anthropic"
        assert spec.model_id == "claude-sonnet-4-6"
        assert spec.protocol == "anthropic"

    def test_remote_unknown_provider_raises(self) -> None:
        """engine='remote' with an unknown provider raises ValueError."""
        # Arrange
        raw = {
            "engine": "remote",
            "provider": "does-not-exist",
            "model_id": "some/model",
            "protocol": "openai",
        }
        # Act / Assert
        with pytest.raises(ValueError, match="unknown provider"):
            _parse_model_spec("test-model", raw)

    def test_remote_with_repo_raises(self) -> None:
        """engine='remote' mixed with local field 'repo' raises ValueError."""
        # Arrange
        raw = {
            "engine": "remote",
            "provider": "fireworks",
            "model_id": "some/model",
            "protocol": "openai",
            "repo": "Org/Model-GGUF",
        }
        # Act / Assert
        with pytest.raises(ValueError, match="local-engine fields"):
            _parse_model_spec("test-model", raw)

    def test_local_engine_with_provider_raises(self) -> None:
        """engine='llamacpp' mixed with remote field 'provider' raises ValueError."""
        # Arrange
        raw = {
            "engine": "llamacpp",
            "repo": "Org/Model-GGUF",
            "port": 8091,
            "vram_gib": 6.0,
            "provider": "fireworks",
        }
        # Act / Assert
        with pytest.raises(ValueError, match="remote-engine fields"):
            _parse_model_spec("test-model", raw)

    def test_machines_parses_correctly(self) -> None:
        """ModelSpec.machines parses a list of hostnames correctly."""
        # Arrange
        raw = {
            "engine": "llamacpp",
            "repo": "Org/Model-GGUF",
            "port": 8091,
            "vram_gib": 6.0,
            "machines": ["roxabitower"],
        }
        # Act
        spec = _parse_model_spec("test-model", raw)
        # Assert
        assert spec.machines == ["roxabitower"]

    def test_all_known_providers_are_valid(self) -> None:
        """Each key in PROVIDERS can be used as provider in a remote spec."""
        # implicit: no ValueError raised — purpose is to confirm all PROVIDERS pass validation.
        for provider_key in PROVIDERS:
            # Anthropic requires protocol='anthropic'; all others use 'openai'.
            protocol = "anthropic" if provider_key == "anthropic" else "openai"
            raw = {
                "engine": "remote",
                "provider": provider_key,
                "model_id": "some/model",
                "protocol": protocol,
            }
            _parse_model_spec(f"test-{provider_key}", raw)

    def test_unknown_provider_in_iteration_raises(self) -> None:
        """A provider key NOT in PROVIDERS is rejected by _parse_model_spec."""
        raw = {
            "engine": "remote",
            "provider": "not-a-real-provider",
            "model_id": "some/model",
            "protocol": "openai",
        }
        with pytest.raises(ValueError, match="unknown provider"):
            _parse_model_spec("test-unknown", raw)

    def test_machines_field_default_is_empty_list(self) -> None:
        """ModelSpec.machines defaults to [] when not specified (direct construction)."""
        # Arrange
        raw = {
            "engine": "remote",
            "provider": "fireworks",
            "model_id": "some/model",
            "protocol": "openai",
        }
        # Act
        spec = _parse_model_spec("test-model", raw)
        # Assert
        assert spec.machines == []

    def test_machines_absent_from_toml_yields_empty_list(self, tmp_path: Path) -> None:
        """ModelSpec.machines is [] when not present in TOML (load-path test)."""
        toml_path = _write_toml(
            tmp_path,
            '[host]\nbind = "0.0.0.0"\n\n[models.kimi]\nengine = "remote"\n'
            'provider = "fireworks"\nmodel_id = "accounts/fireworks/models/kimi"\nprotocol = "openai"\n',
        )
        catalog = load(toml_path)
        assert catalog.models["kimi"].machines == []

    def test_remote_with_extra_unknown_field_raises(self) -> None:
        """engine='remote' with an unknown field raises TypeError from ModelSpec dataclass."""
        raw = {
            "engine": "remote",
            "provider": "fireworks",
            "model_id": "some/model",
            "protocol": "openai",
            "junk": "x",
        }
        with pytest.raises(TypeError):
            _parse_model_spec("test-model", raw)

    def test_remote_missing_provider_raises(self) -> None:
        """engine='remote' without 'provider' raises ValueError matching 'missing required field'."""
        raw = {"engine": "remote", "model_id": "m", "protocol": "openai"}
        with pytest.raises(ValueError, match="missing required field 'provider'"):
            _parse_model_spec("test-model", raw)

    def test_remote_missing_model_id_raises(self) -> None:
        """engine='remote' without 'model_id' raises ValueError matching 'missing required field'."""
        raw = {"engine": "remote", "provider": "fireworks", "protocol": "openai"}
        with pytest.raises(ValueError, match="missing required field 'model_id'"):
            _parse_model_spec("test-model", raw)

    def test_remote_invalid_protocol_raises(self) -> None:
        """engine='remote' with unsupported protocol raises ValueError matching 'invalid protocol'."""
        raw = {"engine": "remote", "provider": "fireworks", "model_id": "m", "protocol": "grpc"}
        with pytest.raises(ValueError, match="invalid protocol"):
            _parse_model_spec("test-model", raw)

    def test_unknown_engine_raises(self) -> None:
        """An unrecognised engine value raises ValueError matching 'unknown engine'."""
        raw = {"engine": "deepspeed", "repo": "Org/M"}
        with pytest.raises(ValueError, match="unknown engine"):
            _parse_model_spec("test-model", raw)

    def test_missing_engine_raises(self) -> None:
        """A spec without 'engine' raises ValueError matching 'missing required field'."""
        raw = {"repo": "Org/M"}
        with pytest.raises(ValueError, match="missing required field 'engine'"):
            _parse_model_spec("test-model", raw)


# ---------------------------------------------------------------------------
# T1 (V1 slice) — HostSettings.port field
# ---------------------------------------------------------------------------


class TestHostSettingsPort:
    def test_host_settings_port_default(self) -> None:
        """HostSettings.port defaults to None when not provided (None means 'absent from TOML')."""
        # Arrange / Act
        hs = HostSettings()
        # Assert
        assert hs.port is None

    def test_host_settings_port_from_toml(self, tmp_path: Path) -> None:
        """[host].port in TOML is parsed onto HostSettings.port."""
        # Arrange — minimal catalog with [host].port set
        cfg = tmp_path / "llmcli.toml"
        cfg.write_text(
            "[host]\n"
            'bind = "0.0.0.0"\n'
            'public_base_url = "http://x.lan"\n'
            'api_key_env = "K"\n'
            "port = 19999\n"
        )
        # Act
        catalog = load(cfg)
        # Assert
        assert catalog.host.port == 19999
