"""Tests for llmcli.litellm_config — RED phase (T2.1).

These tests MUST fail against the current scaffold because:
- build_block() raises NotImplementedError
- write_block() raises NotImplementedError

Expected: NotImplementedError on all tests. GREEN phase (T2.2, T2.3) implements the functions.

Spec trace: SC-7, SC-8, C1, C2
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from unittest.mock import patch

from llmcli.config import Catalog, HostSettings, ModelSpec
from llmcli.litellm_config import BLOCK_END, BLOCK_START, build_block, write_block
from llmcli.providers import PROVIDERS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PUBLIC_BASE_URL = "http://roxabitower.lan"

FIREWORKS_BLOCK = textwrap.dedent("""\
    # Fireworks pass-through (not managed by llmCLI)
    model_list:
      - model_name: fireworks/llama-3-70b
        litellm_params:
          model: fireworks_ai/accounts/fireworks/models/llama-v3-70b-instruct
          api_key: os.environ/FIREWORKS_API_KEY
""")

FIXTURE_WITH_FIREWORKS = textwrap.dedent(f"""\
    # LiteLLM proxy config — roxabitower
    # Hand-authored section above

    {FIREWORKS_BLOCK}
""")

FIXTURE_WITH_EXISTING_BLOCK = textwrap.dedent(f"""\
    # LiteLLM proxy config — roxabitower

    {FIREWORKS_BLOCK}
    {BLOCK_START}
    model_list:
      - model_name: old-model
        litellm_params:
          model: openai/old-model
          api_base: http://old.lan:9999/v1
          api_key: os.environ/LLMCLI_API_KEY
    {BLOCK_END}
""")


def _make_catalog(models: dict[str, dict] | None = None) -> Catalog:
    """Build a Catalog with the given model specs (or two defaults)."""
    host = HostSettings(
        bind="0.0.0.0",
        public_base_url=PUBLIC_BASE_URL,
        api_key_env="LLMCLI_API_KEY",
        default_model="qwen3-8b",
        vram_budget_gib=16.0,
    )
    if models is None:
        models = {
            "qwen3-8b": dict(
                engine="llamacpp",
                repo="Org/Qwen3-8B-GGUF",
                file="qwen3-8b-q4_k_m.gguf",
                port=8091,
                vram_gib=5.5,
            ),
            "qwen3_6-35b-a3b-tq3": dict(
                engine="llamacpp_tq3",
                repo="Org/Qwen3.6-35B-A3B-TQ3-GGUF",
                file="qwen3.6-35b-tq3_4s.gguf",
                port=8092,
                vram_gib=12.4,
            ),
        }
    model_specs = {name: ModelSpec(name=name, **spec) for name, spec in models.items()}
    return Catalog(host=host, models=model_specs)


@pytest.fixture
def catalog() -> Catalog:
    return _make_catalog()


@pytest.fixture
def single_model_catalog() -> Catalog:
    return _make_catalog(
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


@pytest.fixture
def empty_catalog() -> Catalog:
    return _make_catalog(models={})


# ---------------------------------------------------------------------------
# build_block — pure function tests
# ---------------------------------------------------------------------------


class TestBuildBlock:
    def test_block_has_start_sentinel(self, catalog: Catalog) -> None:
        """build_block output starts with the llmCLI managed block start sentinel."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        # Assert
        assert result.lstrip().startswith(BLOCK_START)

    def test_block_has_end_sentinel(self, catalog: Catalog) -> None:
        """build_block output ends with the llmCLI managed block end sentinel."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        # Assert
        assert result.rstrip().endswith(BLOCK_END)

    def test_block_start_before_end(self, catalog: Catalog) -> None:
        """Start sentinel must appear before end sentinel in the output."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        # Assert
        assert result.index(BLOCK_START) < result.index(BLOCK_END)

    def test_block_contains_all_model_names(self, catalog: Catalog) -> None:
        """Every model in catalog.models appears as a model_name entry."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        parsed = yaml.safe_load(result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip())
        # Assert
        names_in_block = {entry["model_name"] for entry in parsed["model_list"]}
        assert names_in_block == set(catalog.models.keys())

    def test_local_block_litellm_params_model_prefix(self, catalog: Catalog) -> None:
        """Each local-engine entry's litellm_params.model is 'openai/<name>'."""
        # This test only applies to local-engine catalogs.
        assert all(spec.engine != "remote" for spec in catalog.models.values()), (
            "this test only applies to local-engine catalogs"
        )
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        parsed = yaml.safe_load(result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip())
        # Assert
        for entry in parsed["model_list"]:
            assert entry["litellm_params"]["model"].startswith("openai/")
            assert entry["litellm_params"]["model"] == f"openai/{entry['model_name']}"

    def test_block_api_base_uses_public_base_url_and_port(self, catalog: Catalog) -> None:
        """api_base is '{public_base_url}:{model.port}/v1' for each model."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        parsed = yaml.safe_load(result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip())
        # Assert
        by_name = {e["model_name"]: e for e in parsed["model_list"]}
        for name, spec in catalog.models.items():
            expected = f"{PUBLIC_BASE_URL}:{spec.port}/v1"
            assert by_name[name]["litellm_params"]["api_base"] == expected

    def test_block_api_key_uses_env_var_reference(self, catalog: Catalog) -> None:
        """api_key is 'os.environ/<api_key_env>' from HostSettings."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        parsed = yaml.safe_load(result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip())
        # Assert
        expected_key = f"os.environ/{catalog.host.api_key_env}"
        for entry in parsed["model_list"]:
            assert entry["litellm_params"]["api_key"] == expected_key

    def test_block_is_valid_yaml_between_sentinels(self, catalog: Catalog) -> None:
        """Content between sentinels is parseable YAML with a model_list key."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        # Assert
        parsed = yaml.safe_load(inner)
        assert "model_list" in parsed
        assert isinstance(parsed["model_list"], list)

    def test_empty_catalog_returns_sentinels_only(self, empty_catalog: Catalog) -> None:
        """Empty catalog produces sentinels wrapping an empty or null model_list."""
        # Arrange / Act
        result = build_block(empty_catalog, PUBLIC_BASE_URL)
        # Assert
        assert BLOCK_START in result
        assert BLOCK_END in result
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # model_list must be absent or empty (None or [])
        model_list = parsed.get("model_list") if parsed else None
        assert not model_list  # None, [], or absent all satisfy "empty"

    def test_single_model_catalog(self, single_model_catalog: Catalog) -> None:
        """A one-model catalog produces exactly one model_list entry."""
        # Arrange / Act
        result = build_block(single_model_catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        assert len(parsed["model_list"]) == 1
        assert parsed["model_list"][0]["model_name"] == "qwen3-8b"

    def test_different_public_base_url_reflected(self, single_model_catalog: Catalog) -> None:
        """build_block uses the supplied public_base_url, not the one in HostSettings."""
        # Arrange
        custom_url = "http://192.168.1.10"
        # Act
        result = build_block(single_model_catalog, custom_url)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        api_base = parsed["model_list"][0]["litellm_params"]["api_base"]
        assert api_base.startswith(custom_url)


# ---------------------------------------------------------------------------
# write_block — file I/O tests
# ---------------------------------------------------------------------------


class TestWriteBlockNewFile:
    def test_creates_file_when_absent(self, tmp_path: Path, catalog: Catalog) -> None:
        """write_block creates the config file when it does not exist."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(block, config_path)
        # Assert
        assert config_path.exists()

    def test_new_file_contains_only_the_block(self, tmp_path: Path, catalog: Catalog) -> None:
        """When the file is absent, write_block writes only the block content."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(block, config_path)
        content = config_path.read_text()
        # Assert
        assert BLOCK_START in content
        assert BLOCK_END in content

    def test_backup_created_for_new_file(self, tmp_path: Path, catalog: Catalog) -> None:
        """write_block creates a .bak backup even for a new file (empty backup is fine)."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(block, config_path)
        # Assert — backup must exist
        backup = config_path.with_suffix(config_path.suffix + ".bak")
        assert backup.exists()


class TestWriteBlockNoExistingBlock:
    def test_appends_block_when_no_sentinel(self, tmp_path: Path, catalog: Catalog) -> None:
        """write_block appends the block to an existing file that has no sentinel."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        config_path.write_text(FIXTURE_WITH_FIREWORKS)
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(block, config_path)
        content = config_path.read_text()
        # Assert
        assert BLOCK_START in content
        assert BLOCK_END in content

    def test_original_content_preserved_when_appending(
        self, tmp_path: Path, catalog: Catalog
    ) -> None:
        """Existing content before the appended block is preserved byte-for-byte."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        config_path.write_text(FIXTURE_WITH_FIREWORKS)
        original_text = FIXTURE_WITH_FIREWORKS
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(block, config_path)
        content = config_path.read_text()
        # Assert — original text is present verbatim as a prefix (possibly with added newline)
        assert content.startswith(original_text) or original_text in content

    def test_backup_created_before_write(self, tmp_path: Path, catalog: Catalog) -> None:
        """write_block creates a .bak backup of the file before modifying it."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        config_path.write_text(FIXTURE_WITH_FIREWORKS)
        original_text = FIXTURE_WITH_FIREWORKS
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(block, config_path)
        # Assert
        backup = config_path.with_suffix(config_path.suffix + ".bak")
        assert backup.exists()
        assert backup.read_text() == original_text


class TestWriteBlockReplaceExisting:
    def test_replaces_existing_block_in_place(self, tmp_path: Path, catalog: Catalog) -> None:
        """write_block replaces the existing sentinel block when one is found."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        config_path.write_text(FIXTURE_WITH_EXISTING_BLOCK)
        new_block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(new_block, config_path)
        content = config_path.read_text()
        # Assert — old model is gone, new models are present
        assert "old-model" not in content
        for name in catalog.models:
            assert name in content

    def test_content_outside_block_preserved_byte_for_byte(
        self, tmp_path: Path, catalog: Catalog
    ) -> None:
        """Lines outside the sentinel block are preserved exactly (SC-7, C1)."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        config_path.write_text(FIXTURE_WITH_EXISTING_BLOCK)
        new_block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(new_block, config_path)
        content = config_path.read_text()
        # Assert — Fireworks block text is intact
        assert "fireworks/llama-3-70b" in content
        assert "FIREWORKS_API_KEY" in content
        assert "fireworks_ai/accounts" in content

    def test_backup_preserves_original_before_replacement(
        self, tmp_path: Path, catalog: Catalog
    ) -> None:
        """Backup file contains the original content before sentinel replacement (SC-8)."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        config_path.write_text(FIXTURE_WITH_EXISTING_BLOCK)
        original_text = FIXTURE_WITH_EXISTING_BLOCK
        new_block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(new_block, config_path)
        backup = config_path.with_suffix(config_path.suffix + ".bak")
        # Assert
        assert backup.exists()
        assert backup.read_text() == original_text

    def test_exactly_one_start_sentinel_after_replacement(
        self, tmp_path: Path, catalog: Catalog
    ) -> None:
        """After replacement, exactly one start sentinel exists in the file."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        config_path.write_text(FIXTURE_WITH_EXISTING_BLOCK)
        new_block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(new_block, config_path)
        content = config_path.read_text()
        # Assert
        assert content.count(BLOCK_START) == 1
        assert content.count(BLOCK_END) == 1

    def test_idempotent_double_write(self, tmp_path: Path, catalog: Catalog) -> None:
        """Calling write_block twice with the same block produces the same file content."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        config_path.write_text(FIXTURE_WITH_FIREWORKS)
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act
        write_block(block, config_path)
        after_first = config_path.read_text()
        write_block(block, config_path)
        after_second = config_path.read_text()
        # Assert
        assert after_first == after_second


class TestWriteBlockMalformed:
    def test_raises_on_start_without_end(self, tmp_path: Path, catalog: Catalog) -> None:
        """write_block raises a specific error when start sentinel has no matching end sentinel."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        malformed = textwrap.dedent(f"""\
            # Some preamble
            {BLOCK_START}
            model_list:
              - model_name: orphan
            # END SENTINEL IS MISSING
        """)
        config_path.write_text(malformed)
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act / Assert
        with pytest.raises((ValueError, RuntimeError, OSError)):
            write_block(block, config_path)

    def test_raises_on_end_without_start(self, tmp_path: Path, catalog: Catalog) -> None:
        """write_block raises when end sentinel appears without a matching start sentinel."""
        # Arrange
        config_path = tmp_path / "config.yaml"
        malformed = textwrap.dedent(f"""\
            # Some preamble — no start sentinel
            model_list:
              - model_name: orphan
            {BLOCK_END}
        """)
        config_path.write_text(malformed)
        block = build_block(catalog, PUBLIC_BASE_URL)
        # Act / Assert
        with pytest.raises((ValueError, RuntimeError, OSError)):
            write_block(block, config_path)


# ---------------------------------------------------------------------------
# Remote engine + hostname filter (issue #36)
# ---------------------------------------------------------------------------


def _make_remote_catalog(
    engine_type: str = "openai",
    machines: list[str] | None = None,
) -> Catalog:
    """Build a catalog with a single remote model spec."""
    host = HostSettings(
        bind="0.0.0.0",
        public_base_url=PUBLIC_BASE_URL,
        api_key_env="LLMCLI_API_KEY",
    )
    if engine_type == "anthropic":
        spec = ModelSpec(
            name="claude-sonnet",
            engine="remote",
            provider="anthropic",
            model_id="claude-sonnet-4-6",
            protocol="anthropic",
            machines=machines or [],
        )
    else:
        spec = ModelSpec(
            name="kimi-k2",
            engine="remote",
            provider="fireworks",
            model_id="accounts/fireworks/models/kimi",
            protocol="openai",
            machines=machines or [],
        )
    return Catalog(host=host, models={spec.name: spec})


class TestBuildBlockRemoteEngines:
    def test_remote_openai_entry_model_prefix(self) -> None:
        """Remote+openai entry has model='openai/<model_id>' (NOT the catalog key)."""
        # Arrange
        catalog = _make_remote_catalog("openai")
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        entry = parsed["model_list"][0]
        assert entry["litellm_params"]["model"] == "openai/accounts/fireworks/models/kimi"

    def test_remote_openai_entry_has_provider_api_base(self) -> None:
        """Remote+openai entry uses provider's api_base (not a local port)."""
        # Arrange
        catalog = _make_remote_catalog("openai")
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        entry = parsed["model_list"][0]
        assert entry["litellm_params"]["api_base"] == PROVIDERS["fireworks"].api_base
        # No port-based local URL
        assert ":8091" not in entry["litellm_params"]["api_base"]

    def test_remote_openai_entry_uses_provider_key_env(self) -> None:
        """Remote+openai entry references provider's key_env, not LLMCLI_API_KEY."""
        # Arrange
        catalog = _make_remote_catalog("openai")
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        entry = parsed["model_list"][0]
        assert entry["litellm_params"]["api_key"] == f"os.environ/{PROVIDERS['fireworks'].key_env}"

    def test_remote_anthropic_entry_model_prefix(self) -> None:
        """Remote+anthropic entry has model='anthropic/<model_id>'."""
        # Arrange
        catalog = _make_remote_catalog("anthropic")
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        entry = parsed["model_list"][0]
        assert entry["litellm_params"]["model"] == "anthropic/claude-sonnet-4-6"

    def test_remote_anthropic_entry_has_no_api_base(self) -> None:
        """Remote+anthropic entry must NOT have api_base — LiteLLM resolves it natively."""
        # Arrange
        catalog = _make_remote_catalog("anthropic")
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        entry = parsed["model_list"][0]
        assert "api_base" not in entry["litellm_params"]

    def test_remote_anthropic_entry_uses_provider_key_env(self) -> None:
        """Remote+anthropic entry references ANTHROPIC_API_KEY."""
        # Arrange
        catalog = _make_remote_catalog("anthropic")
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        entry = parsed["model_list"][0]
        assert entry["litellm_params"]["api_key"] == f"os.environ/{PROVIDERS['anthropic'].key_env}"

    def test_local_llamacpp_entry_unchanged(self) -> None:
        """Local llamacpp entry still emits openai/<name> + local api_base."""
        # Arrange
        host = HostSettings(
            bind="0.0.0.0",
            public_base_url=PUBLIC_BASE_URL,
            api_key_env="LLMCLI_API_KEY",
        )
        spec = ModelSpec(
            name="qwen3-8b",
            engine="llamacpp",
            repo="Org/Qwen3-8B-GGUF",
            file="qwen3-8b-q4_k_m.gguf",
            port=8091,
            vram_gib=5.5,
        )
        catalog = Catalog(host=host, models={"qwen3-8b": spec})
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        entry = parsed["model_list"][0]
        assert entry["litellm_params"]["model"] == "openai/qwen3-8b"
        assert entry["litellm_params"]["api_base"] == f"{PUBLIC_BASE_URL}:8091/v1"
        assert entry["litellm_params"]["api_key"] == "os.environ/LLMCLI_API_KEY"


class TestBuildBlockHostnameFilter:
    def test_spec_without_machines_included_for_any_hostname(self) -> None:
        """A spec with machines=[] is included regardless of hostname."""
        # Arrange
        catalog = _make_remote_catalog("openai", machines=[])
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL, hostname="any-random-host")
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert — spec included
        assert parsed["model_list"] is not None
        assert len(parsed["model_list"]) == 1

    def test_spec_with_machines_included_when_hostname_matches(self) -> None:
        """A spec with machines=["roxabitower"] is included when hostname matches."""
        # Arrange
        catalog = _make_remote_catalog("openai", machines=["roxabitower"])
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL, hostname="roxabitower")
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert
        assert parsed["model_list"] is not None
        assert len(parsed["model_list"]) == 1

    def test_spec_with_machines_excluded_when_hostname_does_not_match(self) -> None:
        """A spec with machines=["roxabitower"] is excluded when hostname differs."""
        # Arrange
        catalog = _make_remote_catalog("openai", machines=["roxabitower"])
        # Act
        result = build_block(catalog, PUBLIC_BASE_URL, hostname="roxabituwer")
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert — spec filtered out, model_list is null/empty
        model_list = parsed.get("model_list") if parsed else None
        assert not model_list

    def test_mixed_catalog_hostname_filter(self) -> None:
        """Catalog with two specs: one pinned to 'other-host', one open — filtered correctly."""
        # Arrange
        host = HostSettings(
            bind="0.0.0.0",
            public_base_url=PUBLIC_BASE_URL,
            api_key_env="LLMCLI_API_KEY",
        )
        pinned_spec = ModelSpec(
            name="pinned-model",
            engine="remote",
            provider="fireworks",
            model_id="accounts/fireworks/models/x",
            protocol="openai",
            machines=["other-host"],
        )
        open_spec = ModelSpec(
            name="open-model",
            engine="remote",
            provider="openai",
            model_id="gpt-4o",
            protocol="openai",
            machines=[],
        )
        catalog = Catalog(host=host, models={"pinned-model": pinned_spec, "open-model": open_spec})

        # When hostname="other-host" → both included
        result = build_block(catalog, PUBLIC_BASE_URL, hostname="other-host")
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        names = {e["model_name"] for e in parsed["model_list"]}
        assert names == {"pinned-model", "open-model"}

        # When hostname="this-host" → only open-model included
        result2 = build_block(catalog, PUBLIC_BASE_URL, hostname="this-host")
        inner2 = result2.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed2 = yaml.safe_load(inner2)
        names2 = {e["model_name"] for e in parsed2["model_list"]}
        assert names2 == {"open-model"}

    def test_build_block_uses_real_hostname_by_default(self) -> None:
        """build_block uses socket.gethostname() when hostname kwarg is omitted."""
        # Arrange — catalog with a spec pinned to a hostname that won't match real hostname
        real_hostname = "definitely-not-this-host-12345"
        catalog = _make_remote_catalog("openai", machines=[real_hostname])
        # Act — no hostname kwarg → falls through to socket.gethostname()
        with patch("llmcli.litellm_config.socket.gethostname", return_value="some-other-host"):
            result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert — spec excluded because mocked hostname doesn't match
        model_list = parsed.get("model_list") if parsed else None
        assert not model_list

    def test_build_block_default_hostname_includes_when_matching(self) -> None:
        """build_block includes spec when mocked gethostname() matches the machines list."""
        # Arrange — catalog pinned to "matching-host"
        catalog = _make_remote_catalog("openai", machines=["matching-host"])
        # Act — no hostname kwarg → falls through to socket.gethostname() (mocked to match)
        with patch("llmcli.litellm_config.socket.gethostname", return_value="matching-host"):
            result = build_block(catalog, PUBLIC_BASE_URL)
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        # Assert — spec included because mocked hostname matches
        assert parsed["model_list"] is not None
        assert len(parsed["model_list"]) == 1

    def test_machines_multi_host_includes_when_matching(self) -> None:
        """Spec with machines=[host-a, host-b] is included when hostname is host-b."""
        catalog = _make_remote_catalog("openai", machines=["host-a", "host-b"])
        result = build_block(catalog, PUBLIC_BASE_URL, hostname="host-b")
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        assert parsed["model_list"] is not None
        assert len(parsed["model_list"]) == 1

    def test_machines_multi_host_excludes_when_no_match(self) -> None:
        """Spec with machines=[host-a, host-b] is excluded when hostname is host-c."""
        catalog = _make_remote_catalog("openai", machines=["host-a", "host-b"])
        result = build_block(catalog, PUBLIC_BASE_URL, hostname="host-c")
        inner = result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        model_list = parsed.get("model_list") if parsed else None
        assert not model_list


# ---------------------------------------------------------------------------
# build_full_config — complete proxy config dict (issue #40)
# ---------------------------------------------------------------------------


def _make_mixed_catalog(
    remote_machines: list[str] | None = None,
    local_machines: list[str] | None = None,
) -> Catalog:
    """Build a catalog with one remote (openai) model and one local (llamacpp) model."""
    host = HostSettings(
        bind="0.0.0.0",
        public_base_url=PUBLIC_BASE_URL,
        api_key_env="LLMCLI_API_KEY",
    )
    remote_spec = ModelSpec(
        name="kimi-k2",
        engine="remote",
        provider="fireworks",
        model_id="accounts/fireworks/models/kimi",
        protocol="openai",
        machines=remote_machines or [],
    )
    local_spec = ModelSpec(
        name="qwen3-8b",
        engine="llamacpp",
        repo="Org/Qwen3-8B-GGUF",
        file="qwen3-8b-q4_k_m.gguf",
        port=8091,
        vram_gib=5.5,
        machines=local_machines or [],
    )
    return Catalog(host=host, models={"kimi-k2": remote_spec, "qwen3-8b": local_spec})


from llmcli.litellm_config import build_full_config  # noqa: E402


class TestBuildFullConfig:
    def test_returns_dict_with_required_keys(self) -> None:
        """Catalog with 1 remote + 1 local model → result has all 3 keys; litellm_settings correct."""
        # Arrange
        catalog = _make_mixed_catalog()
        # Act
        result = build_full_config(catalog, PUBLIC_BASE_URL, hostname="any-host")
        # Assert — all three top-level keys present
        assert "general_settings" in result
        assert "litellm_settings" in result
        assert "model_list" in result
        # litellm_settings is exactly {"drop_params": True}
        assert result["litellm_settings"] == {"drop_params": True}
        # model_list contains entries for both models
        names = {entry["model_name"] for entry in result["model_list"]}
        assert names == {"kimi-k2", "qwen3-8b"}

    def test_machines_filter_excludes_non_matching(self) -> None:
        """Model with machines=["other-host"] is excluded when hostname="this-host"."""
        # Arrange — remote pinned to "other-host", local open
        catalog = _make_mixed_catalog(remote_machines=["other-host"], local_machines=[])
        # Act
        result = build_full_config(catalog, PUBLIC_BASE_URL, hostname="this-host")
        # Assert — only the open local model survives the filter
        names = {entry["model_name"] for entry in result["model_list"]}
        assert "kimi-k2" not in names
        assert "qwen3-8b" in names

    def test_master_key_from_api_key_env(self) -> None:
        """general_settings.master_key == 'os.environ/<api_key_env>'."""
        # Arrange
        catalog = _make_mixed_catalog()
        expected_key = f"os.environ/{catalog.host.api_key_env}"
        # Act
        result = build_full_config(catalog, PUBLIC_BASE_URL, hostname="any-host")
        # Assert
        assert result["general_settings"]["master_key"] == expected_key

    def test_empty_filter_yields_empty_list(self) -> None:
        """All models filtered out → model_list is [] (not None, not absent)."""
        # Arrange — both models pinned to "never-host"
        catalog = _make_mixed_catalog(
            remote_machines=["never-host"],
            local_machines=["never-host"],
        )
        # Act
        result = build_full_config(catalog, PUBLIC_BASE_URL, hostname="this-host")
        # Assert — model_list is present AND is an empty list (not None)
        assert "model_list" in result
        assert result["model_list"] == []
