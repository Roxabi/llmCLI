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

from llmcli.config import Catalog, HostSettings, ModelSpec
from llmcli.litellm_config import BLOCK_END, BLOCK_START, build_block, write_block


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
    model_specs = {
        name: ModelSpec(name=name, **spec) for name, spec in models.items()
    }
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
        parsed = yaml.safe_load(
            result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        )
        # Assert
        names_in_block = {entry["model_name"] for entry in parsed["model_list"]}
        assert names_in_block == set(catalog.models.keys())

    def test_block_litellm_params_model_prefix(self, catalog: Catalog) -> None:
        """Each entry's litellm_params.model is prefixed with 'openai/'."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        parsed = yaml.safe_load(
            result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        )
        # Assert
        for entry in parsed["model_list"]:
            assert entry["litellm_params"]["model"].startswith("openai/")
            assert entry["litellm_params"]["model"] == f"openai/{entry['model_name']}"

    def test_block_api_base_uses_public_base_url_and_port(self, catalog: Catalog) -> None:
        """api_base is '{public_base_url}:{model.port}/v1' for each model."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        parsed = yaml.safe_load(
            result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        )
        # Assert
        by_name = {e["model_name"]: e for e in parsed["model_list"]}
        for name, spec in catalog.models.items():
            expected = f"{PUBLIC_BASE_URL}:{spec.port}/v1"
            assert by_name[name]["litellm_params"]["api_base"] == expected

    def test_block_api_key_uses_env_var_reference(self, catalog: Catalog) -> None:
        """api_key is 'os.environ/<api_key_env>' from HostSettings."""
        # Arrange / Act
        result = build_block(catalog, PUBLIC_BASE_URL)
        parsed = yaml.safe_load(
            result.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        )
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
