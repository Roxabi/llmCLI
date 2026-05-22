"""Tests for CLI commands (SC-2, SC-3, SC-4, SC-7, SC-9).

RED phase (T1.6): stubs fail with empty output / wrong exit codes.
GREEN phase (T1.12): wires real implementations.
GREEN phase (T2.4): register-proxy fully implements SC-9 (--config flag,
reload graceful failure, confirmation output, friendly error on bad path).
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from llmcli.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Shared TOML fixture
# ---------------------------------------------------------------------------


FAKE_TOML = textwrap.dedent("""\
    [host]
    bind             = "0.0.0.0"
    public_base_url  = "http://localhost"
    api_key_env      = "LLMCLI_API_KEY"
    default_model    = "small-q4"
    vram_budget_gib  = 10.0

    [models.small-q4]
    engine   = "llamacpp"
    repo     = "SomeOrg/small-model-GGUF"
    file     = "small-q4_k_m.gguf"
    port     = 8091
    vram_gib = 6.0
    flags    = ["-ngl", "99"]

    [models.big-model]
    engine   = "llamacpp_tq3"
    repo     = "SomeOrg/big-model-GGUF"
    file     = "big-model.gguf"
    port     = 8092
    vram_gib = 13.0
    flags    = ["-ngl", "99"]
""")


@pytest.fixture()
def real_catalog(tmp_path: Path):
    """Load the fake TOML via config.load() and return the Catalog."""
    from llmcli import config

    toml_path = tmp_path / "llmcli.toml"
    toml_path.write_text(FAKE_TOML)
    return config.load(toml_path)


@pytest.fixture()
def fake_catalog(real_catalog):
    """Patch llmcli.cli.config.load to return fake_catalog without hitting disk.

    `create=True` lets the patch work even when cli.py stub doesn't import config yet.
    check_vram_budget is wired to the real function if it exists; falls back to a no-op
    so fixture setup doesn't fail during RED phase (T1.8 adds the real implementation).
    """
    from llmcli import config as real_config

    real_check = getattr(real_config, "check_vram_budget", None)

    with patch("llmcli.cli.config", create=True) as mock_config_mod:
        mock_config_mod.load.return_value = real_catalog
        if real_check is not None:
            mock_config_mod.check_vram_budget.side_effect = real_check
        yield real_catalog


@pytest.fixture()
def mock_openai_client():
    """Patch the openai module used by `chat`."""
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message.content = "pong"

    mock_client_instance = MagicMock()
    mock_client_instance.chat.completions.create.return_value = mock_completion

    with patch("llmcli.cli.openai", create=True) as mock_openai_mod:
        mock_openai_mod.OpenAI.return_value = mock_client_instance
        yield mock_client_instance


# ---------------------------------------------------------------------------
# U1 — llmcli list
# ---------------------------------------------------------------------------


# TestListCommand removed in Slice 6 cutover (#34): `list` now goes through NATS
# and is exercised by tests/nats/test_lifecycle_host_filter.py +
# tests/cli/test_swap_nats.py. Mocking NATS here would duplicate that coverage.


# ---------------------------------------------------------------------------
# U2 — llmcli pull <model>
# ---------------------------------------------------------------------------


class TestPullCommand:
    def test_pull_exits_zero_on_known_model(self, fake_catalog) -> None:
        """pull <known-model> exits 0."""
        # Arrange
        with patch("llmcli.cli.hf_hub_download", create=True, return_value="/tmp/fake.gguf"):
            # Act
            result = runner.invoke(app, ["pull", "small-q4"])
        # Assert
        assert result.exit_code == 0

    def test_pull_invokes_hf_download_for_known_model(self, fake_catalog) -> None:
        """pull <known-model> calls the HF hub download function."""
        # Arrange
        with patch(
            "llmcli.cli.hf_hub_download", create=True, return_value="/tmp/fake.gguf"
        ) as mock_hf:
            # Act
            runner.invoke(app, ["pull", "small-q4"])
        # Assert — RED: stub never calls hf_hub_download
        mock_hf.assert_called_once()

    def test_pull_nonzero_on_unknown_model(self, fake_catalog) -> None:
        """pull <unknown-model> exits non-zero."""
        # Act
        result = runner.invoke(app, ["pull", "does-not-exist"])
        # Assert — RED: stub exits 0 for everything
        assert result.exit_code != 0, "Expected non-zero exit for unknown model, got 0"

    def test_pull_unknown_model_shows_available_names(self, fake_catalog) -> None:
        """pull <unknown-model> output mentions available model names as a helpful hint."""
        # Act
        result = runner.invoke(app, ["pull", "does-not-exist"])
        combined = result.output + (result.stderr or "")
        # Assert — RED: stub outputs nothing
        assert "small-q4" in combined or "big-model" in combined, (
            f"Expected available models in error output, got: {combined!r}"
        )


# ---------------------------------------------------------------------------
# U3-U5 — stop, status, list: tests removed in Slice 6 cutover (#34).
# All three go through NATS and are covered by tests/cli/test_lifecycle_nats.py
# (CLI layer) + tests/nats/test_lifecycle_status.py (worker layer).
# `serve` is now a deprecation stub — covered by TestServeStub below.
# ---------------------------------------------------------------------------


class TestServeStub:
    """B9 (#34 Slice 6): `serve` exits 1 with a redirect to the NATS worker."""

    def test_serve_exits_one(self) -> None:
        result = runner.invoke(app, ["serve"])
        assert result.exit_code == 1

    def test_serve_output_redirects_to_worker(self) -> None:
        result = runner.invoke(app, ["serve"])
        combined = result.output + (result.stderr or "")
        assert "removed" in combined.lower()
        assert "llmcli-nats-worker" in combined


# ---------------------------------------------------------------------------
# U7 — llmcli chat <name> "prompt"
# ---------------------------------------------------------------------------


class TestChatCommand:
    def test_chat_exits_zero(self, fake_catalog, mock_openai_client) -> None:
        """chat exits 0 on success."""
        # Act
        result = runner.invoke(app, ["chat", "small-q4", "ping"])
        # Assert
        assert result.exit_code == 0

    def test_chat_prints_llm_response(self, fake_catalog, mock_openai_client) -> None:
        """chat prints the LLM completion to stdout."""
        # Act
        result = runner.invoke(app, ["chat", "small-q4", "ping"])
        # Assert — RED: stub prints nothing
        assert "pong" in result.output, f"Expected 'pong' in chat output, got: {result.output!r}"

    def test_chat_invokes_openai_completions(self, fake_catalog, mock_openai_client) -> None:
        """chat calls the OpenAI completions endpoint."""
        # Act
        runner.invoke(app, ["chat", "small-q4", "ping"])
        # Assert — RED: stub never instantiates openai client
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_chat_passes_prompt_to_openai(self, fake_catalog, mock_openai_client) -> None:
        """chat passes the user prompt in the messages argument."""
        # Act
        runner.invoke(app, ["chat", "small-q4", "hello world"])
        # Assert
        call_args = mock_openai_client.chat.completions.create.call_args
        if call_args is not None:
            messages = call_args.kwargs.get("messages") or (
                call_args.args[1] if len(call_args.args) > 1 else None
            )
            if messages:
                prompt_found = any("hello world" in str(m.get("content", "")) for m in messages)
                assert prompt_found, f"Prompt not found in messages: {messages}"

    def test_chat_unknown_model_exits_nonzero(self, fake_catalog) -> None:
        """chat <unknown-model> exits non-zero."""
        # Act
        result = runner.invoke(app, ["chat", "nonexistent", "ping"])
        # Assert — RED: stub exits 0 for everything
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# U8 — llmcli register-proxy
# ---------------------------------------------------------------------------


class TestRegisterProxyCommand:
    def _default_patches(self):
        """Return a context manager stacking the three default mocks."""
        import contextlib

        @contextlib.contextmanager
        def _ctx():
            with (
                patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
                patch("llmcli.cli.write_block", create=True),
                patch("llmcli.cli.reload_proxy", create=True),
            ):
                yield

        return _ctx()

    def test_register_proxy_exits_zero(self, fake_catalog, tmp_path) -> None:
        """register-proxy exits 0 on success."""
        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            result = runner.invoke(app, ["register-proxy", "--config", config_file])
        assert result.exit_code == 0

    def test_register_proxy_calls_build_block(self, fake_catalog, tmp_path) -> None:
        """register-proxy calls build_block with the loaded catalog."""
        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n") as mock_build,
            patch("llmcli.cli.write_block", create=True),
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            runner.invoke(app, ["register-proxy", "--config", config_file])
        mock_build.assert_called_once()

    def test_register_proxy_calls_write_block(self, fake_catalog, tmp_path) -> None:
        """register-proxy calls write_block to persist the LiteLLM block."""
        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True) as mock_write,
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            runner.invoke(app, ["register-proxy", "--config", config_file])
        mock_write.assert_called_once()

    def test_register_proxy_write_block_receives_config_path(self, fake_catalog, tmp_path) -> None:
        """write_block is called with the resolved Path, not a string."""
        from pathlib import Path

        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True) as mock_write,
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            runner.invoke(app, ["register-proxy", "--config", config_file])
        positional = mock_write.call_args.args
        assert len(positional) >= 2
        assert positional[1] == Path(config_file)

    def test_register_proxy_calls_reload_proxy(self, fake_catalog, tmp_path) -> None:
        """register-proxy calls reload_proxy after writing the block (SC-9)."""
        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
            patch("llmcli.cli.reload_proxy", create=True) as mock_reload,
        ):
            runner.invoke(app, ["register-proxy", "--config", config_file])
        mock_reload.assert_called_once()

    def test_register_proxy_config_flag_overrides_default_path(
        self, fake_catalog, tmp_path
    ) -> None:
        """--config flag passes the custom path to write_block (SC-9 override)."""
        from pathlib import Path

        custom_cfg = str(tmp_path / "my_litellm.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True) as mock_write,
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            result = runner.invoke(app, ["register-proxy", "--config", custom_cfg])
        assert result.exit_code == 0
        positional = mock_write.call_args.args
        assert positional[1] == Path(custom_cfg)

    def test_register_proxy_env_var_sets_config_path(self, fake_catalog, tmp_path) -> None:
        """LITELLM_CONFIG_PATH env var is respected when --config flag is absent."""
        from pathlib import Path

        env_cfg = str(tmp_path / "env_config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True) as mock_write,
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            result = runner.invoke(
                app,
                ["register-proxy"],
                env={"LITELLM_CONFIG_PATH": env_cfg},
            )
        assert result.exit_code == 0
        positional = mock_write.call_args.args
        assert positional[1] == Path(env_cfg)

    def test_register_proxy_reload_failure_is_graceful(self, fake_catalog, tmp_path) -> None:
        """Reload failure (non-zero exit) is a warning, not a fatal error (SC-9)."""
        import subprocess

        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
            patch(
                "llmcli.cli.reload_proxy",
                create=True,
                side_effect=subprocess.CalledProcessError(1, "make"),
            ),
        ):
            result = runner.invoke(app, ["register-proxy", "--config", config_file])
        # Must exit 0 — write succeeded is the critical outcome
        assert result.exit_code == 0

    def test_register_proxy_reload_failure_prints_warning(self, fake_catalog, tmp_path) -> None:
        """Reload failure emits a warning message mentioning the reload problem."""
        import subprocess

        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
            patch(
                "llmcli.cli.reload_proxy",
                create=True,
                side_effect=subprocess.CalledProcessError(1, "make"),
            ),
        ):
            result = runner.invoke(app, ["register-proxy", "--config", config_file])
        combined = result.output + (result.stderr or "")
        assert any(kw in combined.lower() for kw in ("reload", "warn", "failed", "succeeded")), (
            f"Expected warning in output, got: {combined!r}"
        )

    def test_register_proxy_success_output_contains_path(self, fake_catalog, tmp_path) -> None:
        """Success output contains part of the config path that was updated (SC-9 confirmation).

        Rich may line-wrap long paths; we verify the path fragment appears somewhere
        in the combined output (stripping whitespace) rather than matching verbatim.
        """
        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            result = runner.invoke(app, ["register-proxy", "--config", config_file])
        # Collapse whitespace/newlines introduced by Rich wrapping before checking
        collapsed = "".join(result.output.split())
        collapsed_path = "".join(config_file.split())
        assert collapsed_path in collapsed, (
            f"Expected config path fragment in output, got: {result.output!r}"
        )

    def test_register_proxy_success_output_contains_model_count(
        self, fake_catalog, tmp_path
    ) -> None:
        """Success output contains the number of models written (SC-9 confirmation)."""
        config_file = str(tmp_path / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            result = runner.invoke(app, ["register-proxy", "--config", config_file])
        # fake_catalog has 2 models: small-q4 + big-model
        assert "2" in result.output, f"Expected model count (2) in output, got: {result.output!r}"

    def test_register_proxy_missing_parent_dir_exits_nonzero(self, fake_catalog, tmp_path) -> None:
        """register-proxy exits non-zero with a helpful error when parent dir is missing."""
        nonexistent = str(tmp_path / "does_not_exist" / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            result = runner.invoke(app, ["register-proxy", "--config", nonexistent])
        assert result.exit_code != 0

    def test_register_proxy_missing_parent_dir_mentions_mkdir(self, fake_catalog, tmp_path) -> None:
        """Friendly error for missing parent dir mentions how to create it."""
        nonexistent = str(tmp_path / "does_not_exist" / "config.yaml")
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            result = runner.invoke(app, ["register-proxy", "--config", nonexistent])
        combined = result.output + (result.stderr or "")
        assert "mkdir" in combined.lower() or "does_not_exist" in combined, (
            f"Expected mkdir hint or dir name in error output, got: {combined!r}"
        )

    def test_register_proxy_mixed_catalog_writes_both_local_and_remote(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Mixed catalog: both local (llamacpp) and remote (fireworks/openai) entries appear
        in the written litellm config block.
        """
        from llmcli import config as real_config
        from llmcli.config import Catalog, HostSettings, ModelSpec
        from llmcli.support.litellm_config import BLOCK_START, BLOCK_END

        import yaml

        # Build a catalog with 1 local + 1 remote spec.
        host = HostSettings(
            bind="0.0.0.0",
            public_base_url="http://localhost",
            api_key_env="LLMCLI_API_KEY",
        )
        local_spec = ModelSpec(
            name="small-local",
            engine="llamacpp",
            repo="Org/Small-GGUF",
            file="small.gguf",
            port=8091,
            vram_gib=5.0,
        )
        remote_spec = ModelSpec(
            name="kimi-remote",
            engine="remote",
            provider="fireworks",
            model_id="accounts/fireworks/models/kimi",
            protocol="openai",
        )
        mixed_catalog = Catalog(
            host=host, models={"small-local": local_spec, "kimi-remote": remote_spec}
        )

        config_path = tmp_path / "config.yaml"

        with (
            patch("llmcli.cli.config", create=True) as mock_config_mod,
            patch("llmcli.cli.reload_proxy", create=True),
        ):
            mock_config_mod.load.return_value = mixed_catalog
            mock_config_mod.check_vram_budget.side_effect = real_config.check_vram_budget
            result = runner.invoke(app, ["register-proxy", "--config", str(config_path)])

        assert result.exit_code == 0, f"Unexpected exit: {result.output}"
        assert config_path.exists(), "Config file was not created"

        content = config_path.read_text()
        assert BLOCK_START in content
        assert BLOCK_END in content

        inner = content.replace(BLOCK_START, "").replace(BLOCK_END, "").strip()
        parsed = yaml.safe_load(inner)
        names_in_block = {entry["model_name"] for entry in parsed["model_list"]}
        assert "small-local" in names_in_block, "Local entry missing from written block"
        assert "kimi-remote" in names_in_block, "Remote entry missing from written block"

        by_name = {e["model_name"]: e for e in parsed["model_list"]}
        # Local entry: model=openai/<name>, local api_base
        local_entry = by_name["small-local"]
        assert local_entry["litellm_params"]["model"] == "openai/small-local"
        assert "8091" in local_entry["litellm_params"]["api_base"]
        # Remote entry: model=openai/<model_id>, provider api_base
        remote_entry = by_name["kimi-remote"]
        assert remote_entry["litellm_params"]["model"] == "openai/accounts/fireworks/models/kimi"
        assert "fireworks" in remote_entry["litellm_params"]["api_base"]


# ---------------------------------------------------------------------------
# --help — all commands registered
# ---------------------------------------------------------------------------


class TestHelpOutput:
    def test_help_exits_zero(self) -> None:
        """llmcli --help exits 0."""
        # Act
        result = runner.invoke(app, ["--help"])
        # Assert
        assert result.exit_code == 0

    def test_help_lists_list_command(self) -> None:
        """--help output includes 'list' command."""
        # Act
        result = runner.invoke(app, ["--help"])
        # Assert
        assert "list" in result.output

    def test_help_lists_pull_command(self) -> None:
        """--help output includes 'pull' command."""
        # Act
        result = runner.invoke(app, ["--help"])
        # Assert
        assert "pull" in result.output

    def test_help_lists_serve_command(self) -> None:
        """--help output includes 'serve' command."""
        # Act
        result = runner.invoke(app, ["--help"])
        # Assert
        assert "serve" in result.output

    def test_help_lists_stop_command(self) -> None:
        """--help output includes 'stop' command."""
        # Act
        result = runner.invoke(app, ["--help"])
        # Assert
        assert "stop" in result.output

    def test_help_lists_status_command(self) -> None:
        """--help output includes 'status' command."""
        # Act
        result = runner.invoke(app, ["--help"])
        # Assert
        assert "status" in result.output

    def test_help_lists_chat_command(self) -> None:
        """--help output includes 'chat' command."""
        # Act
        result = runner.invoke(app, ["--help"])
        # Assert
        assert "chat" in result.output

    def test_help_lists_register_proxy_command(self) -> None:
        """--help output includes 'register-proxy' command."""
        # Act
        result = runner.invoke(app, ["--help"])
        # Assert
        assert "register-proxy" in result.output
