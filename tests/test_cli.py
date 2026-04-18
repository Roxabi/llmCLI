"""RED-phase tests for T1.6 — CLI commands (SC-2, SC-3, SC-4, SC-7).

These tests MUST fail against the current scaffold because all commands
just `raise typer.Exit(code=0)` without producing any output.

The stubs in cli.py only import `typer`, so patching names like
`llmcli.cli.config` / `llmcli.cli.daemon_request` / etc. requires
`create=True` — the GREEN implementation will add the real imports and the
patches will then intercept the actual calls.

Expected RED failures (representative):
- test_list_prints_model_name: exit 0 but output is empty
- test_pull_invokes_hf_download_for_known_model: mock never called (stub)
- test_pull_nonzero_on_unknown_model: stub exits 0, expected != 0
- test_serve_rejects_vram_exceeded_nonzero: stub exits 0, expected != 0
- test_stop_sends_shutdown_via_socket: mock never called (stub)
- test_status_shows_running_engine: output is empty
- test_chat_prints_response: output is empty
- test_register_proxy_calls_build_block: mock never called

GREEN phase (T1.12): cli.py wires real implementations → all pass.
"""

from __future__ import annotations

import json
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
def mock_daemon_socket():
    """Patch daemon_request so CLI commands don't need a live AF_UNIX socket."""
    status_payload = json.dumps(
        {
            "small-q4": {
                "pid": 12345,
                "port": 8091,
                "model_name": "small-q4",
                "started_at": 1700000000.0,
            }
        }
    )
    with patch("llmcli.cli.daemon_request", create=True, return_value=status_payload) as mock_req:
        yield mock_req


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


class TestListCommand:
    def test_list_exits_zero(self, fake_catalog, mock_daemon_socket) -> None:
        """list exits 0."""
        # Act
        result = runner.invoke(app, ["list"])
        # Assert
        assert result.exit_code == 0

    def test_list_prints_model_name(self, fake_catalog, mock_daemon_socket) -> None:
        """list output contains the model name from catalog."""
        # Act
        result = runner.invoke(app, ["list"])
        # Assert — RED: stub prints nothing
        assert "small-q4" in result.output, (
            f"Expected 'small-q4' in output but got: {result.output!r}"
        )

    def test_list_prints_engine(self, fake_catalog, mock_daemon_socket) -> None:
        """list output contains the engine type."""
        # Act
        result = runner.invoke(app, ["list"])
        # Assert — RED: stub prints nothing
        assert "llamacpp" in result.output, (
            f"Expected engine name in output but got: {result.output!r}"
        )

    def test_list_prints_vram(self, fake_catalog, mock_daemon_socket) -> None:
        """list output contains the VRAM amount for the model."""
        # Act
        result = runner.invoke(app, ["list"])
        # Assert — RED: stub prints nothing
        assert "6" in result.output, (
            f"Expected VRAM (6.0) in output but got: {result.output!r}"
        )

    def test_list_prints_port(self, fake_catalog, mock_daemon_socket) -> None:
        """list output contains the port number."""
        # Act
        result = runner.invoke(app, ["list"])
        # Assert — RED: stub prints nothing
        assert "8091" in result.output, (
            f"Expected port 8091 in output but got: {result.output!r}"
        )


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
        with patch("llmcli.cli.hf_hub_download", create=True, return_value="/tmp/fake.gguf") as mock_hf:
            # Act
            runner.invoke(app, ["pull", "small-q4"])
        # Assert — RED: stub never calls hf_hub_download
        mock_hf.assert_called_once()

    def test_pull_nonzero_on_unknown_model(self, fake_catalog) -> None:
        """pull <unknown-model> exits non-zero."""
        # Act
        result = runner.invoke(app, ["pull", "does-not-exist"])
        # Assert — RED: stub exits 0 for everything
        assert result.exit_code != 0, (
            "Expected non-zero exit for unknown model, got 0"
        )

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
# U3 — llmcli serve [name]
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_serve_exits_zero_with_named_model(self, fake_catalog) -> None:
        """serve --name <name> exits 0 when model fits VRAM."""
        # Arrange
        with (
            patch("llmcli.cli.Daemon", create=True) as mock_daemon_cls,
            patch("llmcli.cli.hf_hub_download", create=True, return_value="/tmp/fake.gguf"),
        ):
            mock_daemon_cls.return_value = MagicMock()
            # Act
            result = runner.invoke(app, ["serve", "--name", "small-q4"])
        # Assert
        assert result.exit_code == 0

    def test_serve_starts_daemon_with_named_model(self, fake_catalog) -> None:
        """serve --name <name> starts the daemon with the specified model."""
        # Arrange
        with (
            patch("llmcli.cli.Daemon", create=True) as mock_daemon_cls,
            patch("llmcli.cli.hf_hub_download", create=True, return_value="/tmp/fake.gguf"),
        ):
            mock_daemon_instance = MagicMock()
            mock_daemon_cls.return_value = mock_daemon_instance
            # Act
            runner.invoke(app, ["serve", "--name", "small-q4"])
        # Assert — RED: stub never calls Daemon
        mock_daemon_instance.serve.assert_called_once()

    def test_serve_uses_default_model_when_no_name_given(self, fake_catalog) -> None:
        """serve with no name argument uses catalog host.default_model."""
        # Arrange
        with (
            patch("llmcli.cli.Daemon", create=True) as mock_daemon_cls,
            patch("llmcli.cli.hf_hub_download", create=True, return_value="/tmp/fake.gguf"),
        ):
            mock_daemon_instance = MagicMock()
            mock_daemon_cls.return_value = mock_daemon_instance
            # Act
            runner.invoke(app, ["serve"])
        # Assert — RED: stub exits 0 but never calls Daemon.serve
        mock_daemon_instance.serve.assert_called_once()

    def test_serve_rejects_vram_exceeded_exits_nonzero(self, fake_catalog) -> None:
        """serve --name <oversized-model> exits non-zero when VRAM budget exceeded (C2)."""
        # big-model vram_gib=13 > budget=10
        # Act
        result = runner.invoke(app, ["serve", "--name", "big-model"])
        # Assert — RED: stub exits 0 for everything
        assert result.exit_code != 0, (
            "Expected non-zero exit when model exceeds VRAM budget, got 0"
        )

    def test_serve_vram_error_mentions_model_and_budget(self, fake_catalog) -> None:
        """serve --name <oversized-model> output contains helpful VRAM info."""
        # Act
        result = runner.invoke(app, ["serve", "--name", "big-model"])
        combined = result.output + (result.stderr or "")
        # Assert — RED: stub outputs nothing
        assert any(
            keyword in combined
            for keyword in ("big-model", "13", "10", "vram", "VRAM", "budget")
        ), f"Expected VRAM error details in output, got: {combined!r}"

    def test_serve_unknown_model_exits_nonzero(self, fake_catalog) -> None:
        """serve --name <unknown-model> exits non-zero."""
        # Act
        result = runner.invoke(app, ["serve", "--name", "nonexistent-model"])
        # Assert — RED: stub exits 0
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# U4 — llmcli stop
# ---------------------------------------------------------------------------


class TestStopCommand:
    def test_stop_exits_zero(self, mock_daemon_socket) -> None:
        """stop exits 0 when daemon responds."""
        # Act
        result = runner.invoke(app, ["stop"])
        # Assert
        assert result.exit_code == 0

    def test_stop_sends_shutdown_message(self, mock_daemon_socket) -> None:
        """stop calls daemon_request with SHUTDOWN message."""
        # Act
        runner.invoke(app, ["stop"])
        # Assert — RED: stub never calls daemon_request
        mock_daemon_socket.assert_called_once_with("SHUTDOWN")


# ---------------------------------------------------------------------------
# U5 — llmcli status
# ---------------------------------------------------------------------------


class TestStatusCommand:
    def test_status_exits_zero(self, mock_daemon_socket) -> None:
        """status exits 0."""
        # Act
        result = runner.invoke(app, ["status"])
        # Assert
        assert result.exit_code == 0

    def test_status_shows_running_model_name(self, mock_daemon_socket) -> None:
        """status output contains the model name of the running engine."""
        # Act
        result = runner.invoke(app, ["status"])
        # Assert — RED: stub prints nothing
        assert "small-q4" in result.output, (
            f"Expected model name in status output, got: {result.output!r}"
        )

    def test_status_shows_port(self, mock_daemon_socket) -> None:
        """status output contains the port of the running engine."""
        # Act
        result = runner.invoke(app, ["status"])
        # Assert — RED: stub prints nothing
        assert "8091" in result.output, (
            f"Expected port 8091 in status output, got: {result.output!r}"
        )

    def test_status_calls_daemon_socket_with_status(self, mock_daemon_socket) -> None:
        """status sends STATUS to daemon socket."""
        # Act
        runner.invoke(app, ["status"])
        # Assert — RED: stub never calls daemon_request
        mock_daemon_socket.assert_called_once_with("STATUS")


# ---------------------------------------------------------------------------
# U7 — llmcli chat <name> "prompt"
# ---------------------------------------------------------------------------


class TestChatCommand:
    def test_chat_exits_zero(
        self, fake_catalog, mock_daemon_socket, mock_openai_client
    ) -> None:
        """chat exits 0 on success."""
        # Act
        result = runner.invoke(app, ["chat", "small-q4", "ping"])
        # Assert
        assert result.exit_code == 0

    def test_chat_prints_llm_response(
        self, fake_catalog, mock_daemon_socket, mock_openai_client
    ) -> None:
        """chat prints the LLM completion to stdout."""
        # Act
        result = runner.invoke(app, ["chat", "small-q4", "ping"])
        # Assert — RED: stub prints nothing
        assert "pong" in result.output, (
            f"Expected 'pong' in chat output, got: {result.output!r}"
        )

    def test_chat_invokes_openai_completions(
        self, fake_catalog, mock_daemon_socket, mock_openai_client
    ) -> None:
        """chat calls the OpenAI completions endpoint."""
        # Act
        runner.invoke(app, ["chat", "small-q4", "ping"])
        # Assert — RED: stub never instantiates openai client
        mock_openai_client.chat.completions.create.assert_called_once()

    def test_chat_passes_prompt_to_openai(
        self, fake_catalog, mock_daemon_socket, mock_openai_client
    ) -> None:
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
    def test_register_proxy_exits_zero(self, fake_catalog) -> None:
        """register-proxy exits 0 on success."""
        # Arrange
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True),
        ):
            # Act
            result = runner.invoke(app, ["register-proxy"])
        # Assert
        assert result.exit_code == 0

    def test_register_proxy_calls_build_block(self, fake_catalog) -> None:
        """register-proxy calls build_block with the loaded catalog."""
        # Arrange
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n") as mock_build,
            patch("llmcli.cli.write_block", create=True),
        ):
            # Act
            runner.invoke(app, ["register-proxy"])
        # Assert — RED: stub never calls build_block
        mock_build.assert_called_once()

    def test_register_proxy_calls_write_block(self, fake_catalog) -> None:
        """register-proxy calls write_block to persist the LiteLLM block."""
        # Arrange
        with (
            patch("llmcli.cli.build_block", create=True, return_value="# block\n"),
            patch("llmcli.cli.write_block", create=True) as mock_write,
        ):
            # Act
            runner.invoke(app, ["register-proxy"])
        # Assert — RED: stub never calls write_block
        mock_write.assert_called_once()


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
