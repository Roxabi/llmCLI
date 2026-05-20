"""Tests for llmcli.cli.proxy — _validate_provider_keys and _spawn_litellm."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
from rich.console import Console

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


# ---------------------------------------------------------------------------
# TestSpawnLitellm
# ---------------------------------------------------------------------------


class TestSpawnLitellm:
    def test_happy_path_calls_popen_with_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Popen is called with the correct argument list when the binary is found."""
        import llmcli.cli.proxy as proxy_mod
        from llmcli.cli.proxy import _spawn_litellm

        # Arrange
        fake_popen = MagicMock()
        monkeypatch.setattr(proxy_mod.shutil, "which", lambda name: "/usr/local/bin/litellm" if name == "litellm" else None)
        monkeypatch.setattr(proxy_mod.subprocess, "Popen", fake_popen)

        # Act
        _spawn_litellm(Path("/tmp/cfg.yaml"), 18091, "0.0.0.0")

        # Assert
        fake_popen.assert_called_once_with(
            ["/usr/local/bin/litellm", "--config", "/tmp/cfg.yaml", "--port", "18091", "--host", "0.0.0.0"]
        )

    def test_missing_binary_exits_127(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """typer.Exit(127) is raised and stderr contains the expected message when litellm is not on PATH."""
        import llmcli.cli.proxy as proxy_mod
        from llmcli.cli.proxy import _spawn_litellm

        # Arrange
        monkeypatch.setattr(proxy_mod.shutil, "which", lambda name: None)
        stderr_buffer = StringIO()
        fake_err_console = Console(file=stderr_buffer, highlight=False)
        monkeypatch.setattr(proxy_mod, "err_console", fake_err_console)

        # Act + Assert
        with pytest.raises(typer.Exit) as exc_info:
            _spawn_litellm(Path("/tmp/cfg.yaml"), 18091, "0.0.0.0")

        assert exc_info.value.exit_code == 127
        assert "litellm binary not found" in stderr_buffer.getvalue()


# ---------------------------------------------------------------------------
# TestSignalForwarding
# ---------------------------------------------------------------------------


class TestSignalForwarding:
    def test_sigterm_terminates_child_polls_then_returns_when_exited(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handler terminates child and returns without kill when child exits within drain."""
        import signal
        import subprocess
        import time as time_mod

        import llmcli.cli.proxy as proxy_mod
        from llmcli.cli.proxy import _install_signal_handlers

        # Arrange
        child = MagicMock(spec=subprocess.Popen)
        # Simulate child exiting after two polls
        child.poll.side_effect = [None, None, 0]

        captured_handlers: dict = {}

        def fake_signal_signal(signum, handler):
            captured_handlers[signum] = handler

        monkeypatch.setattr(proxy_mod.signal, "signal", fake_signal_signal)
        monkeypatch.setattr(proxy_mod.time, "sleep", lambda _: None)

        # Act — register handlers, then invoke captured SIGTERM handler
        _install_signal_handlers(child, drain_timeout=0.5)
        handler = captured_handlers[signal.SIGTERM]
        handler(signal.SIGTERM, None)

        # Assert
        assert child.terminate.called is True
        assert child.poll.call_count >= 1
        assert child.kill.called is False

    def test_drain_timeout_exceeded_kills_child(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Handler calls kill when child does not exit within drain_timeout."""
        import signal
        import subprocess

        import llmcli.cli.proxy as proxy_mod
        from llmcli.cli.proxy import _install_signal_handlers

        # Arrange
        child = MagicMock(spec=subprocess.Popen)
        # Child never exits — poll always returns None
        child.poll.return_value = None

        captured_handlers: dict = {}

        def fake_signal_signal(signum, handler):
            captured_handlers[signum] = handler

        monkeypatch.setattr(proxy_mod.signal, "signal", fake_signal_signal)
        monkeypatch.setattr(proxy_mod.time, "sleep", lambda _: None)

        # Act — very short timeout so the deadline passes immediately
        _install_signal_handlers(child, drain_timeout=0.0)
        handler = captured_handlers[signal.SIGTERM]
        handler(signal.SIGTERM, None)

        # Assert
        assert child.terminate.called is True
        assert child.kill.called is True

    def test_double_sigint_during_drain_kills_and_exits_130(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second SIGINT during active drain raises SystemExit(130) and kills child."""
        import signal
        import subprocess

        import llmcli.cli.proxy as proxy_mod
        from llmcli.cli.proxy import _install_signal_handlers

        # Arrange
        child = MagicMock(spec=subprocess.Popen)
        # Child never exits so drain stays active
        child.poll.return_value = None

        captured_handlers: dict = {}

        def fake_signal_signal(signum, handler):
            captured_handlers[signum] = handler

        monkeypatch.setattr(proxy_mod.signal, "signal", fake_signal_signal)
        monkeypatch.setattr(proxy_mod.time, "sleep", lambda _: None)

        # Act — zero timeout so first SIGINT drains immediately and finishes
        # then second SIGINT hits the reentrant path
        _install_signal_handlers(child, drain_timeout=0.0)
        handler = captured_handlers[signal.SIGINT]

        # First SIGINT: triggers drain (timeout=0 so kill is called, but
        # drain_state["active"] is set before kill)
        # We need drain_state to be active, so patch time.monotonic to force
        # the deadline to be already past on first invocation too — the first
        # call will set active=True, terminate, then exhaust the loop and kill.
        # To isolate the reentrant path, call handler once to set drain_state
        # active, then call again.
        handler(signal.SIGINT, None)  # first — sets active, drain exhausts, kills
        child.kill.reset_mock()       # reset so we can assert the reentrant kill

        with pytest.raises(SystemExit) as exc_info:
            handler(signal.SIGINT, None)  # second — reentrant path

        assert exc_info.value.code == 130
        assert child.kill.called is True


# ---------------------------------------------------------------------------
# TestConfigOutDryRun
# ---------------------------------------------------------------------------


class TestConfigOutDryRun:
    """Integration tests for the `proxy --config-out` dry-run path."""

    def test_config_out_writes_yaml_and_exits_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """--config-out writes a valid YAML with expected top-level keys and exits 0."""
        import yaml
        from typer.testing import CliRunner
        from llmcli.cli._app import app as typer_app

        # Arrange — env vars required by _validate_provider_keys + catalog.host.api_key_env
        monkeypatch.setenv("LLMCLI_API_KEY", "test-master-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")

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
        out_path = tmp_path / "proxy.yaml"

        runner = CliRunner()

        # Act — patch catalog loader so test is hermetic (no real TOML on disk needed)
        with patch("llmcli.cli.config") as mock_config:
            mock_config.load.return_value = catalog
            result = runner.invoke(typer_app, ["proxy", "--config-out", str(out_path)])

        # Assert
        assert result.exit_code == 0, f"Unexpected exit code; output: {result.output!r}"
        assert out_path.exists(), "Output file was not created"

        cfg = yaml.safe_load(out_path.read_text())
        assert "general_settings" in cfg
        assert "litellm_settings" in cfg
        assert "model_list" in cfg
        assert cfg["general_settings"]["master_key"] == "os.environ/LLMCLI_API_KEY"

    def test_config_out_missing_provider_env_exits_one_no_file(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """--config-out exits 1 with key_env name in output and does NOT write the file."""
        from typer.testing import CliRunner
        from llmcli.cli._app import app as typer_app

        # Arrange — master key present, provider key absent
        monkeypatch.setenv("LLMCLI_API_KEY", "test-master-key")
        key_env = PROVIDERS["fireworks"].key_env  # FIREWORKS_API_KEY
        monkeypatch.delenv(key_env, raising=False)

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
        out_path = tmp_path / "proxy.yaml"

        runner = CliRunner()

        # Act
        with patch("llmcli.cli.config") as mock_config:
            mock_config.load.return_value = catalog
            result = runner.invoke(typer_app, ["proxy", "--config-out", str(out_path)])

        # Assert
        assert result.exit_code == 1, f"Expected exit code 1; output: {result.output!r}"
        combined_output = result.output + result.stderr
        assert key_env in combined_output, (
            f"Expected '{key_env}' in output/stderr but got: {combined_output!r}"
        )
        assert not out_path.exists(), "Output file must not be written on validation failure"


# ---------------------------------------------------------------------------
# TestExitCodePropagation
# ---------------------------------------------------------------------------


class TestExitCodePropagation:
    """Integration tests for exit-code propagation through the proxy lifecycle."""

    def _make_remote_catalog(self) -> "Catalog":
        """Build a Catalog with a single remote model (fireworks) for lifecycle tests."""
        return _make_catalog(
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

    def test_exit_zero_propagates_as_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """proxy returns exit code 0 when litellm exits with code 0."""
        from typer.testing import CliRunner
        from llmcli.cli._app import app as typer_app

        # Arrange
        monkeypatch.setenv("LLMCLI_API_KEY", "test-master-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")
        # Redirect Path.home() to tmp_path so state-dir writes go to a temp location
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        fake_child = MagicMock()
        fake_child.wait.return_value = 0
        fake_spawn = MagicMock(return_value=fake_child)

        catalog = self._make_remote_catalog()
        runner = CliRunner()

        # Act
        with patch("llmcli.cli.config") as mock_config:
            mock_config.load.return_value = catalog
            monkeypatch.setattr("llmcli.cli.proxy._spawn_litellm", fake_spawn)
            monkeypatch.setattr(
                "llmcli.cli.proxy._install_signal_handlers",
                lambda *a, **k: None,
            )
            result = runner.invoke(typer_app, ["proxy"])

        # Assert
        assert result.exit_code == 0, f"Expected 0; got {result.exit_code}; output: {result.output!r}"

    def test_exit_forty_two_propagates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """proxy returns exit code 42 when litellm exits with code 42."""
        from typer.testing import CliRunner
        from llmcli.cli._app import app as typer_app

        # Arrange
        monkeypatch.setenv("LLMCLI_API_KEY", "test-master-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        fake_child = MagicMock()
        fake_child.wait.return_value = 42
        fake_spawn = MagicMock(return_value=fake_child)

        catalog = self._make_remote_catalog()
        runner = CliRunner()

        # Act
        with patch("llmcli.cli.config") as mock_config:
            mock_config.load.return_value = catalog
            monkeypatch.setattr("llmcli.cli.proxy._spawn_litellm", fake_spawn)
            monkeypatch.setattr(
                "llmcli.cli.proxy._install_signal_handlers",
                lambda *a, **k: None,
            )
            result = runner.invoke(typer_app, ["proxy"])

        # Assert
        assert result.exit_code == 42, f"Expected 42; got {result.exit_code}; output: {result.output!r}"

    def test_negative_nine_maps_to_137(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """proxy returns exit code 137 when litellm is SIGKILL-ed (wait() returns -9)."""
        from typer.testing import CliRunner
        from llmcli.cli._app import app as typer_app

        # Arrange
        monkeypatch.setenv("LLMCLI_API_KEY", "test-master-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        fake_child = MagicMock()
        # POSIX: negative return code means killed by signal abs(returncode)
        # -9 = SIGKILL (e.g. OOM), expected propagated code = 128 + 9 = 137
        fake_child.wait.return_value = -9
        fake_spawn = MagicMock(return_value=fake_child)

        catalog = self._make_remote_catalog()
        runner = CliRunner()

        # Act
        with patch("llmcli.cli.config") as mock_config:
            mock_config.load.return_value = catalog
            monkeypatch.setattr("llmcli.cli.proxy._spawn_litellm", fake_spawn)
            monkeypatch.setattr(
                "llmcli.cli.proxy._install_signal_handlers",
                lambda *a, **k: None,
            )
            result = runner.invoke(typer_app, ["proxy"])

        # Assert
        assert result.exit_code == 137, f"Expected 137 (128+9); got {result.exit_code}; output: {result.output!r}"
