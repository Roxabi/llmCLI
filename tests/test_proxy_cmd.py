"""Tests for llmcli.cli.proxy — _validate_provider_keys and _spawn_litellm."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer
import yaml
from rich.console import Console

from llmcli.config import Catalog, HostSettings, ModelSpec
from llmcli.cli.proxy import _validate_provider_keys
from llmcli.support.providers import PROVIDERS


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
        """Engine guard prevents local models from going through remote-key validation.

        Mutation discipline: the local model uses a REAL valid provider string
        ("fireworks") so the unknown-provider branch cannot catch it. The
        fireworks key IS set in the environment, so a removed engine guard would
        NOT cause a missing-key error — it would validate the local model against
        fireworks, which is semantically wrong but invisible without the guard.
        The test must distinguish "engine guard works" from "unknown-provider fallback works."
        """
        # Arrange: set fireworks key so removing the engine guard would not raise a
        # missing-key error — the guard must be the only thing protecting local models.
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
        catalog = _make_catalog(
            models={
                "qwen3-8b": dict(
                    engine="llamacpp",
                    provider="fireworks",
                    repo="Org/Qwen3-8B-GGUF",
                    file="qwen3-8b-q4_k_m.gguf",
                    port=8091,
                    vram_gib=5.5,
                )
            }
        )

        # Spy on PROVIDERS.get to confirm the engine guard short-circuits before
        # PROVIDERS.get is called for this local model.
        import llmcli.cli.proxy as proxy_mod

        providers_mock = MagicMock(wraps=proxy_mod.PROVIDERS)
        monkeypatch.setattr(proxy_mod, "PROVIDERS", providers_mock)

        # Act
        errors = _validate_provider_keys(catalog, hostname="roxabitower")

        # Assert: no errors (engine guard skipped the local model entirely)
        assert errors == []
        # Assert: PROVIDERS.get was never called — the engine guard short-circuited
        providers_mock.get.assert_not_called()

    def test_remote_model_pinned_to_other_host_is_not_validated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Machines filter excludes remote model on non-matching host — no key check.

        Mutation discipline: if the machines filter is removed from _validate_provider_keys,
        the remote fireworks model would be evaluated on "this-host" and the missing
        FIREWORKS_API_KEY would produce an error. The test therefore fails when the filter
        is deleted.
        """
        # Arrange: remote model pinned to "other-host", key NOT in env.
        # If machines filter is removed, this remote model would generate a
        # missing-key error on "this-host". If filter works, no error.
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
        catalog = _make_catalog(
            models={
                "kimi-k2": dict(
                    engine="remote",
                    provider="fireworks",
                    model_id="accounts/fireworks/models/kimi",
                    protocol="openai",
                    machines=["other-host"],
                )
            }
        )
        # Act
        errors = _validate_provider_keys(catalog, hostname="this-host")
        # Assert: no errors — the machines filter excluded the model before key check
        assert errors == []


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
        monkeypatch.setattr(
            proxy_mod.shutil,
            "which",
            lambda name: "/usr/local/bin/litellm" if name == "litellm" else None,
        )
        monkeypatch.setattr(proxy_mod.subprocess, "Popen", fake_popen)

        # Act
        _spawn_litellm(Path("/tmp/cfg.yaml"), 18091, "0.0.0.0")

        # Assert
        fake_popen.assert_called_once_with(
            [
                "/usr/local/bin/litellm",
                "--config",
                "/tmp/cfg.yaml",
                "--port",
                "18091",
                "--host",
                "0.0.0.0",
            ],
            start_new_session=True,
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

    def test_drain_timeout_exceeded_kills_child(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Handler calls kill when child does not exit within drain_timeout."""
        import signal
        import subprocess

        import llmcli.cli.proxy as proxy_mod
        from llmcli.cli.proxy import _install_signal_handlers

        # Arrange
        child = MagicMock(spec=subprocess.Popen)
        # poll() set to None as a safety net but never reached in this scenario:
        # drain_timeout=0.0 makes the deadline already-past on first check, so the
        # while-loop body never executes; kill() is called right after the loop exits.
        child.poll.return_value = None

        captured_handlers: dict = {}

        def fake_signal_signal(signum, handler):
            captured_handlers[signum] = handler

        monkeypatch.setattr(proxy_mod.signal, "signal", fake_signal_signal)
        monkeypatch.setattr(proxy_mod.time, "sleep", lambda _: None)

        # Act — drain_timeout=0.0 → deadline expires immediately, loop skipped
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
        child.kill.reset_mock()  # reset so we can assert the reentrant kill

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
        # Assert file mode is 0o600 (security invariant — credentials path)
        assert oct(out_path.stat().st_mode & 0o777) == oct(0o600)

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
            # Mock shutil.which so the early binary check passes regardless of host env
            import llmcli.cli.proxy as proxy_mod

            monkeypatch.setattr(
                proxy_mod.shutil,
                "which",
                lambda name: "/usr/local/bin/litellm" if name == "litellm" else None,
            )
            monkeypatch.setattr("llmcli.cli.proxy._spawn_litellm", fake_spawn)
            monkeypatch.setattr(
                "llmcli.cli.proxy._install_signal_handlers",
                lambda *a, **k: None,
            )
            result = runner.invoke(typer_app, ["proxy"])

        # Assert
        assert result.exit_code == 0, (
            f"Expected 0; got {result.exit_code}; output: {result.output!r}"
        )

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
            # Mock shutil.which so the early binary check passes regardless of host env
            import llmcli.cli.proxy as proxy_mod

            monkeypatch.setattr(
                proxy_mod.shutil,
                "which",
                lambda name: "/usr/local/bin/litellm" if name == "litellm" else None,
            )
            monkeypatch.setattr("llmcli.cli.proxy._spawn_litellm", fake_spawn)
            monkeypatch.setattr(
                "llmcli.cli.proxy._install_signal_handlers",
                lambda *a, **k: None,
            )
            result = runner.invoke(typer_app, ["proxy"])

        # Assert
        assert result.exit_code == 42, (
            f"Expected 42; got {result.exit_code}; output: {result.output!r}"
        )

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
            # Mock shutil.which so the early binary check passes regardless of host env
            import llmcli.cli.proxy as proxy_mod

            monkeypatch.setattr(
                proxy_mod.shutil,
                "which",
                lambda name: "/usr/local/bin/litellm" if name == "litellm" else None,
            )
            monkeypatch.setattr("llmcli.cli.proxy._spawn_litellm", fake_spawn)
            monkeypatch.setattr(
                "llmcli.cli.proxy._install_signal_handlers",
                lambda *a, **k: None,
            )
            result = runner.invoke(typer_app, ["proxy"])

        # Assert
        assert result.exit_code == 137, (
            f"Expected 137 (128+9); got {result.exit_code}; output: {result.output!r}"
        )

    def test_port_flag_propagates_to_spawn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """`--port 4001` reaches _spawn_litellm via Typer → covers the CLI wiring end-to-end."""
        from typer.testing import CliRunner
        from llmcli.cli._app import app as typer_app

        monkeypatch.setenv("LLMCLI_API_KEY", "test-master-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        fake_child = MagicMock()
        fake_child.wait.return_value = 0
        fake_spawn = MagicMock(return_value=fake_child)

        catalog = self._make_remote_catalog()
        runner = CliRunner()

        with patch("llmcli.cli.config") as mock_config:
            mock_config.load.return_value = catalog
            import llmcli.cli.proxy as proxy_mod

            monkeypatch.setattr(
                proxy_mod.shutil,
                "which",
                lambda name: "/usr/local/bin/litellm" if name == "litellm" else None,
            )
            monkeypatch.setattr("llmcli.cli.proxy._spawn_litellm", fake_spawn)
            monkeypatch.setattr(
                "llmcli.cli.proxy._install_signal_handlers",
                lambda *a, **k: None,
            )
            result = runner.invoke(typer_app, ["proxy", "--port", "4001"])

        assert result.exit_code == 0, (
            f"Expected 0; got {result.exit_code}; output: {result.output!r}"
        )
        fake_spawn.assert_called_once()
        # Signature: _spawn_litellm(config_path, port, host) — port is positional[1].
        called_port = fake_spawn.call_args.args[1]
        assert called_port == 4001, f"Expected port 4001; got {called_port!r}"


# ---------------------------------------------------------------------------
# T1 (V1 slice) — _resolve_port 4-level precedence: env > flag > catalog > default
# ---------------------------------------------------------------------------


class TestResolvePort:
    def test_resolve_port_env_wins(self) -> None:
        """env beats flag, catalog, and default (env=20000 > flag=21000 > catalog=19999).

        _resolve_port is a pure function that takes the parsed env_val directly — the env
        var → int parsing happens in proxy() and is covered by TestProxyEnvPortMalformed +
        TestProxyBaseFailFast. This test only exercises the precedence wiring inside
        _resolve_port itself.
        """
        # Arrange
        from llmcli.cli.proxy import _resolve_port

        # Act
        result = _resolve_port(env_val=20000, flag_val=21000, catalog_port=19999)
        # Assert
        assert result == 20000

    def test_resolve_port_flag_beats_catalog(self) -> None:
        """flag beats catalog and default when env is absent (flag=21000 > catalog=19999)."""
        # Arrange / Act
        from llmcli.cli.proxy import _resolve_port

        result = _resolve_port(env_val=None, flag_val=21000, catalog_port=19999)
        # Assert
        assert result == 21000

    def test_resolve_port_catalog_beats_default(self) -> None:
        """catalog beats the hardcoded default when env and flag are absent (catalog=19999 > 18091)."""
        # Arrange / Act
        from llmcli.cli.proxy import _resolve_port

        result = _resolve_port(env_val=None, flag_val=None, catalog_port=19999)
        # Assert
        assert result == 19999

    def test_resolve_port_default(self) -> None:
        """Falls back to hardcoded default 18091 when env, flag, and catalog are all absent."""
        # Arrange / Act
        from llmcli.cli.proxy import _resolve_port

        result = _resolve_port(env_val=None, flag_val=None, catalog_port=None)
        # Assert
        assert result == 18091


# ---------------------------------------------------------------------------
# T6 — load_proxy_base RED tests
# ---------------------------------------------------------------------------


class TestLoadProxyBase:
    def test_load_proxy_base_absent(self, tmp_path: Path) -> None:
        """Missing file → returns _DEFAULT_PROXY_BASE unchanged."""
        # Arrange
        from llmcli.support.litellm_config import load_proxy_base, _DEFAULT_PROXY_BASE  # lazy

        absent = tmp_path / "nonexistent.yaml"
        # Act
        result = load_proxy_base(absent)
        # Assert
        assert result == _DEFAULT_PROXY_BASE

    def test_load_proxy_base_empty(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        """Empty file → emits warning log and returns _DEFAULT_PROXY_BASE."""
        import logging
        from llmcli.support.litellm_config import load_proxy_base, _DEFAULT_PROXY_BASE  # lazy

        empty_file = tmp_path / "proxy_base.yaml"
        empty_file.write_text("")
        # Act
        with caplog.at_level(logging.WARNING):
            result = load_proxy_base(empty_file)
        # Assert
        assert result == _DEFAULT_PROXY_BASE
        assert len(caplog.records) >= 1

    def test_load_proxy_base_valid(self, tmp_path: Path) -> None:
        """Valid YAML file → returns parsed dict."""
        from llmcli.support.litellm_config import load_proxy_base  # lazy

        content = "general_settings:\n  master_key: os.environ/K\n"
        yaml_file = tmp_path / "proxy_base.yaml"
        yaml_file.write_text(content)
        # Act
        result = load_proxy_base(yaml_file)
        # Assert
        assert result == {"general_settings": {"master_key": "os.environ/K"}}

    def test_load_proxy_base_syntax_error(self, tmp_path: Path) -> None:
        """Malformed YAML → raises yaml.YAMLError."""
        from llmcli.support.litellm_config import load_proxy_base  # lazy

        bad_file = tmp_path / "proxy_base.yaml"
        bad_file.write_text("key: : : :\n")
        # Act + Assert
        with pytest.raises(yaml.YAMLError):
            load_proxy_base(bad_file)

    def test_load_proxy_base_python_tag(self, tmp_path: Path) -> None:
        """Unsafe python tag → raises yaml.YAMLError (ConstructorError is a subclass)."""
        from llmcli.support.litellm_config import load_proxy_base  # lazy

        unsafe_file = tmp_path / "proxy_base.yaml"
        unsafe_file.write_text("foo: !!python/object:os.system []\n")
        # Act + Assert
        with pytest.raises(yaml.YAMLError):
            load_proxy_base(unsafe_file)

    def test_load_proxy_base_non_dict(self, tmp_path: Path) -> None:
        """Non-mapping YAML (int, list, scalar) raises yaml.YAMLError, not silent passthrough."""
        from llmcli.support.litellm_config import load_proxy_base  # lazy

        non_dict_file = tmp_path / "non_dict.yaml"
        non_dict_file.write_text("42\n")  # valid YAML, but it's an int
        with pytest.raises(yaml.YAMLError):
            load_proxy_base(non_dict_file)


# ---------------------------------------------------------------------------
# T7 — merge_proxy_config RED tests
# ---------------------------------------------------------------------------


class TestMergeProxyConfig:
    def test_merge_backfills_missing_general(self) -> None:
        """Base without general_settings → result has default master_key."""
        from llmcli.support.litellm_config import merge_proxy_config  # lazy

        base = {"litellm_settings": {"drop_params": True}}
        computed = []
        # Act
        result = merge_proxy_config(base, computed)
        # Assert
        assert result["general_settings"]["master_key"] == "os.environ/LLMCLI_API_KEY"

    def test_merge_backfills_missing_litellm(self) -> None:
        """Base without litellm_settings → result has drop_params True."""
        from llmcli.support.litellm_config import merge_proxy_config  # lazy

        base = {"general_settings": {"master_key": "os.environ/X"}}
        computed = []
        # Act
        result = merge_proxy_config(base, computed)
        # Assert
        assert result["litellm_settings"]["drop_params"] is True

    def test_merge_preserves_pass_through(self) -> None:
        """pass_through_endpoints and use_chat_completions_url_for_anthropic_messages survive merge."""
        from llmcli.support.litellm_config import merge_proxy_config  # lazy

        base = {
            "general_settings": {
                "master_key": "os.environ/LLMCLI_API_KEY",
                "pass_through_endpoints": [
                    {"path": "/api/messages", "target": "https://api.anthropic.com/v1/messages"}
                ],
            },
            "litellm_settings": {
                "drop_params": True,
                "use_chat_completions_url_for_anthropic_messages": True,
            },
        }
        computed = []
        # Act
        result = merge_proxy_config(base, computed)
        # Assert — both pass-through fields survive
        assert "pass_through_endpoints" in result["general_settings"]
        assert result["litellm_settings"]["use_chat_completions_url_for_anthropic_messages"] is True

    def test_merge_overwrites_stray_model_list(self) -> None:
        """Base model_list is replaced by computed model_list."""
        from llmcli.support.litellm_config import merge_proxy_config  # lazy

        base = {"model_list": [{"model_name": "STALE"}]}
        computed = [{"model_name": "FRESH"}]
        # Act
        result = merge_proxy_config(base, computed)
        # Assert
        assert result["model_list"] == computed

    def test_merge_forward_compat_passthrough(self) -> None:
        """Unknown top-level keys (router_settings, environment_variables) survive unmodified."""
        from llmcli.support.litellm_config import merge_proxy_config  # lazy

        base = {
            "router_settings": {"timeout": 600},
            "environment_variables": {"FOO": "bar"},
        }
        computed = []
        # Act
        result = merge_proxy_config(base, computed)
        # Assert
        assert result["router_settings"] == {"timeout": 600}
        assert result["environment_variables"] == {"FOO": "bar"}

    def test_proxy_base_example_includes_xai_pass_through(self) -> None:
        """deploy/proxy-base.yaml.example exposes /xai → xAI OAuth forwarder."""
        example = Path(__file__).parent.parent / "deploy" / "proxy-base.yaml.example"
        data = yaml.safe_load(example.read_text())
        endpoints = data["general_settings"]["pass_through_endpoints"]
        xai = next(ep for ep in endpoints if ep["path"] == "/xai")
        assert xai["target"] == "http://llmcli-xai-forwarder:18645"
        assert xai["include_subpath"] is True
        assert xai["forward_headers"] is False


# ---------------------------------------------------------------------------
# TestProxyEnvPortMalformed
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TestProxyBaseFailFast — SC-7: CLI-layer fail-fast for malformed proxy-base.yaml
# ---------------------------------------------------------------------------


class TestProxyBaseFailFast:
    """SC-7 — CLI-layer fail-fast for malformed proxy-base.yaml.

    Mutation discipline: if the `except yaml.YAMLError` branch in `proxy()` is
    removed, these tests fail because a raised YAMLError propagates as an
    unhandled exception (non-zero exit, but no "proxy-base.yaml" string in
    output — the runner captures it as a traceback, not the formatted message).
    Deleting the branch also silently loses the actionable user-facing error.
    """

    _EMPTY_CATALOG = _make_catalog()  # no models → provider validation is a no-op

    def _invoke_with_broken_proxy_base(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        proxy_base_content: str,
    ):
        """Shared arrange+act helper: redirect HOME, patch catalog, write broken proxy-base.yaml."""
        from typer.testing import CliRunner
        from llmcli.cli._app import app as typer_app

        # Redirect Path.home() so proxy step 3 reads proxy-base.yaml from tmp_path
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        # Set the master-key env var that catalog.host.api_key_env references
        monkeypatch.setenv("LLMCLI_API_KEY", "test-master-key")

        # Place the broken proxy-base.yaml at the hardcoded location
        proxy_base = tmp_path / ".roxabi" / "llmcli" / "proxy-base.yaml"
        proxy_base.parent.mkdir(parents=True, exist_ok=True)
        proxy_base.write_text(proxy_base_content)

        runner = CliRunner()
        out_path = tmp_path / "out.yaml"

        with patch("llmcli.cli.config") as mock_config:
            mock_config.load.return_value = self._EMPTY_CATALOG
            result = runner.invoke(typer_app, ["proxy", "--config-out", str(out_path)])

        return result

    def test_syntax_error_exits_nonzero_with_filename(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Malformed YAML in proxy-base.yaml → exit 1 + output mentions 'proxy-base.yaml'."""
        # Arrange + Act
        result = self._invoke_with_broken_proxy_base(
            tmp_path,
            monkeypatch,
            proxy_base_content="key: : : :\n",  # syntax error
        )

        # Assert — non-zero exit
        assert result.exit_code != 0, (
            f"Expected non-zero exit; got {result.exit_code}; output: {result.output!r}"
        )
        # Assert — actionable filename appears in user-facing output
        combined = (result.output or "") + (result.stderr or "")
        assert "proxy-base.yaml" in combined, (
            f"Expected 'proxy-base.yaml' in output; got: {combined!r}"
        )

    def test_python_tag_exits_nonzero_with_filename(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """!!python/object tag in proxy-base.yaml (ConstructorError) → exit 1 + filename in output."""
        # Arrange + Act
        result = self._invoke_with_broken_proxy_base(
            tmp_path,
            monkeypatch,
            proxy_base_content="foo: !!python/object:os.system []\n",  # unsafe tag
        )

        # Assert
        assert result.exit_code != 0, (
            f"Expected non-zero exit; got {result.exit_code}; output: {result.output!r}"
        )
        combined = (result.output or "") + (result.stderr or "")
        assert "proxy-base.yaml" in combined, (
            f"Expected 'proxy-base.yaml' in output; got: {combined!r}"
        )

    def test_non_dict_exits_nonzero_with_filename(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """YAML int scalar (not a mapping) raises yaml.YAMLError at CLI layer → exit 1 + filename."""
        # Arrange + Act
        result = self._invoke_with_broken_proxy_base(
            tmp_path,
            monkeypatch,
            proxy_base_content="42\n",  # valid YAML scalar, not a mapping
        )

        # Assert
        assert result.exit_code != 0, (
            f"Expected non-zero exit; got {result.exit_code}; output: {result.output!r}"
        )
        combined = (result.output or "") + (result.stderr or "")
        assert "proxy-base.yaml" in combined, (
            f"Expected 'proxy-base.yaml' in output; got: {combined!r}"
        )


class TestProxyEnvPortMalformed:
    _EMPTY_CATALOG = _make_catalog()  # no models → provider validation is a no-op

    def test_malformed_env_proxy_port_exits_1(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """LLMCLI_PROXY_PORT='abc' produces a user-friendly error and exits 1.

        Uses patch("llmcli.cli.config") to bypass the module-import-time snapshot of
        DEFAULT_CONFIG_PATH; LLMCLI_CONFIG=monkeypatch.setenv would arrive too late.
        Same pattern as TestProxyBaseFailFast / TestConfigOutDryRun.
        """
        from typer.testing import CliRunner
        from llmcli.cli._app import app as typer_app

        # Arrange
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
        monkeypatch.setenv("LLMCLI_API_KEY", "test-master-key")
        monkeypatch.setenv("LLMCLI_PROXY_PORT", "abc")  # malformed

        runner = CliRunner()
        with patch("llmcli.cli.config") as mock_config:
            mock_config.load.return_value = self._EMPTY_CATALOG
            result = runner.invoke(typer_app, ["proxy", "--config-out", str(tmp_path / "out.yaml")])

        # Assert
        assert result.exit_code == 1, (
            f"Expected exit 1; got {result.exit_code}; output: {result.output!r}"
        )
        combined = (result.output or "") + (result.stderr or "")
        assert "LLMCLI_PROXY_PORT" in combined, (
            f"Expected 'LLMCLI_PROXY_PORT' in output; got: {combined!r}"
        )
        assert "abc" in combined, f"Expected 'abc' in output; got: {combined!r}"


# ---------------------------------------------------------------------------
# Model catalogue background refresh (#130)
# ---------------------------------------------------------------------------


class TestModelRefresh:
    def test_parse_model_refresh_interval_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from llmcli.cli.proxy import _parse_model_refresh_interval

        monkeypatch.delenv("LLMCLI_MODEL_REFRESH_SECS", raising=False)
        assert _parse_model_refresh_interval() == 60.0

    def test_parse_model_refresh_interval_rejects_negative(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llmcli.cli.proxy import _parse_model_refresh_interval

        monkeypatch.setenv("LLMCLI_MODEL_REFRESH_SECS", "-1")
        with pytest.raises(typer.Exit) as exc:
            _parse_model_refresh_interval()
        assert exc.value.exit_code == 1

    def test_reload_litellm_child_always_respawns(self) -> None:
        from llmcli.cli.proxy import _reload_litellm_child

        child = MagicMock()
        child.poll.return_value = None
        new_child = MagicMock()
        with patch("llmcli.cli.proxy._spawn_litellm", return_value=new_child) as spawn:
            result = _reload_litellm_child(child, Path("/tmp/cfg.yaml"), 18091, "0.0.0.0")
        child.terminate.assert_called_once()
        child.send_signal.assert_not_called()
        spawn.assert_called_once()
        assert result is new_child

    def test_invalidate_model_cache_invokes_registered_callback(self) -> None:
        from llmcli.support.litellm_config import (
            invalidate_model_cache,
            register_model_refresh_callback,
        )

        calls: list[str] = []

        def _cb() -> None:
            calls.append("refresh")

        register_model_refresh_callback(_cb)
        try:
            invalidate_model_cache()
        finally:
            register_model_refresh_callback(None)
        assert calls == ["refresh"]

    def test_model_refresh_loop_regenerates_config(self) -> None:
        from llmcli.cli.proxy import _start_model_refresh_loop

        catalog = _make_catalog()
        target = Path("/tmp/proxy.config.yaml")
        base: dict = {"general_settings": {}, "litellm_settings": {}}
        child = MagicMock()
        child.poll.return_value = None
        child_state: dict = {"child": child, "stop": False}
        sleep_calls = 0

        def _sleep_then_stop(_interval: float) -> None:
            nonlocal sleep_calls
            sleep_calls += 1
            if sleep_calls >= 2:
                child_state["stop"] = True

        with (
            patch("llmcli.cli.proxy._write_proxy_config") as write_cfg,
            patch("llmcli.cli.proxy._reload_litellm_child", return_value=child) as reload,
            patch("llmcli.cli.proxy.time.sleep", side_effect=_sleep_then_stop),
        ):
            _start_model_refresh_loop(
                child_state,
                catalog=catalog,
                target=target,
                base=base,
                port=18091,
                host="0.0.0.0",
                interval_secs=0.01,
            ).join(timeout=2.0)
        write_cfg.assert_called()
        reload.assert_called()
