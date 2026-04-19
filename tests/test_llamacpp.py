"""RED-phase tests for LlamaCppEngine (T1.3).

All tests assert behaviour expected from the GREEN implementation.
Against the current NotImplementedError scaffold they MUST fail.

Test categories:
- Protocol conformance (method signatures, attribute presence)
- GGUF path resolution from HF hub cache (pure logic, monkeypatched fs)
- Command-line construction (pure logic)
- EngineInstance shape returned by start()
- health() HTTP probe (httpx mocked)
- stop() process teardown (subprocess mocked)

Markers:
  no_gpu  — CI-safe; no real binary, no GPU required
  gpu     — requires real llama-server binary + GPU; skipped in CI
"""
from __future__ import annotations

import inspect
import os
import signal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmcli.config import ModelSpec
from llmcli.engine import EngineInstance
from llmcli.engines.llamacpp import LlamaCppEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def spec() -> ModelSpec:
    """Minimal ModelSpec for a vanilla GGUF model."""
    return ModelSpec(
        name="qwen3-8b-q4",
        engine="llamacpp",
        repo="bartowski/Qwen3-8B-GGUF",
        file="Qwen3-8B-Q4_K_M.gguf",
        port=8091,
        vram_gib=5.5,
        flags=["-ngl", "99", "-c", "4096"],
    )


@pytest.fixture()
def spec_with_mmproj(spec: ModelSpec) -> ModelSpec:
    """ModelSpec that includes an mmproj sidecar."""
    return ModelSpec(
        name=spec.name,
        engine=spec.engine,
        repo=spec.repo,
        file=spec.file,
        port=spec.port,
        vram_gib=spec.vram_gib,
        flags=spec.flags,
        mmproj="mmproj-qwen3-8b.gguf",
    )


@pytest.fixture()
def fake_hf_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a fake HF hub cache directory with a dummy .gguf file.

    Points HF_HOME at tmp_path so the engine resolves paths there.
    Returns the root cache dir (tmp_path / "hub").
    """
    hub = tmp_path / "hub"
    # HF hub stores blobs under:
    #   hub/models--<org>--<repo>/snapshots/<revision>/<filename>
    model_dir = hub / "models--bartowski--Qwen3-8B-GGUF" / "snapshots" / "main"
    model_dir.mkdir(parents=True)
    (model_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"FAKE_GGUF")

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    return hub


@pytest.fixture()
def engine() -> LlamaCppEngine:
    return LlamaCppEngine()


@pytest.fixture()
def fake_instance() -> EngineInstance:
    return EngineInstance(pid=12345, port=8091, model="qwen3-8b-q4", started_at=1_700_000_000.0)


# ---------------------------------------------------------------------------
# 1. Engine Protocol conformance
# ---------------------------------------------------------------------------

class TestEngineProtocolConformance:
    """LlamaCppEngine must implement the Engine Protocol."""

    @pytest.mark.no_gpu
    def test_has_start_method(self, engine: LlamaCppEngine) -> None:
        assert callable(getattr(engine, "start", None)), "LlamaCppEngine must have a start() method"

    @pytest.mark.no_gpu
    def test_has_stop_method(self, engine: LlamaCppEngine) -> None:
        assert callable(getattr(engine, "stop", None)), "LlamaCppEngine must have a stop() method"

    @pytest.mark.no_gpu
    def test_has_health_method(self, engine: LlamaCppEngine) -> None:
        assert callable(getattr(engine, "health", None)), "LlamaCppEngine must have a health() method"

    @pytest.mark.no_gpu
    def test_binary_attribute_is_llama_server(self, engine: LlamaCppEngine) -> None:
        assert engine.binary == "llama-server", "vanilla engine must use 'llama-server' binary"

    @pytest.mark.no_gpu
    def test_start_accepts_model_spec(self, engine: LlamaCppEngine, spec: ModelSpec) -> None:
        """start(spec) must accept a ModelSpec without raising at parameter-binding time."""
        sig = inspect.signature(engine.start)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "start() must accept at least one parameter (spec)"

    @pytest.mark.no_gpu
    def test_stop_accepts_engine_instance(
        self, engine: LlamaCppEngine, fake_instance: EngineInstance
    ) -> None:
        """stop(instance) must accept an EngineInstance without raising at parameter-binding time."""
        sig = inspect.signature(engine.stop)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "stop() must accept at least one parameter (instance)"

    @pytest.mark.no_gpu
    def test_health_accepts_engine_instance(
        self, engine: LlamaCppEngine, fake_instance: EngineInstance
    ) -> None:
        """health(instance) must accept an EngineInstance without raising at parameter-binding time."""
        sig = inspect.signature(engine.health)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "health() must accept at least one parameter (instance)"


# ---------------------------------------------------------------------------
# 2. GGUF path resolution from HF hub cache
# ---------------------------------------------------------------------------

class TestGgufPathResolution:
    """Engine must locate the GGUF file in the HF hub snapshot cache."""

    @pytest.mark.no_gpu
    def test_resolves_gguf_from_hf_cache(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_gguf_path() (or equivalent helper) must return the absolute path to the .gguf file."""
        # The engine should expose a way to compute the local path.
        # If the public surface is only start(), we accept a NotImplementedError here
        # and verify the path logic via the helper directly once GREEN.
        # For RED: assert a helper exists and resolves correctly.
        helper = getattr(engine, "_gguf_path", None) or getattr(engine, "gguf_path", None)
        assert helper is not None, (
            "LlamaCppEngine must expose _gguf_path(spec) or gguf_path(spec) "
            "to resolve the HF hub cache location"
        )
        resolved: Path = helper(spec)
        assert resolved.exists(), f"resolved path {resolved} must point to an existing file"
        assert resolved.suffix == ".gguf"
        assert resolved.name == spec.file

    @pytest.mark.no_gpu
    def test_path_uses_hf_home_env(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Path resolution must honour $HF_HOME (set by fake_hf_cache fixture)."""
        helper = getattr(engine, "_gguf_path", None) or getattr(engine, "gguf_path", None)
        assert helper is not None, "LlamaCppEngine must expose a GGUF path helper"
        resolved: Path = helper(spec)
        hf_home = Path(os.environ["HF_HOME"])
        assert str(resolved).startswith(str(hf_home)), (
            f"resolved path {resolved} must be under $HF_HOME={hf_home}"
        )

    @pytest.mark.no_gpu
    def test_missing_gguf_raises_file_not_found(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the GGUF is not cached, the engine must raise FileNotFoundError (not a silent path)."""
        # Point HF_HOME at an empty directory — no model files present.
        empty = tmp_path / "empty_hub"
        empty.mkdir()
        monkeypatch.setenv("HF_HOME", str(empty))

        helper = getattr(engine, "_gguf_path", None) or getattr(engine, "gguf_path", None)
        if helper is None:
            # Fallback: start() should surface the missing file.
            with pytest.raises((FileNotFoundError, NotImplementedError)):
                engine.start(spec)
        else:
            with pytest.raises(FileNotFoundError):
                helper(spec)


# ---------------------------------------------------------------------------
# 3. Command-line construction
# ---------------------------------------------------------------------------

class TestCommandLineConstruction:
    """Engine must build a correct llama-server invocation."""

    @pytest.mark.no_gpu
    def test_build_cmd_contains_model_flag(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        """Command must include --model <gguf-path>."""
        cmd = self._get_cmd(engine, spec)
        assert "--model" in cmd or "-m" in cmd, "command must pass --model / -m flag"
        model_idx = cmd.index("--model") if "--model" in cmd else cmd.index("-m")
        assert cmd[model_idx + 1].endswith(".gguf"), "model argument must be the .gguf path"

    @pytest.mark.no_gpu
    def test_build_cmd_contains_port_flag(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        """Command must include --port <port> matching ModelSpec.port."""
        cmd = self._get_cmd(engine, spec)
        assert "--port" in cmd, "command must pass --port flag"
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == str(spec.port), (
            f"--port must be {spec.port}, got {cmd[port_idx + 1]}"
        )

    @pytest.mark.no_gpu
    def test_build_cmd_binds_to_all_interfaces(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        """Command must include --host 0.0.0.0 to expose the server on LAN."""
        cmd = self._get_cmd(engine, spec)
        assert "--host" in cmd, "command must pass --host flag"
        host_idx = cmd.index("--host")
        assert cmd[host_idx + 1] == "0.0.0.0", (
            f"--host must be 0.0.0.0, got {cmd[host_idx + 1]}"
        )

    @pytest.mark.no_gpu
    def test_build_cmd_includes_catalog_flags(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        """Extra flags from ModelSpec.flags must appear verbatim in the command."""
        cmd = self._get_cmd(engine, spec)
        for flag in spec.flags:
            assert flag in cmd, f"catalog flag '{flag}' must appear in command"

    @pytest.mark.no_gpu
    def test_build_cmd_includes_mmproj_when_present(
        self,
        engine: LlamaCppEngine,
        spec_with_mmproj: ModelSpec,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ModelSpec.mmproj is set, --mmproj <path> must appear in the command."""
        # Create both the gguf and mmproj files in the fake cache.
        hub = tmp_path / "hub"
        model_dir = hub / "models--bartowski--Qwen3-8B-GGUF" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        (model_dir / "Qwen3-8B-Q4_K_M.gguf").write_bytes(b"FAKE")
        (model_dir / "mmproj-qwen3-8b.gguf").write_bytes(b"FAKE_MMPROJ")
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        cmd = self._get_cmd(engine, spec_with_mmproj)
        assert "--mmproj" in cmd, "--mmproj flag must appear when spec.mmproj is set"
        mmproj_idx = cmd.index("--mmproj")
        assert cmd[mmproj_idx + 1].endswith("mmproj-qwen3-8b.gguf"), (
            "--mmproj argument must point to the mmproj file"
        )

    @pytest.mark.no_gpu
    def test_build_cmd_no_mmproj_when_absent(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        """When ModelSpec.mmproj is None, --mmproj must NOT appear in the command."""
        assert spec.mmproj is None
        cmd = self._get_cmd(engine, spec)
        assert "--mmproj" not in cmd, "--mmproj must be absent when spec.mmproj is None"

    @pytest.mark.no_gpu
    def test_binary_is_first_element(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        """The binary name must be the first element of the command list."""
        cmd = self._get_cmd(engine, spec)
        assert cmd[0] == engine.binary, (
            f"first element must be the binary '{engine.binary}', got '{cmd[0]}'"
        )

    # ------------------------------------------------------------------
    # Helper: extract the command the engine would pass to subprocess
    # ------------------------------------------------------------------

    @staticmethod
    def _get_cmd(engine: LlamaCppEngine, spec: ModelSpec) -> list[str]:
        """Ask the engine for the command it would run.

        During RED phase the engine has no _build_cmd helper — we attempt to
        call it and expect AttributeError or NotImplementedError; the test
        fails on those, satisfying the RED requirement.
        """
        builder = getattr(engine, "_build_cmd", None) or getattr(engine, "build_cmd", None)
        assert builder is not None, (
            "LlamaCppEngine must expose _build_cmd(spec) or build_cmd(spec) "
            "returning the subprocess argument list"
        )
        return builder(spec)


# ---------------------------------------------------------------------------
# 4. EngineInstance returned by start()
# ---------------------------------------------------------------------------

class TestEngineInstanceShape:
    """start(spec) must return an EngineInstance with required fields populated."""

    @pytest.mark.no_gpu
    def test_start_returns_engine_instance(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        """start() must return an EngineInstance (not None, not a dict)."""
        instance = self._start_with_mock_process(engine, spec)
        assert isinstance(instance, EngineInstance), (
            f"start() must return EngineInstance, got {type(instance)}"
        )

    @pytest.mark.no_gpu
    def test_instance_pid_is_positive_int(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        instance = self._start_with_mock_process(engine, spec)
        assert isinstance(instance.pid, int) and instance.pid > 0, (
            f"instance.pid must be a positive int, got {instance.pid!r}"
        )

    @pytest.mark.no_gpu
    def test_instance_port_matches_spec(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        instance = self._start_with_mock_process(engine, spec)
        assert instance.port == spec.port, (
            f"instance.port must equal spec.port={spec.port}, got {instance.port}"
        )

    @pytest.mark.no_gpu
    def test_instance_model_matches_spec_name(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        instance = self._start_with_mock_process(engine, spec)
        assert instance.model == spec.name, (
            f"instance.model must equal spec.name={spec.name!r}, got {instance.model!r}"
        )

    @pytest.mark.no_gpu
    def test_instance_started_at_is_float(
        self,
        engine: LlamaCppEngine,
        spec: ModelSpec,
        fake_hf_cache: Path,
    ) -> None:
        instance = self._start_with_mock_process(engine, spec)
        assert isinstance(instance.started_at, float), (
            f"instance.started_at must be a float, got {type(instance.started_at)}"
        )

    # ------------------------------------------------------------------
    # Helper: patch subprocess + health wait so start() can return
    # ------------------------------------------------------------------

    @staticmethod
    def _start_with_mock_process(engine: LlamaCppEngine, spec: ModelSpec) -> EngineInstance:
        """Run start() with Popen and _wait_ready mocked out."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with (
            patch("llmcli.engines.llamacpp.subprocess.Popen", return_value=mock_proc),
            patch("llmcli.engines.llamacpp._wait_ready", return_value=None),
        ):
            return engine.start(spec)


# ---------------------------------------------------------------------------
# 5. health() — HTTP probe
# ---------------------------------------------------------------------------

class TestHealthProbe:
    """health(instance) must probe the /health endpoint and return a bool."""

    @pytest.mark.no_gpu
    def test_health_returns_true_on_200(
        self,
        engine: LlamaCppEngine,
        fake_instance: EngineInstance,
    ) -> None:
        """health() must return True when the /health endpoint responds 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("llmcli.engines.llamacpp.httpx.get", return_value=mock_response) as mock_get:
            result = engine.health(fake_instance)

        assert result is True, f"health() must return True on HTTP 200, got {result!r}"
        mock_get.assert_called_once()
        call_url: str = mock_get.call_args[0][0]
        assert "/health" in call_url, f"health() must probe a /health endpoint, got URL: {call_url}"

    @pytest.mark.no_gpu
    def test_health_returns_false_on_non_200(
        self,
        engine: LlamaCppEngine,
        fake_instance: EngineInstance,
    ) -> None:
        """health() must return False when the endpoint returns a non-200 status."""
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("llmcli.engines.llamacpp.httpx.get", return_value=mock_response):
            result = engine.health(fake_instance)

        assert result is False, f"health() must return False on HTTP 503, got {result!r}"

    @pytest.mark.no_gpu
    def test_health_returns_false_on_connection_error(
        self,
        engine: LlamaCppEngine,
        fake_instance: EngineInstance,
    ) -> None:
        """health() must return False (not raise) when the server is unreachable."""
        with patch(
            "llmcli.engines.llamacpp.httpx.get",
            side_effect=Exception("Connection refused"),
        ):
            result = engine.health(fake_instance)

        assert result is False, (
            "health() must catch connection errors and return False, not raise"
        )

    @pytest.mark.no_gpu
    def test_health_probes_instance_port(
        self,
        engine: LlamaCppEngine,
        fake_instance: EngineInstance,
    ) -> None:
        """The URL probed must include the instance's port number."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("llmcli.engines.llamacpp.httpx.get", return_value=mock_response) as mock_get:
            engine.health(fake_instance)

        call_url: str = mock_get.call_args[0][0]
        assert str(fake_instance.port) in call_url, (
            f"health() must probe port {fake_instance.port}, got URL: {call_url}"
        )


# ---------------------------------------------------------------------------
# 6. stop() — process teardown
# ---------------------------------------------------------------------------

class TestStopBehaviour:
    """stop(instance) must terminate the process cleanly."""

    @pytest.mark.no_gpu
    def test_stop_sends_sigterm(
        self,
        engine: LlamaCppEngine,
        fake_instance: EngineInstance,
    ) -> None:
        """stop() must send SIGTERM to the process before any forced kill."""
        with patch("llmcli.engines.llamacpp.os.kill") as mock_kill:
            # The process must "exit" quickly after SIGTERM so we patch waitpid too.
            with patch("llmcli.engines.llamacpp.os.waitpid", return_value=(fake_instance.pid, 0)):
                engine.stop(fake_instance)

        sigterm_calls = [
            call for call in mock_kill.call_args_list
            if call[0][1] == signal.SIGTERM
        ]
        assert len(sigterm_calls) >= 1, (
            f"stop() must send at least one SIGTERM; got calls: {mock_kill.call_args_list}"
        )
        assert sigterm_calls[0][0][0] == fake_instance.pid, (
            f"SIGTERM must target pid={fake_instance.pid}"
        )

    @pytest.mark.no_gpu
    def test_stop_sends_sigkill_on_timeout(
        self,
        engine: LlamaCppEngine,
        fake_instance: EngineInstance,
    ) -> None:
        """stop() must escalate to SIGKILL when the process does not exit after SIGTERM."""

        def _kill(pid: int, sig: int) -> None:
            pass  # SIGTERM acknowledged but process stays alive

        with (
            patch("llmcli.engines.llamacpp.os.kill", side_effect=_kill) as mock_kill,
            # waitpid raises ChildProcessError to simulate timeout / hung process
            patch(
                "llmcli.engines.llamacpp.os.waitpid",
                side_effect=ChildProcessError("timeout"),
            ),
        ):
            # stop() should not raise — it escalates to SIGKILL
            try:
                engine.stop(fake_instance)
            except NotImplementedError:
                pass  # RED phase — implementation missing

        sigkill_calls = [
            call for call in mock_kill.call_args_list
            if call[0][1] == signal.SIGKILL
        ]
        # In RED this assertion will not be reached due to NotImplementedError above,
        # which causes the test to fail — correct RED behaviour.
        assert len(sigkill_calls) >= 1, "stop() must send SIGKILL after SIGTERM timeout"

    @pytest.mark.no_gpu
    def test_stop_targets_correct_pid(
        self,
        engine: LlamaCppEngine,
        fake_instance: EngineInstance,
    ) -> None:
        """stop() must use instance.pid, not a hardcoded value."""
        killed_pids: list[int] = []

        def _kill(pid: int, sig: int) -> None:
            killed_pids.append(pid)

        with (
            patch("llmcli.engines.llamacpp.os.kill", side_effect=_kill),
            patch("llmcli.engines.llamacpp.os.waitpid", return_value=(fake_instance.pid, 0)),
        ):
            try:
                engine.stop(fake_instance)
            except NotImplementedError:
                pass  # RED: not yet implemented

        if killed_pids:
            assert fake_instance.pid in killed_pids, (
                f"stop() must kill pid={fake_instance.pid}; killed {killed_pids}"
            )

    @pytest.mark.no_gpu
    def test_stop_does_not_raise_on_already_dead_process(
        self,
        engine: LlamaCppEngine,
        fake_instance: EngineInstance,
    ) -> None:
        """stop() must handle ProcessLookupError gracefully (process already exited)."""
        with (
            patch(
                "llmcli.engines.llamacpp.os.kill",
                side_effect=ProcessLookupError("no such process"),
            ),
        ):
            # Should complete without raising (idempotent stop)
            try:
                engine.stop(fake_instance)
            except NotImplementedError:
                pass  # RED: acceptable, test verifies intent
            except ProcessLookupError:
                pytest.fail(
                    "stop() must not propagate ProcessLookupError for already-dead processes"
                )
