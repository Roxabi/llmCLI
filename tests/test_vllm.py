"""RED-phase tests for VLLMEngine (issue #13).

All tests assert behaviour expected from the GREEN implementation.
Against the current NotImplementedError scaffold they MUST fail.

Test categories:
- Protocol conformance (method signatures)
- Command-line construction (_build_cmd)
- EngineInstance shape returned by start()
- stop() process-group teardown (os.killpg / os.getpgid)
- health() HTTP probe (httpx mocked)
- Import guard (vllm package availability)
- Daemon dispatch routing (engine="vllm" → VLLMEngine)

Markers:
  no_gpu  — CI-safe; no real binary, no GPU required
  gpu     — requires real vllm binary + GPU; skipped in CI
"""
from __future__ import annotations

import inspect
import signal
from unittest.mock import MagicMock, patch

import pytest

from llmcli.config import ModelSpec
from llmcli.engine import EngineInstance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vllm_spec() -> ModelSpec:
    return ModelSpec(
        name="qwen3-27b-nvfp4",
        engine="vllm",
        repo="kaitchup/Qwen3.6-27B-autoround-nvfp4-linearattn-BF16",
        port=8093,
        vram_gib=15.0,
        flags=["--max-model-len", "32768", "--reasoning-parser", "qwen3"],
    )


@pytest.fixture()
def vllm_spec_no_flags() -> ModelSpec:
    return ModelSpec(
        name="qwen3-4b-nvfp4",
        engine="vllm",
        repo="kaitchup/Qwen3-4B-autoround-nvfp4",
        port=8094,
        vram_gib=6.0,
        flags=[],
    )


@pytest.fixture()
def fake_instance() -> EngineInstance:
    return EngineInstance(pid=12345, port=8093, model_name="qwen3-27b-nvfp4", started_at=1_700_000_000.0)


@pytest.fixture()
def engine():
    from llmcli.engines.vllm import VLLMEngine
    return VLLMEngine()


# ---------------------------------------------------------------------------
# 1. Protocol conformance
# ---------------------------------------------------------------------------


class TestVLLMProtocolConformance:
    """VLLMEngine must implement the Engine Protocol."""

    @pytest.mark.no_gpu
    def test_has_start_method(self, engine) -> None:
        assert callable(getattr(engine, "start", None)), "VLLMEngine must have a start() method"

    @pytest.mark.no_gpu
    def test_has_stop_method(self, engine) -> None:
        assert callable(getattr(engine, "stop", None)), "VLLMEngine must have a stop() method"

    @pytest.mark.no_gpu
    def test_has_health_method(self, engine) -> None:
        assert callable(getattr(engine, "health", None)), "VLLMEngine must have a health() method"

    @pytest.mark.no_gpu
    def test_start_accepts_model_spec(self, engine, vllm_spec: ModelSpec) -> None:
        sig = inspect.signature(engine.start)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "start() must accept at least one parameter (spec)"

    @pytest.mark.no_gpu
    def test_stop_accepts_engine_instance(self, engine, fake_instance: EngineInstance) -> None:
        sig = inspect.signature(engine.stop)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "stop() must accept at least one parameter (instance)"

    @pytest.mark.no_gpu
    def test_health_accepts_engine_instance(self, engine, fake_instance: EngineInstance) -> None:
        sig = inspect.signature(engine.health)
        params = list(sig.parameters.keys())
        assert len(params) >= 1, "health() must accept at least one parameter (instance)"


# ---------------------------------------------------------------------------
# 2. Command-line construction
# ---------------------------------------------------------------------------


class TestVLLMCommandBuild:
    """_build_cmd(spec) must produce a correctly ordered argv list."""

    @pytest.mark.no_gpu
    def test_cmd_starts_with_vllm_serve(self, engine, vllm_spec: ModelSpec) -> None:
        # Arrange / Act
        cmd = engine._build_cmd(vllm_spec)
        # Assert
        assert cmd[0] == "vllm", f"first element must be 'vllm', got {cmd[0]!r}"
        assert cmd[1] == "serve", f"second element must be 'serve', got {cmd[1]!r}"

    @pytest.mark.no_gpu
    def test_cmd_contains_repo(self, engine, vllm_spec: ModelSpec) -> None:
        cmd = engine._build_cmd(vllm_spec)
        assert vllm_spec.repo in cmd, f"repo '{vllm_spec.repo}' must appear in command"
        assert cmd[2] == vllm_spec.repo, (
            f"repo must be 3rd element (after 'vllm serve'), got cmd[2]={cmd[2]!r}"
        )

    @pytest.mark.no_gpu
    def test_cmd_contains_port(self, engine, vllm_spec: ModelSpec) -> None:
        cmd = engine._build_cmd(vllm_spec)
        assert "--port" in cmd, "command must include --port flag"
        port_idx = cmd.index("--port")
        assert cmd[port_idx + 1] == str(vllm_spec.port), (
            f"--port value must be {vllm_spec.port!r}, got {cmd[port_idx + 1]!r}"
        )

    @pytest.mark.no_gpu
    def test_cmd_binds_all_interfaces(self, engine, vllm_spec: ModelSpec) -> None:
        cmd = engine._build_cmd(vllm_spec)
        assert "--host" in cmd, "command must include --host flag"
        host_idx = cmd.index("--host")
        assert cmd[host_idx + 1] == "0.0.0.0", (
            f"--host must be '0.0.0.0', got {cmd[host_idx + 1]!r}"
        )

    @pytest.mark.no_gpu
    def test_cmd_ordering_port_before_host(self, engine, vllm_spec: ModelSpec) -> None:
        """Spec: argv order is --port before --host 0.0.0.0."""
        cmd = engine._build_cmd(vllm_spec)
        port_idx = cmd.index("--port")
        host_idx = cmd.index("--host")
        assert port_idx < host_idx, (
            f"--port (idx {port_idx}) must come before --host (idx {host_idx})"
        )

    @pytest.mark.no_gpu
    def test_cmd_appends_flags_after_host(self, engine, vllm_spec: ModelSpec) -> None:
        """spec.flags must appear after --host 0.0.0.0."""
        cmd = engine._build_cmd(vllm_spec)
        host_idx = cmd.index("--host")
        # Flags start after "--host" "0.0.0.0"
        tail = cmd[host_idx + 2:]
        for flag in vllm_spec.flags:
            assert flag in tail, (
                f"catalog flag '{flag}' must appear after '--host 0.0.0.0'; tail={tail}"
            )

    @pytest.mark.no_gpu
    def test_cmd_no_extra_args_when_empty_flags(
        self, engine, vllm_spec_no_flags: ModelSpec
    ) -> None:
        """Empty flags list must not add any extra elements beyond the core argv."""
        cmd = engine._build_cmd(vllm_spec_no_flags)
        # Core argv: ["vllm", "serve", repo, "--port", port, "--host", "0.0.0.0"]
        assert len(cmd) == 7, (
            f"with empty flags, command must have exactly 7 elements, got {len(cmd)}: {cmd}"
        )


# ---------------------------------------------------------------------------
# 3. start()
# ---------------------------------------------------------------------------


class TestVLLMStart:
    """start(spec) must spawn via Popen (start_new_session=True) and return EngineInstance."""

    @pytest.mark.no_gpu
    def test_start_returns_engine_instance(self, engine, vllm_spec: ModelSpec) -> None:
        # Arrange
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with (
            patch("llmcli.engines.vllm.shutil.which", return_value="/usr/bin/vllm"),
            patch.dict("sys.modules", {"vllm": MagicMock()}),
            patch("llmcli.engines.vllm.subprocess.Popen", return_value=mock_proc),
            patch("llmcli.engines.vllm._wait_ready", return_value=None),
        ):
            # Act
            result = engine.start(vllm_spec)

        # Assert
        assert isinstance(result, EngineInstance), (
            f"start() must return EngineInstance, got {type(result)}"
        )

    @pytest.mark.no_gpu
    def test_start_instance_pid_matches_process(self, engine, vllm_spec: ModelSpec) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with (
            patch("llmcli.engines.vllm.shutil.which", return_value="/usr/bin/vllm"),
            patch.dict("sys.modules", {"vllm": MagicMock()}),
            patch("llmcli.engines.vllm.subprocess.Popen", return_value=mock_proc),
            patch("llmcli.engines.vllm._wait_ready", return_value=None),
        ):
            result = engine.start(vllm_spec)

        assert result.pid == 99999, f"instance.pid must equal process pid 99999, got {result.pid}"

    @pytest.mark.no_gpu
    def test_start_instance_port_matches_spec(self, engine, vllm_spec: ModelSpec) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with (
            patch("llmcli.engines.vllm.shutil.which", return_value="/usr/bin/vllm"),
            patch.dict("sys.modules", {"vllm": MagicMock()}),
            patch("llmcli.engines.vllm.subprocess.Popen", return_value=mock_proc),
            patch("llmcli.engines.vllm._wait_ready", return_value=None),
        ):
            result = engine.start(vllm_spec)

        assert result.port == vllm_spec.port, (
            f"instance.port must equal spec.port={vllm_spec.port}, got {result.port}"
        )

    @pytest.mark.no_gpu
    def test_start_instance_model_name_matches_spec(self, engine, vllm_spec: ModelSpec) -> None:
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with (
            patch("llmcli.engines.vllm.shutil.which", return_value="/usr/bin/vllm"),
            patch.dict("sys.modules", {"vllm": MagicMock()}),
            patch("llmcli.engines.vllm.subprocess.Popen", return_value=mock_proc),
            patch("llmcli.engines.vllm._wait_ready", return_value=None),
        ):
            result = engine.start(vllm_spec)

        assert result.model_name == vllm_spec.name, (
            f"instance.model_name must equal spec.name={vllm_spec.name!r}, got {result.model_name!r}"
        )

    @pytest.mark.no_gpu
    def test_start_uses_start_new_session(self, engine, vllm_spec: ModelSpec) -> None:
        """Popen must be called with start_new_session=True for pgid-based stop."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with (
            patch("llmcli.engines.vllm.shutil.which", return_value="/usr/bin/vllm"),
            patch.dict("sys.modules", {"vllm": MagicMock()}),
            patch("llmcli.engines.vllm.subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch("llmcli.engines.vllm._wait_ready", return_value=None),
        ):
            engine.start(vllm_spec)

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs.get("start_new_session") is True, (
            f"Popen must be called with start_new_session=True; got kwargs={call_kwargs}"
        )

    @pytest.mark.no_gpu
    def test_start_propagates_wait_ready_error(self, engine, vllm_spec: ModelSpec) -> None:
        """If _wait_ready raises RuntimeError, start() must propagate it."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        with (
            patch("llmcli.engines.vllm.shutil.which", return_value="/usr/bin/vllm"),
            patch.dict("sys.modules", {"vllm": MagicMock()}),
            patch("llmcli.engines.vllm.subprocess.Popen", return_value=mock_proc),
            patch(
                "llmcli.engines.vllm._wait_ready",
                side_effect=RuntimeError("exited"),
            ),
        ):
            with pytest.raises(RuntimeError, match="exited"):
                engine.start(vllm_spec)


# ---------------------------------------------------------------------------
# 4. stop()
# ---------------------------------------------------------------------------


class TestVLLMStop:
    """stop(instance) must use os.killpg/os.getpgid (not os.kill) for process-group teardown."""

    @pytest.mark.no_gpu
    def test_stop_sends_sigterm_via_killpg(
        self, engine, fake_instance: EngineInstance
    ) -> None:
        """stop() must call os.killpg(os.getpgid(pid), SIGTERM)."""
        with (
            patch("llmcli.engines.vllm.os.getpgid", return_value=55555) as mock_getpgid,
            patch("llmcli.engines.vllm.os.killpg") as mock_killpg,
            patch("llmcli.engines.vllm.os.waitpid", return_value=(12345, 0)),
        ):
            # Simulate process dying after SIGTERM (getpgid raises on second poll)
            mock_getpgid.side_effect = [55555, ProcessLookupError("gone")]
            engine.stop(fake_instance)

        sigterm_calls = [
            c for c in mock_killpg.call_args_list
            if c[0][1] == signal.SIGTERM
        ]
        assert len(sigterm_calls) >= 1, (
            f"stop() must send at least one SIGTERM via killpg; calls={mock_killpg.call_args_list}"
        )
        assert sigterm_calls[0][0][0] == 55555, (
            f"SIGTERM must target pgid=55555; got {sigterm_calls[0][0][0]}"
        )

    @pytest.mark.no_gpu
    def test_stop_escalates_to_sigkill(
        self, engine, fake_instance: EngineInstance
    ) -> None:
        """stop() must escalate to SIGKILL when waitpid raises ChildProcessError after SIGTERM."""
        waitpid_calls = 0

        def fake_waitpid(pid, flags):
            nonlocal waitpid_calls
            waitpid_calls += 1
            if waitpid_calls == 1:
                raise ChildProcessError("process did not exit")
            return (pid, 0)

        with (
            patch("llmcli.engines.vllm.os.getpgid", return_value=55555),
            patch("llmcli.engines.vllm.os.killpg") as mock_killpg,
            patch("llmcli.engines.vllm.os.waitpid", side_effect=fake_waitpid),
        ):
            engine.stop(fake_instance)

        sigkill_calls = [
            c for c in mock_killpg.call_args_list
            if c[0][1] == signal.SIGKILL
        ]
        assert len(sigkill_calls) >= 1, (
            f"stop() must escalate to SIGKILL; calls={mock_killpg.call_args_list}"
        )

    @pytest.mark.no_gpu
    def test_stop_idempotent_when_process_gone(
        self, engine, fake_instance: EngineInstance
    ) -> None:
        """stop() must return without raising when os.getpgid raises ProcessLookupError."""
        with patch(
            "llmcli.engines.vllm.os.getpgid",
            side_effect=ProcessLookupError("no such process"),
        ):
            try:
                engine.stop(fake_instance)
            except ProcessLookupError:
                pytest.fail(
                    "stop() must not propagate ProcessLookupError for already-dead processes"
                )

    @pytest.mark.no_gpu
    def test_stop_uses_getpgid_not_os_kill(
        self, engine, fake_instance: EngineInstance
    ) -> None:
        """stop() must use os.killpg (process-group) rather than os.kill (single pid)."""
        with (
            patch("llmcli.engines.vllm.os.getpgid", return_value=55555),
            patch("llmcli.engines.vllm.os.killpg"),
            patch("llmcli.engines.vllm.os.waitpid", return_value=(12345, 0)),
            patch("llmcli.engines.vllm.os.kill") as mock_os_kill,
        ):
            engine.stop(fake_instance)

        assert mock_os_kill.call_count == 0, (
            "stop() must NOT use os.kill — use os.killpg for process-group signals"
        )


# ---------------------------------------------------------------------------
# 5. health()
# ---------------------------------------------------------------------------


class TestVLLMHealth:
    """health(instance) must probe /health and return a bool."""

    @pytest.mark.no_gpu
    def test_health_returns_true_on_200(
        self, engine, fake_instance: EngineInstance
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("llmcli.engines.vllm.httpx.get", return_value=mock_response) as mock_get:
            result = engine.health(fake_instance)

        assert result is True, f"health() must return True on HTTP 200, got {result!r}"
        mock_get.assert_called_once()
        call_url: str = mock_get.call_args[0][0]
        assert "/health" in call_url, (
            f"health() must probe a /health endpoint, got URL: {call_url}"
        )

    @pytest.mark.no_gpu
    def test_health_returns_false_on_503(
        self, engine, fake_instance: EngineInstance
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("llmcli.engines.vllm.httpx.get", return_value=mock_response):
            result = engine.health(fake_instance)

        assert result is False, f"health() must return False on HTTP 503, got {result!r}"

    @pytest.mark.no_gpu
    def test_health_returns_false_on_connection_error(
        self, engine, fake_instance: EngineInstance
    ) -> None:
        with patch(
            "llmcli.engines.vllm.httpx.get",
            side_effect=Exception("Connection refused"),
        ):
            result = engine.health(fake_instance)

        assert result is False, (
            "health() must catch connection errors and return False, not raise"
        )

    @pytest.mark.no_gpu
    def test_health_probes_instance_port(
        self, engine, fake_instance: EngineInstance
    ) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("llmcli.engines.vllm.httpx.get", return_value=mock_response) as mock_get:
            engine.health(fake_instance)

        call_url: str = mock_get.call_args[0][0]
        assert str(fake_instance.port) in call_url, (
            f"health() must probe port {fake_instance.port}, got URL: {call_url}"
        )


# ---------------------------------------------------------------------------
# 6. Import guard
# ---------------------------------------------------------------------------


class TestVLLMImportGuard:
    """VLLMEngine must import cleanly without vllm installed; start() raises ImportError
    with a helpful install hint when vllm is genuinely unavailable at runtime."""

    @pytest.mark.no_gpu
    def test_import_succeeds_without_vllm_package(self) -> None:
        """Importing VLLMEngine must not require the vllm package at import time."""
        # The import at module level in the fixture already proves this if we got here.
        from llmcli.engines.vllm import VLLMEngine  # noqa: F401  # re-import is fine
        assert True, "Import of VLLMEngine must succeed without vllm installed"

    @pytest.mark.no_gpu
    def test_start_raises_import_error_when_vllm_missing(
        self, engine, vllm_spec: ModelSpec
    ) -> None:
        """When vllm package is unavailable, start() must raise ImportError with install hint."""
        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _fake_import(name, *args, **kwargs):
            if name == "vllm":
                raise ImportError("No module named 'vllm'")
            return original_import(name, *args, **kwargs)

        with (
            patch("llmcli.engines.vllm.shutil.which", return_value="/usr/bin/vllm"),
            patch("builtins.__import__", side_effect=_fake_import),
        ):
            with pytest.raises(ImportError, match="uv sync --group vllm"):
                engine.start(vllm_spec)


# ---------------------------------------------------------------------------
# 7. Daemon dispatch
# ---------------------------------------------------------------------------


class TestDaemonDispatchVLLM:
    """Daemon._engine_for_spec must route engine="vllm" to VLLMEngine."""

    @pytest.mark.no_gpu
    def test_engine_vllm_returns_vllm_engine(self, vllm_spec: ModelSpec) -> None:
        from llmcli.daemon import Daemon
        from llmcli.engines.vllm import VLLMEngine

        # Arrange
        daemon = Daemon()

        # Act
        result = daemon._engine_for_spec(vllm_spec)

        # Assert
        assert isinstance(result, VLLMEngine), (
            f"engine='vllm' must dispatch to VLLMEngine, got {type(result)}"
        )

    @pytest.mark.no_gpu
    def test_engine_llamacpp_returns_llamacpp_engine(self) -> None:
        from llmcli.daemon import Daemon
        from llmcli.engines.llamacpp import LlamaCppEngine

        spec = ModelSpec(
            name="qwen3-8b-q4",
            engine="llamacpp",
            repo="bartowski/Qwen3-8B-GGUF",
            file="Qwen3-8B-Q4_K_M.gguf",
            port=8091,
            vram_gib=5.5,
        )
        daemon = Daemon()
        result = daemon._engine_for_spec(spec)

        assert isinstance(result, LlamaCppEngine), (
            f"engine='llamacpp' must dispatch to LlamaCppEngine, got {type(result)}"
        )

    @pytest.mark.no_gpu
    def test_engine_llamacpp_tq3_returns_tq3_engine(self) -> None:
        from llmcli.daemon import Daemon
        from llmcli.engines.llamacpp_tq3 import LlamaCppTQ3Engine

        spec = ModelSpec(
            name="qwen3-8b-tq3",
            engine="llamacpp_tq3",
            repo="bartowski/Qwen3-8B-GGUF",
            file="Qwen3-8B-TQ3_K_M.gguf",
            port=8092,
            vram_gib=5.0,
        )
        daemon = Daemon()
        result = daemon._engine_for_spec(spec)

        assert isinstance(result, LlamaCppTQ3Engine), (
            f"engine='llamacpp_tq3' must dispatch to LlamaCppTQ3Engine, got {type(result)}"
        )


# ---------------------------------------------------------------------------
# 8. _wait_ready
# ---------------------------------------------------------------------------


class TestWaitReady:
    """Direct unit tests for the _wait_ready polling helper."""

    @pytest.mark.no_gpu
    def test_returns_immediately_on_2xx(self) -> None:
        """_wait_ready should return as soon as /health responds 2xx."""
        from llmcli.engines._common import _wait_ready

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # process still running
        mock_proc.stderr = None
        mock_response = MagicMock()
        mock_response.status_code = 200

        with (
            patch("llmcli.engines._common.httpx.get", return_value=mock_response),
            patch("llmcli.engines._common.time.sleep"),
            patch("llmcli.engines._common.time.monotonic", side_effect=[0.0, 0.1]),
        ):
            _wait_ready("http://localhost:8093/v1", mock_proc, timeout=10.0, engine_name="vllm serve")
        # Should complete without raising

    @pytest.mark.no_gpu
    def test_continues_on_503_then_succeeds_on_200(self) -> None:
        """_wait_ready must continue polling on 503 (warmup) until 2xx."""
        from llmcli.engines._common import _wait_ready

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr = None

        resp_503 = MagicMock()
        resp_503.status_code = 503
        resp_200 = MagicMock()
        resp_200.status_code = 200

        call_count = 0

        def fake_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return resp_200 if call_count >= 3 else resp_503

        with (
            patch("llmcli.engines._common.httpx.get", side_effect=fake_get),
            patch("llmcli.engines._common.time.sleep"),
            patch(
                "llmcli.engines._common.time.monotonic",
                side_effect=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            ),
        ):
            _wait_ready("http://localhost:8093/v1", mock_proc, timeout=60.0, engine_name="vllm serve")

        assert call_count >= 3, "Must poll at least 3 times (2× 503 + 1× 200)"

    @pytest.mark.no_gpu
    def test_raises_on_early_process_exit(self) -> None:
        """_wait_ready must raise RuntimeError with exit code when process exits early."""
        from llmcli.engines._common import _wait_ready

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # process exited immediately
        mock_proc.stderr = None
        mock_proc.wait.return_value = None

        with (
            patch("llmcli.engines._common.httpx.get"),
            patch("llmcli.engines._common.time.monotonic", side_effect=[0.0, 0.5]),
        ):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                _wait_ready("http://localhost:8093/v1", mock_proc, timeout=60.0, engine_name="vllm serve")

    @pytest.mark.no_gpu
    def test_raises_on_timeout(self) -> None:
        """_wait_ready must raise RuntimeError when the deadline passes."""
        from llmcli.engines._common import _wait_ready

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.stderr = None

        resp_503 = MagicMock()
        resp_503.status_code = 503

        with (
            patch("llmcli.engines._common.httpx.get", return_value=resp_503),
            patch("llmcli.engines._common.time.sleep"),
            # Deadline expires immediately on the second call
            patch("llmcli.engines._common.time.monotonic", side_effect=[0.0, 100.0]),
        ):
            with pytest.raises(RuntimeError, match="did not become ready"):
                _wait_ready("http://localhost:8093/v1", mock_proc, timeout=1.0, engine_name="vllm serve")
