"""Tests for fix #24 — daemon.serve(model_name) auto-loads model on startup.

Spec trace:
  #24: daemon.serve() was ignoring model_name parameter (noqa: ARG002).
  Fix: when model_name is non-empty, call _cmd_swap before the accept loop.

Cases:
  1. serve("qwen3-8b") with no engine running → model loaded before accept loop
  2. serve(None) → daemon starts with no model (existing default behaviour)
  3. serve("X") when engine already on "X" → no-op (same-model fast-path, no error)
  4. serve("unknown") → RuntimeError before binding socket (startup guard)

All tests are @pytest.mark.no_gpu — no real llama-server process is spawned.
"""

from __future__ import annotations

import textwrap
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llmcli import config
from llmcli.daemon import Daemon
from llmcli.engine import EngineInstance


# ---------------------------------------------------------------------------
# Shared helpers / fixtures (copied from test_swap.py style)
# ---------------------------------------------------------------------------

FAKE_TOML = textwrap.dedent("""\
    [host]
    bind             = "0.0.0.0"
    public_base_url  = "http://localhost"
    api_key_env      = "LLMCLI_API_KEY"
    default_model    = "model-a"
    vram_budget_gib  = 10.0

    [models.model-a]
    engine   = "llamacpp"
    repo     = "TestOrg/model-a-GGUF"
    file     = "model-a.gguf"
    port     = 8091
    vram_gib = 6.0
    flags    = ["-ngl", "99"]

    [models.model-b]
    engine   = "llamacpp"
    repo     = "TestOrg/model-b-GGUF"
    file     = "model-b.gguf"
    port     = 8092
    vram_gib = 5.0
    flags    = []
""")


@pytest.fixture()
def real_catalog(tmp_path: Path):
    toml_path = tmp_path / "llmcli.toml"
    toml_path.write_text(FAKE_TOML)
    return config.load(toml_path)


def _make_instance(model_name: str, port: int, pid: int = 1000) -> EngineInstance:
    return EngineInstance(pid=pid, port=port, model_name=model_name)


def _make_engine_mock(start_return: EngineInstance) -> MagicMock:
    engine = MagicMock()
    engine.start.return_value = start_return
    engine.stop.return_value = None
    engine.health.return_value = True
    return engine


def _daemon_with_engines(catalog, engine_map: dict[str, MagicMock]) -> Daemon:
    daemon = Daemon(catalog=catalog)
    daemon._engine_for_spec = lambda spec: engine_map[spec.name]  # type: ignore[method-assign]
    return daemon


def _wait_for_socket(sock_path: Path, timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sock_path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Daemon socket did not appear at {sock_path} within {timeout}s")


# ---------------------------------------------------------------------------
# Case 1: serve(model_name) loads model before the accept loop
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestServeWithModelName:
    """serve(model_name) must load the specified model before accepting connections."""

    def test_serve_with_model_name_calls_cmd_swap(
        self, real_catalog: config.Catalog, tmp_path: Path
    ) -> None:
        """serve('model-a') triggers _cmd_swap('model-a') before the accept loop."""
        inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock()})

        sock_path = tmp_path / "llmcli.sock"
        daemon.socket_path = sock_path

        # Patch _cmd_swap to capture the call and then let serve run normally
        # by returning OK (swap would start engine, but here we just verify the call).
        original_cmd_swap = daemon._cmd_swap
        swap_calls: list[str] = []

        def recording_swap(name: str) -> str:
            swap_calls.append(name)
            return original_cmd_swap(name)

        daemon._cmd_swap = recording_swap  # type: ignore[method-assign]

        t = threading.Thread(target=daemon.serve, args=("model-a",), daemon=True)
        t.start()
        _wait_for_socket(sock_path, timeout=5.0)

        assert "model-a" in swap_calls, (
            f"serve('model-a') must call _cmd_swap('model-a') before entering accept loop; "
            f"recorded calls: {swap_calls}"
        )

    def test_serve_with_model_name_registers_instance(
        self, real_catalog: config.Catalog, tmp_path: Path
    ) -> None:
        """After serve(model_name) starts, daemon.instances contains the loaded model."""
        inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock()})

        sock_path = tmp_path / "llmcli.sock"
        daemon.socket_path = sock_path

        t = threading.Thread(target=daemon.serve, args=("model-a",), daemon=True)
        t.start()
        _wait_for_socket(sock_path, timeout=5.0)

        assert "model-a" in daemon.instances, (
            f"daemon.instances must contain 'model-a' after serve('model-a'); "
            f"got keys: {list(daemon.instances)}"
        )

    def test_serve_with_model_name_engine_start_called(
        self, real_catalog: config.Catalog, tmp_path: Path
    ) -> None:
        """engine.start() is called exactly once when serve(model_name) loads the model."""
        inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock()})

        sock_path = tmp_path / "llmcli.sock"
        daemon.socket_path = sock_path

        t = threading.Thread(target=daemon.serve, args=("model-a",), daemon=True)
        t.start()
        _wait_for_socket(sock_path, timeout=5.0)

        engine_a.start.assert_called_once()


# ---------------------------------------------------------------------------
# Case 2: serve(None) starts daemon with no model loaded
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestServeWithoutModelName:
    """serve(None) must preserve existing behaviour: no model loaded on startup."""

    def test_serve_with_none_loads_no_model(
        self, real_catalog: config.Catalog, tmp_path: Path
    ) -> None:
        """serve(None) leaves daemon.instances empty — no auto-load."""
        inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock()})

        sock_path = tmp_path / "llmcli.sock"
        daemon.socket_path = sock_path

        t = threading.Thread(target=daemon.serve, args=(None,), daemon=True)
        t.start()
        _wait_for_socket(sock_path, timeout=5.0)

        assert not daemon.instances, (
            f"serve(None) must not load any model; got instances: {list(daemon.instances)}"
        )
        engine_a.start.assert_not_called()

    def test_serve_with_empty_string_loads_no_model(
        self, real_catalog: config.Catalog, tmp_path: Path
    ) -> None:
        """serve('') is equivalent to serve(None) — no auto-load."""
        inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock()})

        sock_path = tmp_path / "llmcli.sock"
        daemon.socket_path = sock_path

        t = threading.Thread(target=daemon.serve, args=("",), daemon=True)
        t.start()
        _wait_for_socket(sock_path, timeout=5.0)

        assert not daemon.instances, (
            f"serve('') must not load any model; got instances: {list(daemon.instances)}"
        )
        engine_a.start.assert_not_called()


# ---------------------------------------------------------------------------
# Case 3: serve("X") when engine already on "X" → no-op, no error
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestServeWithAlreadyLoadedModel:
    """serve(model_name) when model is already in instances uses the fast-path."""

    def test_serve_same_model_already_running_no_error(
        self, real_catalog: config.Catalog, tmp_path: Path
    ) -> None:
        """serve('model-a') when 'model-a' is pre-loaded does not raise and does not reload."""
        existing_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(existing_inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock()})
        # Pre-load the model to simulate "already running" state
        daemon.instances["model-a"] = existing_inst

        sock_path = tmp_path / "llmcli.sock"
        daemon.socket_path = sock_path

        t = threading.Thread(target=daemon.serve, args=("model-a",), daemon=True)
        t.start()
        _wait_for_socket(sock_path, timeout=5.0)

        # No stop or start should be called (same-model fast-path)
        engine_a.stop.assert_not_called()
        engine_a.start.assert_not_called()
        # Model must still be tracked
        assert "model-a" in daemon.instances

    def test_serve_same_model_socket_still_appears(
        self, real_catalog: config.Catalog, tmp_path: Path
    ) -> None:
        """Daemon socket appears even when same-model fast-path is taken."""
        existing_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(existing_inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock()})
        daemon.instances["model-a"] = existing_inst

        sock_path = tmp_path / "llmcli.sock"
        daemon.socket_path = sock_path

        t = threading.Thread(target=daemon.serve, args=("model-a",), daemon=True)
        t.start()

        # Must not raise — socket appears normally
        _wait_for_socket(sock_path, timeout=5.0)
        assert sock_path.exists()


# ---------------------------------------------------------------------------
# Case 4: serve with unknown model → RuntimeError before socket appears
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestServeWithUnknownModel:
    """serve(unknown_model) must raise RuntimeError before the accept loop."""

    def test_serve_unknown_model_raises_runtime_error(
        self, real_catalog: config.Catalog, tmp_path: Path
    ) -> None:
        """serve('does-not-exist') raises RuntimeError (propagated from _cmd_swap ERR)."""
        daemon = Daemon(catalog=real_catalog)

        sock_path = tmp_path / "llmcli.sock"
        daemon.socket_path = sock_path

        errors: list[Exception] = []

        def _run() -> None:
            try:
                daemon.serve("does-not-exist")
            except Exception as exc:
                errors.append(exc)

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=5.0)

        assert errors, "serve('does-not-exist') must raise an exception"
        assert isinstance(errors[0], RuntimeError), (
            f"Expected RuntimeError, got {type(errors[0])}: {errors[0]}"
        )
        assert "does-not-exist" in str(errors[0]) or "ERR" in str(errors[0]), (
            f"RuntimeError message must reference the failing model, got: {errors[0]}"
        )
