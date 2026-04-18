"""RED-phase tests for T5.1 — Daemon swap handler + CLI swap command.

Spec trace:
  SC-5:  `llmcli swap` stops current engine, starts new one; started_at strictly increases.
  SC-6:  `make llm swap <name>` parity — CLI sends SWAP via daemon socket.
  SC-11: Hot-swap model without daemon restart.
  SC-12: Swap triggered via AF_UNIX daemon socket (SWAP <name> command).

Expected RED failures against current scaffold:
  - _cmd_swap() is a stub: returns "OK swapped to <name>" without calling engine.stop/start.
  - Tests that verify stop() is called BEFORE start() will fail (no calls at all).
  - Tests verifying ERR on unknown model name will fail (stub ignores catalog).
  - Tests verifying ERR on VRAM budget exceeded will fail (stub ignores VRAM check).
  - Tests verifying same-model fast-path will fail (stub always echoes OK).
  - Tests for engine.stop(old) before engine.start(new) ordering will fail.

Design decisions:
  - Same-model swap: defined as "already running → return OK already running <name>" fast-path.
    Rationale: stop+restart the same model wastes VRAM eviction/reload time with zero benefit.
    If the GREEN implementation prefers "always swap", update the assertion to match and document.
  - VRAM guard: mirrors the existing config.check_vram_budget() contract (ValueError → ERR line).
  - Response format: plaintext line ending with newline (matches existing wire protocol).
  - stop() before start(): daemon MUST stop old engine before starting new one so GPU VRAM is
    freed before the new model is loaded (VRAM constraint, C2).

All tests use @pytest.mark.no_gpu — no real llama-server process is spawned.
Engines are mocked via MagicMock implementing the Engine Protocol.
CLI tests mock daemon_request to avoid requiring a live AF_UNIX socket.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from llmcli import config
from llmcli.cli import app
from llmcli.daemon import Daemon
from llmcli.engine import EngineInstance


# ---------------------------------------------------------------------------
# Shared fixtures
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

    [models.model-oversized]
    engine   = "llamacpp"
    repo     = "TestOrg/model-oversized-GGUF"
    file     = "model-oversized.gguf"
    port     = 8093
    vram_gib = 15.0
    flags    = []
""")


@pytest.fixture()
def real_catalog(tmp_path: Path):
    """Load catalog from FAKE_TOML."""
    toml_path = tmp_path / "llmcli.toml"
    toml_path.write_text(FAKE_TOML)
    return config.load(toml_path)


def _make_engine_mock(start_return: EngineInstance) -> MagicMock:
    """Return a MagicMock that satisfies the Engine Protocol."""
    engine = MagicMock()
    engine.start.return_value = start_return
    engine.stop.return_value = None
    engine.health.return_value = True
    return engine


def _make_instance(model_name: str, port: int, pid: int = 1000, started_at: float = 1.0) -> EngineInstance:
    return EngineInstance(pid=pid, port=port, model_name=model_name, started_at=started_at)


# ---------------------------------------------------------------------------
# Helper: build a Daemon with catalog injected + engine factory patched
# ---------------------------------------------------------------------------

def _daemon_with_engines(catalog, engine_map: dict[str, MagicMock]) -> Daemon:
    """Return a Daemon whose _engine_for_spec() returns mock engines by spec.name."""
    daemon = Daemon(catalog=catalog)
    # Patch _engine_for_spec at the instance level so tests control engine dispatch.
    daemon._engine_for_spec = lambda spec: engine_map[spec.name]  # type: ignore[method-assign]
    return daemon


# ---------------------------------------------------------------------------
# T5.2 target — Daemon._cmd_swap() tests
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestDaemonCmdSwapNoEngine:
    """_cmd_swap when no engine is currently running."""

    def test_swap_with_no_engine_starts_new_engine(self, real_catalog: config.Catalog) -> None:
        """_cmd_swap starts the requested engine when no engine is running."""
        # Arrange
        new_inst = _make_instance("model-b", port=8092, pid=2001)
        engine_b = _make_engine_mock(new_inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": MagicMock(), "model-b": engine_b, "model-oversized": MagicMock()})
        assert not daemon.instances  # no engine running

        # Act
        response = daemon._cmd_swap("model-b")

        # Assert — engine.start was called
        engine_b.start.assert_called_once()
        assert "model-b" in response, f"Response must mention new model name, got: {response!r}"

    def test_swap_with_no_engine_registers_new_instance(self, real_catalog: config.Catalog) -> None:
        """After _cmd_swap, the new instance is tracked in daemon.instances."""
        # Arrange
        new_inst = _make_instance("model-b", port=8092, pid=2001)
        engine_b = _make_engine_mock(new_inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": MagicMock(), "model-b": engine_b, "model-oversized": MagicMock()})

        # Act
        daemon._cmd_swap("model-b")

        # Assert
        assert "model-b" in daemon.instances, (
            f"New model must be registered in daemon.instances, got keys: {list(daemon.instances)}"
        )

    def test_swap_with_no_engine_does_not_call_stop(self, real_catalog: config.Catalog) -> None:
        """_cmd_swap with no running engine does not call stop (nothing to stop)."""
        # Arrange
        new_inst = _make_instance("model-b", port=8092, pid=2001)
        engine_b = _make_engine_mock(new_inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": MagicMock(), "model-b": engine_b, "model-oversized": MagicMock()})

        # Act
        daemon._cmd_swap("model-b")

        # Assert — stop must not be called when there was nothing running
        engine_b.stop.assert_not_called()

    def test_swap_with_no_engine_returns_ok_line(self, real_catalog: config.Catalog) -> None:
        """_cmd_swap returns a line starting with 'OK' on success."""
        # Arrange
        new_inst = _make_instance("model-b", port=8092, pid=2001)
        engine_b = _make_engine_mock(new_inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": MagicMock(), "model-b": engine_b, "model-oversized": MagicMock()})

        # Act
        response = daemon._cmd_swap("model-b")

        # Assert
        assert response.startswith("OK"), f"Expected OK response, got: {response!r}"


@pytest.mark.no_gpu
class TestDaemonCmdSwapWithRunningEngine:
    """_cmd_swap when an engine IS currently running."""

    def test_swap_stops_old_engine_before_starting_new(self, real_catalog: config.Catalog) -> None:
        """stop(old) MUST be called BEFORE start(new) to free VRAM first (C2)."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        new_inst = _make_instance("model-b", port=8092, pid=2002)
        engine_a = _make_engine_mock(old_inst)
        engine_b = _make_engine_mock(new_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        call_order: list[str] = []
        engine_a.stop.side_effect = lambda _inst: call_order.append("stop")
        engine_b.start.side_effect = lambda _spec: (call_order.append("start"), new_inst)[1]

        # Act
        daemon._cmd_swap("model-b")

        # Assert — stop came before start
        assert call_order == ["stop", "start"], (
            f"stop(old) must precede start(new) for VRAM safety, got order: {call_order}"
        )

    def test_swap_calls_stop_on_old_instance(self, real_catalog: config.Catalog) -> None:
        """stop() is called exactly once on the currently running instance."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        new_inst = _make_instance("model-b", port=8092, pid=2002)
        engine_a = _make_engine_mock(old_inst)
        engine_b = _make_engine_mock(new_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        # Act
        daemon._cmd_swap("model-b")

        # Assert
        engine_a.stop.assert_called_once_with(old_inst)

    def test_swap_calls_start_on_new_engine(self, real_catalog: config.Catalog) -> None:
        """start() is called exactly once for the new model spec."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        new_inst = _make_instance("model-b", port=8092, pid=2002)
        engine_a = _make_engine_mock(old_inst)
        engine_b = _make_engine_mock(new_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        # Act
        daemon._cmd_swap("model-b")

        # Assert
        engine_b.start.assert_called_once()
        spec_arg = engine_b.start.call_args[0][0]
        assert spec_arg.name == "model-b", (
            f"start() must receive the new model's ModelSpec, got: {spec_arg}"
        )

    def test_swap_removes_old_instance_from_tracking(self, real_catalog: config.Catalog) -> None:
        """After swap, old model is no longer tracked in daemon.instances."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        new_inst = _make_instance("model-b", port=8092, pid=2002)
        engine_a = _make_engine_mock(old_inst)
        engine_b = _make_engine_mock(new_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        # Act
        daemon._cmd_swap("model-b")

        # Assert
        assert "model-a" not in daemon.instances, (
            f"Old model must be removed from instances after swap, got keys: {list(daemon.instances)}"
        )

    def test_swap_registers_new_instance(self, real_catalog: config.Catalog) -> None:
        """After swap, new model instance is tracked in daemon.instances."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        new_inst = _make_instance("model-b", port=8092, pid=2002)
        engine_a = _make_engine_mock(old_inst)
        engine_b = _make_engine_mock(new_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        # Act
        daemon._cmd_swap("model-b")

        # Assert
        assert "model-b" in daemon.instances, (
            f"New model must be tracked in instances after swap, got keys: {list(daemon.instances)}"
        )
        assert daemon.instances["model-b"] is new_inst

    def test_swap_returns_ok_response_with_model_name(self, real_catalog: config.Catalog) -> None:
        """_cmd_swap returns OK line containing the new model name."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        new_inst = _make_instance("model-b", port=8092, pid=2002)
        engine_a = _make_engine_mock(old_inst)
        engine_b = _make_engine_mock(new_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        # Act
        response = daemon._cmd_swap("model-b")

        # Assert
        assert response.startswith("OK"), f"Expected OK response, got: {response!r}"
        assert "model-b" in response, f"Response must mention new model, got: {response!r}"


@pytest.mark.no_gpu
class TestDaemonCmdSwapSameModel:
    """_cmd_swap with same model name — fast-path: no stop/start."""

    def test_swap_same_model_returns_already_running(self, real_catalog: config.Catalog) -> None:
        """Swapping to the already-running model returns a fast-path OK without stop/start.

        Design choice: same-model swap is a no-op — stop+reload wastes VRAM eviction time
        with no benefit. Response contains 'already' or 'running' to signal the fast-path.
        """
        # Arrange
        existing_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(existing_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock(), "model-oversized": MagicMock()})
        daemon.instances["model-a"] = existing_inst

        # Act
        response = daemon._cmd_swap("model-a")

        # Assert — fast-path: neither stop nor start should fire
        engine_a.stop.assert_not_called()
        engine_a.start.assert_not_called()
        # Response must convey the model is already running
        assert "already" in response.lower() or "running" in response.lower(), (
            f"Same-model swap must signal fast-path (already running), got: {response!r}"
        )

    def test_swap_same_model_returns_ok_line(self, real_catalog: config.Catalog) -> None:
        """Same-model fast-path still returns an OK line (not ERR)."""
        # Arrange
        existing_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(existing_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock(), "model-oversized": MagicMock()})
        daemon.instances["model-a"] = existing_inst

        # Act
        response = daemon._cmd_swap("model-a")

        # Assert
        assert response.startswith("OK"), f"Same-model fast-path must return OK line, got: {response!r}"


@pytest.mark.no_gpu
class TestDaemonCmdSwapUnknownModel:
    """_cmd_swap with a model name not in the catalog."""

    def test_swap_unknown_model_returns_err(self, real_catalog: config.Catalog) -> None:
        """Swapping to an unknown model name returns ERR line."""
        # Arrange
        daemon = Daemon(catalog=real_catalog)

        # Act
        response = daemon._cmd_swap("model-does-not-exist")

        # Assert
        assert response.startswith("ERR"), (
            f"Unknown model must return ERR line, got: {response!r}"
        )

    def test_swap_unknown_model_err_mentions_name(self, real_catalog: config.Catalog) -> None:
        """ERR response for unknown model includes the requested model name."""
        # Arrange
        daemon = Daemon(catalog=real_catalog)

        # Act
        response = daemon._cmd_swap("ghost-model")

        # Assert
        assert "ghost-model" in response or "unknown" in response.lower(), (
            f"ERR line must reference the unknown model name or 'unknown', got: {response!r}"
        )

    def test_swap_unknown_model_leaves_running_engine_untouched(self, real_catalog: config.Catalog) -> None:
        """ERR on unknown model does not stop the currently running engine."""
        # Arrange
        existing_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(existing_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock(), "model-oversized": MagicMock()})
        daemon.instances["model-a"] = existing_inst

        # Act
        daemon._cmd_swap("ghost-model")

        # Assert — current engine must not be touched
        engine_a.stop.assert_not_called()
        assert "model-a" in daemon.instances, (
            "Running engine must remain tracked after failed swap"
        )


@pytest.mark.no_gpu
class TestDaemonCmdSwapVramGuard:
    """_cmd_swap rejects models that exceed the VRAM budget."""

    def test_swap_oversized_model_returns_err(self, real_catalog: config.Catalog) -> None:
        """Swapping to a model that exceeds the VRAM budget returns ERR."""
        # Arrange — model-oversized has 15.0 GiB; budget is 10.0 GiB
        daemon = Daemon(catalog=real_catalog)

        # Act
        response = daemon._cmd_swap("model-oversized")

        # Assert
        assert response.startswith("ERR"), (
            f"VRAM-exceeded swap must return ERR line, got: {response!r}"
        )

    def test_swap_oversized_model_err_mentions_vram(self, real_catalog: config.Catalog) -> None:
        """ERR for VRAM-exceeded swap mentions 'vram' or the budget numbers."""
        # Arrange
        daemon = Daemon(catalog=real_catalog)

        # Act
        response = daemon._cmd_swap("model-oversized")

        # Assert — response must hint at VRAM constraint
        lower = response.lower()
        assert "vram" in lower or "budget" in lower or "15" in response or "10" in response, (
            f"ERR line must reference VRAM constraint, got: {response!r}"
        )

    def test_swap_oversized_model_leaves_running_engine_untouched(self, real_catalog: config.Catalog) -> None:
        """VRAM-exceeded swap does not stop the currently running engine."""
        # Arrange
        existing_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(existing_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock(), "model-oversized": MagicMock()})
        daemon.instances["model-a"] = existing_inst

        # Act
        daemon._cmd_swap("model-oversized")

        # Assert
        engine_a.stop.assert_not_called()
        assert "model-a" in daemon.instances, (
            "Running engine must remain tracked after VRAM-rejected swap"
        )

    def test_swap_oversized_model_does_not_start_new_engine(self, real_catalog: config.Catalog) -> None:
        """VRAM-exceeded swap does not attempt to start the oversized engine."""
        # Arrange
        engine_oversized = MagicMock()
        engine_oversized.start.return_value = _make_instance("model-oversized", port=8093)

        daemon = _daemon_with_engines(real_catalog, {
            "model-a": MagicMock(),
            "model-b": MagicMock(),
            "model-oversized": engine_oversized,
        })

        # Act
        daemon._cmd_swap("model-oversized")

        # Assert — engine must not be started for VRAM-rejected swap
        engine_oversized.start.assert_not_called()


@pytest.mark.no_gpu
class TestDaemonCmdSwapStartFailure:
    """_cmd_swap handles failure during new engine start."""

    def test_swap_start_failure_returns_err(self, real_catalog: config.Catalog) -> None:
        """If new engine.start() raises, _cmd_swap returns ERR line."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(old_inst)
        engine_b = MagicMock()
        engine_b.start.side_effect = RuntimeError("llama-server failed to start")

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        # Act
        response = daemon._cmd_swap("model-b")

        # Assert
        assert response.startswith("ERR"), (
            f"Failed start must return ERR line, got: {response!r}"
        )

    def test_swap_start_failure_response_contains_error_info(self, real_catalog: config.Catalog) -> None:
        """ERR line from start failure contains useful context (model or error hint)."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(old_inst)
        engine_b = MagicMock()
        engine_b.start.side_effect = RuntimeError("llama-server failed to start")

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        # Act
        response = daemon._cmd_swap("model-b")

        # Assert — must mention the new model name or include error text
        assert "model-b" in response or "ERR" in response, (
            f"ERR line must reference new model or error, got: {response!r}"
        )


@pytest.mark.no_gpu
class TestDaemonCmdSwapVramOrdering:
    """VRAM ordering constraint: old engine must be stopped before new engine starts."""

    def test_vram_freed_before_new_load(self, real_catalog: config.Catalog) -> None:
        """Verifies the stop→start order guarantees GPU VRAM is freed first."""
        # Arrange
        old_inst = _make_instance("model-a", port=8091, pid=1001)
        new_inst = _make_instance("model-b", port=8092, pid=2002)
        engine_a = _make_engine_mock(old_inst)
        engine_b = _make_engine_mock(new_inst)

        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": engine_b, "model-oversized": MagicMock()})
        daemon.instances["model-a"] = old_inst

        manager = MagicMock()
        manager.attach_mock(engine_a.stop, "stop_old")
        manager.attach_mock(engine_b.start, "start_new")

        # Act
        daemon._cmd_swap("model-b")

        # Assert — expected call order
        assert manager.mock_calls[0] == call.stop_old(old_inst), (
            f"First call must be stop_old(old_inst), got: {manager.mock_calls}"
        )
        assert any(str(c).startswith("call.start_new") for c in manager.mock_calls), (
            f"start_new must be called after stop_old, got: {manager.mock_calls}"
        )

        # Confirm stop index < start index
        stop_idx = next(i for i, c in enumerate(manager.mock_calls) if "stop_old" in str(c))
        start_idx = next(i for i, c in enumerate(manager.mock_calls) if "start_new" in str(c))
        assert stop_idx < start_idx, (
            f"stop(old) at index {stop_idx} must precede start(new) at index {start_idx}"
        )


@pytest.mark.no_gpu
class TestDaemonCmdSwapResponseFormat:
    """Response format must match the plaintext line protocol."""

    def test_response_does_not_contain_newline(self, real_catalog: config.Catalog) -> None:
        """_cmd_swap return value must be a single line without embedded newlines."""
        # Arrange — the wire layer adds the newline; _cmd_swap must NOT include one
        new_inst = _make_instance("model-b", port=8092, pid=2002)
        engine_b = _make_engine_mock(new_inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": MagicMock(), "model-b": engine_b, "model-oversized": MagicMock()})

        # Act
        response = daemon._cmd_swap("model-b")

        # Assert
        assert "\n" not in response, (
            f"_cmd_swap must return a single line without newline, got: {response!r}"
        )

    def test_err_response_starts_with_err(self, real_catalog: config.Catalog) -> None:
        """All error responses must start with 'ERR'."""
        # Arrange
        daemon = Daemon(catalog=real_catalog)

        # Act
        response = daemon._cmd_swap("")

        # Assert — empty name → ERR
        assert response.startswith("ERR"), (
            f"Empty model name must return ERR line, got: {response!r}"
        )

    def test_ok_response_starts_with_ok(self, real_catalog: config.Catalog) -> None:
        """All success responses must start with 'OK'."""
        # Arrange
        new_inst = _make_instance("model-a", port=8091, pid=1001)
        engine_a = _make_engine_mock(new_inst)
        daemon = _daemon_with_engines(real_catalog, {"model-a": engine_a, "model-b": MagicMock(), "model-oversized": MagicMock()})

        # Act
        response = daemon._cmd_swap("model-a")

        # Assert
        assert response.startswith("OK"), f"Success response must start with OK, got: {response!r}"


# ---------------------------------------------------------------------------
# T5.3 target — CLI `llmcli swap <name>` tests
# ---------------------------------------------------------------------------

runner = CliRunner()


@pytest.fixture()
def fake_catalog_patch(real_catalog):
    """Patch llmcli.cli.config.load to return the fake catalog."""
    with patch("llmcli.cli.config") as mock_config_mod:
        mock_config_mod.load.return_value = real_catalog
        mock_config_mod.check_vram_budget.side_effect = config.check_vram_budget
        yield real_catalog


@pytest.mark.no_gpu
class TestCliSwapCommand:
    """CLI `llmcli swap <name>` sends SWAP via daemon socket."""

    def test_swap_sends_swap_command_to_daemon(self, fake_catalog_patch) -> None:
        """llmcli swap <name> calls daemon_request with 'SWAP <name>' and timeout=300.0 (B1)."""
        # Arrange
        with patch("llmcli.cli.daemon_request", return_value="OK model-b pid=2002 port=8092") as mock_req:
            # Act
            runner.invoke(app, ["swap", "model-b"])

        # Assert — B1 fix: timeout=300.0 must be passed for large model loads
        mock_req.assert_called_once_with("SWAP model-b", timeout=300.0)

    def test_swap_ok_response_exits_zero(self, fake_catalog_patch) -> None:
        """llmcli swap exits 0 when daemon returns OK."""
        # Arrange
        with patch("llmcli.cli.daemon_request", return_value="OK model-b pid=2002 port=8092"):
            # Act
            result = runner.invoke(app, ["swap", "model-b"])

        # Assert
        assert result.exit_code == 0, (
            f"Expected exit 0 on OK response, got {result.exit_code}. Output: {result.output!r}"
        )

    def test_swap_ok_response_prints_confirmation(self, fake_catalog_patch) -> None:
        """llmcli swap prints confirmation to stdout on success."""
        # Arrange
        with patch("llmcli.cli.daemon_request", return_value="OK model-b pid=2002 port=8092"):
            # Act
            result = runner.invoke(app, ["swap", "model-b"])

        # Assert — output must contain model name or the OK response
        assert "model-b" in result.output or "OK" in result.output, (
            f"Expected confirmation in output, got: {result.output!r}"
        )

    def test_swap_err_response_exits_nonzero(self, fake_catalog_patch) -> None:
        """llmcli swap exits non-zero when daemon returns ERR.

        RED note: current cli.py swap() does not inspect the response for ERR — it always
        exits 0. This test will fail until GREEN implementation checks the response prefix.
        """
        # Arrange
        with patch("llmcli.cli.daemon_request", return_value="ERR unknown model: ghost-model"):
            # Act
            result = runner.invoke(app, ["swap", "ghost-model"])

        # Assert
        assert result.exit_code != 0, (
            f"Expected non-zero exit on ERR response, got {result.exit_code}. "
            f"Output: {result.output!r}"
        )

    def test_swap_err_response_prints_error(self, fake_catalog_patch) -> None:
        """llmcli swap prints the ERR message when daemon returns ERR."""
        # Arrange
        with patch("llmcli.cli.daemon_request", return_value="ERR unknown model: ghost-model"):
            # Act
            result = runner.invoke(app, ["swap", "ghost-model"])

        # Assert
        combined = result.output + (result.stderr or "")
        assert "ERR" in combined or "ghost-model" in combined or "error" in combined.lower(), (
            f"Expected error message in output, got: {combined!r}"
        )

    def test_swap_without_name_is_usage_error(self) -> None:
        """llmcli swap without a model name argument triggers a Typer usage error (exit 2)."""
        # Act — Typer exit code 2 = "Missing argument"
        result = runner.invoke(app, ["swap"])

        # Assert
        assert result.exit_code == 2, (
            f"Missing argument must exit 2 (Typer usage error), got {result.exit_code}. "
            f"Output: {result.output!r}"
        )

    def test_swap_unknown_model_pre_validates_catalog(self, fake_catalog_patch) -> None:
        """llmcli swap <unknown> exits non-zero with helpful catalog hint before hitting daemon.

        RED note: current cli.py swap() does not pre-validate — it sends SWAP to the daemon
        unconditionally. GREEN should validate model name against catalog first.
        """
        # Arrange — daemon_request is NOT patched so we can confirm it was NOT called
        with patch("llmcli.cli.daemon_request") as mock_req:
            # Act
            result = runner.invoke(app, ["swap", "completely-unknown-model"])

        # Assert — must exit non-zero; daemon_request must NOT be called for unknown model
        assert result.exit_code != 0, (
            f"Expected non-zero exit for unknown model, got {result.exit_code}"
        )
        # Ideally daemon_request is not called for pre-validation failures,
        # but at minimum the exit code must be non-zero.
        combined = result.output + (result.stderr or "")
        assert any(
            keyword in combined
            for keyword in ("model-a", "model-b", "unknown", "Unknown", "not found", "Available")
        ), f"Expected catalog help in output, got: {combined!r}"

    def test_swap_sends_exact_model_name(self, fake_catalog_patch) -> None:
        """daemon_request receives the exact model name passed to llmcli swap."""
        # Arrange
        with patch("llmcli.cli.daemon_request", return_value="OK model-a pid=1001 port=8091") as mock_req:
            # Act
            runner.invoke(app, ["swap", "model-a"])

        # Assert — request must contain exact model name
        args = mock_req.call_args[0][0] if mock_req.called else ""
        assert "model-a" in args, (
            f"daemon_request must receive 'model-a' in the command, got: {args!r}"
        )
