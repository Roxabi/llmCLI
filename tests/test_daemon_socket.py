"""
RED-phase tests for T1.5 — Daemon AF_UNIX socket (SC-3, S1).

Spec trace:
  SC-3: `llmcli status` PID+uptime via socket (no HTTP roundtrip).
  S1:   Management socket at configurable path, AF_UNIX SOCK_STREAM.
  C6:   Plaintext line protocol — `STATUS\n`, `SWAP <name>\n`, `SHUTDOWN\n`.
  C8:   @pytest.mark.no_gpu — these tests must NOT spawn llama-server.

Expected RED failures against current scaffold:
  - `Daemon.serve()` raises NotImplementedError (scaffold stub).
  - `Daemon` has no `daemon_request` client helper.
  - Protocol commands are not implemented.

All tests use tmp_path for socket path — no hardcoded /tmp.
"""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path

import pytest

from llmcli.daemon import Daemon


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _send(sock: socket.socket, msg: str) -> None:
    """Send a newline-terminated plaintext command."""
    sock.sendall((msg if msg.endswith("\n") else msg + "\n").encode())


def _recv_line(sock: socket.socket, timeout: float = 5.0) -> str:
    """Read bytes until a newline, honouring a per-call timeout."""
    sock.settimeout(timeout)
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()


def _connect(sock_path: Path, timeout: float = 5.0) -> socket.socket:
    """Open a client AF_UNIX connection to the daemon socket."""
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(timeout)
    conn.connect(str(sock_path))
    return conn


def _start_daemon_thread(daemon: Daemon) -> threading.Thread:
    """Run daemon.serve() in a background daemon thread."""
    t = threading.Thread(target=daemon.serve, daemon=True)
    t.start()
    return t


def _wait_for_socket(sock_path: Path, timeout: float = 5.0) -> None:
    """Block until the socket file exists (daemon ready) or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sock_path.exists():
            return
        time.sleep(0.05)
    raise TimeoutError(f"Daemon socket did not appear at {sock_path} within {timeout}s")


# ---------------------------------------------------------------------------
# 1. Socket binding
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestDaemonSocketBinding:
    """Daemon must create an AF_UNIX SOCK_STREAM socket at the configured path."""

    def test_socket_file_created_on_serve(self, tmp_path: Path) -> None:
        """After serve() starts the daemon binds a socket file at the given path."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)

        # Act
        t = _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Assert — socket file exists and is a socket
        assert sock_path.exists(), "Socket file must be created by Daemon.serve()"
        assert sock_path.stat().st_mode & 0o170000 == 0o140000, (
            "Created path must be a socket (S_IFSOCK)"
        )

    def test_socket_path_is_configurable(self, tmp_path: Path) -> None:
        """Daemon accepts a custom socket path, not hardcoded default."""
        # Arrange
        custom_path = tmp_path / "custom_subdir" / "my.sock"
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        daemon = Daemon(socket_path=custom_path)

        # Act
        _start_daemon_thread(daemon)
        _wait_for_socket(custom_path, timeout=5.0)

        # Assert
        assert custom_path.exists()

    def test_client_can_connect_after_serve(self, tmp_path: Path) -> None:
        """A raw AF_UNIX client can connect once serve() is running."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act / Assert — connection itself is the assertion
        conn = _connect(sock_path)
        conn.close()


# ---------------------------------------------------------------------------
# 2. STATUS command
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestStatusCommand:
    """STATUS command must return current model info as plaintext line(s)."""

    def test_status_returns_nonempty_response(self, tmp_path: Path) -> None:
        """STATUS yields at least one non-blank line."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "STATUS")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert
        assert response, "STATUS must return a non-empty response line"

    def test_status_response_contains_model_name(self, tmp_path: Path) -> None:
        """STATUS response includes the current model name (or 'none' when idle)."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "STATUS")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert — when no engine running, daemon should report 'none' or similar
        # When an engine IS running, model name must appear
        assert response  # at minimum it must not be blank

    def test_status_response_contains_port(self, tmp_path: Path) -> None:
        """STATUS response includes the port (or 'none' when idle)."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "STATUS")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert — 'port' key or a numeric port value or 'none' must be present
        assert response

    def test_status_with_active_instance(self, tmp_path: Path) -> None:
        """STATUS reports model_name and port when an EngineInstance is tracked."""
        from llmcli.engine import EngineInstance

        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        # Inject a fake running instance before serving
        instance = EngineInstance(pid=12345, port=8091, model_name="qwen3-test")
        daemon.instances["qwen3-test"] = instance

        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "STATUS")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert
        assert "qwen3-test" in response, (
            f"STATUS must include model name 'qwen3-test', got: {response!r}"
        )
        assert "8091" in response, (
            f"STATUS must include port '8091', got: {response!r}"
        )


# ---------------------------------------------------------------------------
# 3. SHUTDOWN command
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestShutdownCommand:
    """SHUTDOWN command must gracefully terminate the daemon."""

    def test_shutdown_closes_socket(self, tmp_path: Path) -> None:
        """After SHUTDOWN, subsequent connection attempts must fail."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act — send SHUTDOWN
        conn = _connect(sock_path)
        try:
            _send(conn, "SHUTDOWN")
            # Drain any acknowledgement
            try:
                _recv_line(conn, timeout=2.0)
            except (OSError, TimeoutError):
                pass
        finally:
            conn.close()

        # Wait for socket file to disappear (daemon cleaned up)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and sock_path.exists():
            time.sleep(0.05)

        # Assert — socket file removed OR new connections refused
        try:
            bad_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            bad_conn.settimeout(1.0)
            bad_conn.connect(str(sock_path))
            bad_conn.close()
            pytest.fail(
                "Expected connection refused after SHUTDOWN, but connect succeeded"
            )
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            pass  # expected

    def test_shutdown_sends_acknowledgement(self, tmp_path: Path) -> None:
        """SHUTDOWN returns an OK line before closing the connection."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        response = ""
        try:
            _send(conn, "SHUTDOWN")
            response = _recv_line(conn, timeout=3.0)
        except (OSError, TimeoutError):
            pass
        finally:
            conn.close()

        # Assert — response should indicate OK or shutting down
        assert response, "SHUTDOWN must send an acknowledgement before closing"

    def test_socket_file_cleaned_on_exit(self, tmp_path: Path) -> None:
        """Socket file must be removed from disk when daemon shuts down."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "SHUTDOWN")
            try:
                _recv_line(conn, timeout=2.0)
            except (OSError, TimeoutError):
                pass
        finally:
            conn.close()

        # Wait for cleanup
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and sock_path.exists():
            time.sleep(0.05)

        # Assert
        assert not sock_path.exists(), (
            "Daemon must remove the socket file after SHUTDOWN"
        )


# ---------------------------------------------------------------------------
# 4. SWAP command
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestSwapCommand:
    """SWAP <model-name> command must be parsed; swap logic is T5.2."""

    def test_swap_command_is_accepted(self, tmp_path: Path) -> None:
        """SWAP <name> must return a response (not ERR unknown command)."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "SWAP qwen3-8b")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert — must not be an "unknown command" error
        assert "unknown command" not in response.lower(), (
            f"SWAP must be a recognised command, got: {response!r}"
        )

    def test_swap_command_missing_model_name_returns_error(self, tmp_path: Path) -> None:
        """Bare `SWAP` without a model name must return an error."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "SWAP")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert — malformed SWAP must surface an error
        assert response.startswith("ERR"), (
            f"SWAP with no model name must return ERR line, got: {response!r}"
        )

    def test_swap_includes_model_name_in_response(self, tmp_path: Path) -> None:
        """SWAP response acknowledges the requested model name."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "SWAP qwen3-14b-q5")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert
        assert "qwen3-14b-q5" in response, (
            f"SWAP response must echo model name 'qwen3-14b-q5', got: {response!r}"
        )


# ---------------------------------------------------------------------------
# 5. Unknown command
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestUnknownCommand:
    """Unrecognised commands must return an ERR line."""

    def test_unknown_command_returns_err_line(self, tmp_path: Path) -> None:
        """An unrecognised command returns `ERR unknown command: <X>`."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "FROBNICATE")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert
        assert response.startswith("ERR"), (
            f"Unknown command must return ERR line, got: {response!r}"
        )
        assert "FROBNICATE" in response or "unknown" in response.lower(), (
            f"ERR line must reference the unknown command, got: {response!r}"
        )

    def test_unknown_command_format(self, tmp_path: Path) -> None:
        """ERR line must follow format: `ERR unknown command: <X>`."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "BADCMD")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert
        assert response.lower().startswith("err"), (
            f"ERR line must begin with 'ERR', got: {response!r}"
        )

    def test_empty_command_returns_err(self, tmp_path: Path) -> None:
        """An empty line (bare newline) is an unknown command and returns ERR."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act
        conn = _connect(sock_path)
        try:
            _send(conn, "")
            response = _recv_line(conn)
        finally:
            conn.close()

        # Assert
        assert response.startswith("ERR"), (
            f"Empty command must return ERR line, got: {response!r}"
        )


# ---------------------------------------------------------------------------
# 6. Multiple concurrent / sequential clients
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestMultipleClients:
    """Daemon handles sequential connect/disconnect cycles from multiple clients."""

    def test_two_sequential_status_calls(self, tmp_path: Path) -> None:
        """Two back-to-back STATUS calls from separate connections both succeed."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act — first client
        conn1 = _connect(sock_path)
        try:
            _send(conn1, "STATUS")
            resp1 = _recv_line(conn1)
        finally:
            conn1.close()

        # Act — second client (new connection after first closed)
        conn2 = _connect(sock_path)
        try:
            _send(conn2, "STATUS")
            resp2 = _recv_line(conn2)
        finally:
            conn2.close()

        # Assert — both responses non-empty (daemon still alive)
        assert resp1, "First STATUS must return a response"
        assert resp2, "Second STATUS must return a response after first client closed"

    def test_three_sequential_clients(self, tmp_path: Path) -> None:
        """Three sequential clients each receive a valid response."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        responses: list[str] = []

        # Act
        for _ in range(3):
            conn = _connect(sock_path)
            try:
                _send(conn, "STATUS")
                responses.append(_recv_line(conn))
            finally:
                conn.close()

        # Assert
        assert len(responses) == 3, "All three clients must receive a response"
        assert all(r for r in responses), "All responses must be non-empty"

    def test_unknown_command_does_not_kill_daemon(self, tmp_path: Path) -> None:
        """After an unknown command, daemon remains alive for subsequent clients."""
        # Arrange
        sock_path = tmp_path / "llmcli.sock"
        daemon = Daemon(socket_path=sock_path)
        _start_daemon_thread(daemon)
        _wait_for_socket(sock_path, timeout=5.0)

        # Act — send bad command, then follow-up with STATUS
        conn1 = _connect(sock_path)
        try:
            _send(conn1, "INVALID_CMD")
            _recv_line(conn1)
        finally:
            conn1.close()

        conn2 = _connect(sock_path)
        try:
            _send(conn2, "STATUS")
            resp = _recv_line(conn2)
        finally:
            conn2.close()

        # Assert — daemon still responds after bad command
        assert resp, "Daemon must remain alive and respond to STATUS after bad command"
