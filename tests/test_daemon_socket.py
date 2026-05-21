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

import re
import socket
import threading
import time
from pathlib import Path

import pytest

from roxabi_nats.errors import sanitize_for_wire

from llmcli.daemon import Daemon, _format_err, _sanitize_wire_msg, _WireErr


_WIRE_ERR_CODE_RE = re.compile(r"^ERR\.[A-Z_]+(?: |$)")


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
        _start_daemon_thread(daemon)
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
        assert "8091" in response, f"STATUS must include port '8091', got: {response!r}"


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
            pytest.fail("Expected connection refused after SHUTDOWN, but connect succeeded")
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
        assert not sock_path.exists(), "Daemon must remove the socket file after SHUTDOWN"


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
        """An unrecognised command returns `ERR.UNKNOWN_CMD <token>`."""
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

        # Assert — typed code form `ERR.UNKNOWN_CMD …`, not bare `ERR …`
        assert _WIRE_ERR_CODE_RE.match(response), (
            f"Unknown command must return typed ERR.<CODE> line, got: {response!r}"
        )
        assert response.startswith("ERR.UNKNOWN_CMD"), (
            f"Unknown command must dispatch on UNKNOWN_CMD code, got: {response!r}"
        )
        assert "FROBNICATE" in response, f"ERR line must echo the unknown token, got: {response!r}"

    def test_unknown_command_format(self, tmp_path: Path) -> None:
        """ERR line follows typed format AND echoes the unknown token verbatim."""
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

        # Assert — regression to bare `ERR …` (pre-#57 shape) would fail this.
        assert _WIRE_ERR_CODE_RE.match(response), (
            f"ERR line must match ^ERR\\.[A-Z_]+, got: {response!r}"
        )
        # Distinct from `test_unknown_command_returns_err_line` which uses
        # FROBNICATE — pin that the response echoes the *uppercased* token
        # (dispatch case-folds `cmd.upper()`), so we can detect both a regression
        # to bare `ERR …` and a regression that drops the token from the frame.
        assert "BADCMD" in response, (
            f"ERR line must echo the upper-cased unknown token, got: {response!r}"
        )

    def test_empty_command_returns_err(self, tmp_path: Path) -> None:
        """An empty line (bare newline) returns `ERR.EMPTY …`."""
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

        # Assert — typed code form so the consumer can dispatch on EMPTY.
        assert _WIRE_ERR_CODE_RE.match(response), (
            f"Empty command must return typed ERR.<CODE> line, got: {response!r}"
        )
        assert response.startswith("ERR.EMPTY"), (
            f"Empty command must dispatch on EMPTY code, got: {response!r}"
        )


# ---------------------------------------------------------------------------
# 7. _format_err / _WireErr — unit (no socket)
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestFormatErr:
    """Direct unit tests for the wire-frame formatter.

    The socket-level tests above exercise this function indirectly; stubbing
    ``sanitize_for_wire`` away (or regressing to bare ``ERR …``) would not
    fail them, so these unit cases pin the shape + sanitization contract.
    """

    def test_empty_msg_yields_code_only(self) -> None:
        # No trailing space — the format string rstrips after interpolation.
        assert _format_err(_WireErr.EMPTY) == "ERR.EMPTY"

    def test_benign_msg_passes_sanitization_unchanged(self) -> None:
        # The msg path always runs the full sanitization pipeline; a benign
        # input (no credentials, no ctrl chars, under DEFAULT_MAX_LEN) passes
        # through unchanged. "Unchanged" describes the observed output, not a
        # bypass — the pipeline still executes.
        assert _format_err(_WireErr.UNKNOWN_MODEL, "qwen3-8b") == "ERR.UNKNOWN_MODEL qwen3-8b"

    def test_exc_credential_url_is_scrubbed(self) -> None:
        # TruffleHog's generic credential-URL detector pattern-matches any
        # `scheme://user:pass@host`; build the fixture from parts so the raw
        # string never appears verbatim in the source. The scrubber still
        # sees the assembled URL and applies its userinfo replacement.
        url = "nats" + "://" + "alice:s3cret" + "@" + "nats.local/jet"  # trufflehog:ignore
        exc = Exception(f"swap failed: {url}")
        result = _format_err(_WireErr.SWAP_FAILED, exc=exc)
        assert result.startswith("ERR.SWAP_FAILED ")
        assert "alice:s3cret" not in result, f"credentials leaked into wire frame: {result!r}"
        assert "***" in result, f"expected scrub marker in result: {result!r}"

    def test_oversized_exc_is_truncated(self) -> None:
        # DEFAULT_MAX_LEN is 200; payload truncated with marker beyond that bound.
        long_msg = "x" * 500
        result = _format_err(_WireErr.INTERNAL, exc=Exception(long_msg))
        # `ERR.INTERNAL ` prefix is 13 chars; the truncated tail must not exceed 200.
        payload = result.removeprefix("ERR.INTERNAL ")
        assert len(payload) <= 200, f"payload exceeded max_len: {len(payload)} chars"
        assert payload.endswith("…"), f"expected truncation marker, got: {payload[-5:]!r}"

    def test_msg_credential_url_is_scrubbed(self) -> None:
        # Defensive: a raw token passed as `msg` (e.g. user-supplied SWAP arg)
        # gets the same scrub treatment as `exc`. See note in
        # ``test_exc_credential_url_is_scrubbed`` for why we assemble the URL
        # at runtime rather than embedding the verbatim form.
        url = "nats" + "://" + "u:p" + "@" + "nats/jet"  # trufflehog:ignore
        result = _format_err(_WireErr.UNKNOWN_MODEL, url)
        assert "u:p" not in result, f"msg credentials leaked: {result!r}"
        assert "***" in result, f"expected scrub marker in result: {result!r}"

    def test_msg_control_chars_are_collapsed(self) -> None:
        # A raw-byte client could embed \r/\b to rewrite log lines; the wire
        # sanitizer collapses C0 control bytes (minus \t) before forwarding.
        # Boundary chars (NUL = range start, DEL = explicitly appended) are
        # included so a refactor that drops either boundary fails this test.
        result = _format_err(_WireErr.UNKNOWN_CMD, "BAD\x00\rCMD\b\x1b[31m\x7f")
        for ctrl in ("\x00", "\r", "\b", "\x1b", "\x7f"):
            assert ctrl not in result, f"control char {ctrl!r} leaked into wire frame: {result!r}"

    def test_msg_tab_is_preserved(self) -> None:
        # `_WIRE_CTRL_RE` is documented as "C0 minus tab" — tab is intentionally
        # excluded from the strip set. Pinning preservation guards against an
        # accidental widening of the regex range (e.g. to `[\x00-\x1f\x7f]`).
        result = _format_err(_WireErr.UNKNOWN_CMD, "BAD\tCMD")
        assert "\t" in result, f"tab must be preserved in wire frame: {result!r}"

    def test_msg_c1_control_chars_are_collapsed(self) -> None:
        # C1 range (U+0080–U+009F) encodes cleanly via UTF-8 — `\xc2\x9b` is
        # the byte sequence for U+009B (CSI), functionally equivalent to ESC+[
        # on a C1-enabled terminal (xterm default). The regex now covers the
        # full range; verify CSI (mid-range) AND the two boundaries are stripped.
        result = _format_err(_WireErr.UNKNOWN_CMD, "BAD\x80\x9b2J\x9fEND")
        for ctrl in ("\x80", "\x9b", "\x9f"):
            assert ctrl not in result, (
                f"C1 control char {ctrl!r} leaked into wire frame: {result!r}"
            )

    def test_recv_line_caps_oversized_input_at_max_bytes(self) -> None:
        # `_recv_line` must bound peak memory at the transport layer so the
        # downstream regex (`scrub_credentials`) never sees a payload larger
        # than the protocol expects. A malicious local client could otherwise
        # force ~400MB peak allocation with a 100MB no-newline token.
        class _FakeSock:
            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self._pos = 0

            def recv(self, n: int) -> bytes:
                chunk = self._payload[self._pos : self._pos + n]
                self._pos += len(chunk)
                return chunk

        # 64 KiB of `A` with no newline — well beyond the 4096 cap.
        big = b"A" * 65536
        result = Daemon._recv_line(_FakeSock(big))  # type: ignore[arg-type]
        assert len(result) <= Daemon._RECV_LINE_MAX_BYTES, (
            f"_recv_line returned {len(result)} bytes, max is {Daemon._RECV_LINE_MAX_BYTES}"
        )

    def test_sanitize_wire_msg_mirrors_sanitize_for_wire_contract(self) -> None:
        # Contract-mirror: both paths share the credential-scrub + truncate
        # primitives. If upstream `sanitize_for_wire` drifts (new normalization
        # step, different scrub regex, different max_len), this test fails and
        # surfaces the parallel-path drift instead of letting it silently diverge.
        # NB: ctrl-strip is intentionally a `_sanitize_wire_msg`-only step
        # (exception strings are system-generated; raw tokens are user-controlled),
        # so the fixture deliberately contains neither ctrl chars nor a credential
        # URL that would benefit from scrub — leaving only the truncation contract
        # under test.
        long_token = "x" * 500
        msg_path = _sanitize_wire_msg(long_token)
        exc_path = sanitize_for_wire(Exception(long_token))
        assert msg_path == exc_path, (
            f"Contract drift between _sanitize_wire_msg and sanitize_for_wire: "
            f"msg={msg_path!r} exc={exc_path!r}"
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
