"""AF_UNIX management socket daemon for llmCLI.

Protocol: plaintext line protocol over AF_UNIX SOCK_STREAM.
Socket path: ~/.local/state/llmcli/llmcli.sock (overridable via env or constructor).

Commands (all newline-terminated):
  STATUS            → OK model=<name> port=<p> uptime=<secs>
  SWAP <name>       → OK swapped to <name>   (swap logic implemented in T5.2)
  SHUTDOWN          → OK shutting down
  <unknown>         → ERR unknown command: <token>
  <empty>           → ERR empty command
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path

from .engine import EngineInstance

SOCKET_PATH = Path(
    os.environ.get(
        "LLMCLI_SOCKET", Path.home() / ".local" / "state" / "llmcli" / "llmcli.sock"
    )
)


class Daemon:
    """AF_UNIX management socket owner. Tracks running engines by model name."""

    def __init__(
        self,
        catalog=None,
        host: str | None = None,
        socket_path: str | Path | None = None,
    ) -> None:
        self.catalog = catalog
        self.host = host
        self.socket_path: Path = Path(socket_path) if socket_path is not None else SOCKET_PATH
        self.instances: dict[str, EngineInstance] = {}
        self._started_at: float = time.monotonic()

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    def serve(self, model_name: str | None = None) -> None:  # noqa: ARG002
        """Bind the AF_UNIX socket and accept commands in a loop.

        Args:
            model_name: (reserved for T1.12 CLI) initial model to load — ignored in
                        this phase; real engine start wired in T1.12.
        """
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket from a previous run.
        self.socket_path.unlink(missing_ok=True)

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(str(self.socket_path))
            srv.listen(5)

            while True:
                conn, _ = srv.accept()
                try:
                    should_stop = self._handle_client(conn)
                except Exception:
                    pass
                else:
                    if should_stop:
                        break
        except Exception:
            raise
        finally:
            srv.close()
            self.socket_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Per-connection handler
    # ------------------------------------------------------------------

    def _handle_client(self, conn: socket.socket) -> bool:
        """Handle one client connection.  Returns True to break the serve loop."""
        conn.settimeout(5.0)
        try:
            raw = self._recv_line(conn)
            response, stop = self._dispatch(raw)
            self._send_line(conn, response)
            return stop
        except Exception:
            try:
                self._send_line(conn, "ERR internal error")
            except Exception:
                pass
            return False
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, raw: str) -> tuple[str, bool]:
        """Parse *raw* (already stripped) and return (response_line, should_stop)."""
        line = raw.strip()

        if not line:
            return "ERR empty command", False

        parts = line.split(None, 1)
        cmd = parts[0].upper()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "STATUS":
            return self._cmd_status(), False

        if cmd == "SWAP":
            return self._cmd_swap(arg), False

        if cmd == "SHUTDOWN":
            return "OK shutting down", True

        return f"ERR unknown command: {cmd}", False

    # ------------------------------------------------------------------
    # Command implementations
    # ------------------------------------------------------------------

    def _cmd_status(self) -> str:
        if not self.instances:
            uptime = int(time.monotonic() - self._started_at)
            return f"OK model=none port=none uptime={uptime}"

        # Report first (or only) running instance.
        instance = next(iter(self.instances.values()))
        uptime = int(time.monotonic() - self._started_at)
        return f"OK model={instance.model_name} port={instance.port} uptime={uptime}"

    def _cmd_swap(self, arg: str) -> str:
        name = arg.strip()
        if not name:
            return "ERR swap requires model name"
        # Real swap logic wired in T5.2.  For now: acknowledge the name.
        return f"OK swapped to {name}"

    # ------------------------------------------------------------------
    # Wire protocol
    # ------------------------------------------------------------------

    @staticmethod
    def _recv_line(conn: socket.socket) -> str:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
        return buf.decode(errors="replace").split("\n")[0]

    @staticmethod
    def _send_line(conn: socket.socket, msg: str) -> None:
        payload = (msg if msg.endswith("\n") else msg + "\n").encode()
        conn.sendall(payload)


# ---------------------------------------------------------------------------
# Client helper
# ---------------------------------------------------------------------------


def daemon_request(
    line: str,
    socket_path: str | Path = SOCKET_PATH,
    timeout: float = 5.0,
) -> str:
    """Send a plaintext command to the daemon and return the response line (stripped)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(socket_path))
        payload = (line if line.endswith("\n") else line + "\n").encode()
        sock.sendall(payload)
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        return buf.decode(errors="replace").split("\n")[0].strip()
    finally:
        sock.close()
