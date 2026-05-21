"""AF_UNIX management socket daemon for llmCLI.

Protocol: plaintext line protocol over AF_UNIX SOCK_STREAM.
Socket path: ~/.local/state/llmcli/llmcli.sock (overridable via env or constructor).

Commands (all newline-terminated):
  STATUS            → OK model=<name> port=<p> uptime=<secs>
  SWAP <name>       → OK swapped to <name>   (swap logic implemented in T5.2)
  SHUTDOWN          → OK shutting down
  <unknown>         → ERR.UNKNOWN_CMD <token>
  <empty>           → ERR.EMPTY empty command

Error frame shape: ``ERR.<CODE> <message>``
Consumers dispatch on the code token (after ``ERR.``) rather than substring-matching
the message.  See ``_WireErr`` for the full code enum.
"""

from __future__ import annotations

import os
import re
import socket
import time
from enum import StrEnum
from pathlib import Path

from roxabi_contracts.errors import scrub_credentials, truncate_with_marker
from roxabi_nats.errors import DEFAULT_MAX_LEN, sanitize_for_wire

from .config import ModelSpec, check_vram_budget
from .engine import Engine, EngineInstance


class _WireErr(StrEnum):
    """Typed error codes for AF_UNIX wire frames.

    Frame shape: ``ERR.<CODE> <message>``. Consumers can dispatch on the code
    token (after ``.``) instead of substring-matching the message.
    """

    EMPTY = "EMPTY"
    UNKNOWN_CMD = "UNKNOWN_CMD"
    MISSING_ARG = "MISSING_ARG"
    UNKNOWN_MODEL = "UNKNOWN_MODEL"
    VRAM_BUDGET = "VRAM_BUDGET"
    SWAP_FAILED = "SWAP_FAILED"
    INTERNAL = "INTERNAL"


# Control chars that break log hygiene on a single-line wire frame —
# C0 (0x00-0x1f, minus tab 0x09), DEL (0x7f), AND C1 (0x80-0x9f). C1 is
# critical: a raw-byte client can encode CSI (U+009B = ESC+[ equivalent
# for C1-enabled terminals like xterm) as valid UTF-8 (0xC2 0x9B) which
# `buf.decode(errors="replace")` accepts cleanly, and then rewrite the
# operator's terminal log line if the C0-only filter let it through.
_WIRE_CTRL_RE = re.compile(r"[\x00-\x08\x0a-\x1f\x7f\x80-\x9f]")


def _sanitize_wire_msg(msg: str, *, max_len: int = DEFAULT_MAX_LEN) -> str:
    """String counterpart to ``sanitize_for_wire(exc)`` for raw client-supplied tokens.

    Mirrors the same primitives (``scrub_credentials`` + ``truncate_with_marker``)
    and additionally collapses single-line-breaking control characters so a
    raw-byte client cannot smuggle ``\\r``/``\\b`` log-rewriting bytes through
    UNKNOWN_CMD/UNKNOWN_MODEL frames where ``msg`` is the verbatim client token.
    """
    return truncate_with_marker(scrub_credentials(_WIRE_CTRL_RE.sub(" ", msg)), max_len)


def _format_err(code: _WireErr, msg: str = "", *, exc: BaseException | None = None) -> str:
    """Format an ERR frame with typed code and sanitized message.

    Both the ``exc`` path (``sanitize_for_wire(exc)``) and the ``msg`` path
    (``_sanitize_wire_msg``) apply the same credential-scrub + length-cap so
    raw client-supplied tokens cannot smuggle ``\\r``/``\\n``-stuffed log
    hygiene attacks or unbounded payloads through the wire.
    """
    payload = sanitize_for_wire(exc) if exc is not None else _sanitize_wire_msg(msg)
    return f"ERR.{code.value} {payload}".rstrip()


SOCKET_PATH = Path(
    os.environ.get("LLMCLI_SOCKET", Path.home() / ".local" / "state" / "llmcli" / "llmcli.sock")
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

    def serve(self, model_name: str | None = None) -> None:
        """Bind the AF_UNIX socket and accept commands in a loop.

        If *model_name* is provided and non-empty, the model is loaded via the
        existing SWAP logic before entering the accept loop.  Same-model fast-path
        in _cmd_swap makes this idempotent (no error if already loaded).

        Args:
            model_name: Initial model to load on startup.  ``None`` or empty string
                        starts the daemon with no model loaded (existing behaviour).
        """
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove stale socket from a previous run.
        self.socket_path.unlink(missing_ok=True)

        # Load the requested model before entering the accept loop so callers
        # don't need to issue a separate SWAP command after startup.
        if model_name:
            result = self._cmd_swap(model_name)
            if result.startswith("ERR"):
                raise RuntimeError(f"Failed to load model on startup: {result}")

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
                self._send_line(conn, _format_err(_WireErr.INTERNAL, "internal error"))
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
            return _format_err(_WireErr.EMPTY, "empty command"), False

        parts = line.split(None, 1)
        cmd = parts[0].upper()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "STATUS":
            return self._cmd_status(), False

        if cmd == "SWAP":
            return self._cmd_swap(arg), False

        if cmd == "SHUTDOWN":
            return "OK shutting down", True

        return _format_err(_WireErr.UNKNOWN_CMD, cmd), False

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

    def _engine_for_spec(self, spec: ModelSpec) -> Engine:
        """Dispatch on spec.engine. Delegates to engines.get_engine after a remote-guard."""
        if spec.engine == "remote":
            raise ValueError(
                f"Model '{spec.name}' uses engine='remote' — cloud-passthrough models are "
                f"managed by LiteLLM, not the local daemon. Use 'llmcli register-proxy' to "
                f"expose this model via the proxy."
            )
        from llmcli.engines import get_engine

        return get_engine(spec)

    def _cmd_swap(self, arg: str) -> str:
        name = arg.strip()
        if not name:
            return _format_err(_WireErr.MISSING_ARG, "swap requires model name")

        # Unknown model guard
        if self.catalog is None or name not in self.catalog.models:
            return _format_err(_WireErr.UNKNOWN_MODEL, name)

        spec = self.catalog.models[name]

        # VRAM budget guard (C2) — reject before touching current engine
        # Remote specs need no local GPU; skip VRAM check.
        if self.catalog is not None and spec.engine != "remote":
            try:
                check_vram_budget(spec, self.catalog.host)
            except ValueError as exc:
                return _format_err(_WireErr.VRAM_BUDGET, exc=exc)

        # Same-model fast-path — no stop/start needed
        if name in self.instances:
            return f"OK already running {name}"

        # Stop all currently running engines (stop-before-start for VRAM safety)
        old_items = list(self.instances.items())
        for old_name, old_inst in old_items:
            old_engine = self._engine_for_spec(self.catalog.models[old_name])
            old_engine.stop(old_inst)
            del self.instances[old_name]

        # Start new engine
        engine = self._engine_for_spec(spec)
        try:
            new_inst = engine.start(spec)
        except Exception as exc:
            return _format_err(_WireErr.SWAP_FAILED, exc=exc)

        self.instances[name] = new_inst
        return f"OK swapped to {name}"

    # ------------------------------------------------------------------
    # Wire protocol
    # ------------------------------------------------------------------

    # Hard cap on a single wire frame — bounds peak memory before any regex
    # work runs on the payload. Matches the recv chunk size; legitimate
    # commands (STATUS, SWAP <name>, SHUTDOWN) fit comfortably below this.
    _RECV_LINE_MAX_BYTES = 4096

    @classmethod
    def _recv_line(cls, conn: socket.socket) -> str:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(4096)
            if not chunk:
                break
            buf += chunk
            if len(buf) >= cls._RECV_LINE_MAX_BYTES:
                # Truncate at the transport layer so downstream regex/scrub
                # never sees a payload larger than the protocol expects.
                buf = buf[: cls._RECV_LINE_MAX_BYTES]
                break
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
