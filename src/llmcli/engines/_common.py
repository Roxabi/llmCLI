"""Shared helpers for llmCLI engine implementations."""

from __future__ import annotations

import collections
import select
import subprocess
import time

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WAIT_INTERVAL = 1.0  # seconds between health polls
_STDERR_TAIL = 20  # lines of stderr to include in early-exit error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_ready(
    base_url: str,
    proc: subprocess.Popen,  # type: ignore[type-arg]
    timeout: float,
    engine_name: str,  # used in error messages e.g. "llama-server", "vllm serve"
) -> None:
    """Poll <base_url>/health until a 2xx response, process exit, or timeout.

    Continues on both connection errors and non-2xx responses (e.g. 503 during
    vLLM model warmup). Only stops on a 2xx response, process early exit, or
    timeout.

    Raises:
        RuntimeError: on process early exit or health-poll timeout.
    """
    deadline = time.monotonic() + timeout
    stderr_lines: collections.deque[str] = collections.deque(maxlen=_STDERR_TAIL)

    while time.monotonic() < deadline:
        # Check if the process exited before becoming ready
        rc = proc.poll()
        if rc is not None:
            # Drain remaining stderr lines if piped
            if proc.stderr is not None:
                for raw in proc.stderr:
                    line = raw.decode(errors="replace").rstrip()
                    stderr_lines.append(line)
            # Reap the zombie — process is already dead, wait() is instant
            proc.wait()
            tail = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
            raise RuntimeError(
                f"{engine_name} exited with code {rc} before becoming ready. "
                f"Last {_STDERR_TAIL} lines of stderr:\n{tail}"
            )

        try:
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code < 300:
                return
            # Non-2xx (e.g. 503 warmup) — continue polling
        except Exception:  # noqa: BLE001
            pass

        # Drain any stderr lines emitted so far (non-blocking)
        if proc.stderr is not None:
            try:
                ready, _, _ = select.select([proc.stderr], [], [], 0)
                if ready:
                    line = proc.stderr.readline()
                    if line:
                        stderr_lines.append(line.decode(errors="replace").rstrip())
            except Exception:  # noqa: BLE001
                pass

        time.sleep(_WAIT_INTERVAL)

    raise RuntimeError(f"{engine_name} did not become ready within {timeout}s ({base_url})")
