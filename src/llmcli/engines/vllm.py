from __future__ import annotations

import collections
import os
import signal
import subprocess
import time

import httpx

from ..config import ModelSpec
from ..engine import EngineInstance

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WAIT_TIMEOUT = 180  # seconds
_WAIT_INTERVAL = 1.0  # seconds between health polls
_STDERR_TAIL = 20  # lines of stderr to include in early-exit error


def _wait_ready(
    base_url: str,
    proc: subprocess.Popen,  # type: ignore[type-arg]
    timeout: float = _WAIT_TIMEOUT,
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
            proc.wait()
            tail = "\n".join(stderr_lines) if stderr_lines else "(no stderr captured)"
            raise RuntimeError(
                f"vllm serve exited with code {rc} before becoming ready. "
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
                import select

                ready, _, _ = select.select([proc.stderr], [], [], 0)
                if ready:
                    line = proc.stderr.readline()
                    if line:
                        stderr_lines.append(line.decode(errors="replace").rstrip())
            except Exception:  # noqa: BLE001
                pass

        time.sleep(_WAIT_INTERVAL)

    raise RuntimeError(f"vllm serve did not become ready within {timeout}s ({base_url})")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class VLLMEngine:
    """vLLM subprocess engine wrapper (HuggingFace repo served via `vllm serve`)."""

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    def _build_cmd(self, spec: ModelSpec) -> list[str]:
        """Build the subprocess argument list for `vllm serve`."""
        return ["vllm", "serve", spec.repo, "--port", str(spec.port), "--host", "0.0.0.0"] + list(spec.flags)

    # ------------------------------------------------------------------
    # Engine Protocol
    # ------------------------------------------------------------------

    def start(self, spec: ModelSpec) -> EngineInstance:
        """Spawn `vllm serve`, wait for readiness, return EngineInstance.

        The vllm import is deferred to this method to avoid ImportError at
        module load time when vLLM is not installed.

        Raises:
            ImportError: when the vllm package is not installed.
            RuntimeError: when the process exits early or the health endpoint
                does not become ready within _WAIT_TIMEOUT seconds.
        """
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise ImportError("vLLM not installed. Run: uv sync --group vllm") from exc

        cmd = self._build_cmd(spec)
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, start_new_session=True)  # noqa: S603
        base_url = f"http://localhost:{spec.port}/v1"
        _wait_ready(base_url, proc)
        return EngineInstance(
            pid=proc.pid,
            port=spec.port,
            model_name=spec.name,
            started_at=time.time(),
        )

    def health(self, instance: EngineInstance) -> bool:
        """Return True iff the vllm /health endpoint responds 2xx."""
        try:
            resp = httpx.get(f"{instance.base_url}/health", timeout=2.0)
            return resp.status_code < 300
        except Exception:  # noqa: BLE001
            return False

    def stop(self, instance: EngineInstance) -> None:
        """Send SIGTERM to the process group; escalate to SIGKILL if it does not exit.

        Uses os.killpg to terminate the entire process group (vLLM spawns GPU worker
        subprocesses in the same session). Always calls os.waitpid() to reap the session
        leader and prevent zombie accumulation across repeated swap cycles.
        Idempotent — silently ignores ProcessLookupError (already dead).
        """
        try:
            os.killpg(os.getpgid(instance.pid), signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            os.waitpid(instance.pid, 0)
        except ChildProcessError:
            # Process did not exit after SIGTERM — escalate to SIGKILL.
            try:
                os.killpg(os.getpgid(instance.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            else:
                # Reap after SIGKILL so no zombie lingers between swap cycles.
                try:
                    os.waitpid(instance.pid, 0)
                except ChildProcessError:
                    pass
