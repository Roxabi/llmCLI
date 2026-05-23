from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time

from ..config import ModelSpec
from ..engine import EngineInstance
from ._common import _wait_ready, default_health

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_WAIT_TIMEOUT = (
    180  # vLLM needs longer: safetensors load + NVFP4 JIT compile can take 60–120 s on first start
    # fallback when ModelSpec.startup_timeout_s is None
)


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
        return ["vllm", "serve", spec.repo, "--port", str(spec.port), "--host", "0.0.0.0"] + list(
            spec.flags
        )

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
                does not become ready within _DEFAULT_WAIT_TIMEOUT seconds.
        """
        if shutil.which("vllm") is None:
            raise RuntimeError(
                "vllm binary not found on PATH. "
                "Ensure your venv is activated or run via `uv run llmcli`. "
                "Install with: uv sync --group vllm"
            )
        try:
            import vllm  # noqa: F401  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("vLLM not installed. Run: uv sync --group vllm") from exc

        cmd = self._build_cmd(spec)
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, start_new_session=True)  # noqa: S603
        base_url = f"http://localhost:{spec.port}/v1"
        _wait_ready(base_url, proc, spec.startup_timeout_s or _DEFAULT_WAIT_TIMEOUT, "vllm serve")
        return EngineInstance(
            pid=proc.pid,
            port=spec.port,
            model_name=spec.name,
            started_at=time.time(),
        )

    def supports_swap(self) -> bool:
        return False

    def supports_hot_reload(self) -> bool:
        return True

    def health(self, instance: EngineInstance) -> bool:
        """Return True iff the vllm /health endpoint responds 2xx."""
        return default_health(instance.base_url)

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
