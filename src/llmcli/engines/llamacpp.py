from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from ..config import ModelSpec
from ..engine import EngineInstance
from ._common import _wait_ready, default_health

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DEFAULT_WAIT_TIMEOUT = 60  # seconds — fallback when ModelSpec.startup_timeout_s is None


def _hf_hub_root() -> Path:
    """Return the HuggingFace hub cache root, honouring $HF_HOME."""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub"


def _repo_to_dir_name(repo: str) -> str:
    """Convert 'Org/Repo' to 'models--Org--Repo' (HF hub convention)."""
    return "models--" + repo.replace("/", "--")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LlamaCppEngine:
    """Vanilla llama.cpp llama-server wrapper (standard GGUF)."""

    binary: str = "llama-server"

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _gguf_path(self, spec: ModelSpec) -> Path:
        """Resolve the GGUF file path inside the HuggingFace hub cache.

        Searches under ``<HF_HOME>/hub/models--<org>--<repo>/snapshots/``
        for any revision that contains ``spec.file``.

        Raises FileNotFoundError with a helpful hint when the model has not
        been pulled yet.
        """
        hub = _hf_hub_root()
        repo_dir = hub / _repo_to_dir_name(spec.repo)
        snapshots = repo_dir / "snapshots"

        if snapshots.is_dir():
            for revision_dir in snapshots.iterdir():
                candidate = revision_dir / spec.file
                if candidate.is_file():
                    return candidate

        raise FileNotFoundError(
            f"GGUF file '{spec.file}' not found in HF hub cache for repo '{spec.repo}'.\n"
            f"Run `llmcli pull {spec.name}` to download the model first."
        )

    # ------------------------------------------------------------------
    # Command builder
    # ------------------------------------------------------------------

    def _build_cmd(self, spec: ModelSpec) -> list[str]:
        """Build the subprocess argument list for llama-server."""
        gguf = self._gguf_path(spec)
        cmd: list[str] = [
            self.binary,
            "--model",
            str(gguf),
            "--host",
            "0.0.0.0",
            "--port",
            str(spec.port),
        ]
        if spec.flags:
            cmd.extend(spec.flags)
        if spec.mmproj is not None:
            mmproj_path = gguf.parent / spec.mmproj
            cmd.extend(["--mmproj", str(mmproj_path)])
        return cmd

    # ------------------------------------------------------------------
    # Engine Protocol
    # ------------------------------------------------------------------

    def start(self, spec: ModelSpec) -> EngineInstance:
        """Spawn llama-server, wait for readiness, return EngineInstance.

        Stderr is piped so that early-exit errors can include a useful tail.
        If the process exits before /health responds, proc.wait() is called
        immediately (inside _wait_ready) to reap the zombie before raising.
        """
        cmd = self._build_cmd(spec)
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)  # noqa: S603
        base_url = f"http://localhost:{spec.port}/v1"
        _wait_ready(base_url, proc, spec.startup_timeout_s or _DEFAULT_WAIT_TIMEOUT, "llama-server")
        return EngineInstance(
            pid=proc.pid,
            port=spec.port,
            model_name=spec.name,
            started_at=time.time(),
        )

    def supports_swap(self) -> bool:
        return True

    def supports_hot_reload(self) -> bool:
        return True

    def health(self, instance: EngineInstance) -> bool:
        """Return True iff the llama-server /health endpoint responds 2xx."""
        return default_health(instance.base_url)

    def stop(self, instance: EngineInstance) -> None:
        """Send SIGTERM; escalate to SIGKILL if the process does not exit.

        Always calls os.waitpid() after terminating to reap the child and
        prevent zombie accumulation across repeated SWAP cycles.
        Idempotent — silently ignores ProcessLookupError (already dead).
        """
        try:
            os.kill(instance.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            os.waitpid(instance.pid, 0)
        except ChildProcessError:
            # Process did not exit after SIGTERM — escalate to SIGKILL.
            try:
                os.kill(instance.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            else:
                # Reap after SIGKILL so no zombie lingers between swap cycles.
                try:
                    os.waitpid(instance.pid, 0)
                except ChildProcessError:
                    pass
