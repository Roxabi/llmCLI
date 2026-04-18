from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from ..config import ModelSpec
from ..engine import EngineInstance

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WAIT_TIMEOUT = 60  # seconds
_WAIT_INTERVAL = 0.5  # seconds between health polls
_STOP_GRACE = 5  # seconds to wait after SIGTERM before SIGKILL


def _hf_hub_root() -> Path:
    """Return the HuggingFace hub cache root, honouring $HF_HOME."""
    hf_home = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    return Path(hf_home) / "hub"


def _repo_to_dir_name(repo: str) -> str:
    """Convert 'Org/Repo' to 'models--Org--Repo' (HF hub convention)."""
    return "models--" + repo.replace("/", "--")


def _wait_ready(base_url: str, timeout: float = _WAIT_TIMEOUT) -> None:
    """Poll <base_url>/health until a 2xx response or timeout is exceeded.

    Raises RuntimeError on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = httpx.get(f"{base_url}/health", timeout=2.0)
            if resp.status_code < 300:
                return
        except Exception:  # noqa: BLE001
            pass
        time.sleep(_WAIT_INTERVAL)
    raise RuntimeError(f"llama-server did not become ready within {timeout}s ({base_url})")


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
            "--model", str(gguf),
            "--host", "0.0.0.0",
            "--port", str(spec.port),
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
        """Spawn llama-server, wait for readiness, return EngineInstance."""
        cmd = self._build_cmd(spec)
        proc = subprocess.Popen(cmd)  # noqa: S603
        base_url = f"http://localhost:{spec.port}/v1"
        _wait_ready(base_url)
        return EngineInstance(
            pid=proc.pid,
            port=spec.port,
            model_name=spec.name,
            started_at=time.time(),
        )

    def health(self, instance: EngineInstance) -> bool:
        """Return True iff the llama-server /health endpoint responds 2xx."""
        try:
            resp = httpx.get(f"{instance.base_url}/health", timeout=2.0)
            return resp.status_code < 300
        except Exception:  # noqa: BLE001
            return False

    def stop(self, instance: EngineInstance) -> None:
        """Send SIGTERM; escalate to SIGKILL if the process does not exit.

        Idempotent — silently ignores ProcessLookupError (already dead).
        """
        try:
            os.kill(instance.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            os.waitpid(instance.pid, 0)
        except ChildProcessError:
            # Process did not exit after SIGTERM — escalate.
            try:
                os.kill(instance.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
