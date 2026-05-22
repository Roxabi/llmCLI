"""VRAM probing and monitoring utilities for llmCLI."""

from __future__ import annotations

import logging
import subprocess
import threading
from typing import Self

logger = logging.getLogger(__name__)


def probe_free_vram_gib() -> float:
    """Return free VRAM in GiB for the first CUDA device.

    Probe order:
    1. pynvml (preferred — no subprocess, GPU-index aware)
    2. nvidia-smi subprocess fallback
    3. Returns 0.0 if neither works (caller must skip the dynamic check).
    """
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_gib = mem.free / (1024**3)
        pynvml.nvmlShutdown()
        return free_gib
    except Exception:  # noqa: BLE001
        pass

    # Fallback: nvidia-smi
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.strip().splitlines()[0].strip()
            free_mib = float(first_line)
            return free_mib / 1024.0
    except Exception:  # noqa: BLE001
        pass

    logger.warning(
        "Could not probe free VRAM (pynvml unavailable, nvidia-smi failed). "
        "Skipping dynamic VRAM check."
    )
    return 0.0


class VRAMSampler:
    """Background thread that polls GPU VRAM usage and tracks peak.

    Holds a single nvmlInit() open for its lifetime — NOT the same as
    probe_free_vram_gib() which calls init/shutdown on every call.

    Usage::

        sampler = VRAMSampler()
        sampler.start()
        # ... do work ...
        peak = sampler.stop()  # returns float GiB or None if no GPU
    """

    def __init__(self, poll_interval: float = 0.2) -> None:
        self._poll_interval = poll_interval
        self._peak: float | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._nvml_available = False
        self._handle = None

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("VRAMSampler already started")
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
        except Exception:  # noqa: BLE001
            self._nvml_available = False

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        """Stop polling and return peak VRAM used in GiB, or None if no GPU."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._nvml_available:
            try:
                import pynvml  # type: ignore[import-untyped]

                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
        return self._peak

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._sample_vram()
            if sample is not None:
                self._peak = max(self._peak or 0.0, sample)
            self._stop_event.wait(self._poll_interval)

    def _sample_vram(self) -> float | None:
        """Return current VRAM used in GiB, or None on failure."""
        if self._nvml_available:
            try:
                import pynvml  # type: ignore[import-untyped]

                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                return mem.used / (1024**3)
            except Exception:  # noqa: BLE001
                pass
        # Fallback: nvidia-smi (used memory, not free)
        try:
            result = subprocess.run(  # noqa: S603, S607
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            if result.returncode == 0:
                return float(result.stdout.strip().splitlines()[0]) / 1024.0
        except Exception:  # noqa: BLE001
            pass
        return None


class VRAMMonitor:
    """Long-lived nvml context manager with cached device handle.

    Use VRAMSampler for one-shot bench peaks. Use VRAMMonitor when you
    need repeated (free, used) samples without re-initializing nvml each
    time — e.g. NATS heartbeat payloads.

    Usage::

        with VRAMMonitor() as vm:
            free_mb, used_mb = vm.sample()
            # ... many more samples over the lifetime ...

    Or for long-lived objects (e.g. adapters whose lifecycle is not a
    single ``with`` block), use the explicit API which delegates to the
    same primitives::

        vm = VRAMMonitor()
        vm.open()
        free_mb, used_mb = vm.sample()
        vm.close()
    """

    def __init__(self, device_index: int = 0) -> None:
        self._index = device_index
        self._handle: object | None = None
        self._init_failed = False

    def __enter__(self) -> Self:
        # Re-entry is sticky in both directions: already-opened short-circuits
        # so we don't double-init (and orphan the previous handle), and a prior
        # init failure short-circuits too — GPU/driver availability does not
        # change at runtime, so retrying nvmlInit() on every open() is wasteful
        # noise.
        if self._handle is not None or self._init_failed:
            return self
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._index)
        except Exception:  # noqa: BLE001
            self._init_failed = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            try:
                import pynvml  # type: ignore[import-untyped]

                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
            self._handle = None

    def open(self) -> Self:
        """Explicit lifecycle counterpart to ``__enter__`` for non-``with`` callers.

        Double-call is absorbed silently by ``__enter__``'s re-entry guard
        (idempotent context-manager semantics, valid for nested ``with``),
        but on the explicit-API path a second ``open()`` without a matching
        ``close()`` is unambiguously caller misuse — log a warning so
        adapter lifecycle bugs surface in operator logs rather than as
        latent never-released-handle leaks.
        """
        if self._handle is not None:
            logger.warning(
                "VRAMMonitor.open() called while already open — call ignored. "
                "Caller likely missed a matching close()."
            )
        return self.__enter__()

    def close(self) -> None:
        """Explicit lifecycle counterpart to ``__exit__`` for non-``with`` callers."""
        self.__exit__(None, None, None)

    def sample(self) -> tuple[float, float]:
        """Return (free_mb, used_mb). (0.0, 0.0) when nvml unavailable."""
        if self._handle is None:
            return (0.0, 0.0)
        try:
            import pynvml  # type: ignore[import-untyped]

            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return (mem.free / (1024**2), mem.used / (1024**2))
        except Exception:  # noqa: BLE001
            return (0.0, 0.0)
