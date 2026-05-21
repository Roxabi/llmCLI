"""GPU utilities for llmCLI — VRAM probing and KV-cache overhead estimation."""

from __future__ import annotations

import logging
import re
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
        if self._handle is not None:
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
        """Explicit lifecycle counterpart to ``__enter__`` for non-``with`` callers."""
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


_QUANT_BITS: dict[str, float] = {
    "q2_k": 2.0,
    "q2_k_s": 2.0,
    "iq3_xxs": 3.0,
    "iq3_xs": 3.0,
    "iq3_s": 3.0,
    "iq3_m": 3.0,
    "tq3_0": 3.0,
    "q4_0": 4.0,
    "q4_1": 4.0,
    "q4_k": 4.0,
    "q4_k_s": 4.0,
    "q4_k_m": 4.0,
    "iq4_nl": 4.0,
    "iq4_xs": 4.0,
    "q5_0": 5.0,
    "q5_1": 5.0,
    "q5_k": 5.0,
    "q5_k_s": 5.0,
    "q5_k_m": 5.0,
    "q6_k": 6.0,
    "q8_0": 8.0,
    "f16": 16.0,
    "bf16": 16.0,
    "f32": 32.0,
}
_BASELINE_BITS = 4.0  # empirical baseline: q4_0


def _quant_multiplier(flags: list[str]) -> float:
    """Return a VRAM scaling factor relative to q4_0 from -ctk/-ctv flags.

    Averages the key-cache and value-cache bit widths and normalises against
    the q4_0 baseline. Falls back to 1.0 (no adjustment) when flags are absent.
    """
    key_bits: float | None = None
    val_bits: float | None = None

    i = 0
    while i < len(flags) - 1:
        flag = flags[i]
        if flag == "-ctk":
            key_bits = _QUANT_BITS.get(flags[i + 1].lower())
        elif flag == "-ctv":
            val_bits = _QUANT_BITS.get(flags[i + 1].lower())
        i += 1

    bits = [b for b in (key_bits, val_bits) if b is not None]
    if not bits:
        return 1.0
    return sum(bits) / len(bits) / _BASELINE_BITS


def kv_overhead_gib(flags: list[str]) -> float:
    """Estimate KV-cache VRAM overhead in GiB from llama-server flags.

    Heuristic: 0.5 GiB per 4096 tokens at q4_0, scaled by actual KV quant
    type parsed from -ctk/-ctv flags (e.g. q8_0 = 2× overhead vs q4_0).

    Returns 0.0 when -c / --ctx-size is absent.
    """
    ctx_tokens: int | None = None

    i = 0
    while i < len(flags):
        token = flags[i]
        if token in ("-c", "--ctx-size") and i + 1 < len(flags):
            try:
                ctx_tokens = int(flags[i + 1])
            except ValueError:
                pass
            break
        m = re.match(r"^(?:-c|--ctx-size)=(\d+)$", token)
        if m:
            ctx_tokens = int(m.group(1))
            break
        i += 1

    if ctx_tokens is None:
        return 0.0

    return 0.5 * (ctx_tokens / 4096.0) * _quant_multiplier(flags)
