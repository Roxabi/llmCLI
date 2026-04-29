"""GPU utilities for llmCLI — VRAM probing and KV-cache overhead estimation."""

from __future__ import annotations

import logging
import re
import subprocess

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
