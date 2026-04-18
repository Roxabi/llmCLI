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


def kv_overhead_gib(flags: list[str]) -> float:
    """Estimate KV-cache VRAM overhead in GiB from llama-server flags.

    Heuristic: 0.5 GiB per 4096 tokens of context window, derived from
    empirical measurements of Qwen3-family GQA models at q4_0/tq3_0 KV
    cache quantisation. Not exact — treats it as a conservative safety margin.

    Returns 0.0 when -c / --ctx-size is absent from flags (no context budget
    specified, so we cannot estimate overhead safely).
    """
    ctx_tokens: int | None = None

    # Parse -c <N> or --ctx-size <N> from the flags list
    i = 0
    while i < len(flags):
        token = flags[i]
        if token in ("-c", "--ctx-size") and i + 1 < len(flags):
            try:
                ctx_tokens = int(flags[i + 1])
            except ValueError:
                pass
            break
        # Also handle --ctx-size=<N> form
        m = re.match(r"^(?:-c|--ctx-size)=(\d+)$", token)
        if m:
            ctx_tokens = int(m.group(1))
            break
        i += 1

    if ctx_tokens is None:
        return 0.0

    # heuristic: 0.5 GiB per 4096 tokens of context
    return 0.5 * (ctx_tokens / 4096.0)
