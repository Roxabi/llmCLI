from __future__ import annotations

import os
import shutil

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "no_gpu: tests that do not require a GPU (safe for CI)",
    )
    config.addinivalue_line(
        "markers",
        "gpu: tests that require a live GPU (skipped when no GPU is available)",
    )


def _gpu_available() -> bool:
    """Return True if a GPU is available for tests.

    Detection order:
    1. ``LLMCLI_NO_GPU=1``  → no GPU (explicit override)
    2. ``SKIP_GPU_TESTS=1`` → no GPU (CI legacy env var)
    3. ``shutil.which("nvidia-smi")`` absent → no GPU (no CUDA toolchain on PATH)
    4. Otherwise → GPU assumed available
    """
    if os.environ.get("LLMCLI_NO_GPU") == "1":
        return False
    if os.environ.get("SKIP_GPU_TESTS") == "1":
        return False
    if shutil.which("nvidia-smi") is None:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip @pytest.mark.gpu tests when no GPU is available."""
    if _gpu_available():
        return

    skip_marker = pytest.mark.skip(
        reason=(
            "GPU not available — set LLMCLI_NO_GPU=0 and ensure nvidia-smi is on PATH "
            "to run GPU integration tests"
        )
    )
    for item in items:
        if "gpu" in item.keywords and "no_gpu" not in item.keywords:
            item.add_marker(skip_marker)
