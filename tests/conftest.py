from __future__ import annotations

import importlib.util
import os
import shutil
from contextlib import ExitStack
from unittest.mock import patch

import pytest


def pytest_ignore_collect(collection_path, config):  # noqa: ARG001
    """Skip NATS test dirs when the nats optional-extra is not installed."""
    if importlib.util.find_spec("roxabi_contracts") is not None:
        return None
    p = str(collection_path)
    if "/tests/nats" in p or p.endswith(("test_lifecycle_nats.py", "test_swap_nats.py")):
        return True
    return None


@pytest.fixture(autouse=True)
def _stub_free_vram_probe():
    # Keep unit tests deterministic: the B2 dynamic VRAM check would otherwise
    # read live NVML and fail on VRAM-constrained hosts. Tests that exercise
    # the constrained path install their own patch, which wins over this stub.
    with patch("llmcli.config.probe_free_vram_gib", return_value=1024.0):
        yield


@pytest.fixture(autouse=True)
def _stub_remote_upstream_probe(request: pytest.FixtureRequest):
    # Remote TOML entries are health-gated; default to healthy so existing
    # build_block/build_full_config tests stay deterministic without live APIs.
    if request.node.get_closest_marker("no_probe_stub"):
        yield
        return
    with patch("llmcli.support.litellm_config.probe_remote_model", return_value=True):
        yield


@pytest.fixture(autouse=True)
def _reset_model_discovery_cache():
    from llmcli.support.litellm_config import (
        _MODEL_DISCOVERY_CACHE,
        register_model_refresh_callback,
    )

    _MODEL_DISCOVERY_CACHE.invalidate()
    register_model_refresh_callback(None)
    yield
    _MODEL_DISCOVERY_CACHE.invalidate()
    register_model_refresh_callback(None)


@pytest.fixture(autouse=True)
def _isolate_xai_credentials_path(tmp_path_factory):
    # OAuth model injection (litellm_config.build_model_list) is gated on
    # XAI_CREDENTIALS_PATH.exists(). Point it at a tmp path so tests are
    # deterministic regardless of whether `llmcli xai login` was run on the
    # host running pytest. Tests that exercise the OAuth path can re-patch.
    from pathlib import Path
    fake = tmp_path_factory.mktemp("no-xai-creds") / "xai.json"
    with ExitStack() as stack:
        stack.enter_context(
            patch("llmcli.support.litellm_config._XAI_CREDENTIALS_PATH", Path(fake))
        )
        if importlib.util.find_spec("roxabi_contracts") is not None:
            stack.enter_context(
                patch("llmcli.nats._lifecycle.XAI_CREDENTIALS_PATH", Path(fake))
            )
        yield


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "no_gpu: tests that do not require a GPU (safe for CI)",
    )
    config.addinivalue_line(
        "markers",
        "gpu: tests that require a live GPU (skipped when no GPU is available)",
    )
    config.addinivalue_line(
        "markers",
        "no_probe_stub: disable autouse upstream health probe stub",
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
