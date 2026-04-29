"""RED-phase tests for get_engine() dispatch (#16).

Tests assert behaviour expected from the GREEN implementation.
Against the current state (get_engine not yet exported) they MUST fail.

Test categories:
- Dispatch to LlamaCppEngine for engine="llamacpp"
- Dispatch to LlamaCppTQ3Engine for engine="llamacpp_tq3"
- Dispatch to VLLMEngine for engine="vllm"
- ValueError raised for unknown engine name

Markers:
  no_gpu — CI-safe; no binary, no GPU required
"""
from __future__ import annotations

import pytest

from llmcli.config import ModelSpec
from llmcli.engines import LlamaCppEngine, LlamaCppTQ3Engine, VLLMEngine
from llmcli.engines import get_engine  # noqa: F401 — will fail until T3 implements this


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def minimal_spec() -> ModelSpec:
    """Minimal ModelSpec suitable for dispatch tests."""
    return ModelSpec(
        name="test-model",
        engine="llamacpp",
        repo="org/test-model-gguf",
        port=8091,
        vram_gib=5.0,
    )


# ---------------------------------------------------------------------------
# get_engine() dispatch
# ---------------------------------------------------------------------------


class TestGetEngineDispatch:
    """get_engine() must return the correct engine class instance for each engine name."""

    @pytest.mark.no_gpu
    def test_get_engine_dispatch_llamacpp(self, minimal_spec: ModelSpec) -> None:
        """get_engine returns a LlamaCppEngine instance when engine='llamacpp'."""
        # Arrange
        spec = ModelSpec(
            name=minimal_spec.name,
            engine="llamacpp",
            repo=minimal_spec.repo,
            port=minimal_spec.port,
            vram_gib=minimal_spec.vram_gib,
        )

        # Act
        engine = get_engine(spec)

        # Assert
        assert isinstance(engine, LlamaCppEngine), (
            f"get_engine() must return LlamaCppEngine for engine='llamacpp', got {type(engine)}"
        )

    @pytest.mark.no_gpu
    def test_get_engine_dispatch_tq3(self, minimal_spec: ModelSpec) -> None:
        """get_engine returns a LlamaCppTQ3Engine instance when engine='llamacpp_tq3'."""
        # Arrange
        spec = ModelSpec(
            name=minimal_spec.name,
            engine="llamacpp_tq3",
            repo=minimal_spec.repo,
            port=minimal_spec.port,
            vram_gib=minimal_spec.vram_gib,
        )

        # Act
        engine = get_engine(spec)

        # Assert
        assert isinstance(engine, LlamaCppTQ3Engine), (
            f"get_engine() must return LlamaCppTQ3Engine for engine='llamacpp_tq3', got {type(engine)}"
        )

    @pytest.mark.no_gpu
    def test_get_engine_dispatch_vllm(self, minimal_spec: ModelSpec) -> None:
        """get_engine returns a VLLMEngine instance when engine='vllm'."""
        # Arrange
        spec = ModelSpec(
            name=minimal_spec.name,
            engine="vllm",
            repo=minimal_spec.repo,
            port=minimal_spec.port,
            vram_gib=minimal_spec.vram_gib,
        )

        # Act
        engine = get_engine(spec)

        # Assert
        assert isinstance(engine, VLLMEngine), (
            f"get_engine() must return VLLMEngine for engine='vllm', got {type(engine)}"
        )

    @pytest.mark.no_gpu
    def test_get_engine_unknown_raises(self, minimal_spec: ModelSpec) -> None:
        """get_engine raises ValueError for an unrecognised engine name."""
        # Arrange
        spec = ModelSpec(
            name=minimal_spec.name,
            engine="unknown_engine",
            repo=minimal_spec.repo,
            port=minimal_spec.port,
            vram_gib=minimal_spec.vram_gib,
        )

        # Act / Assert
        with pytest.raises(ValueError, match="unknown_engine"):
            get_engine(spec)
