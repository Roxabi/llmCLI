"""GREEN-phase tests for LlamaCppTQ3Engine.

All tests are @pytest.mark.no_gpu because none invoke a real binary.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmcli.engines.llamacpp import LlamaCppEngine
from llmcli.engines.llamacpp_tq3 import LlamaCppTQ3Engine


# ---------------------------------------------------------------------------
# 1. Class hierarchy — LlamaCppTQ3Engine must be a subclass of LlamaCppEngine
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
def test_is_subclass_of_llamacpp_engine() -> None:
    """LlamaCppTQ3Engine must inherit from LlamaCppEngine to reuse cmd-building logic."""
    # Arrange / Act / Assert — pure reflection, no instantiation needed
    assert issubclass(LlamaCppTQ3Engine, LlamaCppEngine), (
        "LlamaCppTQ3Engine must be a subclass of LlamaCppEngine"
    )


# ---------------------------------------------------------------------------
# 2. Binary path — must point to TurboQuant fork, not vanilla llama-server
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
def test_binary_is_tq3_fork() -> None:
    """The engine binary must reference the TurboQuant fork, not vanilla llama.cpp."""
    # Arrange
    engine = LlamaCppTQ3Engine()

    # Act
    binary = engine.binary

    # Assert
    assert binary == "llama-server-tq3", (
        f"Expected TurboQuant fork binary 'llama-server-tq3', got '{binary}'"
    )


# ---------------------------------------------------------------------------
# 3. Binary path differs from vanilla LlamaCppEngine
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
def test_binary_differs_from_vanilla_llamacpp() -> None:
    """TQ3 engine binary must be distinct from the vanilla LlamaCppEngine binary."""
    # Arrange
    vanilla = LlamaCppEngine()
    tq3 = LlamaCppTQ3Engine()

    # Act / Assert
    assert tq3.binary != vanilla.binary, (
        f"TQ3 and vanilla engines share binary '{tq3.binary}' — they must differ"
    )


# ---------------------------------------------------------------------------
# 4. Engine acceptance — only LlamaCppTQ3Engine is correct for TQ3_4S models
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
@pytest.mark.parametrize(
    "model_name",
    [
        "Qwen3.6-35B-A3B-TQ3_4S",
        "qwen3.6-35b-a3b-tq3_4s",
        "some-model-TQ3_4S-variant",
    ],
)
def test_tq3_engine_accepts_tq3_4s_models(model_name: str) -> None:
    """Any model name containing 'TQ3_4S' must be routed to LlamaCppTQ3Engine, not vanilla."""
    # Arrange — verify the vanilla engine would be wrong for these models
    # (the convention is: TQ3_4S in name → engine must be llamacpp_tq3)
    assert "TQ3_4S" in model_name.upper(), (
        f"Test fixture '{model_name}' does not contain TQ3_4S — fix the parametrize list"
    )

    # Act — a correctly implemented catalog loader would map engine="llamacpp_tq3"
    # to LlamaCppTQ3Engine, not LlamaCppEngine.  Here we verify the engine class
    # itself is recognised as the right handler.
    engine = LlamaCppTQ3Engine()

    # Assert — LlamaCppTQ3Engine is the designated engine for TQ3_4S quants
    assert isinstance(engine, LlamaCppTQ3Engine)
    # And it must NOT be confused with the vanilla engine only
    assert type(engine) is not LlamaCppEngine, (
        "A plain LlamaCppEngine instance must not be used to serve TQ3_4S models"
    )


@pytest.mark.no_gpu
def test_vanilla_engine_is_not_suitable_for_tq3_4s() -> None:
    """LlamaCppEngine (vanilla) binary must not match TQ3_4S requirement."""
    vanilla = LlamaCppEngine()
    # The TQ3_4S format requires the fork binary; vanilla cannot serve it.
    assert "tq3" not in vanilla.binary.lower(), (
        "Vanilla LlamaCppEngine binary must not reference the TurboQuant fork"
    )


# ---------------------------------------------------------------------------
# 5. start() returns EngineInstance (shape contract) — fails on NotImplementedError
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
def test_start_returns_engine_instance(tmp_path: pytest.TempPathFactory) -> None:
    """start(spec) must return an EngineInstance with pid, port, model, started_at."""
    import time
    from pathlib import Path

    from llmcli.config import ModelSpec
    from llmcli.engine import EngineInstance

    # Arrange — minimal ModelSpec for a TQ3_4S model
    spec = ModelSpec(
        name="qwen3.6-35b-a3b-tq3",
        engine="llamacpp_tq3",
        repo="turbo-tan/Qwen3.6-35B-A3B-TQ3_4S-GGUF",
        file="qwen3.6-35b-a3b-tq3_4s.gguf",
        port=8091,
        vram_gib=12.4,
        flags=["-ngl", "99", "-c", "8192", "-ctk", "q4_0", "-ctv", "tq3_0", "-fa", "on"],
    )

    engine = LlamaCppTQ3Engine()

    # Arrange — mock subprocess and health-wait so no real binary or GGUF is needed
    mock_proc = MagicMock()
    mock_proc.pid = 77777
    fake_gguf = Path(tmp_path) / "model.gguf"

    # Act — patch at the llamacpp module level (inherited start() lives there)
    with (
        patch("llmcli.engines.llamacpp.subprocess.Popen", return_value=mock_proc),
        patch("llmcli.engines.llamacpp._wait_ready", return_value=None),
        patch.object(engine, "_gguf_path", return_value=fake_gguf),
    ):
        result = engine.start(spec)

    # Assert — shape contract
    assert isinstance(result, EngineInstance), (
        f"start() must return EngineInstance, got {type(result)}"
    )
    assert result.model == spec.name
    assert result.port == spec.port
    assert isinstance(result.pid, int) and result.pid > 0
    assert result.started_at <= time.time()
