"""RED-phase tests for get_engine() dispatch and bench helpers (#16).

Tests assert behaviour expected from the GREEN implementation.
Against the current state (get_engine not yet exported) they MUST fail.

Test categories:
- Dispatch to LlamaCppEngine for engine="llamacpp"
- Dispatch to LlamaCppTQ3Engine for engine="llamacpp_tq3"
- Dispatch to VLLMEngine for engine="vllm"
- ValueError raised for unknown engine name
- run_single() returns RunResult with correct fields
- render_table() handles vram_peak=None gracefully

Markers:
  no_gpu — CI-safe; no binary, no GPU required
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from llmcli.config import ModelSpec
from llmcli.engines import LlamaCppEngine, LlamaCppTQ3Engine, VLLMEngine
from llmcli.engines import get_engine  # noqa: F401 — will fail until T3 implements this
from llmcli.cli.bench import (  # noqa: F401 — will fail until run_single/render_table implemented
    BenchConfig,
    DepthStats,
    RunResult,
    render_table,
    run_single,
)


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bench_config(
    model_name: str = "test-model",
    pp_tokens: int = 16,
    tg_tokens: int = 8,
    depths: list[int] | None = None,
    runs: int = 1,
) -> BenchConfig:
    return BenchConfig(
        model_name=model_name,
        pp_tokens=pp_tokens,
        tg_tokens=tg_tokens,
        depths=depths if depths is not None else [0],
        runs=runs,
    )


def _make_sse_chunks(tokens: list[str]) -> list[bytes]:
    """Build fake SSE byte lines simulating streaming completions."""
    import json

    lines: list[bytes] = []
    for i, tok in enumerate(tokens):
        finish = "stop" if i == len(tokens) - 1 else None
        payload = json.dumps({"choices": [{"text": tok, "finish_reason": finish}]})
        lines.append(f"data: {payload}\n".encode())
    lines.append(b"data: [DONE]\n")
    return lines


# ---------------------------------------------------------------------------
# run_single() — RED tests
# ---------------------------------------------------------------------------


class TestRunSingle:
    """run_single() must return a RunResult with timing metrics populated."""

    @pytest.mark.no_gpu
    def test_run_single_returns_run_result(self) -> None:
        """run_single returns a RunResult instance with positive timing metrics."""
        # Arrange
        config = _make_bench_config()
        chunks = _make_sse_chunks(["hello", " world"])
        mock_response = MagicMock()
        mock_response.__iter__ = lambda self: iter(chunks)
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_cm.__exit__ = MagicMock(return_value=False)

        # Act
        with patch("httpx.stream", return_value=mock_cm):
            result = run_single(
                base_url="http://localhost:8091",
                config=config,
                depth=0,
                sampler=None,
            )

        # Assert
        assert isinstance(result, RunResult)
        assert result.ttft_ms > 0
        assert result.tg_tok_per_s > 0
        assert result.pp_tok_per_s > 0
        assert result.vram_peak_gib is None

    @pytest.mark.no_gpu
    def test_run_single_ttft_measured_from_first_chunk(self) -> None:
        """run_single measures TTFT from request start to first token arrival."""
        # Arrange
        config = _make_bench_config()
        chunks = _make_sse_chunks(["hello", " world"])
        delay_s = 0.1

        def _slow_iter(self: object) -> object:
            time.sleep(delay_s)
            return iter(chunks)

        mock_response = MagicMock()
        mock_response.__iter__ = _slow_iter
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_cm.__exit__ = MagicMock(return_value=False)

        # Act
        with patch("httpx.stream", return_value=mock_cm):
            result = run_single(
                base_url="http://localhost:8091",
                config=config,
                depth=0,
                sampler=None,
            )

        # Assert — TTFT must reflect the 100 ms pre-first-chunk delay
        assert result.ttft_ms >= delay_s * 1000

    @pytest.mark.no_gpu
    def test_run_single_tg_tok_per_s_calculation(self) -> None:
        """run_single computes tg_tok_per_s within 20% of N / elapsed."""
        # Arrange
        n_tokens = 10
        tokens = [f"tok{i}" for i in range(n_tokens)]
        config = _make_bench_config(tg_tokens=n_tokens)
        chunk_delay_s = 0.01  # 10 ms per token → ~100 tok/s

        original_iter = iter(_make_sse_chunks(tokens))

        def _paced_iter(self: object) -> object:
            for chunk in _make_sse_chunks(tokens):
                time.sleep(chunk_delay_s)
                yield chunk

        mock_response = MagicMock()
        mock_response.__iter__ = _paced_iter
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_response)
        mock_cm.__exit__ = MagicMock(return_value=False)

        # Act
        with patch("httpx.stream", return_value=mock_cm):
            result = run_single(
                base_url="http://localhost:8091",
                config=config,
                depth=0,
                sampler=None,
            )

        # Assert — within 20% of the theoretical rate
        theoretical = 1.0 / chunk_delay_s  # tok/s for sequential chunks
        assert result.tg_tok_per_s > 0
        assert result.tg_tok_per_s < theoretical * 1.2


# ---------------------------------------------------------------------------
# render_table() — vram_peak=None rendering
# ---------------------------------------------------------------------------


class TestRenderTableVramNone:
    """render_table() must display a placeholder when vram_peak is None."""

    @pytest.mark.no_gpu
    def test_vram_none_renders_dash(self) -> None:
        """render_table outputs '—' or 'N/A' in the VRAM column when vram_peak is None."""
        # Arrange
        stats = [
            DepthStats(
                depth=0,
                pp_mean=500.0,
                pp_std=5.0,
                tg_mean=80.0,
                tg_std=2.0,
                ttft_mean=120.0,
                vram_peak=None,
            )
        ]

        # Act
        rendered = render_table(stats, engine="llamacpp")

        # Assert — a placeholder must be present (implementation chooses '—' or 'N/A')
        assert rendered is not None
        assert "—" in rendered or "N/A" in rendered
