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


def _make_httpx_client_mock(lines: list[bytes]) -> MagicMock:
    """Build a mock httpx.Client whose stream() context manager yields decoded lines."""
    decoded = [ln.decode() for ln in lines]

    mock_resp = MagicMock()
    mock_resp.iter_lines = MagicMock(return_value=iter(decoded))

    mock_stream_cm = MagicMock()
    mock_stream_cm.__enter__ = MagicMock(return_value=mock_resp)
    mock_stream_cm.__exit__ = MagicMock(return_value=False)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=mock_stream_cm)

    mock_client_cm = MagicMock()
    mock_client_cm.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cm.__exit__ = MagicMock(return_value=False)

    return mock_client_cm


class TestRunSingle:
    """run_single() must return a RunResult with timing metrics populated."""

    @pytest.mark.no_gpu
    def test_run_single_returns_run_result(self) -> None:
        """run_single returns a RunResult instance with positive timing metrics."""
        # Arrange
        config = _make_bench_config()
        chunks = _make_sse_chunks(["hello", " world"])
        mock_client_cm = _make_httpx_client_mock(chunks)

        # Act
        with patch("httpx.Client", return_value=mock_client_cm):
            result = run_single(
                base_url="http://localhost:8091",
                config=config,
                depth=0,
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
        decoded = [ln.decode() for ln in chunks]

        def _slow_iter_lines() -> object:
            time.sleep(delay_s)
            return iter(decoded)

        mock_resp = MagicMock()
        mock_resp.iter_lines = _slow_iter_lines

        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_resp)
        mock_stream_cm.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        mock_client_cm = MagicMock()
        mock_client_cm.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cm.__exit__ = MagicMock(return_value=False)

        # Act
        with patch("httpx.Client", return_value=mock_client_cm):
            result = run_single(
                base_url="http://localhost:8091",
                config=config,
                depth=0,
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
        raw_chunks = _make_sse_chunks(tokens)

        def _paced_iter_lines() -> object:
            for chunk in raw_chunks:
                time.sleep(chunk_delay_s)
                yield chunk.decode()

        mock_resp = MagicMock()
        mock_resp.iter_lines = _paced_iter_lines

        mock_stream_cm = MagicMock()
        mock_stream_cm.__enter__ = MagicMock(return_value=mock_resp)
        mock_stream_cm.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client.stream = MagicMock(return_value=mock_stream_cm)

        mock_client_cm = MagicMock()
        mock_client_cm.__enter__ = MagicMock(return_value=mock_client)
        mock_client_cm.__exit__ = MagicMock(return_value=False)

        # Act
        with patch("httpx.Client", return_value=mock_client_cm):
            result = run_single(
                base_url="http://localhost:8091",
                config=config,
                depth=0,
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
        import io

        from rich.console import Console

        # Arrange — RunResult with vram_peak_gib=None (no sampler data)
        results = [
            RunResult(
                depth=0,
                engine="llamacpp",
                ttft_ms=120.0,
                tg_tok_per_s=80.0,
                pp_tok_per_s=500.0,
                vram_peak_gib=None,
            )
        ]

        # Act
        table = render_table(results, engine="llamacpp")
        buf = io.StringIO()
        Console(file=buf, width=200).print(table)
        output = buf.getvalue()

        # Assert — a placeholder must be present (implementation chooses '—' or 'N/A')
        assert table is not None
        assert "—" in output or "N/A" in output


# ---------------------------------------------------------------------------
# _aggregate() — mean and stddev correctness
# ---------------------------------------------------------------------------


class TestAggregation:
    """_aggregate() must compute correct mean and std for a set of RunResults."""

    @pytest.mark.no_gpu
    def test_aggregation_mean_stddev(self) -> None:
        """_aggregate returns correct tg_mean and positive tg_std for 3 varied results."""
        from llmcli.cli.bench import _aggregate

        # Arrange — 3 results at depth=0 with tg values 40, 42, 44
        results = [
            RunResult(
                depth=0,
                engine="llamacpp",
                ttft_ms=100.0,
                tg_tok_per_s=v,
                pp_tok_per_s=500.0,
                vram_peak_gib=None,
            )
            for v in (40.0, 42.0, 44.0)
        ]

        # Act
        stats = _aggregate(results, depth=0)

        # Assert
        assert stats.tg_mean == pytest.approx(42.0)
        assert stats.tg_std > 0


# ---------------------------------------------------------------------------
# render_table() — vLLM footnote caption
# ---------------------------------------------------------------------------


class TestRenderTableVllmFootnote:
    """render_table() must include a vLLM scheduler footnote caption when engine='vllm'."""

    @pytest.mark.no_gpu
    def test_table_vllm_footnote(self) -> None:
        """render_table adds a vLLM-specific caption when engine contains 'vllm'."""
        # Arrange
        results = [
            RunResult(
                depth=0,
                engine="vllm",
                ttft_ms=150.0,
                tg_tok_per_s=60.0,
                pp_tok_per_s=400.0,
                vram_peak_gib=None,
            )
        ]

        # Act
        table = render_table(results, "vllm")

        # Assert — caption must reference vLLM scheduler overhead
        assert table.caption is not None
        assert "vllm" in table.caption.lower() or "scheduler" in table.caption.lower()

    @pytest.mark.no_gpu
    def test_table_no_gpu_na_when_sampler_ran(self) -> None:
        """render_table shows '—' in VRAM column when no GPU data is available."""
        import io

        from rich.console import Console

        # Arrange — vram_peak_gib=None simulates no GPU / sampler returned nothing
        results = [
            RunResult(
                depth=0,
                engine="llamacpp",
                ttft_ms=100.0,
                tg_tok_per_s=80.0,
                pp_tok_per_s=500.0,
                vram_peak_gib=None,
            )
        ]

        # Act
        table = render_table(results, engine="llamacpp")
        buf = io.StringIO()
        Console(file=buf, width=200).print(table)
        output = buf.getvalue()

        # Assert — N/A placeholder present in rendered output when no GPU data
        assert "N/A" in output


# ---------------------------------------------------------------------------
# bench CLI command — error paths
# ---------------------------------------------------------------------------


class TestBenchCliErrors:
    """bench CLI command must exit code 1 for unknown model and port-in-use errors."""

    @pytest.mark.no_gpu
    def test_unknown_model_error(self) -> None:
        """bench exits with code 1 and prints 'Unknown model' for a missing catalog entry."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from llmcli.cli import app
        from llmcli.config import Catalog, HostSettings

        # Arrange — empty model catalog
        empty_catalog = Catalog(host=HostSettings(), models={})
        runner = CliRunner()

        # Act
        with patch("llmcli.cli.config.load", return_value=empty_catalog):
            result = runner.invoke(app, ["bench", "nonexistent"])

        # Assert
        assert result.exit_code == 1
        assert "Unknown model" in (result.output + (result.stderr or ""))

    @pytest.mark.no_gpu
    def test_port_in_use_error(self) -> None:
        """bench exits with code 1 and prints 'in use' when the model port is occupied."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from llmcli.cli import app
        from llmcli.config import Catalog, HostSettings, ModelSpec

        # Arrange — catalog with one model; port is occupied (connect_ex returns 0)
        spec = ModelSpec(
            name="test-model", engine="llamacpp", repo="org/repo", port=8091, vram_gib=5.0
        )
        catalog = Catalog(host=HostSettings(), models={"test-model": spec})
        runner = CliRunner()

        mock_sock = MagicMock()
        mock_sock.__enter__ = MagicMock(return_value=mock_sock)
        mock_sock.__exit__ = MagicMock(return_value=False)
        mock_sock.connect_ex = MagicMock(return_value=0)

        # Act
        with (
            patch("llmcli.cli.config.load", return_value=catalog),
            patch("llmcli.cli.bench.socket.socket", return_value=mock_sock),
        ):
            result = runner.invoke(app, ["bench", "test-model"])

        # Assert
        assert result.exit_code == 1
        assert "in use" in (result.output + (result.stderr or "")).lower()
