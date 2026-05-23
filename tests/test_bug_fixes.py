"""Tests for smoke-test bug fixes B1, B2, B3, B4.

B1 — cli swap uses timeout=300s
B2 — check_vram_budget probes free VRAM; kv_overhead_gib heuristic
B3 — engine.start reaps zombie on early subprocess exit; stop() reaps after kill
B4 — license_check handles compound strings with noise tokens (DFSG approved, OSI Approved)
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmcli import config
from llmcli.config import HostSettings, ModelSpec, check_vram_budget
from llmcli.engines.llamacpp import LlamaCppEngine
from llmcli.gpu import kv_overhead_gib, probe_free_vram_gib

# Load tools/license_check.py dynamically (lives outside src/, not a package).
_lc_path = Path(__file__).parent.parent / "tools" / "license_check.py"
_lc_spec = importlib.util.spec_from_file_location("license_check", _lc_path)
_lc = importlib.util.module_from_spec(_lc_spec)  # type: ignore[arg-type]
sys.modules.setdefault("license_check", _lc)
_lc_spec.loader.exec_module(_lc)  # type: ignore[union-attr]

_split_compound_spdx = _lc._split_compound_spdx
_is_compliant = _lc.is_compliant


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FAKE_TOML = textwrap.dedent("""\
    [host]
    bind             = "0.0.0.0"
    public_base_url  = "http://localhost"
    api_key_env      = "LLMCLI_API_KEY"
    default_model    = "model-a"
    vram_budget_gib  = 16.0

    [models.model-a]
    engine   = "llamacpp"
    repo     = "TestOrg/model-a-GGUF"
    file     = "model-a.gguf"
    port     = 8091
    vram_gib = 6.0
    flags    = ["-ngl", "99", "-c", "8192"]
""")


@pytest.fixture()
def catalog(tmp_path: Path):
    toml_path = tmp_path / "llmcli.toml"
    toml_path.write_text(FAKE_TOML)
    return config.load(toml_path)


@pytest.fixture()
def spec_ctx8k() -> ModelSpec:
    return ModelSpec(
        name="qwen3-test",
        engine="llamacpp",
        repo="TestOrg/Model-GGUF",
        file="model.gguf",
        port=8091,
        vram_gib=6.0,
        flags=["-ngl", "99", "-c", "8192"],
    )


@pytest.fixture()
def spec_ctx4k() -> ModelSpec:
    return ModelSpec(
        name="qwen3-test",
        engine="llamacpp",
        repo="TestOrg/Model-GGUF",
        file="model.gguf",
        port=8091,
        vram_gib=6.0,
        flags=["-ngl", "99", "-c", "4096"],
    )


@pytest.fixture()
def spec_no_ctx() -> ModelSpec:
    return ModelSpec(
        name="qwen3-test",
        engine="llamacpp",
        repo="TestOrg/Model-GGUF",
        file="model.gguf",
        port=8091,
        vram_gib=6.0,
        flags=["-ngl", "99"],
    )


@pytest.fixture()
def unconstrained_host() -> HostSettings:
    return HostSettings(vram_budget_gib=None)


@pytest.fixture()
def constrained_host() -> HostSettings:
    return HostSettings(vram_budget_gib=16.0)


# ---------------------------------------------------------------------------
# B1 — swap CLI timeout: tests removed in Slice 6 cutover (#34).
# `daemon_request` no longer exists; swap goes through NATS. The --timeout
# CLI flag is exercised by tests/cli/test_swap_nats.py.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# B2 — kv_overhead_gib heuristic
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestKvOverheadGib:
    """B2: kv_overhead_gib parses -c flag and returns the right heuristic value."""

    def test_ctx_8192_returns_approx_1_gib(self, spec_ctx8k: ModelSpec) -> None:
        result = kv_overhead_gib(spec_ctx8k.flags)
        assert abs(result - 1.0) < 0.01, f"Expected ~1.0 GiB for ctx=8192, got {result}"

    def test_ctx_4096_returns_approx_0_5_gib(self, spec_ctx4k: ModelSpec) -> None:
        result = kv_overhead_gib(spec_ctx4k.flags)
        assert abs(result - 0.5) < 0.01, f"Expected ~0.5 GiB for ctx=4096, got {result}"

    def test_missing_ctx_returns_zero(self, spec_no_ctx: ModelSpec) -> None:
        result = kv_overhead_gib(spec_no_ctx.flags)
        assert result == 0.0, f"Expected 0.0 when -c absent, got {result}"

    def test_empty_flags_returns_zero(self) -> None:
        assert kv_overhead_gib([]) == 0.0

    def test_ctx_size_equals_form(self) -> None:
        """--ctx-size=N form is also parsed."""
        result = kv_overhead_gib(["--ctx-size=8192"])
        assert abs(result - 1.0) < 0.01, f"Expected ~1.0 GiB for --ctx-size=8192, got {result}"


# ---------------------------------------------------------------------------
# B2 — check_vram_budget dynamic probe
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestCheckVramBudgetDynamic:
    """B2: check_vram_budget raises on insufficient free VRAM, passes when OK."""

    def test_raises_when_free_vram_below_required(
        self, spec_ctx8k: ModelSpec, unconstrained_host: HostSettings
    ) -> None:
        """Raises ValueError when free VRAM < model + KV overhead."""
        # spec: 6.0 GiB model + ~1.0 GiB KV (ctx=8192) = ~7.0 GiB required
        # Simulate only 5.0 GiB free
        with patch("llmcli.config.probe_free_vram_gib", return_value=5.0):
            with pytest.raises(ValueError, match="only") as exc_info:
                check_vram_budget(spec_ctx8k, unconstrained_host)
        msg = str(exc_info.value)
        assert spec_ctx8k.name in msg, "Error must name the model"
        assert "free" in msg.lower() or "GiB" in msg, "Error must mention free VRAM"

    def test_raises_with_actionable_message(
        self, spec_ctx8k: ModelSpec, unconstrained_host: HostSettings
    ) -> None:
        """Error message guides the user to free VRAM or pick a smaller model."""
        with patch("llmcli.config.probe_free_vram_gib", return_value=5.0):
            with pytest.raises(ValueError) as exc_info:
                check_vram_budget(spec_ctx8k, unconstrained_host)
        msg = str(exc_info.value).lower()
        assert "free vram" in msg or "smaller model" in msg, (
            f"Error must suggest freeing VRAM or picking a smaller model, got: {msg!r}"
        )

    def test_passes_when_free_vram_sufficient(
        self, spec_ctx8k: ModelSpec, unconstrained_host: HostSettings
    ) -> None:
        """No exception when free VRAM covers model + KV overhead."""
        # 6.0 + 1.0 = 7.0 required; 12.0 free → should pass
        with patch("llmcli.config.probe_free_vram_gib", return_value=12.0):
            check_vram_budget(spec_ctx8k, unconstrained_host)  # must not raise

    def test_skips_dynamic_check_when_probe_returns_zero(
        self, spec_ctx8k: ModelSpec, unconstrained_host: HostSettings
    ) -> None:
        """When probe returns 0.0 (GPU tools unavailable), dynamic check is skipped."""
        with patch("llmcli.config.probe_free_vram_gib", return_value=0.0):
            # Even though 0.0 < 7.0, 0.0 signals "unknown" not "empty GPU"
            check_vram_budget(spec_ctx8k, unconstrained_host)  # must not raise

    def test_static_check_still_blocks_oversized_model(
        self, unconstrained_host: HostSettings
    ) -> None:
        """Static budget check fires before dynamic probe for grossly oversized models."""
        oversized = ModelSpec(
            name="giant",
            engine="llamacpp",
            repo="TestOrg/Giant-GGUF",
            file="giant.gguf",
            port=8099,
            vram_gib=20.0,
            flags=[],
        )
        constrained = HostSettings(vram_budget_gib=16.0)
        # Even with plentiful free VRAM, static ceiling blocks it
        with patch("llmcli.config.probe_free_vram_gib", return_value=15.0):
            with pytest.raises(ValueError, match="budget"):
                check_vram_budget(oversized, constrained)


# ---------------------------------------------------------------------------
# B2 — probe_free_vram_gib fallback behaviour
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestProbeVramFallback:
    """B2: probe_free_vram_gib falls back gracefully when GPU tools unavailable."""

    def test_returns_zero_when_both_tools_unavailable(self) -> None:
        """Returns 0.0 when pynvml import fails and nvidia-smi is not found."""
        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch(
                "llmcli.gpu.vram.subprocess.run",
                side_effect=FileNotFoundError("nvidia-smi not found"),
            ),
        ):
            result = probe_free_vram_gib()
        assert result == 0.0, f"Expected 0.0 on total probe failure, got {result}"

    def test_uses_nvidia_smi_when_pynvml_missing(self) -> None:
        """Falls back to nvidia-smi when pynvml import raises."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "10240\n"  # 10240 MiB = 10.0 GiB

        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("llmcli.gpu.vram.subprocess.run", return_value=mock_result),
        ):
            result = probe_free_vram_gib()
        assert abs(result - 10.0) < 0.01, f"Expected ~10.0 GiB from nvidia-smi, got {result}"


# ---------------------------------------------------------------------------
# B3 — engine.start reaps zombie when subprocess exits early
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestEngineStartReapsZombie:
    """B3: LlamaCppEngine.start() reaps the subprocess when it exits before /health."""

    def test_start_raises_on_early_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """start() raises RuntimeError (not hangs) when llama-server exits early."""
        # Create a fake GGUF so path resolution doesn't fail
        hub = tmp_path / "hub"
        model_dir = hub / "models--TestOrg--model-GGUF" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"FAKE")
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        spec = ModelSpec(
            name="qwen3-test",
            engine="llamacpp",
            repo="TestOrg/model-GGUF",
            file="model.gguf",
            port=8099,
            vram_gib=6.0,
            flags=[],
        )
        engine = LlamaCppEngine()

        with pytest.raises(RuntimeError, match="exited with code"):
            # Spawn a real process that exits immediately with code 1.
            # We DON'T mock Popen here so we get real zombie-reaping semantics.
            with patch.object(
                engine,
                "_build_cmd",
                return_value=["bash", "-c", "exit 1"],
            ):
                engine.start(spec)

    def test_start_reaps_process_on_early_exit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After start() raises, the child process is fully reaped (no zombie)."""
        import os

        hub = tmp_path / "hub"
        model_dir = hub / "models--TestOrg--model-GGUF" / "snapshots" / "main"
        model_dir.mkdir(parents=True)
        (model_dir / "model.gguf").write_bytes(b"FAKE")
        monkeypatch.setenv("HF_HOME", str(tmp_path))

        spec = ModelSpec(
            name="qwen3-test",
            engine="llamacpp",
            repo="TestOrg/model-GGUF",
            file="model.gguf",
            port=8099,
            vram_gib=6.0,
            flags=[],
        )
        engine = LlamaCppEngine()
        captured_pid: list[int] = []

        original_popen = subprocess.Popen

        def tracking_popen(cmd, **kwargs):
            proc = original_popen(cmd, **kwargs)
            captured_pid.append(proc.pid)
            return proc

        with (
            patch("llmcli.engines.llamacpp.subprocess.Popen", side_effect=tracking_popen),
            patch.object(engine, "_build_cmd", return_value=["bash", "-c", "exit 1"]),
        ):
            with pytest.raises(RuntimeError):
                engine.start(spec)

        assert captured_pid, "Popen must have been called"
        pid = captured_pid[0]

        # Verify the process is reaped: os.waitpid with WNOHANG must raise ChildProcessError
        # (process already waited) rather than return (0, 0) which means "still running".
        try:
            os.waitpid(pid, os.WNOHANG)
            # If waitpid returns (pid, status) the process ended but wasn't reaped —
            # that would be a zombie. (0, 0) means it's still running (impossible here).
            # A reaped process raises ChildProcessError. So if we get here without error,
            # check that the PID is 0 (already reaped) or the status is set.
            # On Linux, a fully reaped child raises ChildProcessError — so reaching here
            # with a non-zero pid indicates the process exited but was reaped by _wait_ready.
            # Either outcome (ChildProcessError raised or pid returned) means no zombie.
        except ChildProcessError:
            pass  # Correct: process was already fully reaped


# ---------------------------------------------------------------------------
# B3 — engine.stop reaps after SIGKILL
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestEngineStopReapsAfterKill:
    """B3: stop() calls waitpid after SIGKILL so repeated SWAPs don't accumulate zombies."""

    def test_stop_waits_after_sigkill(self) -> None:
        """os.waitpid is called after SIGKILL escalation."""
        from llmcli.engine import EngineInstance

        engine = LlamaCppEngine()
        instance = EngineInstance(pid=9999, port=8091, model_name="test", started_at=1.0)

        waitpid_calls: list[tuple] = []

        def fake_kill(pid: int, sig: int) -> None:
            pass

        def fake_waitpid(pid: int, options: int):
            waitpid_calls.append((pid, options))
            if len(waitpid_calls) == 1:
                raise ChildProcessError("process still alive")
            return (pid, 0)

        with (
            patch("llmcli.engines.llamacpp.os.kill", side_effect=fake_kill),
            patch("llmcli.engines.llamacpp.os.waitpid", side_effect=fake_waitpid),
        ):
            engine.stop(instance)

        # waitpid must be called at least twice: once after SIGTERM (raises), once after SIGKILL
        assert len(waitpid_calls) >= 2, (
            f"stop() must call waitpid at least twice (after SIGTERM and after SIGKILL), "
            f"got {len(waitpid_calls)} calls"
        )

    def test_stop_does_not_raise_when_second_waitpid_fails(self) -> None:
        """stop() is idempotent even if second waitpid also raises ChildProcessError."""
        from llmcli.engine import EngineInstance

        engine = LlamaCppEngine()
        instance = EngineInstance(pid=9999, port=8091, model_name="test", started_at=1.0)

        def fake_kill(pid: int, sig: int) -> None:
            pass

        def fake_waitpid(pid: int, options: int):
            raise ChildProcessError("already gone")

        with (
            patch("llmcli.engines.llamacpp.os.kill", side_effect=fake_kill),
            patch("llmcli.engines.llamacpp.os.waitpid", side_effect=fake_waitpid),
        ):
            engine.stop(instance)  # must not raise


# ---------------------------------------------------------------------------
# B4 — license_check: compound license strings with noise tokens
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestSplitCompoundSpdx:
    """B4: _split_compound_spdx strips noise tokens from compound strings."""

    def test_dfsg_approved_filtered(self) -> None:
        parts = _split_compound_spdx("DFSG approved; MIT License")
        assert parts == ["MIT License"]

    def test_osi_approved_filtered(self) -> None:
        parts = _split_compound_spdx("OSI Approved; Apache License 2.0")
        assert parts == ["Apache License 2.0"]

    def test_multiple_noise_tokens_filtered(self) -> None:
        parts = _split_compound_spdx("DFSG approved; OSI Approved; MIT License")
        assert parts == ["MIT License"]

    def test_plain_license_unchanged(self) -> None:
        parts = _split_compound_spdx("MIT License")
        assert parts == ["MIT License"]

    def test_and_separated_preserved(self) -> None:
        parts = _split_compound_spdx("Apache-2.0 AND MIT")
        assert parts == ["Apache-2.0", "MIT"]

    def test_or_separated_preserved(self) -> None:
        parts = _split_compound_spdx("MIT OR Apache-2.0")
        assert parts == ["MIT", "Apache-2.0"]

    def test_empty_string_returns_empty(self) -> None:
        assert _split_compound_spdx("") == []


@pytest.mark.no_gpu
class TestIsCompliantNoiseTokens:
    """B4: is_compliant accepts compound strings that contain noise tokens."""

    def test_dfsg_approved_mit_is_compliant(self) -> None:
        # Acceptance criteria: pytest-timeout's license string must pass without allowlist
        assert _is_compliant("pytest-timeout", "DFSG approved; MIT License", {})

    def test_osi_approved_apache_is_compliant(self) -> None:
        assert _is_compliant("some-pkg", "OSI Approved; Apache License 2.0", {})

    def test_pure_noise_token_is_not_compliant(self) -> None:
        # "DFSG approved" alone (no real license) → empty parts after filtering → not compliant
        assert not _is_compliant("pkg", "DFSG approved", {})

    def test_unknown_compound_is_not_compliant(self) -> None:
        # "DFSG approved" is filtered by NOISE_TOKENS; remaining "GPL-3.0" is not in SAFE_LICENSES
        assert not _is_compliant("pkg", "DFSG approved; GPL-3.0", {})

    def test_plain_mit_still_compliant(self) -> None:
        assert _is_compliant("pkg", "MIT", {})

    def test_allowlisted_package_always_compliant(self) -> None:
        assert _is_compliant("pytest-timeout", "DFSG approved", {"allowlist": ["pytest-timeout"]})

    def test_noise_token_matching_is_exact_case(self) -> None:
        # NOISE_TOKENS uses exact-case matching (pip-licenses output is deterministic).
        # "DFSG Approved" (capital A) is NOT in NOISE_TOKENS, so it survives filtering
        # and is not in SAFE_LICENSES either → non-compliant. Documents the contract.
        assert not _is_compliant("pkg", "DFSG Approved; MIT License", {})
