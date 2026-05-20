"""Integration tests for VRAM guard CLI error (SC-13, C2, T4.3).

Simulates a prod scenario: host budget 10 GiB (roxabituwer RTX 3080),
model requires 12 GiB — serve must refuse with exit code != 0 and a
user-friendly, actionable error message.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from llmcli.cli import app

runner = CliRunner()


PROD_SCENARIO_TOML = textwrap.dedent("""\
    [host]
    bind             = "0.0.0.0"
    public_base_url  = "http://roxabituwer.lan"
    api_key_env      = "LLMCLI_API_KEY"
    default_model    = "qwen3-14b-q5"
    vram_budget_gib  = 10.0

    [models.qwen3-14b-q5]
    engine   = "llamacpp"
    repo     = "SomeOrg/Qwen3-14B-GGUF"
    file     = "qwen3-14b-q5_k_m.gguf"
    port     = 8091
    vram_gib = 12.0
    flags    = ["-ngl", "99"]
""")


@pytest.fixture()
def prod_catalog(tmp_path: Path):
    """Catalog with an oversized model (12 GiB) against a 10 GiB prod budget."""
    from llmcli import config

    toml_path = tmp_path / "llmcli.toml"
    toml_path.write_text(PROD_SCENARIO_TOML)
    return config.load(toml_path)


@pytest.fixture()
def patched_prod_catalog(prod_catalog):
    """Patch config.load in cli module to return the prod catalog."""
    from llmcli import config as real_config

    real_check = getattr(real_config, "check_vram_budget", None)

    with patch("llmcli.cli.config", create=True) as mock_config_mod:
        mock_config_mod.load.return_value = prod_catalog
        if real_check is not None:
            mock_config_mod.check_vram_budget.side_effect = real_check
        yield prod_catalog


class TestVramGuardProdScenario:
    """End-to-end VRAM guard verification: 12 GiB model vs 10 GiB host budget."""

    def test_serve_exits_nonzero_when_model_exceeds_prod_budget(self, patched_prod_catalog) -> None:
        """serve exits non-zero when model VRAM exceeds prod host budget (SC-13)."""
        result = runner.invoke(app, ["serve", "--name", "qwen3-14b-q5"])
        assert result.exit_code != 0, (
            f"Expected non-zero exit for oversized model, got {result.exit_code}. "
            f"Output: {result.output!r}"
        )

    def test_serve_vram_error_names_model(self, patched_prod_catalog) -> None:
        """Error output contains the model name (SC-13)."""
        result = runner.invoke(app, ["serve", "--name", "qwen3-14b-q5"])
        combined = result.output + (result.stderr or "")
        assert "qwen3-14b-q5" in combined, (
            f"Expected model name 'qwen3-14b-q5' in output. Got: {combined!r}"
        )

    def test_serve_vram_error_states_required_vram(self, patched_prod_catalog) -> None:
        """Error output contains the required VRAM amount (12 GiB) (SC-13)."""
        result = runner.invoke(app, ["serve", "--name", "qwen3-14b-q5"])
        combined = result.output + (result.stderr or "")
        assert "12" in combined, f"Expected required VRAM (12) in output. Got: {combined!r}"

    def test_serve_vram_error_states_available_budget(self, patched_prod_catalog) -> None:
        """Error output contains the host VRAM budget (10 GiB) (SC-13)."""
        result = runner.invoke(app, ["serve", "--name", "qwen3-14b-q5"])
        combined = result.output + (result.stderr or "")
        assert "10" in combined, f"Expected budget (10) in output. Got: {combined!r}"

    def test_serve_vram_error_includes_remediation_hint(self, patched_prod_catalog) -> None:
        """Error output includes a remediation hint (smaller model / deployment docs) (SC-13)."""
        result = runner.invoke(app, ["serve", "--name", "qwen3-14b-q5"])
        combined = result.output + (result.stderr or "")
        assert any(
            kw in combined.lower() for kw in ("smaller", "budget", "deployment", "docs", "catalog")
        ), f"Expected remediation hint in output. Got: {combined!r}"

    def test_serve_vram_error_does_not_start_daemon(self, patched_prod_catalog) -> None:
        """Daemon.serve is never called when VRAM guard rejects the model (SC-13)."""
        with patch("llmcli.cli.Daemon", create=True) as mock_daemon_cls:
            mock_daemon_instance = mock_daemon_cls.return_value
            runner.invoke(app, ["serve", "--name", "qwen3-14b-q5"])
        mock_daemon_instance.serve.assert_not_called()

    def test_serve_default_model_also_blocked_by_vram_guard(self, patched_prod_catalog) -> None:
        """serve with no --name uses default_model and still applies VRAM guard (SC-13)."""
        # default_model = "qwen3-14b-q5" which is 12 GiB > 10 GiB budget
        result = runner.invoke(app, ["serve"])
        assert result.exit_code != 0, (
            f"Expected non-zero exit for oversized default model. Output: {result.output!r}"
        )
