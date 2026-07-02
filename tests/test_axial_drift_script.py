"""Subprocess integration tests for tools/check_axial_drift.sh (ADR-006).

The script greps file *contents* with:
  PRIMARY:   engines/(llamacpp|llamacpp_tq3|vllm).*def.*(wait|poll|ready)
  SECONDARY: if.*engine_type.*(vllm|llamacpp)

Both patterns must appear on a single line inside the targeted source tree.
These tests verify that the script exits non-zero and reports the offending
path when a pattern is present, and exits 0 when all source files are clean.

A regex regression in the script would silently disable the guard — these
tests make such a regression visible by failing when the guard is removed.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "tools" / "check_axial_drift.sh"

# Content that triggers SECONDARY: a line with "if.*engine_type.*(vllm|llamacpp)".
_SECONDARY_OFFENDING_LINE = "if engine_type in ('vllm', 'llamacpp'):\n    handler()\n"

# Clean content — no lines matching either pattern.
_CLEAN_ENGINE_CONTENT = (
    "class LlamaCppEngine:\n    def start(self, spec: object) -> None:\n        pass\n"
)

_CLEAN_NATS_CONTENT = "def dispatch(msg: object) -> None:\n    pass\n"


def _init_fake_repo(tmp_path: Path) -> None:
    """Initialise a minimal fake git repo + directory skeleton the script expects."""
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)

    (tmp_path / "src" / "llmcli" / "engines").mkdir(parents=True)
    (tmp_path / "src" / "llmcli" / "nats").mkdir(parents=True)
    (tmp_path / "src" / "llmcli" / "cli").mkdir(parents=True)


def _run_script(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


class TestCheckAxialDriftScript:
    """Integration tests for tools/check_axial_drift.sh."""

    @pytest.mark.parametrize(
        "engine_name, offending_line",
        [
            (
                "llamacpp",
                "# re-implemented from engines/llamacpp def _wait_ready — do not keep\n"
                "def _wait_ready_local(self) -> bool:\n"
                "    return True\n",
            ),
            (
                "llamacpp_tq3",
                "# re-implemented from engines/llamacpp_tq3 def _wait_ready — do not keep\n"
                "def _wait_ready_local(self) -> bool:\n"
                "    return True\n",
            ),
            (
                "vllm",
                "# re-implemented from engines/vllm def _poll_ready — do not keep\n"
                "def _poll_ready_local(self) -> bool:\n"
                "    return True\n",
            ),
        ],
    )
    def test_primary_detects_stage_method_breadcrumb(
        self, tmp_path: Path, engine_name: str, offending_line: str
    ) -> None:
        """Non-zero exit + offender path in output when PRIMARY pattern matches.

        Negative-test guard: if the PRIMARY grep pattern is removed from the
        script, this test will pass with exit 0, causing the assertion to fail.
        Parametrized over all three engine variants: llamacpp, llamacpp_tq3, vllm.
        """
        # Arrange
        _init_fake_repo(tmp_path)
        engine_file = tmp_path / "src" / "llmcli" / "engines" / f"{engine_name}.py"
        engine_file.write_text(offending_line)

        # Clean nats + cli so the secondary check passes
        (tmp_path / "src" / "llmcli" / "nats" / "__init__.py").write_text(_CLEAN_NATS_CONTENT)
        (tmp_path / "src" / "llmcli" / "cli" / "__init__.py").write_text(_CLEAN_NATS_CONTENT)

        # Act
        result = _run_script(tmp_path)

        # Assert
        assert result.returncode != 0, (
            "Script should exit non-zero when PRIMARY pattern fires.\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
        combined = result.stdout + result.stderr
        assert f"{engine_name}.py" in combined, (
            f"Offending file path should appear in script output.\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )

    def test_secondary_detects_dispatch_on_type(self, tmp_path: Path) -> None:
        """Non-zero exit when SECONDARY pattern (dispatch-on-type) fires in nats/.

        Negative-test guard: if the SECONDARY grep pattern is removed, this
        test will pass with exit 0, causing the assertion to fail.
        """
        # Arrange
        _init_fake_repo(tmp_path)
        (tmp_path / "src" / "llmcli" / "engines" / "llamacpp.py").write_text(_CLEAN_ENGINE_CONTENT)
        nats_file = tmp_path / "src" / "llmcli" / "nats" / "worker.py"
        nats_file.write_text(_SECONDARY_OFFENDING_LINE)
        (tmp_path / "src" / "llmcli" / "cli" / "__init__.py").write_text(_CLEAN_NATS_CONTENT)

        # Act
        result = _run_script(tmp_path)

        # Assert
        assert result.returncode != 0, (
            "Script should exit non-zero when SECONDARY pattern fires.\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )

    def test_passes_for_clean_source_tree(self, tmp_path: Path) -> None:
        """Exit 0 when neither pattern matches anywhere in the source tree."""
        # Arrange
        _init_fake_repo(tmp_path)
        (tmp_path / "src" / "llmcli" / "engines" / "llamacpp.py").write_text(_CLEAN_ENGINE_CONTENT)
        (tmp_path / "src" / "llmcli" / "nats" / "__init__.py").write_text(_CLEAN_NATS_CONTENT)
        (tmp_path / "src" / "llmcli" / "cli" / "__init__.py").write_text(_CLEAN_NATS_CONTENT)

        # Act
        result = _run_script(tmp_path)

        # Assert
        assert result.returncode == 0, (
            f"Script should exit 0 for a clean source tree.\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )
