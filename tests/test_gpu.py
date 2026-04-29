"""Tests for VRAMSampler in llmcli.gpu (#16).

Test categories:
- pynvml available — returns float GiB
- pynvml absent — falls back to nvidia-smi
- no GPU at all — returns None

Markers:
  no_gpu — CI-safe; no binary, no GPU required
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from llmcli.gpu import VRAMSampler


# ---------------------------------------------------------------------------
# VRAMSampler
# ---------------------------------------------------------------------------


class TestVRAMSampler:
    """VRAMSampler polls GPU VRAM in a background thread and returns peak GiB."""

    @pytest.mark.no_gpu
    def test_vram_sampler_returns_float_with_pynvml(self) -> None:
        """VRAMSampler returns a float GiB value when pynvml is available."""
        # Arrange
        mem_mock = MagicMock()
        mem_mock.used = 4 * 1024**3  # 4 GiB in bytes

        handle_mock = MagicMock()

        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.return_value = None
        pynvml_mock.nvmlDeviceGetHandleByIndex.return_value = handle_mock
        pynvml_mock.nvmlDeviceGetMemoryInfo.return_value = mem_mock
        pynvml_mock.nvmlShutdown.return_value = None

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            sampler = VRAMSampler(poll_interval=0.05)

            # Act
            sampler.start()
            time.sleep(0.3)
            peak = sampler.stop()

        # Assert
        assert isinstance(peak, float)
        assert peak == pytest.approx(4.0, rel=0.1)

    @pytest.mark.no_gpu
    def test_vram_sampler_fallback_nvidia_smi_when_pynvml_absent(self) -> None:
        """VRAMSampler falls back to nvidia-smi when pynvml is not importable."""
        # Arrange — pynvml absent; nvidia-smi returns 4096 MiB (4 GiB)
        smi_result = MagicMock()
        smi_result.returncode = 0
        smi_result.stdout = "4096\n"

        with (
            patch.dict("sys.modules", {"pynvml": None}),
            patch("subprocess.run", return_value=smi_result) as mock_run,
        ):
            sampler = VRAMSampler(poll_interval=0.05)

            # Act
            sampler.start()
            time.sleep(0.3)
            peak = sampler.stop()

        # Assert
        assert mock_run.called
        assert peak == pytest.approx(4.0, rel=0.1)

    @pytest.mark.no_gpu
    def test_vram_sampler_no_gpu_returns_none(self) -> None:
        """VRAMSampler returns None when both pynvml and nvidia-smi fail."""
        # Arrange — pynvml raises on nvmlInit; nvidia-smi returns returncode=1
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.side_effect = Exception("no GPU")

        smi_result = MagicMock()
        smi_result.returncode = 1
        smi_result.stdout = ""

        with (
            patch.dict("sys.modules", {"pynvml": pynvml_mock}),
            patch("subprocess.run", return_value=smi_result),
        ):
            sampler = VRAMSampler(poll_interval=0.05)

            # Act
            sampler.start()
            time.sleep(0.2)
            peak = sampler.stop()

        # Assert
        assert peak is None
