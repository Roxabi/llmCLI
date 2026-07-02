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

from llmcli.gpu import VRAMMonitor, VRAMSampler


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


# ---------------------------------------------------------------------------
# VRAMMonitor
# ---------------------------------------------------------------------------


@pytest.mark.no_gpu
class TestVRAMMonitor:
    """``VRAMMonitor`` is a long-lived nvml context manager with cached handle.

    Distinct from ``VRAMSampler``: no background thread, no peak tracking,
    one-shot ``sample() -> (free_mb, used_mb)`` reads with init/shutdown
    bound to the context-manager lifetime. Used by the NATS adapter for
    heartbeat payloads (#53).
    """

    @staticmethod
    def _make_pynvml_mock(
        free_bytes: int = 8 * 1024**3, used_bytes: int = 4 * 1024**3
    ) -> MagicMock:
        mem_mock = MagicMock()
        mem_mock.free = free_bytes
        mem_mock.used = used_bytes

        handle_mock = MagicMock(name="nvml-handle")

        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.return_value = None
        pynvml_mock.nvmlDeviceGetHandleByIndex.return_value = handle_mock
        pynvml_mock.nvmlDeviceGetMemoryInfo.return_value = mem_mock
        pynvml_mock.nvmlShutdown.return_value = None
        return pynvml_mock

    def test_enter_with_pynvml_initialises_handle(self) -> None:
        pynvml_mock = self._make_pynvml_mock()

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            vm = VRAMMonitor()
            vm.__enter__()
            try:
                assert vm._handle is not None
                assert vm._init_failed is False
                pynvml_mock.nvmlInit.assert_called_once()
                pynvml_mock.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
            finally:
                vm.__exit__(None, None, None)

    def test_enter_without_pynvml_sets_init_failed_flag(self) -> None:
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.side_effect = Exception("no GPU")

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            vm = VRAMMonitor()
            vm.__enter__()
            try:
                # Init failed → handle stays None and the failure is sticky.
                assert vm._handle is None
                assert vm._init_failed is True
            finally:
                vm.__exit__(None, None, None)

        # On the failure path, __exit__ must short-circuit on `_handle is None`
        # and NEVER call nvmlShutdown — otherwise the shutdown would itself
        # throw against an uninitialised library.
        pynvml_mock.nvmlShutdown.assert_not_called()

    def test_sample_with_handle_returns_free_used_mb(self) -> None:
        # 8 GiB free / 4 GiB used → 8192 MiB / 4096 MiB on the wire.
        pynvml_mock = self._make_pynvml_mock(free_bytes=8 * 1024**3, used_bytes=4 * 1024**3)

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            with VRAMMonitor() as vm:
                free_mb, used_mb = vm.sample()

        assert free_mb == pytest.approx(8192.0, rel=1e-3)
        assert used_mb == pytest.approx(4096.0, rel=1e-3)

    def test_sample_without_handle_returns_zero_zero(self) -> None:
        # nvml init failed → handle is None → sample is the no-op `(0.0, 0.0)`
        # contract callers rely on to keep heartbeat payloads emitting.
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.side_effect = Exception("no GPU")

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            with VRAMMonitor() as vm:
                free_mb, used_mb = vm.sample()

        assert (free_mb, used_mb) == (0.0, 0.0)

    def test_exit_is_idempotent(self) -> None:
        pynvml_mock = self._make_pynvml_mock()

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            vm = VRAMMonitor()
            vm.__enter__()
            vm.__exit__(None, None, None)
            # Second __exit__ must not call nvmlShutdown again or raise.
            vm.__exit__(None, None, None)

        assert vm._handle is None
        # Exactly one shutdown — the second __exit__ short-circuited on `_handle is None`.
        pynvml_mock.nvmlShutdown.assert_called_once()

    def test_open_close_delegate_to_context_manager(self) -> None:
        pynvml_mock = self._make_pynvml_mock()

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            vm = VRAMMonitor()
            vm.open()
            assert vm._handle is not None
            vm.close()
            assert vm._handle is None

        pynvml_mock.nvmlInit.assert_called_once()
        pynvml_mock.nvmlShutdown.assert_called_once()

    def test_double_open_logs_warning_and_does_not_double_init(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # `__enter__` is idempotent (valid nested-`with` semantics), but the
        # explicit `open()` API treats double-call as caller misuse and surfaces
        # it via a WARNING log so adapter lifecycle bugs do not vanish silently
        # as never-released handles.
        pynvml_mock = self._make_pynvml_mock()

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            vm = VRAMMonitor()
            vm.open()
            with caplog.at_level("WARNING", logger="llmcli.gpu"):
                vm.open()
            vm.close()

        # No double-init at the nvml layer (guard still holds).
        pynvml_mock.nvmlInit.assert_called_once()
        # Warning visible to operators.
        assert any(
            "VRAMMonitor.open() called while already open" in rec.message for rec in caplog.records
        ), f"expected double-open warning, got: {[r.message for r in caplog.records]!r}"

    def test_reentry_guard_does_not_double_init(self) -> None:
        # Second open() must short-circuit so we never orphan the previous
        # handle by calling nvmlInit() again without a matching nvmlShutdown().
        pynvml_mock = self._make_pynvml_mock()

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            vm = VRAMMonitor()
            vm.open()
            vm.open()
            try:
                pynvml_mock.nvmlInit.assert_called_once()
                pynvml_mock.nvmlDeviceGetHandleByIndex.assert_called_once_with(0)
            finally:
                vm.close()

    def test_failed_init_is_sticky_does_not_retry(self) -> None:
        # nvml/driver availability does not change at runtime; a failed init
        # must stick so we don't burn cycles re-attempting nvmlInit() on every
        # open() call. Without the `_init_failed` branch in the re-entry guard
        # this assertion would fail (nvmlInit called twice).
        pynvml_mock = MagicMock()
        pynvml_mock.nvmlInit.side_effect = Exception("no GPU")

        with patch.dict("sys.modules", {"pynvml": pynvml_mock}):
            vm = VRAMMonitor()
            vm.open()
            vm.open()
            try:
                assert vm._init_failed is True
                pynvml_mock.nvmlInit.assert_called_once()
            finally:
                vm.close()
