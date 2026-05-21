"""Direct unit tests for `llmcli.engines._common` (#58 item 5).

The per-engine health tests in `test_llamacpp.py` and `test_vllm.py` patch
`llmcli.engines._common.httpx.get` and call `engine.health(instance)` — which
exercises `default_health` only transitively. Stubbing `default_health` to a
no-op there would not fail those tests. These cases pin the function shape
directly so the shared health-probe contract has its own regression net.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llmcli.engines._common import default_health


@pytest.mark.no_gpu
class TestDefaultHealth:
    """``default_health(base_url)`` → bool. 2xx True, non-2xx False, exception False."""

    def test_returns_true_on_2xx(self) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("llmcli.engines._common.httpx.get", return_value=mock_response) as mock_get:
            result = default_health("http://127.0.0.1:8091")

        assert result is True
        mock_get.assert_called_once()
        call_url: str = mock_get.call_args[0][0]
        assert call_url == "http://127.0.0.1:8091/health", (
            f"default_health must probe the /health endpoint, got: {call_url!r}"
        )

    def test_returns_false_on_non_2xx(self) -> None:
        # 503 is the canonical warmup/transient unready response.
        mock_response = MagicMock()
        mock_response.status_code = 503

        with patch("llmcli.engines._common.httpx.get", return_value=mock_response):
            result = default_health("http://127.0.0.1:8091")

        assert result is False

    def test_returns_false_on_transport_exception(self) -> None:
        # ConnectError, timeouts, DNS failures, etc. — any exception swallowed.
        with patch(
            "llmcli.engines._common.httpx.get",
            side_effect=Exception("Connection refused"),
        ):
            result = default_health("http://127.0.0.1:8091")

        assert result is False, "default_health must catch transport errors, not raise"
