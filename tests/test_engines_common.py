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

    @pytest.mark.parametrize(
        ("status_code", "expected"),
        [
            (200, True),  # canonical OK
            (201, True),  # Created — common first-load response
            (299, True),  # boundary: last 2xx
            (300, False),  # boundary: first non-2xx
            (404, False),  # standard not-found
            (503, False),  # canonical warmup/transient unready
        ],
    )
    def test_returns_bool_on_status_code(self, status_code: int, expected: bool) -> None:
        # Parametrizing the 2xx / non-2xx boundary catches off-by-one regressions
        # (e.g. `< 200` or `<= 200`) that the previous 200/503-only pair missed.
        mock_response = MagicMock()
        mock_response.status_code = status_code

        with patch("llmcli.engines._common.httpx.get", return_value=mock_response) as mock_get:
            result = default_health("http://127.0.0.1:8091")

        assert result is expected, (
            f"default_health({status_code}) expected {expected}, got {result!r}"
        )
        mock_get.assert_called_once()
        call_url: str = mock_get.call_args[0][0]
        assert call_url == "http://127.0.0.1:8091/health", (
            f"default_health must probe the /health endpoint, got: {call_url!r}"
        )

    def test_returns_false_on_transport_exception(self) -> None:
        # ConnectError, timeouts, DNS failures, etc. — any exception swallowed.
        with patch(
            "llmcli.engines._common.httpx.get",
            side_effect=Exception("Connection refused"),
        ):
            result = default_health("http://127.0.0.1:8091")

        assert result is False, "default_health must catch transport errors, not raise"
