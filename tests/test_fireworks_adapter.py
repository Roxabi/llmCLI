"""Tests for FireworksAdapter.

Pins the behaviour:
- api_base constant
- transform_request: forwards body unchanged (relabel removed after Fireworks patch)
- extra_headers: anthropic-version + User-Agent keys present and non-empty
- health: key_present reflects FIREWORKS_API_KEY env var
- ForwardAdapter structural conformance
"""

from __future__ import annotations

import json

import pytest

from llmcli.proxy_forwarder._common import ForwardAdapter
from llmcli.proxy_forwarder.fireworks_adapter import FireworksAdapter


# ---------------------------------------------------------------------------
# Test 1 — api_base constant
# ---------------------------------------------------------------------------


def test_fireworks_api_base() -> None:
    """FireworksAdapter.api_base must be the Fireworks inference base URL."""
    adapter = FireworksAdapter()
    assert adapter.api_base == "https://api.fireworks.ai/inference"


# ---------------------------------------------------------------------------
# Test 2 — transform_request is a pass-through (no relabel)
# ---------------------------------------------------------------------------


def test_transform_request_passes_through_body() -> None:
    """transform_request returns the exact bytes unchanged for /v1/messages."""
    adapter = FireworksAdapter()
    body = json.dumps(
        {
            "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        }
    ).encode()

    result = adapter.transform_request(body, "/v1/messages")
    assert result == body, "body should be forwarded unchanged"


# ---------------------------------------------------------------------------
# Test 3 — pass-through for any path
# ---------------------------------------------------------------------------


def test_transform_request_passes_through_any_path() -> None:
    """transform_request returns body bytes unchanged for any path."""
    adapter = FireworksAdapter()
    body = json.dumps({"messages": [{"role": "system", "content": "x"}]}).encode()

    assert adapter.transform_request(body, "/v1/chat/completions") == body
    assert adapter.transform_request(body, "/v1/messages") == body
    assert adapter.transform_request(b"not json", "/v1/messages") == b"not json"


# ---------------------------------------------------------------------------
# Test 4 — extra_headers contains anthropic-version and User-Agent (non-empty)
# ---------------------------------------------------------------------------


def test_extra_headers_has_version_and_ua() -> None:
    """extra_headers() must contain 'anthropic-version' and 'User-Agent', both non-empty."""
    adapter = FireworksAdapter()
    headers = adapter.extra_headers()
    assert "anthropic-version" in headers, (
        f"'anthropic-version' missing from extra_headers(): {headers!r}"
    )
    assert headers["anthropic-version"], (
        f"'anthropic-version' must be non-empty, got {headers['anthropic-version']!r}"
    )
    assert "User-Agent" in headers, f"'User-Agent' missing from extra_headers(): {headers!r}"
    assert headers["User-Agent"], f"'User-Agent' must be non-empty, got {headers['User-Agent']!r}"


# ---------------------------------------------------------------------------
# Test 5 — health key_present True when FIREWORKS_API_KEY is set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_key_present_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """health() returns key_present=True when FIREWORKS_API_KEY is set."""
    adapter = FireworksAdapter()
    monkeypatch.setenv("FIREWORKS_API_KEY", "x")
    result = await adapter.health()
    assert result.get("status") == "ok", f"expected status=ok, got {result!r}"
    assert result.get("key_present") is True, (
        f"expected key_present=True when env var is set, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 6 — health key_present False when FIREWORKS_API_KEY is absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_key_present_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """health() returns key_present=False when FIREWORKS_API_KEY is not set."""
    adapter = FireworksAdapter()
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    result = await adapter.health()
    assert result.get("status") == "ok", f"expected status=ok, got {result!r}"
    assert result.get("key_present") is False, (
        f"expected key_present=False when env var is absent, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 7 — FireworksAdapter conforms to ForwardAdapter Protocol
# ---------------------------------------------------------------------------


def test_fireworks_conforms_to_forwardadapter() -> None:
    """FireworksAdapter() must be an instance of ForwardAdapter (runtime_checkable Protocol)."""
    adapter = FireworksAdapter()
    assert isinstance(adapter, ForwardAdapter), (
        f"FireworksAdapter does not satisfy the ForwardAdapter Protocol: {type(adapter)}"
    )


# ---------------------------------------------------------------------------
# Test 8 — execute raises when FIREWORKS_API_KEY is missing
# ---------------------------------------------------------------------------


async def test_execute_raises_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute refuses to forward a blank `Bearer ` when FIREWORKS_API_KEY is unset.

    The guard fires before any upstream request, so the (dummy) session is never used.
    """
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    adapter = FireworksAdapter()
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        await adapter.execute(
            None,  # type: ignore[arg-type]  # guard raises before session use
            "POST",
            "https://api.fireworks.ai/inference/v1/messages",
            b"{}",
            {},
        )
