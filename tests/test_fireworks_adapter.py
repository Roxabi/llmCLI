"""Tests for FireworksAdapter — RED until T7 lands (fireworks_adapter.py does not exist yet).

These tests pin the exact behaviour specified in issue #103:
- api_base constant
- transform_request: relabel system→user on /v1/messages only
- extra_headers: anthropic-version + User-Agent keys present and non-empty
- health: key_present reflects FIREWORKS_API_KEY env var
- ForwardAdapter structural conformance
"""

from __future__ import annotations

import json

import pytest

# F1 — direct imports (no try/except guards); collection error is the expected RED state
from llmcli.proxy_forwarder._common import ForwardAdapter
from llmcli.proxy_forwarder.fireworks_adapter import FireworksAdapter


# ---------------------------------------------------------------------------
# Test 1 — api_base constant
# ---------------------------------------------------------------------------


def test_fireworks_api_base() -> None:
    """FireworksAdapter.api_base must be the Fireworks inference base URL."""
    # Arrange
    adapter = FireworksAdapter()

    # Act / Assert
    assert adapter.api_base == "https://api.fireworks.ai/inference"


# ---------------------------------------------------------------------------
# Test 2 — relabel single system role on /v1/messages
# ---------------------------------------------------------------------------


def test_relabel_single_system_role() -> None:
    """transform_request on /v1/messages rewrites a single system entry to user."""
    # Arrange
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

    # Act
    result = adapter.transform_request(body, "/v1/messages")

    # Assert — parse result; system entry becomes user, user entry untouched
    parsed = json.loads(result)
    assert parsed["messages"][0]["role"] == "user", (
        f"expected system→user, got {parsed['messages'][0]['role']!r}"
    )
    assert parsed["messages"][0]["content"] == "You are helpful."
    assert parsed["messages"][1]["role"] == "user"
    assert parsed["messages"][1]["content"] == "Hello"


# ---------------------------------------------------------------------------
# Test 3 — relabel multiple system roles; non-system roles untouched
# ---------------------------------------------------------------------------


def test_relabel_multiple_system_roles() -> None:
    """All system entries are relabelled; user/assistant entries are left untouched."""
    # Arrange
    adapter = FireworksAdapter()
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Sys1"},
                {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "A1"},
                {"role": "system", "content": "Sys2"},
            ]
        }
    ).encode()

    # Act
    result = adapter.transform_request(body, "/v1/messages")

    # Assert
    parsed = json.loads(result)
    msgs = parsed["messages"]
    assert msgs[0] == {"role": "user", "content": "Sys1"}, (
        f"first sys entry not relabelled: {msgs[0]}"
    )
    assert msgs[1] == {"role": "user", "content": "U1"}, f"user entry mutated: {msgs[1]}"
    assert msgs[2] == {"role": "assistant", "content": "A1"}, f"assistant entry mutated: {msgs[2]}"
    assert msgs[3] == {"role": "user", "content": "Sys2"}, (
        f"second sys entry not relabelled: {msgs[3]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — no-op for non-/v1/messages path (exact bytes returned)
# ---------------------------------------------------------------------------


def test_relabel_noop_wrong_path() -> None:
    """transform_request returns body bytes unchanged for paths other than /v1/messages."""
    # Arrange
    adapter = FireworksAdapter()
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "You are a bot."},
                {"role": "user", "content": "Hi"},
            ]
        }
    ).encode()

    # Act
    result = adapter.transform_request(body, "/v1/chat/completions")

    # Assert — exact bytes unchanged
    assert result == body, "body should be returned unchanged for /v1/chat/completions"


# ---------------------------------------------------------------------------
# Test 5 — no-op for non-JSON body (exact bytes returned)
# ---------------------------------------------------------------------------


def test_relabel_noop_non_json() -> None:
    """transform_request returns body bytes unchanged when body is not valid JSON."""
    # Arrange
    adapter = FireworksAdapter()
    body = b"not json"

    # Act
    result = adapter.transform_request(body, "/v1/messages")

    # Assert — exact bytes unchanged
    assert result == b"not json", f"non-JSON body should be returned unchanged, got {result!r}"


# ---------------------------------------------------------------------------
# Test 6 — no-op when body has no messages key (exact bytes returned)
# ---------------------------------------------------------------------------


def test_relabel_noop_no_messages() -> None:
    """transform_request returns body bytes unchanged when messages key is absent."""
    # Arrange
    adapter = FireworksAdapter()
    body = b'{"model":"x"}'

    # Act
    result = adapter.transform_request(body, "/v1/messages")

    # Assert — exact bytes unchanged
    assert result == body, f"body without messages should be returned unchanged, got {result!r}"


# ---------------------------------------------------------------------------
# Test 7 — idempotent: double-apply == single-apply
# ---------------------------------------------------------------------------


def test_relabel_idempotent() -> None:
    """Applying transform_request twice produces the same result as applying it once."""
    # Arrange
    adapter = FireworksAdapter()
    body = json.dumps(
        {
            "messages": [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ]
        }
    ).encode()

    # Act
    once = adapter.transform_request(body, "/v1/messages")
    twice = adapter.transform_request(once, "/v1/messages")

    # Assert — compare semantically (key ordering may differ)
    assert json.loads(once) == json.loads(twice), (
        "double-apply must equal single-apply (idempotent)"
    )


# ---------------------------------------------------------------------------
# Test 8 — extra_headers contains anthropic-version and User-Agent (non-empty)
# ---------------------------------------------------------------------------


def test_extra_headers_has_version_and_ua() -> None:
    """extra_headers() must contain 'anthropic-version' and 'User-Agent', both non-empty."""
    # Arrange
    adapter = FireworksAdapter()

    # Act
    headers = adapter.extra_headers()

    # Assert — keys present and non-empty
    assert "anthropic-version" in headers, (
        f"'anthropic-version' missing from extra_headers(): {headers!r}"
    )
    assert headers["anthropic-version"], (
        f"'anthropic-version' must be non-empty, got {headers['anthropic-version']!r}"
    )
    assert "User-Agent" in headers, f"'User-Agent' missing from extra_headers(): {headers!r}"
    assert headers["User-Agent"], f"'User-Agent' must be non-empty, got {headers['User-Agent']!r}"


# ---------------------------------------------------------------------------
# Test 9 — health key_present True when FIREWORKS_API_KEY is set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_key_present_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """health() returns key_present=True when FIREWORKS_API_KEY is set."""
    # Arrange
    adapter = FireworksAdapter()
    monkeypatch.setenv("FIREWORKS_API_KEY", "x")

    # Act
    result = await adapter.health()

    # Assert
    assert result.get("status") == "ok", f"expected status=ok, got {result!r}"
    assert result.get("key_present") is True, (
        f"expected key_present=True when env var is set, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 10 — health key_present False when FIREWORKS_API_KEY is absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_key_present_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """health() returns key_present=False when FIREWORKS_API_KEY is not set."""
    # Arrange
    adapter = FireworksAdapter()
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

    # Act
    result = await adapter.health()

    # Assert
    assert result.get("status") == "ok", f"expected status=ok, got {result!r}"
    assert result.get("key_present") is False, (
        f"expected key_present=False when env var is absent, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Test 11 — FireworksAdapter conforms to ForwardAdapter Protocol
# ---------------------------------------------------------------------------


def test_fireworks_conforms_to_forwardadapter() -> None:
    """FireworksAdapter() must be an instance of ForwardAdapter (runtime_checkable Protocol)."""
    # Arrange
    adapter = FireworksAdapter()

    # Act / Assert
    assert isinstance(adapter, ForwardAdapter), (
        f"FireworksAdapter does not satisfy the ForwardAdapter Protocol: {type(adapter)}"
    )


# ---------------------------------------------------------------------------
# F9 guard-edge tests — fail when the defensive guards in transform_request
# are removed (kills tautology risk on isinstance/get("role") guards)
# ---------------------------------------------------------------------------


def test_relabel_skips_non_dict_message_entries() -> None:
    """Non-dict entries in messages are preserved; dict entries with role=system are relabelled.

    Guards isinstance(msg, dict) — if removed, the string entry would cause AttributeError
    or be incorrectly mutated, and the system dict entry relabelling would also break.
    """
    # Arrange
    adapter = FireworksAdapter()
    body = json.dumps(
        {
            "messages": [
                "not-a-dict",
                {"role": "system", "content": "x"},
            ]
        }
    ).encode()

    # Act — must not raise
    result = adapter.transform_request(body, "/v1/messages")

    # Assert — string entry preserved, dict system entry relabelled to user
    parsed = json.loads(result)
    msgs = parsed["messages"]
    assert msgs[0] == "not-a-dict", f"non-dict entry should be preserved unchanged, got {msgs[0]!r}"
    assert msgs[1]["role"] == "user", (
        f"system dict entry should be relabelled to 'user', got {msgs[1]['role']!r}"
    )


def test_relabel_skips_message_without_role_key() -> None:
    """Dict entries missing the role key are left untouched; entries with role=system are relabelled.

    Guards msg.get("role") — if removed, a KeyError would occur on dicts without 'role',
    or entries without a role would be incorrectly mutated.
    """
    # Arrange
    adapter = FireworksAdapter()
    body = json.dumps(
        {
            "messages": [
                {"content": "no role"},
                {"role": "system", "content": "x"},
            ]
        }
    ).encode()

    # Act — must not raise KeyError
    result = adapter.transform_request(body, "/v1/messages")

    # Assert — role-less entry untouched, system entry relabelled to user
    parsed = json.loads(result)
    msgs = parsed["messages"]
    assert msgs[0] == {"content": "no role"}, (
        f"entry without 'role' key should be untouched, got {msgs[0]!r}"
    )
    assert msgs[1]["role"] == "user", (
        f"system entry should be relabelled to 'user', got {msgs[1]['role']!r}"
    )


def test_relabel_noop_empty_messages_list() -> None:
    """transform_request on /v1/messages with an empty messages list round-trips cleanly."""
    # Arrange
    adapter = FireworksAdapter()
    body = json.dumps({"messages": []}).encode()

    # Act — must not raise
    result = adapter.transform_request(body, "/v1/messages")

    # Assert — round-trips to the same structure (no error, no mutation)
    assert json.loads(result) == {"messages": []}, (
        f"empty messages list should round-trip unchanged, got {json.loads(result)!r}"
    )
