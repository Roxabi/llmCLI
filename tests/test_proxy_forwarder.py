"""Contract and regression tests for proxy_forwarder generalization (issue #103).

Tests 1–4: ForwardAdapter Protocol contract (GREEN now — _common.py is done).
Tests 5–7: XaiAdapter regression under the new contract (RED until T4/T7 land).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmcli.proxy_forwarder._common import ALLOWED_PATHS, ForwardAdapter, OAuthAdapter


# ---------------------------------------------------------------------------
# Test 1 — ForwardAdapter declares the four required methods and api_base
# ---------------------------------------------------------------------------


def test_forwardadapter_contract_members() -> None:
    """ForwardAdapter Protocol declares transform_request, extra_headers, execute, health + api_base."""
    # Arrange
    required_methods = {"transform_request", "extra_headers", "execute", "health"}

    # Act — collect names from the Protocol (annotations + dir)
    protocol_members = set(dir(ForwardAdapter))

    # Assert — all four methods are present on the Protocol
    for method in required_methods:
        assert method in protocol_members, (
            f"ForwardAdapter Protocol is missing required method: {method!r}"
        )

    # Assert — api_base is declared as an annotation on the Protocol
    annotations = getattr(ForwardAdapter, "__annotations__", {})
    assert "api_base" in annotations, (
        f"ForwardAdapter.__annotations__ does not contain 'api_base'; got: {annotations}"
    )


# ---------------------------------------------------------------------------
# Test 2 — credential_path and refresh are NOT part of ForwardAdapter
# ---------------------------------------------------------------------------


def test_forwardadapter_protocol_excludes_credentials() -> None:
    """credential_path and refresh are NOT part of the ForwardAdapter Protocol contract."""
    # Arrange
    credentials_attrs = {"credential_path", "refresh"}

    # Act
    annotations = getattr(ForwardAdapter, "__annotations__", {})
    # Protocol abstract members live in __protocol_attrs__ on runtime_checkable Protocols
    protocol_attrs = getattr(ForwardAdapter, "__protocol_attrs__", set())

    # Assert — neither credential_path nor refresh appear in annotations
    for attr in credentials_attrs:
        assert attr not in annotations, (
            f"{attr!r} must NOT be in ForwardAdapter.__annotations__; found in: {annotations}"
        )

    # Assert — neither credential_path nor refresh appear in protocol abstract members
    for attr in credentials_attrs:
        assert attr not in protocol_attrs, (
            f"{attr!r} must NOT be an abstract protocol member of ForwardAdapter; "
            f"__protocol_attrs__={protocol_attrs}"
        )


# ---------------------------------------------------------------------------
# Test 3 — /v1/messages is in ALLOWED_PATHS
# ---------------------------------------------------------------------------


def test_messages_path_allowed() -> None:
    """/v1/messages must be in ALLOWED_PATHS (generalization requirement)."""
    # Arrange / Act — ALLOWED_PATHS is a module-level constant
    # Assert
    assert "/v1/messages" in ALLOWED_PATHS, (
        f"/v1/messages missing from ALLOWED_PATHS; got: {ALLOWED_PATHS}"
    )


# ---------------------------------------------------------------------------
# Test 4 — OAuthAdapter is the ForwardAdapter alias (back-compat)
# ---------------------------------------------------------------------------


def test_oauthadapter_is_forwardadapter_alias() -> None:
    """OAuthAdapter must be the exact same object as ForwardAdapter (back-compat alias)."""
    # Arrange / Act — both are imported at module level above
    # Assert
    assert OAuthAdapter is ForwardAdapter, (
        f"OAuthAdapter is not ForwardAdapter: OAuthAdapter={OAuthAdapter!r}, "
        f"ForwardAdapter={ForwardAdapter!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — XaiAdapter conforms to ForwardAdapter (runtime_checkable isinstance)
# RED until XaiAdapter gains execute/health/transform_request/extra_headers.
# ---------------------------------------------------------------------------


def test_xai_adapter_conforms_to_forwardadapter() -> None:
    """XaiAdapter must satisfy isinstance(adapter, ForwardAdapter) (runtime_checkable)."""
    # Arrange
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    adapter = XaiAdapter()

    # Act + Assert — Protocol conformance
    assert isinstance(adapter, ForwardAdapter), (
        "XaiAdapter does not satisfy the ForwardAdapter Protocol. "
        "Missing methods: "
        + str(
            {
                m
                for m in ("transform_request", "extra_headers", "execute", "health", "api_base")
                if not hasattr(adapter, m)
            }
        )
    )

    # Assert — public credential_path attribute still present (regression)
    assert hasattr(adapter, "credential_path"), (
        "XaiAdapter lost public 'credential_path' attribute — breaks existing test monkeypatching"
    )

    # Assert — public refresh method still present (regression)
    assert hasattr(adapter, "refresh"), (
        "XaiAdapter lost public 'refresh' method — breaks existing callers"
    )


# ---------------------------------------------------------------------------
# Test 6 — /health returns {status, logged_in, expires_at} via TestClient
# RED until XaiAdapter.health() is implemented.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_xai_forwarder_health_via_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /health returns 200 with status/logged_in/expires_at; logged_in=False when no creds."""
    # Arrange
    from aiohttp.test_utils import TestClient, TestServer

    from llmcli.proxy_forwarder._server import create_app
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    adapter = XaiAdapter()
    absent_creds = tmp_path / "absent.json"
    # absent_creds does NOT exist — adapter must handle missing file gracefully
    monkeypatch.setattr(adapter, "credential_path", absent_creds)

    app = create_app(adapter)

    # Act
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/health")

        # Assert — status 200
        assert resp.status == 200, f"expected 200, got {resp.status}"

        body = await resp.json()

    # Assert — required keys present
    assert "status" in body, f"'status' key missing from health response: {body}"
    assert "logged_in" in body, f"'logged_in' key missing from health response: {body}"
    assert "expires_at" in body, f"'expires_at' key missing from health response: {body}"

    # Assert — no creds → logged_in is False
    assert body["logged_in"] is False, (
        f"expected logged_in=False when creds file absent, got {body['logged_in']!r}"
    )


# ---------------------------------------------------------------------------
# Test 7 — XaiAdapter.transform_request is identity; extra_headers returns {}
# RED until those methods are added to XaiAdapter.
# ---------------------------------------------------------------------------


def test_xai_transform_is_identity() -> None:
    """XaiAdapter.transform_request must return the body unchanged; extra_headers must return {}."""
    # Arrange
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    adapter = XaiAdapter()
    payload = b'{"x":1}'

    # Act
    result = adapter.transform_request(payload, "/v1/messages")

    # Assert — body is returned unchanged (identity transform)
    assert result == payload, (
        f"XaiAdapter.transform_request modified the body: input={payload!r}, output={result!r}"
    )

    # Act
    headers = adapter.extra_headers()

    # Assert — no extra headers injected by xAI adapter (auth is handled in execute)
    assert headers == {}, f"XaiAdapter.extra_headers() expected {{}}, got {headers!r}"
