"""Contract and regression tests for proxy_forwarder generalization (issue #103).

Tests 1–4: ForwardAdapter Protocol contract (GREEN now — _common.py is done).
Tests 5–7: XaiAdapter regression under the new contract (RED until T4/T7 land).
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llmcli.auth.store import XaiCredentials
from llmcli.proxy_forwarder._common import ALLOWED_PATHS, ForwardAdapter, OAuthAdapter


# ---------------------------------------------------------------------------
# Test 1 — ForwardAdapter declares the four required methods and api_base
# ---------------------------------------------------------------------------


def test_forwardadapter_contract_members() -> None:
    """ForwardAdapter Protocol declares transform_request, extra_headers, execute, health + api_base."""
    # Arrange
    required_methods = {"transform_request", "extra_headers", "execute", "health"}

    # Act — Python 3.12 tracks Protocol abstract members in __protocol_attrs__
    protocol_attrs = getattr(ForwardAdapter, "__protocol_attrs__", set())

    # Assert — all four methods are present in the Protocol's abstract member set
    for method in required_methods:
        assert method in protocol_attrs, (
            f"ForwardAdapter Protocol is missing required method: {method!r}; "
            f"__protocol_attrs__={protocol_attrs}"
        )

    # Assert — api_base is declared as an abstract protocol member
    assert "api_base" in protocol_attrs, (
        f"ForwardAdapter.__protocol_attrs__ does not contain 'api_base'; got: {protocol_attrs}"
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


# ---------------------------------------------------------------------------
# Test 8 — fail-loud back-off: a known-dead refresh token is NOT re-POSTed (#114)
# ---------------------------------------------------------------------------


def _dead_creds(refresh_token: str = "DEAD_RTK") -> XaiCredentials:
    """Expired credentials whose refresh token the token endpoint will 4xx."""
    return XaiCredentials(
        access_token="A",
        refresh_token=refresh_token,
        id_token="I",
        expires_at=int(time.time()) - 60,  # expired → triggers proactive refresh
        token_type="Bearer",
        scope="openid",
    )


@pytest.mark.asyncio
async def test_xai_adapter_backs_off_on_dead_refresh_token() -> None:
    """After a 4xx proves the refresh token dead, refresh() backs off — no re-POST."""
    # Arrange
    from llmcli.auth.store import ReauthRequired
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    adapter = XaiAdapter()
    dead = _dead_creds()

    post_calls = 0

    def _fake_post(url: str, **kwargs):  # noqa: ARG001 — signature match
        nonlocal post_calls
        post_calls += 1
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "invalid_grant"
        return resp

    # Act — two refreshes of the same dead token
    with patch("httpx.post", side_effect=_fake_post):
        with pytest.raises(ReauthRequired):
            await adapter.refresh(dead)  # real POST → 400 → flag set
        with pytest.raises(ReauthRequired):
            await adapter.refresh(dead)  # backed off → no POST

    # Assert — exactly one upstream POST despite two refresh attempts
    assert post_calls == 1, f"expected one POST (back-off), got {post_calls}"
    assert adapter._reauth_required is True


@pytest.mark.asyncio
async def test_xai_adapter_clears_flag_after_reauth() -> None:
    """A new refresh_token (operator re-authed) clears back-off and POSTs again."""
    # Arrange — drive the adapter into the dead state first
    from llmcli.auth.store import ReauthRequired
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    adapter = XaiAdapter()

    def _post_400(url: str, **kwargs):  # noqa: ARG001
        resp = MagicMock()
        resp.status_code = 400
        resp.text = "invalid_grant"
        return resp

    with patch("httpx.post", side_effect=_post_400):
        with pytest.raises(ReauthRequired):
            await adapter.refresh(_dead_creds("DEAD_RTK"))
    assert adapter._reauth_required is True

    # Act — operator re-authed: a DIFFERENT refresh_token now succeeds
    token_resp = {
        "access_token": "NEW_ATK",
        "refresh_token": "NEW_RTK",
        "id_token": "NEW_ITK",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "openid",
    }

    def _post_ok(url: str, **kwargs):  # noqa: ARG001
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = token_resp
        return resp

    with patch("httpx.post", side_effect=_post_ok):
        new = await adapter.refresh(_dead_creds("NEW_RTK"))

    # Assert — refreshed + flag cleared
    assert new.access_token == "NEW_ATK"
    assert adapter._reauth_required is False


# ---------------------------------------------------------------------------
# Test 9 — /health returns 503 when re-auth required (deterministic unhealthy) (#114)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_503_when_reauth_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /health → 503 (not 200) when the adapter flags reauth_required."""
    # Arrange
    from aiohttp.test_utils import TestClient, TestServer

    from llmcli.proxy_forwarder._server import create_app
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    adapter = XaiAdapter()
    # Absent creds → _sync_reauth_state cannot clear the flag (no fresh token on disk).
    monkeypatch.setattr(adapter, "credential_path", tmp_path / "absent.json")
    adapter._reauth_required = True
    adapter._dead_refresh_token = "DEAD_RTK"

    app = create_app(adapter)

    # Act
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/health")

        # Assert — 503 so the container HEALTHCHECK flips unhealthy
        assert resp.status == 503, f"expected 503, got {resp.status}"
        body = await resp.json()
    assert body["reauth_required"] is True


# ---------------------------------------------------------------------------
# Test 10 — proxy maps ReauthRequired → 503 + X-Llmcli-Reauth header (#114)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_503_on_reauth_required(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ReauthRequired from execute() surfaces as HTTP 503 + X-Llmcli-Reauth: required."""
    # Arrange
    from aiohttp.test_utils import TestClient, TestServer

    from llmcli.auth.store import ReauthRequired
    from llmcli.proxy_forwarder._server import create_app
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    adapter = XaiAdapter()

    async def _boom(*_args, **_kwargs):
        raise ReauthRequired("refresh token dead")

    monkeypatch.setattr(adapter, "execute", _boom)

    app = create_app(adapter)

    # Act
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/chat/completions", data=b"{}")

        # Assert — hard 503 (not a soft 401) + the re-auth signal header
        assert resp.status == 503, f"expected 503, got {resp.status}"
        assert resp.headers.get("X-Llmcli-Reauth") == "required"


# ---------------------------------------------------------------------------
# Test 11 — /health 503 on corrupted creds (review fix #114)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_503_on_corrupted_creds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corrupted xai.json → /health 503 (was 200) so monitoring sees unhealthy."""
    from aiohttp.test_utils import TestClient, TestServer

    from llmcli.proxy_forwarder._server import create_app
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    creds_file = tmp_path / "xai.json"
    creds_file.write_text('{"access_token": "abc"')  # partial JSON → CredentialsCorruptError
    adapter = XaiAdapter()
    monkeypatch.setattr(adapter, "credential_path", creds_file)

    app = create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/health")
        assert resp.status == 503, f"expected 503, got {resp.status}"
        body = await resp.json()
    assert body["status"] == "corrupted"
    assert body["logged_in"] is False


# ---------------------------------------------------------------------------
# Test 12 — transient (non-ReauthRequired) refresh failure → 502, no reauth header
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_proxy_502_on_transient_refresh_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain RuntimeError (auth.x.ai 5xx) → 502, NOT 401+reauth (creds not known-bad)."""
    from aiohttp.test_utils import TestClient, TestServer

    from llmcli.proxy_forwarder._server import create_app
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    adapter = XaiAdapter()

    async def _boom(*_args, **_kwargs):
        raise RuntimeError("xAI token refresh failed (HTTP 503)")

    monkeypatch.setattr(adapter, "execute", _boom)

    app = create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        resp = await client.post("/v1/chat/completions", data=b"{}")
        assert resp.status == 502, f"expected 502, got {resp.status}"
        assert resp.headers.get("X-Llmcli-Reauth") is None


# ---------------------------------------------------------------------------
# Test 13 — /health self-heals once the operator re-auths (new refresh_token)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_selfheal_clears_flag_on_new_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After re-auth (new refresh_token on disk), /health flips 503 → 200."""
    from aiohttp.test_utils import TestClient, TestServer

    from llmcli.auth import store
    from llmcli.proxy_forwarder._server import create_app
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    creds_file = tmp_path / "xai.json"
    adapter = XaiAdapter()
    monkeypatch.setattr(adapter, "credential_path", creds_file)
    # Simulate a prior dead-token episode, then the operator re-authing (new token).
    store.save(_dead_creds("NEW_RTK"), path=creds_file)
    adapter._reauth_required = True
    adapter._dead_refresh_token = "OLD_DEAD_RTK"

    app = create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/health")
        assert resp.status == 200, f"expected self-heal to 200, got {resp.status}"
        body = await resp.json()
    assert body["reauth_required"] is False
