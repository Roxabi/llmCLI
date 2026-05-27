"""Tests for xAI OAuth PKCE flow and credential store."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# F1 — direct imports (no try/except guards)
from llmcli.auth import store as auth_store
from llmcli.auth.store import CredentialsCorruptError, XaiCredentials
from llmcli.auth.xai_oauth import (
    XAI_OAUTH_CLIENT_ID,
    _build_authorize_url,
    exchange_code,
)
from llmcli.proxy_forwarder._common import _REFRESH_LOCK, lazy_retry_on_401  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_fake_creds(
    access_token: str = "ATK",
    refresh_token: str = "RTK",
    id_token: str = "ITK",
    expires_delta: int = 3600,
) -> XaiCredentials:
    """Build an XaiCredentials for test usage."""
    return XaiCredentials(
        access_token=access_token,
        refresh_token=refresh_token,
        id_token=id_token,
        expires_at=int(time.time()) + expires_delta,
        token_type="Bearer",
        scope="openid",
    )


# ---------------------------------------------------------------------------
# Test 1 — PKCE code exchange
# ---------------------------------------------------------------------------


def test_pkce_code_exchange(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """exchange_code POSTs to auth.x.ai/oauth/token with correct body; returns XaiCredentials."""
    # Arrange
    fake_response_json = {
        "access_token": "ATK",
        "refresh_token": "RTK",
        "id_token": "ITK",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "openid",
    }
    fake_verifier = "s" * 43  # minimum PKCE verifier length

    captured_kwargs: dict = {}

    def _fake_post(url: str, **kwargs):  # type: ignore[return]
        captured_kwargs["url"] = url
        captured_kwargs.update(kwargs)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = fake_response_json
        return mock_resp

    # Act — patch httpx.post used by exchange_code
    with patch("httpx.post", side_effect=_fake_post):
        before = int(time.time())
        result = exchange_code(code="test_code", verifier=fake_verifier)
        after = int(time.time())

    # Assert — POST was made to auth.x.ai/oauth/token
    assert captured_kwargs["url"] == "https://auth.x.ai/oauth/token"

    # Assert — request body contains required fields (httpx uses data= as dict)
    body = captured_kwargs.get("data") or {}
    if isinstance(body, bytes):
        body = dict(item.split("=") for item in body.decode().split("&"))
    assert body.get("grant_type") == "authorization_code", f"body={body}"
    assert body.get("code") == "test_code", f"body={body}"
    assert body.get("code_verifier") == fake_verifier, f"body={body}"
    assert body.get("client_id") == XAI_OAUTH_CLIENT_ID, f"body={body}"
    # F4 — assert code_challenge in POST body
    assert body.get("code_challenge") is not None, f"code_challenge missing from body: {body}"

    # Assert — returned XaiCredentials matches mock response
    assert result.access_token == "ATK"
    assert result.refresh_token == "RTK"
    assert result.id_token == "ITK"
    # expires_at must be roughly time.time() + 3600 (±5 s tolerance)
    assert before + 3595 <= result.expires_at <= after + 3605, (
        f"expires_at={result.expires_at} not in expected range"
    )

    # Assert — authorize URL contains plan=generic (build helper)
    # F3 — removed tautological assert XAI_OAUTH_PLAN == "generic"
    auth_url = _build_authorize_url(
        challenge="challenge_abc",
        state="state_xyz",
        nonce="nonce_123",
    )
    assert "plan=generic" in auth_url, f"plan=generic missing from URL: {auth_url}"
    # F4 — assert state and nonce in URL
    assert "state=state_xyz" in auth_url, f"state missing from URL: {auth_url}"
    assert "nonce=nonce_123" in auth_url, f"nonce missing from URL: {auth_url}"
    # F11 — assert client_id constant (not hardcoded literal)
    assert f"client_id={XAI_OAUTH_CLIENT_ID}" in auth_url, (
        f"client_id missing from URL: {auth_url}"
    )


# ---------------------------------------------------------------------------
# Test 2 — lazy 401 retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_refresh_on_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """lazy_retry_on_401 retries once on 401; new credentials persisted to disk."""
    # Arrange — initial (expired) credentials on disk
    creds_file = tmp_path / "xai.json"
    monkeypatch.setattr("llmcli.auth.store.XAI_CREDENTIALS_PATH", creds_file)
    # F8 — isolate _REFRESH_LOCK per test
    monkeypatch.setattr("llmcli.proxy_forwarder._common._REFRESH_LOCK", asyncio.Lock())

    old_creds = _make_fake_creds(access_token="OLD_ATK", expires_delta=-60)
    auth_store.save(old_creds, path=creds_file)

    new_token_resp = {
        "access_token": "NEW_ATK",
        "refresh_token": "NEW_RTK",
        "id_token": "NEW_ITK",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "openid",
    }
    chat_ok_resp = {"choices": [{"message": {"content": "pong"}}]}

    api_call_count = 0
    auth_call_count = 0
    # F9 — accumulate tokens received
    tokens_received: list[str] = []

    async def _json_ok() -> dict:
        return chat_ok_resp

    async def request_fn(token: str) -> MagicMock:
        nonlocal api_call_count
        api_call_count += 1
        tokens_received.append(token)
        resp = MagicMock()
        if api_call_count == 1:
            resp.status = 401
        else:
            resp.status = 200
            resp.json = _json_ok
        return resp

    async def refresh_fn(creds: XaiCredentials) -> XaiCredentials:
        nonlocal auth_call_count
        auth_call_count += 1
        return XaiCredentials(
            access_token=new_token_resp["access_token"],
            refresh_token=new_token_resp["refresh_token"],
            id_token=new_token_resp["id_token"],
            expires_at=int(time.time()) + new_token_resp["expires_in"],
            token_type=new_token_resp["token_type"],
            scope=new_token_resp["scope"],
        )

    # Act
    store_load = lambda: auth_store.load(path=creds_file)  # noqa: E731
    store_save = lambda c: auth_store.save(c, path=creds_file)  # noqa: E731
    await lazy_retry_on_401(request_fn, refresh_fn, store_load, store_save)

    # Assert — 2 calls to api.x.ai (1 initial 401 + 1 retry), 1 refresh POST
    assert api_call_count == 2, f"expected 2 api calls, got {api_call_count}"
    assert auth_call_count == 1, f"expected 1 refresh call, got {auth_call_count}"

    # Assert — new credentials persisted to disk
    persisted = auth_store.load(path=creds_file)
    assert persisted is not None
    assert persisted.access_token == "NEW_ATK"

    # F9 — assert token values used on each call
    assert tokens_received[0] == "OLD_ATK", (
        f"first call should use old token, got {tokens_received[0]!r}"
    )
    assert tokens_received[1] == "NEW_ATK", (
        f"retry should use refreshed token, got {tokens_received[1]!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 — concurrent 401 dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_refresh_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5 concurrent 401s result in exactly ONE refresh POST (asyncio.Lock dedup)."""
    # Arrange — expired credentials on disk
    creds_file = tmp_path / "xai.json"
    monkeypatch.setattr("llmcli.auth.store.XAI_CREDENTIALS_PATH", creds_file)
    # F8 — isolate _REFRESH_LOCK per test
    monkeypatch.setattr("llmcli.proxy_forwarder._common._REFRESH_LOCK", asyncio.Lock())

    old_creds = _make_fake_creds(access_token="OLD_ATK", expires_delta=-60)
    auth_store.save(old_creds, path=creds_file)

    api_call_count = 0
    refresh_call_count = 0
    lock = asyncio.Lock()

    async def request_fn(token: str) -> MagicMock:
        nonlocal api_call_count
        async with lock:
            api_call_count += 1
            current_count = api_call_count
        resp = MagicMock()
        # First 5 calls return 401; subsequent calls (retries with new token) return 200
        if current_count <= 5:
            resp.status = 401
        else:
            resp.status = 200
        return resp

    async def refresh_fn(creds: XaiCredentials) -> XaiCredentials:
        nonlocal refresh_call_count
        refresh_call_count += 1
        # Small delay to make race condition visible
        await asyncio.sleep(0.01)
        return XaiCredentials(
            access_token="REFRESHED_ATK",
            refresh_token="REFRESHED_RTK",
            id_token="REFRESHED_ITK",
            expires_at=int(time.time()) + 3600,
            token_type="Bearer",
            scope="openid",
        )

    store_load = lambda: auth_store.load(path=creds_file)  # noqa: E731
    store_save = lambda c: auth_store.save(c, path=creds_file)  # noqa: E731

    # Act — 5 concurrent callers all hit 401 simultaneously
    await asyncio.gather(
        *[lazy_retry_on_401(request_fn, refresh_fn, store_load, store_save) for _ in range(5)]
    )

    # Assert — EXACTLY ONE refresh POST (the _REFRESH_LOCK dedup property)
    assert refresh_call_count == 1, (
        f"expected exactly 1 refresh, got {refresh_call_count} — _REFRESH_LOCK dedup broken"
    )

    # F10 — assert disk persistence + final api count
    persisted = auth_store.load(path=creds_file)
    assert persisted is not None
    assert persisted.access_token == "REFRESHED_ATK"

    assert api_call_count >= 5, (
        f"expected ≥5 api calls (initial 401s), got {api_call_count}"
    )


# ---------------------------------------------------------------------------
# Test 4 — corrupted credentials
# ---------------------------------------------------------------------------


def test_credentials_corrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """store.load() raises CredentialsCorruptError when xai.json contains partial JSON."""
    # Arrange — write malformed JSON to the credentials file
    xai_creds = tmp_path / "xai.json"
    xai_creds.write_text('{"access_token": "abc"')  # truncated — no closing brace
    monkeypatch.setattr("llmcli.auth.store.XAI_CREDENTIALS_PATH", xai_creds)

    # Act + Assert — CredentialsCorruptError raised
    with pytest.raises(CredentialsCorruptError) as exc_info:
        auth_store.load(path=xai_creds)

    # Assert — error message contains "corrupted" and "re-run"
    message = str(exc_info.value).lower()
    assert "corrupted" in message, f"expected 'corrupted' in error message, got: {message!r}"
    assert "re-run" in message, f"expected 're-run' in error message, got: {message!r}"


# ---------------------------------------------------------------------------
# Test 5 — AC14: credentials repr redacts tokens  (F5)
# ---------------------------------------------------------------------------


def test_credentials_repr_redacts_tokens() -> None:
    """AC14: XaiCredentials.__repr__ MUST redact access/refresh/id tokens."""
    creds = XaiCredentials(
        access_token="eyJhbGciOiJIUzI1NiJ9.payload.sig",  # JWT-like prefix
        refresh_token="xai-refresh-abc123",
        id_token="eyJhbGciOiJIUzI1NiJ9.id.sig",
        expires_at=9999999,
        token_type="Bearer",
        scope="openid",
    )
    r = repr(creds)
    assert "eyJ" not in r, f"JWT prefix leaked in repr: {r}"
    assert "xai-refresh" not in r, f"refresh token leaked in repr: {r}"
    assert "access_token=***" in r
    assert "refresh_token=***" in r
    assert "id_token=***" in r
    assert "expires_at=9999999" in r  # non-secret int is fine


# ---------------------------------------------------------------------------
# Test 6 — empty verifier guard  (F6)
# ---------------------------------------------------------------------------


def test_exchange_code_empty_verifier_raises() -> None:
    """`exchange_code` MUST reject empty verifier — guard at xai_oauth.py."""
    with pytest.raises(ValueError, match="PKCE code_verifier"):
        exchange_code(code="some_code", verifier="")


# ---------------------------------------------------------------------------
# Test 7 — AC4 second half: /health returns logged_in:false on corrupt creds  (F7)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_credentials_corrupted_health_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test-AC4 second half: forwarder /health reports logged_in:false when xai.json is corrupted."""
    from aiohttp.test_utils import TestClient, TestServer

    from llmcli.proxy_forwarder._server import create_app
    from llmcli.proxy_forwarder.xai_adapter import XaiAdapter

    creds_file = tmp_path / "xai.json"
    creds_file.write_text('{"access_token": "abc"')  # partial JSON

    adapter = XaiAdapter()
    monkeypatch.setattr(adapter, "credential_path", creds_file)

    app = create_app(adapter)
    async with TestClient(TestServer(app)) as client:
        resp = await client.get("/health")
        assert resp.status == 200
        body = await resp.json()
        assert body["logged_in"] is False


# ---------------------------------------------------------------------------
# Test 8 — lazy_retry still 401 propagates  (F8)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lazy_refresh_still_401_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the retry is ALSO 401, lazy_retry_on_401 returns the 401 (caller handles X-Llmcli-Reauth)."""
    creds_file = tmp_path / "xai.json"
    monkeypatch.setattr("llmcli.auth.store.XAI_CREDENTIALS_PATH", creds_file)
    monkeypatch.setattr("llmcli.proxy_forwarder._common._REFRESH_LOCK", asyncio.Lock())

    old = XaiCredentials(
        access_token="OLD_ATK",
        refresh_token="OLD_RTK",
        id_token="OLD_ITK",
        expires_at=int(time.time()) - 60,  # expired
        token_type="Bearer",
        scope="openid",
    )
    auth_store.save(old, path=creds_file)

    api_calls = 0

    async def request_fn(token: str) -> MagicMock:
        nonlocal api_calls
        api_calls += 1
        resp = MagicMock()
        resp.status = 401
        return resp

    async def refresh_fn(creds: XaiCredentials) -> XaiCredentials:
        # Returns NEW creds — but the api will STILL return 401
        return XaiCredentials(
            access_token="NEW_BUT_STILL_REJECTED",
            refresh_token="NEW_RTK",
            id_token="NEW_ITK",
            expires_at=int(time.time()) + 3600,
            token_type="Bearer",
            scope="openid",
        )

    resp = await lazy_retry_on_401(
        request_fn,
        refresh_fn,
        lambda: auth_store.load(path=creds_file),
        lambda c: auth_store.save(c, path=creds_file),
    )
    assert resp.status == 401, f"expected final 401, got {resp.status}"
    assert api_calls == 2, f"expected 2 api calls (initial + retry), got {api_calls}"
