"""RED-phase tests for xAI OAuth PKCE flow and credential store.

Tests 1 and 4 (test_pkce_code_exchange, test_credentials_corrupted) exercise
auth/xai_oauth.py + auth/store.py from Wave 1 and are expected to PASS after
backend-dev-A completes T2 + T3.

Tests 2 and 3 (test_lazy_refresh_on_401, test_concurrent_refresh_dedup) exercise
proxy_forwarder/_common.py from Wave 2 and are SKIPPED until that module lands
(import-guarded via importlib.util.find_spec at module level).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Wave 1 import guard — auth modules (required for T1/T4 at RG1)
# findspec on a submodule of a non-existent parent raises ModuleNotFoundError,
# so we guard with a try/except rather than find_spec directly.
# ---------------------------------------------------------------------------

try:
    from llmcli.auth.store import CredentialsCorruptError, XaiCredentials
    from llmcli.auth import store as auth_store

    _AUTH_STORE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _AUTH_STORE_AVAILABLE = False

try:
    from llmcli.auth.xai_oauth import (
        XAI_OAUTH_CLIENT_ID,
        XAI_OAUTH_PLAN,
        _build_authorize_url,
        exchange_code,
    )

    _AUTH_OAUTH_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _AUTH_OAUTH_AVAILABLE = False

_AUTH_AVAILABLE = _AUTH_STORE_AVAILABLE and _AUTH_OAUTH_AVAILABLE

# ---------------------------------------------------------------------------
# Wave 2 import guard — proxy_forwarder._common (required for T2/T3 at RG2)
# ---------------------------------------------------------------------------

try:
    from llmcli.proxy_forwarder._common import _REFRESH_LOCK, lazy_retry_on_401  # noqa: F401

    _FORWARDER_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _FORWARDER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def creds_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Monkeypatch XAI_CREDENTIALS_PATH to a temp file; return the path."""
    xai_creds = tmp_path / "xai.json"
    if _AUTH_STORE_AVAILABLE:
        monkeypatch.setattr("llmcli.auth.store.XAI_CREDENTIALS_PATH", xai_creds)
    return xai_creds


def _make_fake_creds(
    access_token: str = "ATK",
    refresh_token: str = "RTK",
    id_token: str = "ITK",
    expires_delta: int = 3600,
) -> "XaiCredentials":
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


@pytest.mark.skipif(not _AUTH_AVAILABLE, reason="auth modules not yet implemented (Wave 1)")
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

    # Assert — returned XaiCredentials matches mock response
    assert result.access_token == "ATK"
    assert result.refresh_token == "RTK"
    assert result.id_token == "ITK"
    # expires_at must be roughly time.time() + 3600 (±5 s tolerance)
    assert before + 3595 <= result.expires_at <= after + 3605, (
        f"expires_at={result.expires_at} not in expected range"
    )

    # Assert — authorize URL contains plan=generic (build helper)
    assert XAI_OAUTH_PLAN == "generic"
    auth_url = _build_authorize_url(
        challenge="challenge_abc",
        state="state_xyz",
        nonce="nonce_123",
    )
    assert "plan=generic" in auth_url, f"plan=generic missing from URL: {auth_url}"


# ---------------------------------------------------------------------------
# Test 2 — lazy 401 retry (Wave 2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FORWARDER_AVAILABLE, reason="proxy_forwarder not yet implemented (Wave 2)")
@pytest.mark.asyncio
async def test_lazy_refresh_on_401(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """lazy_retry_on_401 retries once on 401; new credentials persisted to disk."""
    # Arrange — initial (expired) credentials on disk
    creds_file = tmp_path / "xai.json"
    monkeypatch.setattr("llmcli.auth.store.XAI_CREDENTIALS_PATH", creds_file)

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

    async def request_fn(token: str) -> MagicMock:
        nonlocal api_call_count
        api_call_count += 1
        resp = MagicMock()
        if api_call_count == 1:
            resp.status = 401
        else:
            resp.status = 200
            resp.json = asyncio.coroutine(lambda: chat_ok_resp)  # type: ignore[attr-defined]
        return resp

    async def refresh_fn(creds: "XaiCredentials") -> "XaiCredentials":
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


# ---------------------------------------------------------------------------
# Test 3 — concurrent 401 dedup (Wave 2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FORWARDER_AVAILABLE, reason="proxy_forwarder not yet implemented (Wave 2)")
@pytest.mark.asyncio
async def test_concurrent_refresh_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5 concurrent 401s result in exactly ONE refresh POST (asyncio.Lock dedup)."""
    # Arrange — expired credentials on disk
    creds_file = tmp_path / "xai.json"
    monkeypatch.setattr("llmcli.auth.store.XAI_CREDENTIALS_PATH", creds_file)

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

    async def refresh_fn(creds: "XaiCredentials") -> "XaiCredentials":
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


# ---------------------------------------------------------------------------
# Test 4 — corrupted credentials
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AUTH_STORE_AVAILABLE, reason="auth.store not yet implemented (Wave 1)")
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
