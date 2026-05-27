"""xAI OAuth PKCE flow for llmCLI.

Ported from hermes_cli/auth.py (Hermes Agent). Omits multi-account
credential_pool — single-account only.
"""
import base64
import hashlib
import hmac
import logging
import secrets
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from llmcli.auth.store import XaiCredentials, XAI_CREDENTIALS_PATH, save

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — load-bearing values, do NOT modify without testing against xAI
# ---------------------------------------------------------------------------
XAI_OAUTH_ISSUER = "https://auth.x.ai"
XAI_OAUTH_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_OAUTH_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_OAUTH_REDIRECT_PORT = 56121
XAI_OAUTH_REDIRECT_PATH = "/callback"
XAI_OAUTH_PLAN = "generic"  # load-bearing — DO NOT remove

_XAI_REDIRECT_HOST = "127.0.0.1"
_XAI_TOKEN_ENDPOINT = f"{XAI_OAUTH_ISSUER}/oauth/token"
_XAI_AUTHORIZE_ENDPOINT = f"{XAI_OAUTH_ISSUER}/oauth/authorize"


# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------

def _s256_challenge(verifier: str) -> str:
    """Compute the S256 PKCE code_challenge from a verifier string."""
    return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")


@dataclass(frozen=True)
class PkceVerifier:
    """PKCE parameters for one OAuth session."""

    code_verifier: str
    code_challenge: str
    state: str
    nonce: str


def _generate_pkce_verifier() -> PkceVerifier:
    """Generate a fresh PKCE verifier using S256 challenge method."""
    verifier = secrets.token_urlsafe(64)
    state = secrets.token_hex(16)
    nonce = secrets.token_hex(16)
    return PkceVerifier(
        code_verifier=verifier,
        code_challenge=_s256_challenge(verifier),
        state=state,
        nonce=nonce,
    )


def _build_authorize_url(challenge: str, state: str, nonce: str) -> str:
    """Build the xAI OAuth authorization URL.

    MUST include plan=generic — without it auth.x.ai rejects loopback OAuth
    from non-allowlisted clients (ref: Hermes auth.py line 6393).
    """
    redirect_uri = (
        f"http://{_XAI_REDIRECT_HOST}:{XAI_OAUTH_REDIRECT_PORT}{XAI_OAUTH_REDIRECT_PATH}"
    )
    params = {
        "response_type": "code",
        "client_id": XAI_OAUTH_CLIENT_ID,
        "redirect_uri": redirect_uri,
        "scope": XAI_OAUTH_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "nonce": nonce,
        "plan": XAI_OAUTH_PLAN,
    }
    return f"{_XAI_AUTHORIZE_ENDPOINT}?{urlencode(params)}"


# ---------------------------------------------------------------------------
# Loopback callback server
# ---------------------------------------------------------------------------

def _loopback_server(timeout: float = 120.0) -> tuple[str, str]:
    """Run a single-request HTTP server on 127.0.0.1:56121.

    Blocks until the /callback request arrives (or timeout).
    Returns (code, state) extracted from the redirect URL.
    Raises RuntimeError on timeout or OAuth error.
    """
    result: dict = {"code": None, "state": None, "error": None}
    result_lock = threading.Lock()
    expected_path = XAI_OAUTH_REDIRECT_PATH

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != expected_path:
                self.send_response(404)
                self.end_headers()
                return
            params = parse_qs(parsed.query)
            incoming = {
                "code": params.get("code", [None])[0],
                "state": params.get("state", [None])[0],
                "error": params.get("error", [None])[0],
            }
            with result_lock:
                if not (result["code"] or result["error"]):
                    result.update(incoming)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            if incoming.get("error"):
                body = b"<html><body><h1>xAI authorization failed.</h1>Close this tab.</body></html>"
            else:
                body = b"<html><body><h1>xAI authorization received.</h1>Close this tab.</body></html>"
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 — match BaseHTTPRequestHandler signature
            pass  # silence request logs

    class _ReuseHTTPServer(HTTPServer):
        allow_reuse_address = True

    server = _ReuseHTTPServer((_XAI_REDIRECT_HOST, XAI_OAUTH_REDIRECT_PORT), _CallbackHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    thread.start()

    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            if result["code"] or result["error"]:
                break
            time.sleep(0.1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    if result.get("error"):
        raise RuntimeError(f"xAI authorization failed: {result['error']}")
    if not result["code"]:
        raise RuntimeError("xAI authorization timed out waiting for callback.")

    return str(result["code"]), str(result["state"] or "")


# ---------------------------------------------------------------------------
# Token exchange and refresh
# ---------------------------------------------------------------------------

def exchange_code(code: str, verifier: str) -> XaiCredentials:
    """Exchange an authorization code for tokens via POST to auth.x.ai/oauth/token.

    Also sends code_challenge + code_challenge_method as defense-in-depth
    (xAI validates challenge at token step, not only at authorize step).
    """
    if not verifier:
        raise ValueError("PKCE code_verifier is required for token exchange.")

    redirect_uri = (
        f"http://{_XAI_REDIRECT_HOST}:{XAI_OAUTH_REDIRECT_PORT}{XAI_OAUTH_REDIRECT_PATH}"
    )
    # Recompute challenge from verifier for the defense-in-depth parameter
    challenge = _s256_challenge(verifier)

    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": XAI_OAUTH_CLIENT_ID,
        "code_verifier": verifier,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    response = httpx.post(
        _XAI_TOKEN_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        logger.debug("xai token endpoint error body: %s", response.text)
        raise RuntimeError(f"xAI token exchange failed (HTTP {response.status_code})")
    payload = response.json()
    expires_in = int(payload.get("expires_in") or 3600)
    return XaiCredentials(
        access_token=str(payload.get("access_token", "")).strip(),
        refresh_token=str(payload.get("refresh_token", "")).strip(),
        id_token=str(payload.get("id_token", "") or "").strip(),
        expires_at=int(time.time()) + expires_in,
        token_type=str(payload.get("token_type") or "Bearer").strip() or "Bearer",
        scope=str(payload.get("scope", "") or "").strip(),
    )


def refresh_credentials(creds: XaiCredentials) -> XaiCredentials:
    """Refresh an expired access_token using the stored refresh_token.

    Returns a new XaiCredentials with updated tokens.
    Raises RuntimeError on 4xx (e.g. refresh_token expired after >30d offline).
    """
    response = httpx.post(
        _XAI_TOKEN_ENDPOINT,
        data={
            "grant_type": "refresh_token",
            "client_id": XAI_OAUTH_CLIENT_ID,
            "refresh_token": creds.refresh_token,
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=30.0,
    )
    if response.status_code != 200:
        logger.debug("xai token endpoint error body: %s", response.text)
        raise RuntimeError(f"xAI token refresh failed (HTTP {response.status_code})")
    payload = response.json()
    expires_in = int(payload.get("expires_in") or 3600)
    return XaiCredentials(
        access_token=str(payload.get("access_token", "")).strip(),
        refresh_token=str(payload.get("refresh_token") or creds.refresh_token).strip(),
        id_token=str(payload.get("id_token", "") or "").strip(),
        expires_at=int(time.time()) + expires_in,
        token_type=str(payload.get("token_type") or "Bearer").strip() or "Bearer",
        scope=str(payload.get("scope", "") or creds.scope).strip(),
    )


# ---------------------------------------------------------------------------
# Login flow orchestrator
# ---------------------------------------------------------------------------

def login_flow() -> XaiCredentials:
    """Run the full xAI OAuth PKCE login flow.

    1. Generate PKCE verifier + challenge
    2. Spawn loopback listener on :56121
    3. Open browser at auth.x.ai/oauth/authorize?plan=generic&...
    4. Wait for /callback?code=...&state=...
    5. Verify state, exchange code for tokens
    6. Persist credentials to XAI_CREDENTIALS_PATH (0600)
    7. Return XaiCredentials
    """
    pkce = _generate_pkce_verifier()
    authorize_url = _build_authorize_url(pkce.code_challenge, pkce.state, pkce.nonce)

    print(f"Opening browser at:\n  {authorize_url}")
    print(f"(loopback listener on {_XAI_REDIRECT_HOST}:{XAI_OAUTH_REDIRECT_PORT})")
    print()

    try:
        webbrowser.open(authorize_url)
    except Exception:
        print("Could not open browser automatically — open the URL above manually.")

    print("Waiting for authorization callback (timeout 120s)…")
    code, received_state = _loopback_server(timeout=120.0)

    if not hmac.compare_digest(received_state, pkce.state):
        raise RuntimeError("OAuth state mismatch — possible CSRF")

    print("Authorization code received, exchanging for tokens…")
    creds = exchange_code(code, pkce.code_verifier)

    save(creds, XAI_CREDENTIALS_PATH)

    print(f"Logged in. Credentials stored at {XAI_CREDENTIALS_PATH}")
    print(f"  expires_at: {creds.expires_at}")
    print("  refresh_token: stored (long-lived)")

    return creds
