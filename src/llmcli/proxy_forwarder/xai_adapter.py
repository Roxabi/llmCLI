"""xAI provider adapter for the llmCLI OAuth forwarder.

Implements ForwardAdapter for a single-account SuperGrok credential.
The adapter is intentionally thin: it holds only provider-specific
configuration (api_base, credential_path) and delegates to the auth
module for token refresh.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from llmcli.auth import store
from llmcli.auth.store import (
    CredentialsCorruptError,
    ReauthRequired,
    XAI_CREDENTIALS_PATH,
    XaiCredentials,
)
from llmcli.auth.xai_oauth import refresh_credentials

from ._common import _Resp401, lazy_retry_on_401

logger = logging.getLogger(__name__)


class XaiAdapter:
    """ForwardAdapter implementation for xAI / SuperGrok OAuth credentials.

    Attributes
    ----------
    api_base:
        xAI host-only base URL. The ``/v1/...`` prefix lives on the inbound
        ``path`` (see ALLOWED_PATHS), so concatenation with ``api_base`` MUST
        NOT double it.
    credential_path:
        Path to the JSON file written by ``llmcli xai login``.
    """

    api_base: str = "https://api.x.ai"
    credential_path = XAI_CREDENTIALS_PATH

    def __init__(self) -> None:
        # Flipped once a 4xx refresh proves the refresh token is dead. While set,
        # the adapter backs off (no re-POST to auth.x.ai) and /health reports 503.
        # Cleared when the operator re-auths (refresh_token on disk changes).
        self._reauth_required: bool = False
        self._dead_refresh_token: str | None = None

    def _sync_reauth_state(self, creds: XaiCredentials | None) -> None:
        """Clear the re-auth flag once the operator has re-authed (token changed)."""
        if (
            self._reauth_required
            and creds is not None
            and creds.refresh_token != self._dead_refresh_token
        ):
            logger.info("xai forwarder: fresh credentials detected — re-auth flag cleared")
            self._reauth_required = False
            self._dead_refresh_token = None

    def transform_request(self, body: bytes, path: str) -> bytes:  # noqa: ARG002
        """Return body unchanged — xAI does not require request body mutation."""
        return body

    def extra_headers(self) -> dict[str, str]:
        """Return STATIC per-provider headers — xAI requires none at this level.

        MUST NOT include ``Authorization`` — auth is injected inside ``execute()``.
        """
        return {}

    async def execute(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        body: bytes,
        headers: dict[str, str],
    ) -> aiohttp.ClientResponse | _Resp401:
        """Perform the upstream request with OAuth single-flight retry on 401.

        Authorization header is injected per-call from the current token.
        On a 401 response, refreshes the token once (under a process-level
        lock) and retries.
        """

        def request_fn(token: str):
            return session.request(
                method,
                url,
                data=body if body else None,
                headers={**headers, "Authorization": f"Bearer {token}"},
                allow_redirects=False,
            )

        return await lazy_retry_on_401(
            request_fn,
            self.refresh,
            lambda: store.load(self.credential_path),
            lambda c: store.save(c, self.credential_path),
        )

    async def health(self) -> dict[str, Any]:
        """Return credential-presence health payload for GET /health.

        Returns
        -------
        dict
            Always contains ``status``, ``logged_in``, and ``expires_at``.
            ``logged_in`` is False when credentials are absent or corrupted.
        """
        try:
            creds = store.load(self.credential_path)
        except CredentialsCorruptError:
            return {
                "status": "corrupted",
                "logged_in": False,
                "expires_at": None,
                "reauth_required": self._reauth_required,
            }
        self._sync_reauth_state(creds)
        return {
            "status": "reauth_required" if self._reauth_required else "ok",
            "logged_in": creds is not None,
            "expires_at": creds.expires_at if creds else None,
            "reauth_required": self._reauth_required,
        }

    async def refresh(self, creds: XaiCredentials) -> XaiCredentials:
        """Refresh *creds* via the xAI token endpoint (off-loop executor).

        Backs off when the refresh token is already known-dead: re-POSTing a
        rejected token on every request hammers auth.x.ai and starves the event
        loop — the silent 42 h failure mode of issue #114. On a fresh 4xx, logs
        ERROR once and flips the re-auth flag; on success, clears it.

        Raises
        ------
        ReauthRequired
            When the refresh token is rejected (4xx), now or previously.
        """
        # Back-off — refresh token already proven dead and unchanged on disk.
        if self._reauth_required and creds.refresh_token == self._dead_refresh_token:
            raise ReauthRequired(
                "xAI re-auth required — run `llmcli xai login` (cached; not re-POSTing)"
            )
        loop = asyncio.get_running_loop()
        try:
            new = await loop.run_in_executor(None, refresh_credentials, creds)
        except ReauthRequired:
            if not self._reauth_required:
                logger.error(
                    "RE-AUTH REQUIRED: xAI rejected the refresh token (HTTP 4xx). "
                    "Run `llmcli xai login` on this host — serving 503 until re-auth."
                )
            self._reauth_required = True
            self._dead_refresh_token = creds.refresh_token
            raise
        self._reauth_required = False
        self._dead_refresh_token = None
        return new
