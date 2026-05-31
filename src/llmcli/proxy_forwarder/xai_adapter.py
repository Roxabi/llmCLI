"""xAI provider adapter for the llmCLI OAuth forwarder.

Implements ForwardAdapter for a single-account SuperGrok credential.
The adapter is intentionally thin: it holds only provider-specific
configuration (api_base, credential_path) and delegates to the auth
module for token refresh.
"""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

from llmcli.auth import store
from llmcli.auth.store import CredentialsCorruptError, XAI_CREDENTIALS_PATH, XaiCredentials
from llmcli.auth.xai_oauth import refresh_credentials

from ._common import _Resp401, lazy_retry_on_401


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
            return {
                "status": "ok",
                "logged_in": creds is not None,
                "expires_at": creds.expires_at if creds else None,
            }
        except CredentialsCorruptError:
            return {"status": "ok", "logged_in": False, "expires_at": None}

    async def refresh(self, creds: XaiCredentials) -> XaiCredentials:
        """Refresh *creds* via the xAI token endpoint.

        Delegates to the synchronous ``refresh_credentials`` function via
        ``run_in_executor`` to avoid blocking the event loop during the
        HTTPS POST to auth.x.ai.

        Parameters
        ----------
        creds:
            Credentials whose ``refresh_token`` will be exchanged for a new
            ``access_token``.

        Returns
        -------
        XaiCredentials
            Fresh credentials with updated ``access_token`` and ``expires_at``.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, refresh_credentials, creds)
