"""xAI provider adapter for the llmCLI OAuth forwarder.

Implements OAuthAdapter for a single-account SuperGrok credential.
The adapter is intentionally thin: it holds only provider-specific
configuration (api_base, credential_path) and delegates to the auth
module for token refresh.
"""

from __future__ import annotations

import asyncio

from llmcli.auth.store import XAI_CREDENTIALS_PATH, XaiCredentials
from llmcli.auth.xai_oauth import refresh_credentials


class XaiAdapter:
    """OAuthAdapter implementation for xAI / SuperGrok OAuth credentials.

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
