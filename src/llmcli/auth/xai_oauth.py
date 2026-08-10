"""Back-compat shim — OAuth flow lives in ``roxabi_xai.oauth``.

Re-exports public *and* test-facing symbols (PKCE helpers, constants) so
existing llmCLI tests keep working without forking hermes-derived flow.

``login_flow`` syncs monkeypatched shim attributes into ``roxabi_xai.oauth``
before calling through (tests patch ``llmcli.auth.xai_oauth.*``).
"""

from __future__ import annotations

import roxabi_xai.oauth as _oauth
from roxabi_xai.oauth import (  # noqa: F401
    API_BASE,
    XAI_OAUTH_CLIENT_ID,
    XAI_OAUTH_ISSUER,
    XAI_OAUTH_PLAN,
    XAI_OAUTH_REDIRECT_PATH,
    XAI_OAUTH_REDIRECT_PORT,
    XAI_OAUTH_SCOPE,
    PkceVerifier,
    _build_authorize_url,
    _generate_pkce_verifier,
    _parse_manual_input,
    _s256_challenge,
    exchange_code,
    refresh_credentials,
)
from roxabi_xai.store import XAI_CREDENTIALS_PATH  # noqa: F401 — tests monkeypatch this


def login_flow(manual: bool = False):
    """Delegate to roxabi_xai after applying any test monkeypatches on this module."""
    # Tests patch these names on *this* module; push them into the implementation.
    _oauth.XAI_CREDENTIALS_PATH = XAI_CREDENTIALS_PATH
    _oauth._generate_pkce_verifier = _generate_pkce_verifier
    return _oauth.login_flow(manual=manual)


__all__ = [
    "API_BASE",
    "XAI_CREDENTIALS_PATH",
    "XAI_OAUTH_CLIENT_ID",
    "XAI_OAUTH_ISSUER",
    "XAI_OAUTH_PLAN",
    "XAI_OAUTH_REDIRECT_PATH",
    "XAI_OAUTH_REDIRECT_PORT",
    "XAI_OAUTH_SCOPE",
    "PkceVerifier",
    "_build_authorize_url",
    "_generate_pkce_verifier",
    "_parse_manual_input",
    "_s256_challenge",
    "exchange_code",
    "login_flow",
    "refresh_credentials",
]
