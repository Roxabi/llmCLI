"""OAuth credential management — re-exports shared ``roxabi_xai`` (inference monorepo).

Canonical implementation: ``Roxabi/roxabi-inference`` package ``roxabi-xai``.
This module keeps import paths stable for the xAI forwarder and ``llmcli xai`` CLI.
"""

from roxabi_xai import (
    CredentialsCorruptError,
    ReauthRequired,
    XaiCredentials,
    load,
    login_flow,
    refresh_credentials,
    save,
)
from roxabi_xai import store
from roxabi_xai.store import XAI_CREDENTIALS_PATH

__all__ = [
    "XaiCredentials",
    "CredentialsCorruptError",
    "ReauthRequired",
    "XAI_CREDENTIALS_PATH",
    "load",
    "save",
    "login_flow",
    "refresh_credentials",
    "store",
]
