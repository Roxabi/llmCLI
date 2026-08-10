"""Back-compat shim — credentials live in ``roxabi_xai.store``."""

from roxabi_xai.store import (  # noqa: F401
    CREDENTIALS_DIR,
    XAI_CREDENTIALS_PATH,
    CredentialsCorruptError,
    ReauthRequired,
    XaiCredentials,
    load,
    resolve_credentials_path,
    save,
)

__all__ = [
    "CREDENTIALS_DIR",
    "XAI_CREDENTIALS_PATH",
    "CredentialsCorruptError",
    "ReauthRequired",
    "XaiCredentials",
    "load",
    "save",
    "resolve_credentials_path",
]
