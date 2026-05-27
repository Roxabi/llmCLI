"""Credential storage for xAI OAuth tokens.

Atomic-write with fcntl flock. Credentials dir mode 0700, file mode 0600.
"""
import fcntl
import json
import os
from dataclasses import dataclass
from pathlib import Path

CREDENTIALS_DIR = Path.home() / ".roxabi" / "llmcli" / "credentials"
XAI_CREDENTIALS_PATH = CREDENTIALS_DIR / "xai.json"


class CredentialsCorruptError(RuntimeError):
    """Raised when credentials JSON cannot be parsed."""


@dataclass(frozen=True)
class XaiCredentials:
    """Immutable OAuth credentials for xAI / SuperGrok."""

    access_token: str
    refresh_token: str
    id_token: str
    expires_at: int  # unix epoch seconds
    token_type: str = "Bearer"
    scope: str = ""

    def __repr__(self) -> str:
        return (
            f"XaiCredentials(access_token=***, refresh_token=***, "
            f"id_token=***, expires_at={self.expires_at}, "
            f"token_type={self.token_type!r}, scope={self.scope!r})"
        )

    def __str__(self) -> str:
        return self.__repr__()


def load(path: Path = XAI_CREDENTIALS_PATH) -> XaiCredentials | None:
    """Load credentials from *path*.

    Returns None if the file does not exist.
    Raises CredentialsCorruptError if the JSON cannot be parsed.
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CredentialsCorruptError(
            f"credentials corrupted at {path} — re-run `llmcli xai login`"
        ) from exc
    return XaiCredentials(**data)


def save(creds: XaiCredentials, path: Path = XAI_CREDENTIALS_PATH) -> None:
    """Atomically write *creds* to *path*.

    Parent directory is created with mode 0700 if absent.
    Uses an exclusive flock + os.replace for atomicity; final file mode 0600.
    """
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_suffix(".tmp")
    payload = {
        "access_token": creds.access_token,
        "refresh_token": creds.refresh_token,
        "id_token": creds.id_token,
        "expires_at": creds.expires_at,
        "token_type": creds.token_type,
        "scope": creds.scope,
    }
    with open(tmp, "w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        json.dump(payload, f)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)  # atomic rename
